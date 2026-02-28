from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.dataset.token_dataset import (
    TokenDatasetConfig,
    TokenPairDataset,
    load_tokens,
    token_pair_collate,
)
from src.model.embedder import TokenEmbedder, TokenEmbedderConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline contrastive embedder.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to training config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run training on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/checkpoints"),
        help="Where to store checkpoints and train summary.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save epoch checkpoint every N epochs.",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Optional limit for quick smoke runs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for data.batch_size from config.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for data.num_workers from config.",
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected mapping in {path}, got: {type(data)}")
    return data


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(device_flag: str) -> torch.device:
    if device_flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_flag == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_flag)


def _build_datasets(config: dict[str, Any]) -> tuple[TokenPairDataset, TokenPairDataset | None]:
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    seed = int(config.get("seed", 42))

    tokens_root = Path(data_cfg.get("tokens_root", "data/tokens"))
    splits_root = Path(data_cfg.get("splits_root", "data/splits"))
    min_crop_ratio = float(train_cfg.get("min_crop_ratio", 0.6))
    max_crop_ratio = float(train_cfg.get("max_crop_ratio", 1.0))
    dropout_prob = float(train_cfg.get("dropout_prob", 0.02))
    pad_value = int(train_cfg.get("pad_value", 0))

    train_split = Path(data_cfg.get("train_split", splits_root / "train.txt"))
    val_split = Path(data_cfg.get("val_split", splits_root / "val.txt"))

    train_dataset = TokenPairDataset(
        TokenDatasetConfig(
            tokens_root=tokens_root,
            split_file=train_split,
            seed=seed,
            min_crop_ratio=min_crop_ratio,
            max_crop_ratio=max_crop_ratio,
            dropout_prob=dropout_prob,
            pad_value=pad_value,
        )
    )

    val_dataset: TokenPairDataset | None = None
    if val_split.exists():
        val_dataset = TokenPairDataset(
            TokenDatasetConfig(
                tokens_root=tokens_root,
                split_file=val_split,
                seed=seed + 1,
                min_crop_ratio=min_crop_ratio,
                max_crop_ratio=max_crop_ratio,
                dropout_prob=dropout_prob,
                pad_value=pad_value,
            )
        )
    return train_dataset, val_dataset


def _infer_quantizers(dataset: TokenPairDataset) -> int:
    sample_track = dataset.track_ids[0]
    sample_file = dataset.track_to_files[sample_track][0]
    sample_tokens = load_tokens(sample_file)
    if sample_tokens.dim() != 2:
        raise RuntimeError(f"Expected [Q,T] tokens, got shape={tuple(sample_tokens.shape)}")
    return int(sample_tokens.shape[0])


def _build_model(config: dict[str, Any], dataset: TokenPairDataset) -> tuple[TokenEmbedder, TokenEmbedderConfig]:
    model_cfg = config.get("model", {})
    inferred_q = _infer_quantizers(dataset)
    embedder_cfg = TokenEmbedderConfig(
        vocab_size=int(model_cfg.get("vocab_size", 4096)),
        max_quantizers=max(int(model_cfg.get("max_quantizers", inferred_q)), inferred_q),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    return TokenEmbedder(embedder_cfg), embedder_cfg


def _nt_xent_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, temperature: float) -> torch.Tensor:
    if emb_a.shape != emb_b.shape:
        raise ValueError(f"Embedding shapes mismatch: {emb_a.shape} vs {emb_b.shape}")
    if emb_a.ndim != 2:
        raise ValueError(f"Expected [B,D] embeddings, got {emb_a.shape}")
    if emb_a.shape[0] < 2:
        raise ValueError("Batch size must be >= 2 for contrastive loss.")

    z = torch.cat([emb_a, emb_b], dim=0)  # [2B,D]
    logits = torch.matmul(z, z.t()) / temperature

    n = emb_a.shape[0]
    diag_mask = torch.eye(2 * n, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(diag_mask, -1e9)

    targets = torch.arange(2 * n, device=logits.device)
    targets = (targets + n) % (2 * n)
    return F.cross_entropy(logits, targets)


def _run_epoch(
    model: TokenEmbedder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    temperature: float,
    max_steps: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    start = time.perf_counter()
    total_loss = 0.0
    total_steps = 0

    for step, batch in enumerate(loader, start=1):
        if max_steps is not None and step > max_steps:
            break

        tokens_a = batch["tokens_a"].to(device, non_blocking=True)
        tokens_b = batch["tokens_b"].to(device, non_blocking=True)
        mask_a = batch["mask_a"].to(device, non_blocking=True)
        mask_b = batch["mask_b"].to(device, non_blocking=True)

        if tokens_a.shape[0] < 2:
            continue

        try:
            emb_a = model(tokens_a, mask_a)
            emb_b = model(tokens_b, mask_b)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda error" in msg:
                raise RuntimeError(
                    "Training failed due to GPU memory pressure. "
                    "Try smaller batch size: --batch-size 2 or 4, and/or "
                    "shorter chunks during tokenization (--chunk-seconds 8)."
                ) from exc
            raise
        loss = _nt_xent_loss(emb_a, emb_b, temperature)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1

    elapsed = time.perf_counter() - start
    mean_loss = total_loss / max(total_steps, 1)
    return {
        "loss": mean_loss,
        "steps": float(total_steps),
        "seconds": elapsed,
        "steps_per_sec": float(total_steps / elapsed) if elapsed > 0 else 0.0,
    }


def _save_checkpoint(
    path: Path,
    model: TokenEmbedder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: TokenEmbedderConfig,
    raw_config: dict[str, Any],
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": raw_config,
            "history": history,
        },
        path,
    )


def main() -> None:
    args = _parse_args()
    raw_config = _read_yaml(args.config)
    seed = int(raw_config.get("seed", 42))
    _set_seed(seed)

    device = _resolve_device(args.device)
    train_dataset, val_dataset = _build_datasets(raw_config)
    model, model_config = _build_model(raw_config, train_dataset)
    model = model.to(device)

    data_cfg = raw_config.get("data", {})
    train_cfg = raw_config.get("train", {})
    config_batch_size = int(data_cfg.get("batch_size", 64))
    config_num_workers = int(data_cfg.get("num_workers", 4))
    batch_size = int(args.batch_size) if args.batch_size is not None else config_batch_size
    num_workers = int(args.num_workers) if args.num_workers is not None else config_num_workers
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    # Colab/containers often expose fewer effective CPUs than os.cpu_count reports.
    # Cap workers conservatively to reduce freezes/warnings.
    cpu_limit = max(1, (os.cpu_count() or 2))
    num_workers = min(num_workers, cpu_limit)
    temperature = float(train_cfg.get("temperature", 0.07))
    epochs = int(train_cfg.get("epochs", 10))
    learning_rate = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=token_pair_collate,
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=token_pair_collate,
            drop_last=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_val = float("inf")

    print(f"Training on device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Val samples:   {len(val_dataset)}")
    print(f"Batch size:    {batch_size}")
    print(f"Num workers:   {num_workers}")
    print(f"Epochs:        {epochs}")

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            temperature=temperature,
            max_steps=args.max_steps_per_epoch,
        )
        log_entry = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_steps_per_sec": train_metrics["steps_per_sec"],
        }

        line = (
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_steps_per_sec={train_metrics['steps_per_sec']:.2f}"
        )

        if val_loader is not None:
            with torch.inference_mode():
                val_metrics = _run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    temperature=temperature,
                    max_steps=args.max_steps_per_epoch,
                )
            log_entry["val_loss"] = val_metrics["loss"]
            line += f" val_loss={val_metrics['loss']:.4f}"
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                _save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    model_config,
                    raw_config,
                    history + [log_entry],
                )

        history.append(log_entry)
        print(line)

        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_checkpoint(
                args.output_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                model_config,
                raw_config,
                history,
            )

    _save_checkpoint(
        args.output_dir / "last.pt",
        model,
        optimizer,
        epochs,
        model_config,
        raw_config,
        history,
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Saved checkpoints and history to: {args.output_dir}")


if __name__ == "__main__":
    main()
