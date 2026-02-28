from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.dataset.augmentations import TokenAugmentConfig, make_augmented_view
from src.dataset.token_dataset import (
    build_track_index,
    load_tokens,
    load_track_ids,
    pad_token_batch,
)
from src.model.embedder import TokenEmbedder, TokenEmbedderConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline retrieval evaluation for token embedder.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Training/model config path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional model checkpoint from train_contrastive.py.",
    )
    parser.add_argument(
        "--tokens-root",
        type=Path,
        default=Path("data/tokens"),
        help="Directory with token files.",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path("data/splits"),
        help="Directory with split txt files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--topk",
        type=str,
        default="1,10,100",
        help="Comma-separated K values for Recall@K.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Embedding device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic query augmentation.",
    )
    parser.add_argument(
        "--query-min-crop",
        type=float,
        default=0.8,
        help="Min crop ratio for query augmentation.",
    )
    parser.add_argument(
        "--query-max-crop",
        type=float,
        default=1.0,
        help="Max crop ratio for query augmentation.",
    )
    parser.add_argument(
        "--query-dropout",
        type=float,
        default=0.0,
        help="Token dropout prob for query augmentation.",
    )
    parser.add_argument(
        "--use-faiss",
        action="store_true",
        help="Also compute ANN metrics using faiss.IndexFlatIP.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for metrics json.",
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


def _resolve_device(device_flag: str) -> torch.device:
    if device_flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_flag == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_flag)


def _parse_topk(values: str) -> list[int]:
    parsed = sorted({int(part.strip()) for part in values.split(",") if part.strip()})
    if not parsed or any(v <= 0 for v in parsed):
        raise ValueError(f"Invalid top-k list: {values}")
    return parsed


def _build_eval_views(
    tokens_root: Path,
    split_file: Path,
    aug_cfg: TokenAugmentConfig,
    seed: int,
) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor]]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")

    random.seed(seed)
    track_ids = load_track_ids(split_file)
    track_to_files = build_track_index(tokens_root)

    kept_ids: list[str] = []
    query_views: list[torch.Tensor] = []
    corpus_views: list[torch.Tensor] = []

    for track_id in track_ids:
        files = track_to_files.get(track_id)
        if not files:
            continue
        tokens = load_tokens(files[0])
        if tokens.dim() != 2:
            continue
        kept_ids.append(track_id)
        corpus_views.append(tokens)
        query_views.append(make_augmented_view(tokens, aug_cfg))

    if len(kept_ids) < 2:
        raise RuntimeError("Need at least 2 tracks in split to compute retrieval metrics.")
    return kept_ids, query_views, corpus_views


def _encode_sequences(
    model: TokenEmbedder,
    sequences: list[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(sequences), batch_size):
            chunk = sequences[i : i + batch_size]
            tokens, mask = pad_token_batch(chunk)
            emb = model(
                tokens=tokens.to(device, non_blocking=True),
                mask=mask.to(device, non_blocking=True),
            )
            outputs.append(emb.cpu())
    return torch.cat(outputs, dim=0)


def _compute_metrics(indices: torch.Tensor, truth: torch.Tensor, topks: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    n, retrieved = indices.shape

    for k in topks:
        kk = min(k, retrieved)
        hit = (indices[:, :kk] == truth.unsqueeze(1)).any(dim=1).float()
        metrics[f"recall@{k}"] = float(hit.mean().item())

    rr = torch.zeros(n, dtype=torch.float32)
    for i in range(n):
        matches = torch.where(indices[i] == truth[i])[0]
        if len(matches) > 0:
            rr[i] = 1.0 / float(matches[0].item() + 1)
    metrics["mrr"] = float(rr.mean().item())
    return metrics


def _exact_search(query_emb: torch.Tensor, corpus_emb: torch.Tensor, max_k: int) -> torch.Tensor:
    sims = torch.matmul(query_emb, corpus_emb.t())
    k = min(max_k, sims.shape[1])
    return torch.topk(sims, k=k, dim=1, largest=True, sorted=True).indices


def _faiss_search(query_emb: torch.Tensor, corpus_emb: torch.Tensor, max_k: int) -> torch.Tensor:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss is not installed. Disable --use-faiss or install faiss-cpu.") from exc

    q = query_emb.detach().cpu().numpy().astype("float32")
    c = corpus_emb.detach().cpu().numpy().astype("float32")
    dim = c.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(c)
    _, idx = index.search(q, min(max_k, c.shape[0]))
    return torch.from_numpy(idx).long()


def _build_model(
    raw_config: dict[str, Any],
    num_quantizers: int,
    checkpoint: Path | None,
    device: torch.device,
) -> tuple[TokenEmbedder, dict[str, Any]]:
    model_cfg = raw_config.get("model", {})
    resolved_config = TokenEmbedderConfig(
        vocab_size=int(model_cfg.get("vocab_size", 4096)),
        max_quantizers=max(int(model_cfg.get("max_quantizers", num_quantizers)), num_quantizers),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )

    state_loaded = False
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu")
        ckpt_cfg = payload.get("model_config")
        if isinstance(ckpt_cfg, dict):
            resolved_config = TokenEmbedderConfig(
                vocab_size=int(ckpt_cfg.get("vocab_size", resolved_config.vocab_size)),
                max_quantizers=max(
                    int(ckpt_cfg.get("max_quantizers", resolved_config.max_quantizers)),
                    num_quantizers,
                ),
                hidden_dim=int(ckpt_cfg.get("hidden_dim", resolved_config.hidden_dim)),
                embedding_dim=int(ckpt_cfg.get("embedding_dim", resolved_config.embedding_dim)),
                num_layers=int(ckpt_cfg.get("num_layers", resolved_config.num_layers)),
                num_heads=int(ckpt_cfg.get("num_heads", resolved_config.num_heads)),
                dropout=float(ckpt_cfg.get("dropout", resolved_config.dropout)),
            )

        model = TokenEmbedder(resolved_config)
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint has no model_state_dict: {checkpoint}")
        model.load_state_dict(state, strict=True)
        state_loaded = True
    else:
        model = TokenEmbedder(resolved_config)

    model = model.to(device).eval()
    info = {
        "model_config": asdict(resolved_config),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_loaded": state_loaded,
    }
    return model, info


def _print_table(method_to_metrics: dict[str, dict[str, float]], topks: list[int]) -> None:
    headers = ["method"] + [f"recall@{k}" for k in topks] + ["mrr"]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for method, metrics in method_to_metrics.items():
        row = [method]
        for k in topks:
            row.append(f"{metrics[f'recall@{k}']:.4f}")
        row.append(f"{metrics['mrr']:.4f}")
        print(" | ".join(row))


def main() -> None:
    args = _parse_args()
    raw_config = _read_yaml(args.config)
    topks = _parse_topk(args.topk)
    max_k = max(topks)

    split_file = args.splits_root / f"{args.split}.txt"
    aug_cfg = TokenAugmentConfig(
        min_crop_ratio=args.query_min_crop,
        max_crop_ratio=args.query_max_crop,
        dropout_prob=args.query_dropout,
        pad_value=0,
    )
    track_ids, query_sequences, corpus_sequences = _build_eval_views(
        tokens_root=args.tokens_root,
        split_file=split_file,
        aug_cfg=aug_cfg,
        seed=args.seed,
    )

    num_quantizers = int(corpus_sequences[0].shape[0])
    device = _resolve_device(args.device)
    model, model_info = _build_model(
        raw_config=raw_config,
        num_quantizers=num_quantizers,
        checkpoint=args.checkpoint,
        device=device,
    )

    query_emb = _encode_sequences(model, query_sequences, args.batch_size, device)
    corpus_emb = _encode_sequences(model, corpus_sequences, args.batch_size, device)
    truth = torch.arange(len(track_ids), dtype=torch.long)

    results: dict[str, dict[str, float]] = {}
    exact_idx = _exact_search(query_emb, corpus_emb, max_k=max_k)
    results["exact"] = _compute_metrics(exact_idx, truth=truth, topks=topks)

    if args.use_faiss:
        ann_idx = _faiss_search(query_emb, corpus_emb, max_k=max_k)
        results["faiss_flat_ip"] = _compute_metrics(ann_idx, truth=truth, topks=topks)

    print(f"Evaluated split={args.split} with {len(track_ids)} tracks")
    print(f"Device: {device}")
    _print_table(results, topks)

    if args.output_json is not None:
        payload = {
            "split": args.split,
            "num_tracks": len(track_ids),
            "topks": topks,
            "results": results,
            "model": model_info,
            "config_path": str(args.config),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"Saved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
