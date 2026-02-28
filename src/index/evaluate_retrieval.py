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
        default=16,
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
        "--eval-protocol",
        type=str,
        default="same_file_aug",
        choices=["same_file_aug", "cross_chunk"],
        help=(
            "Evaluation protocol. "
            "'same_file_aug' is easy sanity-check mode; "
            "'cross_chunk' is stricter and uses one chunk as query, "
            "other chunks of the same track as positives."
        ),
    )
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help=(
            "Exclude exact same chunk from retrieval candidates for each query. "
            "Recommended with --eval-protocol cross_chunk."
        ),
    )
    parser.add_argument(
        "--use-faiss",
        action="store_true",
        help="Also compute retrieval metrics with faiss.IndexFlatIP (exact IP baseline).",
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
    eval_protocol: str,
    exclude_self: bool,
) -> tuple[
    list[str],
    list[torch.Tensor],
    list[torch.Tensor],
    list[set[int]],
    list[int | None],
]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")

    random.seed(seed)
    track_ids = load_track_ids(split_file)
    track_to_files = build_track_index(tokens_root)

    if eval_protocol == "same_file_aug" and exclude_self:
        raise ValueError(
            "--exclude-self cannot be used with --eval-protocol same_file_aug. "
            "Use --eval-protocol cross_chunk."
        )

    if eval_protocol == "same_file_aug":
        query_track_ids: list[str] = []
        query_views: list[torch.Tensor] = []
        corpus_views: list[torch.Tensor] = []
        positive_sets: list[set[int]] = []
        self_indices: list[int | None] = []

        for track_id in track_ids:
            files = track_to_files.get(track_id)
            if not files:
                continue
            tokens = load_tokens(files[0])
            if tokens.dim() != 2:
                continue
            corpus_idx = len(corpus_views)
            query_track_ids.append(track_id)
            corpus_views.append(tokens)
            query_views.append(make_augmented_view(tokens, aug_cfg))
            positive_sets.append({corpus_idx})
            self_indices.append(corpus_idx)

        if len(query_track_ids) < 2:
            raise RuntimeError("Need at least 2 tracks in split to compute retrieval metrics.")
        return query_track_ids, query_views, corpus_views, positive_sets, self_indices

    # cross_chunk: one query chunk per track, positives are other chunks of the same track.
    eligible_tracks: list[tuple[str, list[torch.Tensor]]] = []
    for track_id in track_ids:
        files = track_to_files.get(track_id)
        if not files:
            continue
        valid_tokens: list[torch.Tensor] = []
        for file_path in files:
            tokens = load_tokens(file_path)
            if tokens.dim() == 2:
                valid_tokens.append(tokens)
        if len(valid_tokens) >= 2:
            eligible_tracks.append((track_id, valid_tokens))

    if len(eligible_tracks) < 2:
        raise RuntimeError(
            "cross_chunk protocol requires at least 2 tracks with >=2 token files each "
            "(usually from --chunk-seconds during tokenization)."
        )

    corpus_views: list[torch.Tensor] = []
    track_to_corpus_indices: dict[str, list[int]] = {}
    for track_id, token_list in eligible_tracks:
        idxs: list[int] = []
        for tokens in token_list:
            idxs.append(len(corpus_views))
            corpus_views.append(tokens)
        track_to_corpus_indices[track_id] = idxs

    query_track_ids = []
    query_views = []
    positive_sets = []
    self_indices = []

    for track_id, token_list in eligible_tracks:
        positives = set(track_to_corpus_indices[track_id])
        self_idx = track_to_corpus_indices[track_id][0]
        if exclude_self:
            positives.discard(self_idx)
        if not positives:
            # Should not happen for eligible tracks, but keep this guard explicit.
            continue

        query_tokens = token_list[0]
        query_views.append(make_augmented_view(query_tokens, aug_cfg))
        query_track_ids.append(track_id)
        positive_sets.append(positives)
        self_indices.append(self_idx if exclude_self else None)

    if len(query_track_ids) < 2:
        raise RuntimeError("Need at least 2 query tracks to compute retrieval metrics.")
    return query_track_ids, query_views, corpus_views, positive_sets, self_indices


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


def _compute_metrics(
    indices: torch.Tensor,
    positive_sets: list[set[int]],
    topks: list[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    n, retrieved = indices.shape
    if len(positive_sets) != n:
        raise ValueError("positive_sets size must match number of queries.")

    for k in topks:
        kk = min(k, retrieved)
        hit_values: list[float] = []
        for i in range(n):
            ranked = indices[i, :kk].tolist()
            hit = any((idx >= 0 and idx in positive_sets[i]) for idx in ranked)
            hit_values.append(1.0 if hit else 0.0)
        metrics[f"recall@{k}"] = float(sum(hit_values) / max(len(hit_values), 1))

    rr_values: list[float] = []
    for i in range(n):
        reciprocal_rank = 0.0
        for rank, idx in enumerate(indices[i].tolist(), start=1):
            if idx >= 0 and idx in positive_sets[i]:
                reciprocal_rank = 1.0 / float(rank)
                break
        rr_values.append(reciprocal_rank)
    metrics["mrr"] = float(sum(rr_values) / max(len(rr_values), 1))
    return metrics


def _filter_excluded_indices(
    indices: torch.Tensor,
    max_k: int,
    excluded_corpus_indices: list[int | None] | None,
) -> torch.Tensor:
    n, found_k = indices.shape
    out_k = min(max_k, found_k)
    filtered = torch.full((n, out_k), -1, dtype=torch.long)
    if excluded_corpus_indices is None:
        return indices[:, :out_k]
    if len(excluded_corpus_indices) != n:
        raise ValueError("excluded_corpus_indices size must match number of queries.")

    for i in range(n):
        banned = excluded_corpus_indices[i]
        write = 0
        for idx in indices[i].tolist():
            if banned is not None and idx == banned:
                continue
            if write >= out_k:
                break
            filtered[i, write] = idx
            write += 1
    return filtered


def _exact_search(
    query_emb: torch.Tensor,
    corpus_emb: torch.Tensor,
    max_k: int,
    excluded_corpus_indices: list[int | None] | None = None,
) -> torch.Tensor:
    sims = torch.matmul(query_emb, corpus_emb.t())
    extra = 1 if excluded_corpus_indices is not None else 0
    search_k = min(max_k + extra, sims.shape[1])
    raw = torch.topk(sims, k=search_k, dim=1, largest=True, sorted=True).indices
    return _filter_excluded_indices(raw, max_k=max_k, excluded_corpus_indices=excluded_corpus_indices)


def _faiss_search(
    query_emb: torch.Tensor,
    corpus_emb: torch.Tensor,
    max_k: int,
    excluded_corpus_indices: list[int | None] | None = None,
) -> torch.Tensor:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss is not installed. Disable --use-faiss or install faiss-cpu.") from exc

    q = query_emb.detach().cpu().numpy().astype("float32")
    c = corpus_emb.detach().cpu().numpy().astype("float32")
    dim = c.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(c)
    extra = 1 if excluded_corpus_indices is not None else 0
    search_k = min(max_k + extra, c.shape[0])
    _, idx = index.search(q, search_k)
    raw = torch.from_numpy(idx).long()
    return _filter_excluded_indices(raw, max_k=max_k, excluded_corpus_indices=excluded_corpus_indices)


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
    query_track_ids, query_sequences, corpus_sequences, positive_sets, self_indices = _build_eval_views(
        tokens_root=args.tokens_root,
        split_file=split_file,
        aug_cfg=aug_cfg,
        seed=args.seed,
        eval_protocol=args.eval_protocol,
        exclude_self=args.exclude_self,
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

    results: dict[str, dict[str, float]] = {}
    excluded = self_indices if args.exclude_self else None
    exact_idx = _exact_search(
        query_emb,
        corpus_emb,
        max_k=max_k,
        excluded_corpus_indices=excluded,
    )
    results["exact"] = _compute_metrics(exact_idx, positive_sets=positive_sets, topks=topks)

    if args.use_faiss:
        ann_idx = _faiss_search(
            query_emb,
            corpus_emb,
            max_k=max_k,
            excluded_corpus_indices=excluded,
        )
        results["faiss_flat_ip"] = _compute_metrics(ann_idx, positive_sets=positive_sets, topks=topks)

    print(
        f"Evaluated split={args.split} protocol={args.eval_protocol} "
        f"queries={len(query_track_ids)} corpus_items={len(corpus_sequences)} "
        f"exclude_self={args.exclude_self}"
    )
    print(f"Device: {device}")
    _print_table(results, topks)

    if args.output_json is not None:
        payload = {
            "split": args.split,
            "eval_protocol": args.eval_protocol,
            "exclude_self": bool(args.exclude_self),
            "num_queries": len(query_track_ids),
            "num_corpus_items": len(corpus_sequences),
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
