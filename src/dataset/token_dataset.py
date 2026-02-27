from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import TokenAugmentConfig, make_augmented_view


def normalize_track_id(stem: str) -> str:
    return re.sub(r"__chunk\d{4}$", "", stem)


def load_track_ids(split_file: Path) -> list[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_tokens_from_pt(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "tokens" in payload:
        tokens = payload["tokens"]
    else:
        tokens = payload
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens)
    return tokens.long()


def _load_tokens_from_npz(path: Path) -> torch.Tensor:
    arr = np.load(path, allow_pickle=False)
    if "tokens" not in arr:
        raise RuntimeError(f"No 'tokens' key in npz file: {path}")
    return torch.from_numpy(arr["tokens"]).long()


def load_tokens(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".pt":
        return _load_tokens_from_pt(path)
    if path.suffix.lower() == ".npz":
        return _load_tokens_from_npz(path)
    raise RuntimeError(f"Unsupported token extension: {path.suffix}")


def build_track_index(tokens_root: Path) -> dict[str, list[Path]]:
    files = sorted([p for p in tokens_root.rglob("*") if p.suffix.lower() in {".pt", ".npz"}])
    if not files:
        raise RuntimeError(f"No token files found in: {tokens_root}")

    index: dict[str, list[Path]] = {}
    for path in files:
        track_id = normalize_track_id(path.stem)
        index.setdefault(track_id, []).append(path)
    return index


@dataclass
class TokenDatasetConfig:
    tokens_root: Path
    split_file: Path
    seed: int = 42
    min_crop_ratio: float = 0.6
    max_crop_ratio: float = 1.0
    dropout_prob: float = 0.02
    pad_value: int = 0


class TokenPairDataset(Dataset):
    """
    Produces positive pairs for contrastive learning:
    two augmented views from the same track.
    """

    def __init__(self, config: TokenDatasetConfig) -> None:
        self.config = config
        self.track_ids = load_track_ids(config.split_file)
        self.track_to_files = build_track_index(config.tokens_root)
        self.track_ids = [tid for tid in self.track_ids if tid in self.track_to_files]
        if not self.track_ids:
            raise RuntimeError("No split track_ids found in token index.")

        self.rng = random.Random(config.seed)
        self.aug_cfg = TokenAugmentConfig(
            min_crop_ratio=config.min_crop_ratio,
            max_crop_ratio=config.max_crop_ratio,
            dropout_prob=config.dropout_prob,
            pad_value=config.pad_value,
        )

    def __len__(self) -> int:
        return len(self.track_ids)

    def _sample_file(self, track_id: str) -> Path:
        files = self.track_to_files[track_id]
        return files[self.rng.randrange(len(files))]

    def _sample_view(self, track_id: str) -> torch.Tensor:
        token_file = self._sample_file(track_id)
        tokens = load_tokens(token_file)
        if tokens.dim() != 2:
            raise RuntimeError(
                f"Expected [Q,T] tokens, got shape={tuple(tokens.shape)} from {token_file}"
            )
        return make_augmented_view(tokens, self.aug_cfg)

    def __getitem__(self, index: int) -> dict[str, Any]:
        track_id = self.track_ids[index]
        tokens_a = self._sample_view(track_id)
        tokens_b = self._sample_view(track_id)
        return {
            "track_id": track_id,
            "tokens_a": tokens_a,
            "tokens_b": tokens_b,
        }


def pad_token_batch(
    sequences: list[torch.Tensor],
    pad_value: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pads [Q,T_i] sequences to [B,Q,T_max] and builds mask [B,T_max].
    """
    if not sequences:
        raise ValueError("Empty sequences list.")
    q = sequences[0].shape[0]
    max_t = max(seq.shape[1] for seq in sequences)
    batch = torch.full((len(sequences), q, max_t), pad_value, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_t), dtype=torch.long)

    for i, seq in enumerate(sequences):
        if seq.shape[0] != q:
            raise RuntimeError("Inconsistent number of quantizers across batch.")
        t = seq.shape[1]
        batch[i, :, :t] = seq
        mask[i, :t] = 1
    return batch, mask


def token_pair_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tokens_a = [item["tokens_a"] for item in batch]
    tokens_b = [item["tokens_b"] for item in batch]
    track_ids = [item["track_id"] for item in batch]

    a_padded, a_mask = pad_token_batch(tokens_a)
    b_padded, b_mask = pad_token_batch(tokens_b)
    return {
        "track_id": track_ids,
        "tokens_a": a_padded,
        "tokens_b": b_padded,
        "mask_a": a_mask,
        "mask_b": b_mask,
    }


def dump_dataset_preview(path: Path, item: dict[str, Any]) -> None:
    """Utility to quickly inspect one sample structure for debugging."""
    preview = {
        "track_id": item["track_id"],
        "tokens_a_shape": list(item["tokens_a"].shape),
        "tokens_b_shape": list(item["tokens_b"].shape),
    }
    path.write_text(json.dumps(preview, indent=2), encoding="utf-8")

