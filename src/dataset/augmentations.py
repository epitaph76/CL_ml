from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class TokenAugmentConfig:
    min_crop_ratio: float = 0.6
    max_crop_ratio: float = 1.0
    dropout_prob: float = 0.02
    pad_value: int = 0


def random_time_crop(tokens: torch.Tensor, min_ratio: float, max_ratio: float) -> torch.Tensor:
    """
    Randomly crops token sequence along time axis.
    Expected shape: [num_quantizers, time_steps].
    """
    if tokens.dim() != 2:
        raise ValueError(f"Expected 2D tokens [Q,T], got shape={tuple(tokens.shape)}")
    q, t = tokens.shape
    if t <= 1:
        return tokens

    ratio = random.uniform(min_ratio, max_ratio)
    crop_len = max(1, int(t * ratio))
    if crop_len >= t:
        return tokens

    start = random.randint(0, t - crop_len)
    end = start + crop_len
    return tokens[:, start:end]


def random_token_dropout(tokens: torch.Tensor, dropout_prob: float, pad_value: int = 0) -> torch.Tensor:
    """
    Replaces random token positions with pad value to improve robustness.
    """
    if dropout_prob <= 0:
        return tokens
    keep_mask = torch.rand_like(tokens.float()) > dropout_prob
    return torch.where(keep_mask, tokens, torch.full_like(tokens, pad_value))


def make_augmented_view(tokens: torch.Tensor, cfg: TokenAugmentConfig) -> torch.Tensor:
    cropped = random_time_crop(tokens, cfg.min_crop_ratio, cfg.max_crop_ratio)
    dropped = random_token_dropout(cropped, cfg.dropout_prob, cfg.pad_value)
    return dropped

