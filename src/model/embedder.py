from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TokenEmbedderConfig:
    vocab_size: int = 4096
    max_quantizers: int = 32
    hidden_dim: int = 256
    embedding_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1


class TokenEmbedder(nn.Module):
    """
    Baseline encoder for token tensors with shape [B, Q, T].
    Produces L2-normalized track embeddings with shape [B, D].
    """

    def __init__(self, config: TokenEmbedderConfig) -> None:
        super().__init__()
        if config.vocab_size <= 1:
            raise ValueError("vocab_size must be > 1 (0 is reserved for padding).")
        if config.max_quantizers <= 0:
            raise ValueError("max_quantizers must be positive.")

        self.config = config
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim,
            padding_idx=0,
        )
        self.quantizer_embedding = nn.Embedding(
            num_embeddings=config.max_quantizers,
            embedding_dim=config.hidden_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=max(1, config.num_layers),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.hidden_dim, config.embedding_dim)

    def _remap_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Keep 0 as padding and map non-zero ids into [1, vocab_size - 1]
        to avoid index errors when token ids exceed configured vocab.
        """
        if tokens.dtype != torch.long:
            tokens = tokens.long()
        if self.config.vocab_size == 2:
            return (tokens > 0).long()

        mapped = tokens.clone()
        non_pad = mapped > 0
        mapped[non_pad] = torch.remainder(mapped[non_pad] - 1, self.config.vocab_size - 1) + 1
        mapped[~non_pad] = 0
        return mapped

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"Expected tokens shape [B,Q,T], got {tuple(tokens.shape)}")

        batch, quantizers, _ = tokens.shape
        if quantizers > self.config.max_quantizers:
            raise ValueError(
                f"Input uses {quantizers} quantizers, but max_quantizers={self.config.max_quantizers}"
            )

        if mask is None:
            # Valid timestep if any quantizer has non-pad token.
            mask = (tokens > 0).any(dim=1).long()
        if mask.dim() != 2 or mask.shape[0] != batch:
            raise ValueError(f"Expected mask shape [B,T], got {tuple(mask.shape)}")

        tokens = self._remap_tokens(tokens)
        token_emb = self.token_embedding(tokens)  # [B,Q,T,H]

        q_ids = torch.arange(quantizers, device=tokens.device)
        q_emb = self.quantizer_embedding(q_ids).view(1, quantizers, 1, -1)  # [1,Q,1,H]
        mixed = token_emb + q_emb
        time_features = mixed.mean(dim=1)  # [B,T,H]
        time_features = self.dropout(time_features)

        key_padding_mask = mask == 0
        encoded = self.transformer(time_features, src_key_padding_mask=key_padding_mask)

        mask_f = mask.to(encoded.dtype).unsqueeze(-1)  # [B,T,1]
        summed = (encoded * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = summed / counts
        projected = self.projection(self.dropout(pooled))
        return F.normalize(projected, p=2, dim=-1)
