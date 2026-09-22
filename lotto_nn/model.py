"""Transformer-based generator model for UK Lotto draws."""
from __future__ import annotations

import torch
import torch.nn as nn


class LottoTransformer(nn.Module):
    """A small transformer encoder that models sequences of 6 lotto numbers.

    vocab_size = 60 allows token indices 0-59; lotto numbers use 1-59 and 0 is
    reserved as a padding/unknown token (e.g. during autoregressive sampling).
    """

    def __init__(
        self,
        vocab_size: int = 60,
        seq_len: int = 6,
        embed_dim: int = 32,
        nhead: int = 4,
        ff_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len] integer tokens -> logits: [batch, seq_len, vocab_size]."""
        batch_size, seq_len = x.shape
        pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x_emb = self.embed(x) + self.pos(pos_idx)
        encoded = self.encoder(x_emb)
        logits = self.fc(encoded)
        return logits
