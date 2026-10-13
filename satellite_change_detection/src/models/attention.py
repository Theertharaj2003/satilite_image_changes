from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionBlock(nn.Module):
    """Self-attention over the concatenated T1/T2 bottleneck tokens."""

    def __init__(self, dim: int = 256, heads: int = 8, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.temporal_embedding = nn.Parameter(torch.zeros(2, 1, dim))
        nn.init.normal_(self.temporal_embedding, std=0.02)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor):
        b, c, h, w = f1.shape
        n = h * w
        q1 = f1.flatten(2).transpose(1, 2)
        q2 = f2.flatten(2).transpose(1, 2)
        x = torch.cat([q1 + self.temporal_embedding[0], q2 + self.temporal_embedding[1]], dim=1)
        x = self.encoder(x)
        return x[:, :n], x[:, n:]


class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention between T1 and T2 features."""

    def __init__(self, dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn_12 = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_21 = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor):
        c12, _ = self.attn_12(query=f1, key=f2, value=f2, need_weights=False)
        c21, _ = self.attn_21(query=f2, key=f1, value=f1, need_weights=False)
        z1 = self.norm1(f1 + c12)
        z2 = self.norm2(f2 + c21)
        fused = self.proj(torch.cat([z1, z2], dim=-1))
        return fused
