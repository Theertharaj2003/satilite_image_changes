from __future__ import annotations

import torch
from torch import nn

from .encoder import SharedCNNEncoder
from .attention import TemporalAttentionBlock, CrossAttentionBlock
from .decoder import MultiScaleDecoder


class SiameseCNNChangeDetector(nn.Module):
    """CNN baseline: shared encoder + absolute temporal difference + decoder."""

    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        self.encoder = SharedCNNEncoder(channels=channels)
        self.decoder = MultiScaleDecoder(channels=channels, use_multiscale=True)
        self.bottleneck_proj = nn.Conv2d(channels[-1], channels[-1], 1)

    def forward(self, image_t1, image_t2):
        a = self.encoder(image_t1); b = self.encoder(image_t2)
        fused = self.bottleneck_proj(torch.abs(a["f5"] - b["f5"]))
        skips = tuple(torch.abs(a[k] - b[k]) for k in ("f1", "f2", "f3", "f4"))
        return self.decoder(fused, skips)


class TemporalTransformerChangeDetector(nn.Module):
    """Transformer baseline using temporal self-attention without cross-attention."""

    def __init__(self, channels=(32, 64, 128, 256), heads=8, layers=2, dropout=0.1):
        super().__init__()
        dim = channels[-1]
        self.encoder = SharedCNNEncoder(channels=channels)
        self.temporal = TemporalAttentionBlock(dim, heads, layers, dropout)
        self.merge = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim))
        self.decoder = MultiScaleDecoder(channels=channels, use_multiscale=True)

    def forward(self, image_t1, image_t2):
        a = self.encoder(image_t1); b = self.encoder(image_t2)
        f1, f2 = self.temporal(a["f5"], b["f5"])
        x = self.merge(torch.cat([f1, f2], dim=-1))
        bsz, n, ch = x.shape
        h, w = a["f5"].shape[-2:]
        x = x.transpose(1, 2).reshape(bsz, ch, h, w)
        skips = tuple(torch.abs(a[k] - b[k]) for k in ("f1", "f2", "f3", "f4"))
        return self.decoder(x, skips)


class ProposedTemporalTransformer(nn.Module):
    """Proposed architecture: shared CNN + temporal self-attention + cross-attention + multiscale decoder."""

    def __init__(
        self,
        channels=(32, 64, 128, 256),
        heads=8,
        layers=2,
        dropout=0.1,
        use_temporal_self_attention=True,
        use_cross_attention=True,
        use_multiscale=True,
    ):
        super().__init__()
        dim = channels[-1]
        self.encoder = SharedCNNEncoder(channels=channels)
        self.use_temporal_self_attention = use_temporal_self_attention
        self.use_cross_attention = use_cross_attention
        self.use_multiscale = use_multiscale
        self.temporal = TemporalAttentionBlock(dim, heads, layers, dropout) if use_temporal_self_attention else None
        self.cross = CrossAttentionBlock(dim, heads, dropout) if use_cross_attention else None
        self.no_cross_proj = nn.Linear(dim * 2, dim)
        self.decoder = MultiScaleDecoder(channels=channels, use_multiscale=use_multiscale)

    def forward(self, image_t1, image_t2):
        a = self.encoder(image_t1); b = self.encoder(image_t2)
        if self.temporal is not None:
            f1, f2 = self.temporal(a["f5"], b["f5"])
        else:
            f1 = a["f5"].flatten(2).transpose(1, 2)
            f2 = b["f5"].flatten(2).transpose(1, 2)

        if self.cross is not None:
            x = self.cross(f1, f2)
        else:
            x = self.no_cross_proj(torch.cat([f1, f2], dim=-1))

        bsz, n, ch = x.shape
        h, w = a["f5"].shape[-2:]
        x = x.transpose(1, 2).reshape(bsz, ch, h, w)

        skips = tuple(torch.abs(a[k] - b[k]) for k in ("f1", "f2", "f3", "f4"))
        return self.decoder(x, skips if self.use_multiscale else (None, None, None, None))
