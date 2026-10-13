from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SharedCNNEncoder(nn.Module):
    """Shared-weight encoder used for both temporal images."""

    def __init__(self, in_channels: int = 3, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stage1 = ConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.stage4 = ConvBlock(c3, c4)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c4, c4)
        self.channels = channels

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(self.pool1(f1))
        f3 = self.stage3(self.pool2(f2))
        f4 = self.stage4(self.pool3(f3))
        f5 = self.bottleneck(self.pool4(f4))
        return {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5}
