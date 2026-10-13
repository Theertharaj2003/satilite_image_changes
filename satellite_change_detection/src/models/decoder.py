from __future__ import annotations

import torch
from torch import nn


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.up(x))


class MultiScaleDecoder(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), use_multiscale=True):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.use_multiscale = use_multiscale

        self.up1 = nn.ConvTranspose2d(c4, c4, 2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(c4 + (c4 if use_multiscale else 0), c4, 3, padding=1, bias=False),
            nn.BatchNorm2d(c4), nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, 3, padding=1, bias=False),
            nn.BatchNorm2d(c4), nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c3 + (c3 if use_multiscale else 0), c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
        )
        self.up3 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2 + (c2 if use_multiscale else 0), c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
        )
        self.up4 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(c1 + (c1 if use_multiscale else 0), c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(c1, 1, 1)

    @staticmethod
    def _cat(x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x, skips):
        f1, f2, f3, f4 = skips
        x = self.up1(x)
        if self.use_multiscale: x = self.conv1(self._cat(x, f4))
        else: x = self.conv1(x)
        x = self.up2(x)
        if self.use_multiscale: x = self.conv2(self._cat(x, f3))
        else: x = self.conv2(x)
        x = self.up3(x)
        if self.use_multiscale: x = self.conv3(self._cat(x, f2))
        else: x = self.conv3(x)
        x = self.up4(x)
        if self.use_multiscale: x = self.conv4(self._cat(x, f1))
        else: x = self.conv4(x)
        return self.head(x)
