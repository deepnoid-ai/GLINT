"""UNETR-style decoder without skip connections, isolating encoder representation quality across backbones."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ConvBlock3d(nn.Module):
    """Conv3d → InstanceNorm3d → LeakyReLU (×2), following UNETR convention."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock3d(nn.Module):
    """Upsample(×2) + ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int, upsample_factor: tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=upsample_factor,
            stride=upsample_factor,
            bias=False,
        )
        self.conv_block = ConvBlock3d(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv_block(x)
        return x


class UNETRNoSkipDecoder(nn.Module):
    """One ×2 upsample stage per log2(max patch dim); channels halve per stage down to base_channels."""

    def __init__(
        self,
        patch_size_3d: tuple[int, int, int],
        embed_dim: int,
        num_classes: int,
        base_channels: int = 32,
    ):
        super().__init__()

        # e.g. patch_size_3d=(2, 16, 16) -> 4 stages; an axis stops upsampling once its factor is used up
        stages_per_axis = [int(math.log2(p)) for p in patch_size_3d]
        num_stages = max(stages_per_axis)

        self.upsample_factors = []
        remaining = list(patch_size_3d)
        for _ in range(num_stages):
            factor = []
            for ax in range(3):
                if remaining[ax] >= 2:
                    factor.append(2)
                    remaining[ax] //= 2
                else:
                    factor.append(1)
            self.upsample_factors.append(tuple(factor))

        channels = [embed_dim]
        for _ in range(num_stages):
            next_ch = max(channels[-1] // 2, base_channels)
            channels.append(next_ch)

        self.initial_conv = ConvBlock3d(embed_dim, channels[0])

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(UpsampleBlock3d(channels[i], channels[i + 1], upsample_factor=self.upsample_factors[i]))

        self.out_conv = nn.Conv3d(channels[-1], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, d, h, w) patch-grid features -> (B, num_classes, D, H, W) logits."""
        x = self.initial_conv(x)
        for stage in self.stages:
            x = stage(x)
        return self.out_conv(x)
