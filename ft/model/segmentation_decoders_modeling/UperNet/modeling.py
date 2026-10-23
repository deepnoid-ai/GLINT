import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6], out_channels=512):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.out_channels = out_channels

        self.stages = nn.ModuleList()
        for pool_size in pool_sizes:
            stage = nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.stages.append(stage)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * out_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramid_features = [x]

        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
            pyramid_features.append(upsampled)

        out = torch.cat(pyramid_features, dim=1)
        out = self.bottleneck(out)
        return out


class FPNHead(nn.Module):
    """Feature Pyramid Network Head with progressive upsampling"""

    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.lateral_bn = nn.BatchNorm2d(out_channels)

        self.fpn_stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ),
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ),
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ),
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ),
            ]
        )

        self.final_conv = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        lateral = self.lateral_conv(x)
        lateral = self.lateral_bn(lateral)

        fpn_out = lateral
        for stage in self.fpn_stages:
            fpn_out = stage(fpn_out)

        out = self.final_conv(fpn_out)
        return out


class SimpleUperNetDecoder(nn.Module):
    """UPerNet-like decoder (PPM + progressive upsampling) on single-scale (B, C, H, W) features."""

    def __init__(
        self,
        in_channels=768,
        out_channels=1,
        ppm_pool_sizes=[1, 2, 3, 6],
        ppm_out_channels=512,
        fpn_out_channels=512,
    ):
        super().__init__()

        self.ppm = PPM(in_channels=in_channels, pool_sizes=ppm_pool_sizes, out_channels=ppm_out_channels)

        self.fpn_head = FPNHead(
            in_channels=in_channels,
            out_channels=fpn_out_channels,
            num_classes=out_channels,
        )

    def forward(self, x):
        ppm_out = self.ppm(x)
        out = self.fpn_head(ppm_out)
        return out
