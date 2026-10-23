"""Adapted from MAVL (https://github.com/hieuvmphan/CVPR2024_MAVL).
No upstream license; rights remain with the original authors.
"""

import torch
import torchvision.models as models
from einops import rearrange
from torch import nn

from .utils import load_encoder_weights


class MAVLImageEncoder(nn.Module):
    """MAVL image encoder: ResNet-50 up to layer3 -> 1x1 conv projections -> (B, N, 256)."""

    def __init__(self, d_model=256):
        super().__init__()
        resnet = models.resnet50(weights=None)
        num_ftrs = int(resnet.fc.in_features / 2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.res_l1 = nn.Sequential(nn.Conv2d(num_ftrs, num_ftrs, kernel_size=1), nn.PReLU())
        self.res_l2 = nn.Conv2d(num_ftrs, d_model, kernel_size=1)

    def forward_features(self, x):
        x = self.res_l2(self.res_l1(self.backbone(x)))
        return rearrange(x, "b d n1 n2 -> b (n1 n2) d")


def build_mavl_resnet(checkpoint_path, d_model=256):
    """MAVL image encoder from the official ``checkpoint_full_40.pth``."""
    model = MAVLImageEncoder(d_model=d_model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    load_encoder_weights(model, state, "MAVL")
    return model
