"""Adapted from KAD (https://github.com/xiaoman-zhang/KAD).
Copyright (c) 2022 Rajpurkar Lab. MIT License (LICENSES/MIT.txt)
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange
from torch import nn

from .utils import load_encoder_weights


class KADImageEncoder(nn.Module):
    """KAD ``ModelRes512`` image encoder: ResNet-50 feature map -> per-patch 2-layer MLP -> (B, N, 768)."""

    def __init__(self, d_model=768):
        super().__init__()
        resnet = models.resnet50(weights=None)
        num_ftrs = int(resnet.fc.in_features)
        self.res_features = nn.Sequential(*list(resnet.children())[:-2])
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, d_model)

    def forward_features(self, x):
        batch_size = x.shape[0]
        h = rearrange(self.res_features(x), "b d n1 n2 -> (b n1 n2) d")
        h = self.res_l2(F.relu(self.res_l1(h)))
        return rearrange(h, "(b n) d -> b n d", b=batch_size)


def build_kad_resnet(checkpoint_path, d_model=768):
    """KAD-512 image encoder from the official ``best_valid.pt``."""
    model = KADImageEncoder(d_model=d_model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("module."): v for k, v in ckpt["image_encoder"].items()}
    load_encoder_weights(model, state, "KAD")
    return model
