"""Adapted from MedKLIP (https://github.com/MediaBrain-SJTU/MedKLIP).
No upstream license; rights remain with the original authors.
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange
from torch import nn

from .utils import load_encoder_weights


class MedKLIPImageEncoder(nn.Module):
    """MedKLIP image encoder: ResNet-50 up to layer3 -> per-patch 2-layer MLP -> (B, N, 256)."""

    def __init__(self, d_model=256):
        super().__init__()
        resnet = models.resnet50(weights=None)
        num_ftrs = int(resnet.fc.in_features / 2)
        self.res_features = nn.Sequential(*list(resnet.children())[:-3])
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, d_model)

    def forward_features(self, x):
        batch_size = x.shape[0]
        h = rearrange(self.res_features(x), "b d n1 n2 -> (b n1 n2) d")
        h = self.res_l2(F.relu(self.res_l1(h)))
        return rearrange(h, "(b n) d -> b n d", b=batch_size)


def build_medklip_resnet(checkpoint_path, d_model=256):
    """MedKLIP image encoder from the official ``checkpoint_final.pth``."""
    model = MedKLIPImageEncoder(d_model=d_model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    load_encoder_weights(model, state, "MedKLIP")
    return model
