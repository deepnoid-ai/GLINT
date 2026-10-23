"""Adapted from CARZero (https://github.com/laihaoran/CARZero; Apache License 2.0, LICENSES/Apache-2.0.txt), which ports
the MRM encoder (https://github.com/RL4M/MRM-pytorch; Copyright (c) 2023 DopamineLcy, MIT License, LICENSES/MIT.txt),
itself built on MAE (https://github.com/facebookresearch/mae; Copyright (c) Meta Platforms, Inc. and affiliates,
CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/).
"""

from functools import partial

import torch
from torch import nn

from .utils import load_encoder_weights


class MRMEncoder(nn.Module):
    """ViT-B/16 image encoder of MRM (masked record modeling); the MAE/report decoders are dropped."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        from timm.models.vision_transformer import Block, PatchEmbed

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # fixed sin-cos embedding, restored from the checkpoint
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)]
        )
        self.norm = norm_layer(embed_dim)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def build_mrm_vit_b16(checkpoint_path):
    """MRM ViT-B/16 encoder from the official ``MRM.pth``."""
    model = MRMEncoder()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_encoder_weights(model, ckpt.get("model", ckpt), "MRM")
    return model
