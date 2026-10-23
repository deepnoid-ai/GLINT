"""Adapted from fVLM (https://github.com/alibaba-damo-academy/fvlm; no upstream license, rights remain with the
original authors), lavis/models/blip_models/{vit,patchembedding,selfattention,transformerblock}.py, which modify the
MONAI ViT (https://github.com/Project-MONAI/MONAI). Copyright (c) MONAI Consortium. Apache License 2.0
(LICENSES/Apache-2.0.txt)

Only the image encoder path is kept: conv patch embedding with a learnable position embedding (trilinearly
resized when the token grid differs from 7 x 16 x 11), pre-norm transformer blocks and the final norm.
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from monai.networks.blocks.mlp import MLPBlock
from monai.networks.layers import trunc_normal_
from torch import nn

from .utils import load_encoder_weights


class PatchEmbeddingBlock(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, hidden_size, dropout_rate=0.0):
        super().__init__()
        self.n_patches = int(np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)]))
        self.patch_embeddings = nn.Conv3d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)

    def forward(self, x):
        x = self.patch_embeddings(x)
        dim, nd, nh, nw = x.shape[1:]
        x = x.flatten(2).transpose(-1, -2)

        if nd == 7 and nh == 16 and nw == 11:
            embeddings = x + self.position_embeddings
        else:
            pos_embed = self.position_embeddings.permute(0, 2, 1).view(1, dim, -1, 16, 11)
            pos_embed = F.interpolate(pos_embed, size=(nd, nh, nw), mode="trilinear", align_corners=False)
            embeddings = x + pos_embed.flatten(2).transpose(1, 2)

        return self.dropout(embeddings)


class SABlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.0, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads
        inner_dim = self.dim_head * num_heads

        self.out_proj = nn.Linear(inner_dim, hidden_size)
        self.qkv = nn.Linear(hidden_size, inner_dim * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.scale = self.dim_head**-0.5

    def forward(self, x):
        q, k, v = self.input_rearrange(self.qkv(x))
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        x = torch.einsum("bhxy,bhyd->bhxd", self.drop_weights(att_mat), v)
        return self.drop_output(self.out_proj(self.out_rearrange(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, mlp_dim, num_heads, dropout_rate=0.0, qkv_bias=False):
        super().__init__()
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class FVLMViT(nn.Module):
    """fVLM ViT-B image encoder over (B, 1, 112, 256, 352) volumes -> (B, 7 * 16 * 11, 768) tokens."""

    def __init__(
        self,
        in_channels=1,
        img_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        dropout_rate=0.0,
        qkv_bias=True,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbeddingBlock(in_channels, img_size, patch_size, hidden_size, dropout_rate)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def build_fvlm_encoder(checkpoint_path, **kwargs):
    """fVLM image encoder (``visual_encoder``) of the official CT-RATE checkpoint ``model.pth``."""
    model = FVLMViT(**kwargs)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prefix = "visual_encoder."
    state = {k.removeprefix(prefix): v for k, v in ckpt["model"].items() if k.startswith(prefix)}
    load_encoder_weights(model, state, "fVLM")
    return model
