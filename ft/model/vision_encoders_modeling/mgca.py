"""Adapted from MGCA (https://github.com/HKU-MedAI/MGCA).
No upstream license; rights remain with the original authors.
"""

from functools import partial

import torch
from torch import nn

from .utils import load_encoder_weights


def build_mgca_vit_b16(checkpoint_path):
    """ViT-B/16 image encoder (``img_encoder_q.model``) of the MGCA checkpoint ``vit_base.ckpt``."""
    from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn

    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    prefix = "img_encoder_q.model."
    encoder_sd = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    load_encoder_weights(model, checkpoint_filter_fn(encoder_sd, model), "MGCA")
    return model
