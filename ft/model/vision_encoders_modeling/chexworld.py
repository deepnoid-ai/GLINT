"""Adapted from CheXWorld (https://github.com/LeapLabTHU/CheXWorld).
No upstream license; rights remain with the original authors.
"""

from functools import partial

import torch
from torch import nn

from .utils import load_encoder_weights


def build_chexworld_vit_b16(checkpoint_path):
    """ViT-B/16 context encoder (``encoder``) of the CheXWorld checkpoint ``chexworld_pretrained.tar``."""
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
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    encoder_sd = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}

    # the I-JEPA-style encoder has no CLS token: keep timm's initial CLS slot of the position embedding
    pos_embed = encoder_sd.get("pos_embed")
    if pos_embed is not None and pos_embed.shape[1] == model.patch_embed.num_patches:
        encoder_sd["pos_embed"] = torch.cat([model.pos_embed.data[:, :1], pos_embed], dim=1)

    load_encoder_weights(model, checkpoint_filter_fn(encoder_sd, model), "CheXWorld")
    return model
