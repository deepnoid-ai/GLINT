"""Adapted from COLIPRI (https://huggingface.co/microsoft/colipri), src/colipri/model/image.py and its backbone
config; Primus is from dynamic-network-architectures (https://github.com/MIC-DKFZ/dynamic-network-architectures).
Copyright (c) Microsoft Corporation. MIT License (LICENSES/MIT.txt)
"""

from pathlib import Path

import safetensors.torch
from einops import rearrange
from torch import nn

from .utils import load_encoder_weights


class CoLiPriEncoder(nn.Module):
    """(B, 1, D, H, W) volume -> {"last_hidden_state": (B, D/8 * H/8 * W/8, 864)} Primus-M patch tokens."""

    def __init__(self, checkpoint_path=None, embed_dim=864, img_size=(160, 160, 160)):
        super().__init__()
        from dynamic_network_architectures.architectures.primus import Primus

        self.backbone = Primus(
            input_channels=1,
            num_classes=1,
            eva_depth=16,
            eva_numheads=12,
            embed_dim=embed_dim,
            patch_embed_size=[8, 8, 8],
            input_shape=list(img_size),
            use_rot_pos_emb=True,
            use_abs_pos_embed=False,
            drop_path_rate=0.2,
            init_values=0.1,
            scale_attn_inner=True,
            num_register_tokens=0,
        )
        # as in COLIPRI's ImageEncoder: drop the Primus decoder and keep the token feature map
        self.backbone.up_projection = nn.Identity()

        if checkpoint_path:
            path = Path(checkpoint_path)
            if path.is_dir():
                path = path / "model.safetensors"
            tensors = safetensors.torch.load_file(str(path))
            prefix = "image_encoder.backbone."
            state = {k.removeprefix(prefix): v for k, v in tensors.items() if k.startswith(prefix)}
            load_encoder_weights(self.backbone, state, "COLIPRI")

    def forward(self, x):
        features = self.backbone(x)
        return {"last_hidden_state": rearrange(features, "b c d h w -> b (d h w) c")}
