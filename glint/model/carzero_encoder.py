"""CARZero's M3AE encoder (ViT-B/16) and inference transform, for the alternative-initialization experiments.

Adapted from CARZero (https://github.com/laihaoran/CARZero), transformer_backbones.py and datasets/utils.py.
Apache License 2.0 (LICENSES/Apache-2.0.txt). Weights: ``VITB-16-M3AE_last.ckpt`` from the CARZero release, under
``$PRETRAINED_DIR/CARZero/``.
"""

import os
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from common import logger, save_json

from .vjepa21_encoder import VisionEncoderOutput

CARZERO_CKPT = "CARZero/VITB-16-M3AE_last.ckpt"


# 2D sin-cos position embedding (CARZero / MAE)
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class MRMEncoder(nn.Module):
    """Encoder half of CARZero's MRM (ViT-B/16, fixed sin-cos position embedding)."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        from timm.models.vision_transformer import Block, PatchEmbed

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)]
        )
        self.norm = norm_layer(embed_dim)
        self.embed_dim = embed_dim

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_tokens(self, x, collect_hidden_states=False):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        hidden_states = [x] if collect_hidden_states else None
        for blk in self.blocks:
            x = blk(x)
            if collect_hidden_states:
                hidden_states.append(x)

        return self.norm(x), hidden_states


def load_carzero_weights(model, ckpt_path):
    """CARZero's ``load_weight``: keep only ``vision_encoder.*`` keys, strip the prefix."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "vision_encoder."
    model_dict = model.state_dict()
    matched = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix) and k[len(prefix) :] in model_dict}

    missing = set(model_dict) - set(matched)
    if missing:
        raise RuntimeError(
            f"CARZero checkpoint {ckpt_path} is missing {len(missing)} encoder tensors: {sorted(missing)[:8]}"
        )

    model_dict.update(matched)
    model.load_state_dict(model_dict)
    logger.info(f"CARZero M3AE: loaded {len(matched)}/{len(model_dict)} tensors from {ckpt_path}")
    return model


def _interpolate_pos_embed(pos_embed, new_grid):
    """Bicubic-resize the patch part of a [1, 1+N, D] position embedding to new_grid x new_grid."""
    cls_pos, patch_pos = pos_embed[:, :1, :], pos_embed[:, 1:, :]
    old_grid = int(round(patch_pos.shape[1] ** 0.5))
    if old_grid == new_grid:
        return pos_embed
    dim = patch_pos.shape[-1]
    patch_pos = patch_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
    return torch.cat([cls_pos, patch_pos], dim=1)


class CARZeroM3AEEncoder(nn.Module):
    """Config: pretrained_name_or_path (relative to pretrained_dir), img_size (224 native), random_init."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        img_size = int(getattr(config, "img_size", 224) or 224)
        random_init = bool(getattr(config, "random_init", False))

        self.encoder = MRMEncoder(
            img_size=img_size,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self._hidden_size = self.encoder.embed_dim
        self._grad_ckpt = False

        if random_init:
            self.encoder.initialize_weights()
            logger.info(f"CARZero M3AE: random init at {img_size}px (no checkpoint)")
            return

        ckpt_path = getattr(config, "pretrained_name_or_path", None) or CARZERO_CKPT
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(getattr(config, "pretrained_dir", ""), ckpt_path)

        if img_size == 224:
            load_carzero_weights(self.encoder, ckpt_path)
        else:
            # load at 224 so the [1, 197, 768] pos_embed matches, then resize it to this grid
            ref = MRMEncoder(img_size=224, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            load_carzero_weights(ref, ckpt_path)
            state = ref.state_dict()
            state["pos_embed"] = _interpolate_pos_embed(state["pos_embed"], img_size // 16)
            self.encoder.load_state_dict(state)
            logger.info(f"CARZero M3AE: pos_embed interpolated 14x14 -> {img_size // 16}x{img_size // 16}")

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def gradient_checkpointing(self):
        return self._grad_ckpt

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self._grad_ckpt = bool(value)

    def save_pretrained(self, save_directory, **kwargs):
        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        save_file(self.state_dict(), os.path.join(save_directory, "model.safetensors"))
        logger.info(f"CARZeroM3AEEncoder saved to {save_directory}")

    def forward(self, pixel_values, output_hidden_states=False, **kwargs):
        if self._grad_ckpt and self.training:
            from torch.utils.checkpoint import checkpoint

            x = self.encoder.patch_embed(pixel_values)
            x = x + self.encoder.pos_embed[:, 1:, :]
            cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
            x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            hidden_states = [x] if output_hidden_states else None
            for blk in self.encoder.blocks:
                x = checkpoint(blk, x, use_reentrant=False)
                if output_hidden_states:
                    hidden_states.append(x)
            tokens = self.encoder.norm(x)
        else:
            tokens, hidden_states = self.encoder.forward_tokens(
                pixel_values, collect_hidden_states=output_hidden_states
            )

        return VisionEncoderOutput(
            last_hidden_state=tokens,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
        )


class M3AEImageProcessor:
    """CARZero's inference transform; zero-shot evaluation keys the similarity-map inversion on this class name."""

    def __init__(
        self,
        resize_size=256,
        crop_size=224,
        image_mean=(0.4978,),
        image_std=(0.2449,),
    ):
        import torchvision.transforms as transforms

        self.resize_size = resize_size
        self.crop_size = crop_size
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)

        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )

    def __call__(self, images, **kwargs):
        if not isinstance(images, (list, tuple)):
            images = [images]

        processed = []
        for image in images:
            if isinstance(image, Image.Image):
                image = np.array(image.convert("L"), dtype=np.uint8)
            image = carzero_resize_img(image, self.resize_size)
            processed.append(self.transform(Image.fromarray(image).convert("RGB")))

        return {"pixel_values": torch.stack(processed)}

    def save_pretrained(self, save_dir, **kwargs):
        os.makedirs(save_dir, exist_ok=True)
        save_json(
            {
                "image_processor_type": "M3AEImageProcessor",
                "resize_size": self.resize_size,
                "crop_size": self.crop_size,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
            },
            os.path.join(save_dir, "preprocessor_config.json"),
        )


def carzero_resize_img(img, scale):
    """Longest-edge resize to `scale`, then zero-pad to a square (from CARZero datasets/utils.py)."""
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    if max_ind == 0:
        wpercent = scale / float(size[0])
        desireable_size = (scale, int(float(size[1]) * float(wpercent)))
    else:
        hpercent = scale / float(size[1])
        desireable_size = (int(float(size[0]) * float(hpercent)), scale)
    resized_img = cv2.resize(img, desireable_size[::-1], interpolation=cv2.INTER_AREA)

    if max_ind == 0:
        pad_size = scale - resized_img.shape[1]
        left, right, top, bottom = int(np.floor(pad_size / 2)), int(np.ceil(pad_size / 2)), 0, 0
    else:
        pad_size = scale - resized_img.shape[0]
        left, right, top, bottom = 0, 0, int(np.floor(pad_size / 2)), int(np.ceil(pad_size / 2))

    return np.pad(resized_img, [(top, bottom), (left, right)], "constant", constant_values=0)
