"""V-JEPA 2.1 encoder.

Adapted from https://github.com/facebookresearch/vjepa2
Copyright (c) Meta Platforms, Inc. and affiliates. MIT License (LICENSES/MIT.txt)
drop_path from timm (https://github.com/huggingface/pytorch-image-models).
Copyright 2020 Ross Wightman. Apache License 2.0 (LICENSES/Apache-2.0.txt)
"""

import math

import torch

from common import logger


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)], dim=0)
    return x


def apply_masks(x, masks, concat=True):
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)


import torch.nn as nn  # noqa: E402


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        patch_size=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402


def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def rotate_queries_or_keys(x, pos, n_registers, has_cls_first):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    n_cls = 1 if has_cls_first else 0
    start_ctx = n_cls
    end_ctx = N - n_registers

    x_cls = x[..., :n_cls, :] if n_cls else None
    x_ctx = x[..., start_ctx:end_ctx, :]
    x_reg = x[..., end_ctx:, :] if n_registers > 0 else None

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)

    emb_sin = freq.sin()
    emb_cos = freq.cos()

    emb_sin = emb_sin.repeat_interleave(2, dim=-1)
    emb_cos = emb_cos.repeat_interleave(2, dim=-1)

    y = x_ctx.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(
        dim=-1,
    )
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    out_ctx = (x_ctx * emb_cos) + (y * emb_sin)

    parts = []
    if n_cls:
        parts.append(x_cls)
    parts.append(out_ctx)
    if n_registers:
        parts.append(x_reg)
    out = torch.cat(parts, dim=-2)

    return out


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        drop=0.0,
        wide_silu=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        patch_size=16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal
        self.n_registers = n_registers
        self.has_cls_first = has_cls_first
        self.interpolate_rope = interpolate_rope
        self.pretrained_patch_size = patch_size
        if patch_size == 14:
            self.pretrained_grid_size = int(252 / patch_size)
        elif patch_size == 16:
            self.pretrained_grid_size = int(256 / patch_size)

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(
        self,
        x,
        mask=None,
        T=None,
        H_patches=None,
        W_patches=None,
        return_attn=False,
    ):
        B, N, C = x.size()
        N_ctx = N - self.n_registers
        grid_depth = int(N_ctx // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        if self.interpolate_rope:
            if H_patches is None:
                H_patches = int(self.grid_size)
            if W_patches is None:
                W_patches = int(self.grid_size)
            h_mask = h_mask * (self.pretrained_grid_size - 1) / (H_patches - 1)
            w_mask = w_mask * (self.pretrained_grid_size - 1) / (W_patches - 1)

        s = 0
        qd = rotate_queries_or_keys(
            q[..., s : s + self.d_dim],
            pos=d_mask,
            n_registers=self.n_registers,
            has_cls_first=self.has_cls_first,
        )
        kd = rotate_queries_or_keys(
            k[..., s : s + self.d_dim],
            pos=d_mask,
            n_registers=self.n_registers,
            has_cls_first=self.has_cls_first,
        )
        s += self.d_dim
        qh = rotate_queries_or_keys(
            q[..., s : s + self.h_dim],
            pos=h_mask,
            n_registers=self.n_registers,
            has_cls_first=self.has_cls_first,
        )
        kh = rotate_queries_or_keys(
            k[..., s : s + self.h_dim],
            pos=h_mask,
            n_registers=self.n_registers,
            has_cls_first=self.has_cls_first,
        )
        s += self.h_dim
        qw = rotate_queries_or_keys(
            q[..., s : s + self.w_dim],
            pos=w_mask,
            n_registers=self.n_registers,
            has_cls_first=self.has_cls_first,
        )
        kw = rotate_queries_or_keys(
            k[..., s : s + self.w_dim],
            pos=w_mask,
            n_registers=self.n_registers,
            has_cls_first=self.has_cls_first,
        )
        s += self.w_dim

        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if self.use_sdpa:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        else:
            return x, None


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        patch_size=16,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_rope = use_rope
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
                n_registers=n_registers,
                has_cls_first=has_cls_first,
                interpolate_rope=interpolate_rope,
                patch_size=patch_size,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                wide_silu=wide_silu,
                drop=drop,
            )
        else:
            self.mlp = MLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )

    def forward(
        self,
        x,
        mask=None,
        T=None,
        H_patches=None,
        W_patches=None,
        return_attn=False,
        mode="video",
    ):
        if self.use_rope:
            y, attn = self.attn(
                self.norm1(x),
                mask=mask,
                T=T,
                H_patches=H_patches,
                W_patches=W_patches,
                return_attn=return_attn,
            )
        else:
            y = self.attn(self.norm1(x))
            attn = None
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attn:
            return x, attn
        else:
            return x, None


class Lambda_LinearWarmupHold:
    def __init__(self, lambda_value: float, start_iter: int = 15_000, end_iter: int = 30_000):
        assert end_iter > start_iter, "end_iter must be > start_iter"
        self.lambda_value = float(lambda_value)
        self.start = int(start_iter)
        self.end = int(end_iter)
        self.span = self.end - self.start

    def value(self, global_iter: int) -> float:
        if global_iter < self.start:
            return 0.0
        if global_iter >= self.end:
            return self.lambda_value
        alpha = (global_iter - self.start) / self.span
        return self.lambda_value * alpha


from functools import partial  # noqa: E402


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        is_causal=False,
        use_rope=False,
        init_type: str = "default",
        handle_nonsquare_inputs=True,
        img_temporal_dim_size=None,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        modality_embedding=True,
        n_output_distillation=4,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.init_type = init_type
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.img_temporal_dim_size = img_temporal_dim_size

        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            self.num_patches = (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
        else:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=1,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        if self.img_temporal_dim_size is not None:
            if not isinstance(self.img_temporal_dim_size, int):
                raise ValueError(f"img_temporal_dim_size must be an int, got {self.img_temporal_dim_size}")
            self.patch_embed_img = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=1,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed_img = None

        self.uniform_power = uniform_power

        self.use_rope = use_rope
        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=img_size[0] // patch_size,
                    grid_depth=num_frames // tubelet_size,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=use_sdpa,
                    is_causal=is_causal,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    n_registers=n_registers,
                    has_cls_first=has_cls_first,
                    interpolate_rope=interpolate_rope,
                    patch_size=patch_size,
                )
                for i in range(depth)
            ]
        )

        self.attn_out = False
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        if depth == 12:
            self.hierarchical_layers = [2, 5, 8, 11]
            if n_output_distillation == 4:
                self.out_layers_distillation = [2, 5, 8, 11]
            elif n_output_distillation == 1:
                self.out_layers_distillation = [11]

        elif depth == 24:
            self.hierarchical_layers = [5, 11, 17, 23]
            if n_output_distillation == 4:
                self.out_layers_distillation = [5, 11, 17, 23]
            elif n_output_distillation == 1:
                self.out_layers_distillation = [23]

        elif depth == 40:
            self.hierarchical_layers = [9, 19, 29, 39]
            if n_output_distillation == 4:
                self.out_layers_distillation = [9, 19, 29, 39]
            elif n_output_distillation == 1:
                self.out_layers_distillation = [39]

        elif depth == 48:
            self.hierarchical_layers = [11, 23, 37, 47]
            if n_output_distillation == 4:
                self.out_layers_distillation = [11, 23, 37, 47]
            elif n_output_distillation == 1:
                self.out_layers_distillation = [47]
        else:
            print("Check the code! ;)")
        self.norms_block = nn.ModuleList([norm_layer(embed_dim) for _ in range(len(self.hierarchical_layers))])

        self.cls_token = None
        self.return_hierarchical = False

        self.modality_embedding = False
        if modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.img_mod_embed, std=1e-6)
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            self.modality_embedding = True

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            return
        if self.init_type == "default":
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=self.init_std)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        elif self.init_type == "xavier_uniform":
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif self.init_type == "xavier_normal":
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            raise ValueError(f"Unknown init type {self.init_type}")

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def check_temporal_dim(self, shape) -> bool:
        if self.img_temporal_dim_size is not None:
            if shape[2] == self.img_temporal_dim_size:
                return True

        return False

    def forward(self, x, masks=None, training=False):
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        if x.ndim == 4:
            _, _, H, W = x.shape
            T = 1
        elif x.ndim == 5:
            _, _, T, H, W = x.shape
            if self.check_temporal_dim(x.shape):
                T = T // 1
            else:
                T = T // self.tubelet_size

        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if not self.handle_nonsquare_inputs:
            T = H_patches = W_patches = None

        if self.check_temporal_dim(x.shape):
            assert self.patch_embed_img is not None
            x = self.patch_embed_img(x)
            mode = "img"
            if self.modality_embedding:
                x += self.img_mod_embed.repeat(x.shape[0], 1, 1)
        else:
            x = self.patch_embed(x)
            mode = "video"
            if self.modality_embedding:
                x += self.video_mod_embed.repeat(x.shape[0], 1, 1)

        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        outs = []
        hier = []
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x, attn = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    use_reentrant=False,
                    return_attn=self.attn_out,
                    mode=mode,
                )
            else:
                x, attn = blk(
                    x,
                    mask=masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    return_attn=self.attn_out,
                    mode=mode,
                )

            if self.out_layers is not None and i in self.out_layers:
                out_idx = self.hierarchical_layers.index(i)
                out_norm = self.norms_block[out_idx](x)
                outs.append(out_norm)

            if i in self.out_layers_distillation:
                out_idx = self.hierarchical_layers.index(i)
                hier.append(self.norms_block[out_idx](x))

        if self.out_layers is not None:
            return outs

        if training or self.return_hierarchical:
            hier = torch.cat(hier, dim=2)
            return hier
        else:
            x = self.norms_block[-1](x)
            return x


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_giant_xformers(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=22,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_gigantic_xformers(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1664,
        depth=48,
        num_heads=26,
        mlp_ratio=64 / 13,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


VIT_EMBED_DIMS = {
    "vit_base": 768,
    "vit_large": 1024,
    "vit_huge": 1280,
    "vit_giant_xformers": 1408,
    "vit_gigantic_xformers": 1664,
}

ARCH_MAP = {
    "vjepa2_1_vit_base_384": (vit_base, 1),
    "vjepa2_1_vit_large_384": (vit_large, 1),
    "vjepa2_1_vit_giant_384": (vit_giant_xformers, 4),
    "vjepa2_1_vit_gigantic_384": (vit_gigantic_xformers, 4),
}


class VisionEncoderOutput:
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class VJEPA21Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        arch = getattr(config, "arch", "vjepa2_1_vit_large_384")
        img_size = getattr(config, "img_size", 384)
        num_frames = getattr(config, "num_frames", 80)
        n_out = getattr(config, "n_output_distillation", None)
        use_ckpt = getattr(config, "use_activation_checkpointing", False)

        if arch not in ARCH_MAP:
            raise ValueError(f"Unknown arch '{arch}'. Choose from {list(ARCH_MAP.keys())}")
        factory_fn, default_n_out = ARCH_MAP[arch]
        if n_out is None:
            n_out = default_n_out

        self.encoder = factory_fn(
            img_size=(img_size, img_size),
            num_frames=num_frames,
            tubelet_size=2,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
            n_output_distillation=n_out,
            use_activation_checkpointing=use_ckpt,
        )

        self.use_hierarchical = getattr(config, "use_hierarchical", False)
        self.encoder.return_hierarchical = self.use_hierarchical

        n_levels = len(self.encoder.out_layers_distillation)
        self._hidden_size = self.encoder.embed_dim * n_levels if self.use_hierarchical else self.encoder.embed_dim

        ckpt_path = getattr(config, "pretrained_checkpoint", None)
        if ckpt_path:
            import os

            if not os.path.isabs(ckpt_path):
                pretrained_dir = getattr(config, "pretrained_dir", "")
                ckpt_path = os.path.join(pretrained_dir, ckpt_path)
            load_vjepa21_checkpoint(self, ckpt_path)

    @property
    def hidden_size(self):
        return self._hidden_size

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.use_activation_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.encoder.use_activation_checkpointing = False

    def save_pretrained(self, save_directory, **kwargs):
        import os

        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)

        self.config.save_pretrained(save_directory)

        save_file(self.state_dict(), os.path.join(save_directory, "model.safetensors"))
        logger.info(f"VJEPA21Encoder saved to {save_directory}")

    def forward(self, pixel_values, output_hidden_states=False, **kwargs):
        x = pixel_values.permute(0, 2, 1, 3, 4).contiguous()

        if output_hidden_states:
            if self.encoder.check_temporal_dim(x.shape):
                x_tok = self.encoder.patch_embed_img(x)
                if self.encoder.modality_embedding:
                    x_tok = x_tok + self.encoder.img_mod_embed
            else:
                x_tok = self.encoder.patch_embed(x)
                if self.encoder.modality_embedding:
                    x_tok = x_tok + self.encoder.video_mod_embed

            _, _, T, H, W = x.shape
            T_p = T // (1 if self.encoder.check_temporal_dim(x.shape) else self.encoder.tubelet_size)
            H_p, W_p = H // self.encoder.patch_size, W // self.encoder.patch_size

            hidden_states = [x_tok]
            h = x_tok
            for blk in self.encoder.blocks:
                h, _ = blk(h, mask=None, T=T_p, H_patches=H_p, W_patches=W_p)
                hidden_states.append(h)

            if self.use_hierarchical:
                hier = []
                for idx in self.encoder.out_layers_distillation:
                    out_idx = self.encoder.hierarchical_layers.index(idx)
                    hier.append(self.encoder.norms_block[out_idx](hidden_states[idx]))
                features = torch.cat(hier, dim=2)
            else:
                features = self.encoder.norms_block[-1](hidden_states[-1])

            return VisionEncoderOutput(last_hidden_state=features, hidden_states=tuple(hidden_states))

        features = self.encoder(x, masks=None, training=self.use_hierarchical and self.training)
        return VisionEncoderOutput(last_hidden_state=features, hidden_states=None)


def load_vjepa21_checkpoint(model, checkpoint_path, strict=False):
    import os

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"V-JEPA 2.1 checkpoint not found: {checkpoint_path}")

    if os.path.isdir(checkpoint_path):
        st_file = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(st_file):
            raise FileNotFoundError(f"Directory {checkpoint_path} is missing model.safetensors")
        from safetensors.torch import load_file

        logger.info(f"Detected VJEPA21Encoder safetensors dir: {checkpoint_path}")
        state_dict = load_file(st_file)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        n_loaded = len(state_dict) - len(unexpected)
        logger.info(
            f"V-JEPA 2.1 safetensors loaded: {n_loaded} keys, {len(missing)} missing, {len(unexpected)} unexpected"
        )
        if missing:
            logger.info(f"  Missing (first 10): {missing[:10]}")
        if unexpected:
            logger.info(f"  Unexpected (first 10): {unexpected[:10]}")
        return model

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "ema_encoder" in state_dict:
        logger.info("Detected V-JEPA 2.1 format (ema_encoder)")
        state_dict = state_dict["ema_encoder"]
    elif "target_encoder" in state_dict:
        logger.info("Target Encoder only detected")
        state_dict = state_dict["target_encoder"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "").replace("backbone.", "")] = v

    mapped = {f"encoder.{k}": v for k, v in cleaned.items()}

    missing, unexpected = model.load_state_dict(mapped, strict=strict)
    n_loaded = len(mapped) - len(unexpected)
    logger.info(f"V-JEPA 2.1 loaded: {n_loaded} keys, {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        logger.info(f"  Missing (first 10): {missing[:10]}")
    if unexpected:
        logger.info(f"  Unexpected (first 10): {unexpected[:10]}")
    return model
