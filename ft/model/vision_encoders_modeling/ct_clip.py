"""Adapted from CT-CLIP (https://github.com/ibrahimethemhamamci/CT-CLIP), transformer_maskgit/ctvit.py and attention.py.
CT-CLIP is released under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/); this file is shared
under the same license. CTViT originates from phenaki-pytorch (https://github.com/lucidrains/phenaki-pytorch),
Copyright (c) 2022 Phil Wang. MIT License (LICENSES/MIT.txt)

Only the CTViT encoder path used by CT-CLIP's image tower is kept (patch embedding, spatial/temporal
transformers, vector quantizer); the VQ-GAN decoder, discriminator, MaskGIT and text parts are dropped.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn

from .utils import load_encoder_weights


def l2norm(t):
    return F.normalize(t, dim=-1)


class LayerNorm(nn.Module):
    """Bias-less layer norm."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, dropout=0.0):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PEG(nn.Module):
    """Position-generating depthwise 3D conv."""

    def __init__(self, dim, causal=False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    def forward(self, x, shape: Tuple[int, int, int, int] = None):
        needs_shape = x.ndim == 3
        orig_shape = x.shape

        if needs_shape:
            x = x.reshape(*shape, -1)

        x = rearrange(x, "b ... d -> b d ...")
        frame_padding = (2, 0) if self.causal else (1, 1)
        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value=0.0)
        x = self.dsconv(x)
        x = rearrange(x, "b d ... -> b ... d")

        if needs_shape:
            x = rearrange(x, "b ... d -> b (...) d")

        return x.reshape(orig_shape)


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_null_kv=0, dropout=0.0, scale=8):
        super().__init__()
        self.heads = heads
        self.scale = scale
        inner_dim = dim_head * heads

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim)

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, attn_bias=None):
        batch = x.shape[0]
        # as upstream: keys/values come from the un-normalized input, queries from the normalized one
        kv_input = x
        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        nk, nv = repeat(self.null_kv, "h (n r) d -> b h n r d", b=batch, r=2).unbind(dim=-2)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if attn_bias is not None:
            sim = sim + F.pad(attn_bias, (self.num_null_kv, 0), value=0.0)

        attn = self.attn_dropout(sim.softmax(dim=-1))
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class ContinuousPositionBias(nn.Module):
    """Log-distance continuous relative position bias (https://arxiv.org/abs/2111.09883)."""

    def __init__(self, *, dim, heads, num_dims=2, layers=2, log_dist=True):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([nn.Sequential(nn.Linear(num_dims, dim), nn.LeakyReLU(0.1))])
        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(0.1)))
        self.net.append(nn.Linear(dim, heads))

    def forward(self, *dimensions, device=torch.device("cpu")):
        positions = [torch.arange(d, device=device) for d in dimensions]
        grid = rearrange(torch.stack(torch.meshgrid(*positions, indexing="ij")), "c ... -> (...) c")
        rel_pos = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")
        if self.log_dist:
            rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

        rel_pos = rel_pos.to(torch.float32)
        for layer in self.net:
            rel_pos = layer(rel_pos.float())
        return rearrange(rel_pos, "i j h -> h i j")


class Transformer(nn.Module):
    def __init__(
        self, dim, *, depth, dim_head=64, heads=8, ff_mult=4, peg_causal=False, attn_dropout=0.0, ff_dropout=0.0
    ):
        super().__init__()
        # [PEG, self-attention, cross-attention (unused, kept for checkpoint key layout), feed-forward]
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PEG(dim=dim, causal=peg_causal),
                        Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout),
                        None,
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm_out = LayerNorm(dim)

    def forward(self, x, video_shape: Tuple[int, int, int, int] = None, attn_bias=None):
        for peg, self_attn, _, ff in self.layers:
            x = peg(x, shape=video_shape) + x
            x = self_attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x
        return self.norm_out(x)


class CTViT(nn.Module):
    """CT-CLIP image encoder: factorized spatial/temporal ViT over (B, 1, D, H, W) volumes, followed by the VQ codebook.

    Returns the quantized tokens (B, D / temporal_patch_size, H / patch_size, W / patch_size, dim), i.e. what
    CT-CLIP's ``CTViT(..., return_encoded_tokens=True)`` returns.
    """

    def __init__(
        self,
        *,
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
        channels=1,
    ):
        super().__init__()
        from vector_quantize_pytorch import VectorQuantize

        self.image_size = (image_size, image_size)
        self.patch_size = (patch_size, patch_size)
        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        patch_dim = channels * patch_size * patch_size * temporal_patch_size
        self.to_patch_emb = nn.Sequential(
            Rearrange(
                "b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)",
                p1=patch_size,
                p2=patch_size,
                pt=temporal_patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        transformer_kwargs = dict(dim=dim, dim_head=dim_head, heads=heads, peg_causal=True)
        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)
        self.vq = VectorQuantize(dim=dim, codebook_size=codebook_size, use_cosine_sim=True)

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(self, tokens):
        b = tokens.shape[0]
        h, w = self.patch_height_width
        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        tokens = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)
        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)

        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")
        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)
        return rearrange(tokens, "(b h w) t d -> b t h w d", b=b, h=h, w=w)

    def forward(self, video):
        assert tuple(video.shape[-2:]) == self.image_size, f"CTViT expects {self.image_size} slices"

        tokens = self.to_patch_emb(video)
        *_, h, w, _ = tokens.shape
        tokens = self.encode(tokens)

        tokens, _ = pack([tokens], "b * d")
        tokens, _, _ = self.vq(tokens)
        return rearrange(tokens, "b (t h w) d -> b t h w d", h=h, w=w)


def build_ct_clip_encoder(checkpoint_path):
    """CTViT image encoder (``visual_transformer``) of the official ``CT-CLIP_v2.pt``."""
    model = CTViT()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prefix = "visual_transformer."
    state = {k.removeprefix(prefix): v for k, v in ckpt.items() if k.startswith(prefix)}
    load_encoder_weights(model, state, "CT-CLIP")
    return model
