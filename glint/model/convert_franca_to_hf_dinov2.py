"""Convert Franca's ViT-B/14 checkpoint (https://github.com/valeoai/Franca, Apache-2.0) into an HF Dinov2Model directory
usable as ``vision_config.model_type: dinov2`` (``--validate`` proves the remap on Meta's DINOv2 checkpoints).

    python -m glint.model.convert_franca_to_hf_dinov2 --validate --ckpt url --out $PRETRAINED_DIR/franca-vitb14-in21k
"""

import argparse
import os
import re
import sys

import torch

FRANCA_URL = "https://github.com/valeoai/Franca/releases/download/v1.0.0/franca_vitb14_In21K.pth"
META_DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
META_DINOV2G_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth"
BLOCK_RE = re.compile(r"^blocks\.(?:(\d+)\.)?(\d+)\.(.+)$")


def strip_prefixes(state_dict):
    """Franca stores {'teacher': {...}} with module./backbone. prefixes and a RASA head."""
    if "teacher" in state_dict:
        state_dict = state_dict["teacher"]
    out = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        if k.startswith("rasa_head.") or k.startswith("dino_head.") or k.startswith("ibot_head."):
            continue
        out[k] = v
    return out


def remap(state_dict):
    """Meta/Franca DINOv2 keys -> HF Dinov2Model keys; the in-chunk block index is already absolute."""
    top = {
        "cls_token": "embeddings.cls_token",
        "mask_token": "embeddings.mask_token",
        "pos_embed": "embeddings.position_embeddings",
        "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
        "patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
        "norm.weight": "layernorm.weight",
        "norm.bias": "layernorm.bias",
    }
    within = {
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
        "attn.proj.weight": "attention.output.dense.weight",
        "attn.proj.bias": "attention.output.dense.bias",
        "ls1.gamma": "layer_scale1.lambda1",
        "ls2.gamma": "layer_scale2.lambda1",
        "mlp.fc1.weight": "mlp.fc1.weight",
        "mlp.fc1.bias": "mlp.fc1.bias",
        "mlp.fc2.weight": "mlp.fc2.weight",
        "mlp.fc2.bias": "mlp.fc2.bias",
        # SwiGLU FFN (Franca ViT-B); HF Dinov2SwiGLUFFN is DINOv2's SwiGLUFFNFused, renamed
        "mlp.w12.weight": "mlp.weights_in.weight",
        "mlp.w12.bias": "mlp.weights_in.bias",
        "mlp.w3.weight": "mlp.weights_out.weight",
        "mlp.w3.bias": "mlp.weights_out.bias",
    }

    seen_layers = set()
    for key in state_dict:
        m = BLOCK_RE.match(key)
        if m:
            seen_layers.add(int(m.group(2)))
    if seen_layers and seen_layers != set(range(len(seen_layers))):
        raise ValueError(f"block indices are not a contiguous 0..N-1 run: {sorted(seen_layers)}")

    out, unhandled = {}, []
    for key, val in state_dict.items():
        if key in top:
            out[top[key]] = val
            continue

        m = BLOCK_RE.match(key)
        if not m:
            unhandled.append(key)
            continue

        prefix = f"encoder.layer.{int(m.group(2))}"
        rest = m.group(3)

        if rest in within:
            out[f"{prefix}.{within[rest]}"] = val
        elif rest in ("attn.qkv.weight", "attn.qkv.bias"):
            # HF keeps query/key/value as three separate projections.
            dim = val.shape[0] // 3
            suffix = rest.rsplit(".", 1)[1]
            for i, name in enumerate(("query", "key", "value")):
                out[f"{prefix}.attention.attention.{name}.{suffix}"] = val[i * dim : (i + 1) * dim]
        else:
            unhandled.append(key)

    if unhandled:
        raise ValueError(f"{len(unhandled)} keys have no HF mapping: {sorted(unhandled)[:10]}")
    return out


def build_hf_model(hf_state_dict, base="facebook/dinov2-base"):
    from transformers import Dinov2Config, Dinov2Model

    # base config matches ViT-B/14 @518; only the FFN type is read from the remapped keys
    config = Dinov2Config.from_pretrained(base)
    config.use_swiglu_ffn = any(".mlp.weights_in." in k for k in hf_state_dict)
    model = Dinov2Model(config)
    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    missing = [k for k in missing if not k.startswith("pooler.")]
    if missing or unexpected:
        raise ValueError(f"state_dict mismatch — missing={missing[:8]} unexpected={unexpected[:8]}")
    return model, config


def load_url(url, name):
    from torch.hub import load_state_dict_from_url

    print(f"  downloading {name} ...")
    return load_state_dict_from_url(url, map_location="cpu", weights_only=True, file_name=name)


def validate(arch="base", chunked=False):
    """The same remap on Meta's checkpoint must reproduce HF's own conversion exactly (giant covers SwiGLU)."""
    from transformers import Dinov2Model

    url, name, hf_id = {
        "base": (META_DINOV2_URL, "dinov2_vitb14_pretrain.pth", "facebook/dinov2-base"),
        "giant": (META_DINOV2G_URL, "dinov2_vitg14_pretrain.pth", "facebook/dinov2-giant"),
    }[arch]

    print(f"Validating the remap ({arch}{', chunked' if chunked else ''}) against {hf_id}")
    meta = strip_prefixes(load_url(url, name))
    print(f"  Meta checkpoint tensors: {len(meta)}")

    if chunked:
        # Franca uses BlockChunk naming; re-chunk so that branch is the one under test
        meta = {
            (re.sub(r"^blocks\.(\d+)\.", lambda m: f"blocks.{int(m.group(1)) // 3}.{m.group(1)}.", k)): v
            for k, v in meta.items()
        }
        print(f"  re-chunked to BlockChunk naming, e.g. {[k for k in meta if k.startswith('blocks.1.')][0]}")

    ours, cfg = build_hf_model(remap(meta), base=hf_id)
    print(f"  use_swiglu_ffn detected: {cfg.use_swiglu_ffn}")
    ours.eval()
    official = Dinov2Model.from_pretrained(hf_id).eval()

    a, b = ours.state_dict(), official.state_dict()
    shared = [k for k in b if not k.startswith("pooler.")]
    diffs = {k: (a[k] - b[k]).abs().max().item() for k in shared if k in a}
    worst = max(diffs.values())
    print(f"  weights compared: {len(diffs)}   max|delta| = {worst:.3e}")

    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        fa = ours(x).last_hidden_state
        fb = official(x).last_hidden_state
    delta = (fa - fb).abs().max().item()
    print(f"  forward output {tuple(fa.shape)}   max|delta| = {delta:.3e}")

    ok = worst == 0.0 and delta == 0.0
    print("\nPASS — the remap reproduces the official conversion exactly" if ok else "\nFAIL")
    return ok


def convert(ckpt, out_dir):
    from transformers import AutoImageProcessor

    print(f"Converting {ckpt}")
    if ckpt == "url":
        raw = load_url(FRANCA_URL, "franca_vitb14_In21K.pth")
    else:
        raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    src = strip_prefixes(raw)
    print(f"  Franca tensors after prefix strip: {len(src)}")

    model, config = build_hf_model(remap(src))
    pos = model.embeddings.position_embeddings.shape
    print(f"  built Dinov2Model — pos_embed {tuple(pos)}, patch {config.patch_size}, hidden {config.hidden_size}")

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    # same processor as the DINOv2 arms
    AutoImageProcessor.from_pretrained("facebook/dinov2-base").save_pretrained(out_dir)
    print(f"  saved -> {out_dir}")
    print(f"  contents: {sorted(os.listdir(out_dir))}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--validate", action="store_true", help="prove the remap against facebook/dinov2-base (standard MLP)"
    )
    p.add_argument(
        "--validate-swiglu",
        action="store_true",
        help="also prove the SwiGLU path against facebook/dinov2-giant (4.5 GB)",
    )
    p.add_argument("--ckpt", help="Franca .pth path, or 'url' to fetch the release asset")
    p.add_argument("--out", help="output HF model directory")
    args = p.parse_args()

    ok = True
    if args.validate:
        ok &= validate("base")
        ok &= validate("base", chunked=True)
    if args.validate_swiglu:
        ok &= validate("giant")
    if args.ckpt:
        if not args.out:
            p.error("--ckpt requires --out")
        ok &= convert(args.ckpt, args.out)
    if not (args.validate or args.validate_swiglu or args.ckpt):
        p.error("nothing to do — pass --validate and/or --ckpt/--out")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
