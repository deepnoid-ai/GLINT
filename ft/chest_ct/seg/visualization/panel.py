"""Seg panels: GT-containing slices as rows of CT | GT overlay | Pred overlay."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _get_font(size: int = 28) -> ImageFont.ImageFont:
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            _FONT_CACHE[size] = ImageFont.load_default(size=size)
    return _FONT_CACHE[size]


def ct_slice_to_pil(slice_u8: np.ndarray) -> Image.Image:
    return Image.fromarray(slice_u8).convert("RGB")


def overlay_mask(
    slice_img: Image.Image, mask_np: np.ndarray, color: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.45
) -> Image.Image:
    mask_bool = mask_np.astype(bool)
    if not mask_bool.any():
        return slice_img.copy()
    base = slice_img.copy().convert("RGBA")
    overlay = np.zeros((*mask_bool.shape, 4), dtype=np.uint8)
    overlay[mask_bool] = [*color, int(255 * alpha)]
    overlay_img = Image.fromarray(overlay, "RGBA").resize(base.size, Image.NEAREST)
    return Image.alpha_composite(base, overlay_img).convert("RGB")


def prob_to_color_pil(prob: np.ndarray, vmin: float = 0.0, vmax: float = 1.0, cmap_name: str = "magma") -> Image.Image:
    normalized = np.clip((prob - vmin) / (vmax - vmin + 1e-8), 0, 1)
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(normalized)
    return Image.fromarray((colored[..., :3] * 255).astype(np.uint8))


def create_side_by_side(
    images: list[Image.Image],
    labels: list[str] | None = None,
    padding: int = 4,
    label_height: int = 40,
    font_size: int = 28,
) -> Image.Image:
    target_h = max(img.size[1] for img in images)
    resized = []
    for img in images:
        if img.size[1] != target_h:
            ratio = target_h / img.size[1]
            img = img.resize((int(img.size[0] * ratio), target_h), Image.BILINEAR)
        resized.append(img)
    total_w = sum(img.size[0] for img in resized) + padding * (len(resized) - 1)
    extra_h = label_height if labels else 0
    canvas = Image.new("RGB", (total_w, target_h + extra_h), (255, 255, 255))
    font = _get_font(font_size)
    draw = ImageDraw.Draw(canvas)
    x = 0
    for i, img in enumerate(resized):
        if labels and i < len(labels):
            draw.text((x + 4, 4), labels[i], fill=(0, 0, 0), font=font)
        canvas.paste(img, (x, extra_h))
        x += img.size[0] + padding
    return canvas


def create_multi_row(
    rows: list[Image.Image],
    row_padding: int = 6,
    header: str | None = None,
    header_height: int = 60,
    header_font_size: int = 40,
) -> Image.Image | None:
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    max_w = max(r.size[0] for r in rows)
    extra = header_height if header else 0
    total_h = sum(r.size[1] for r in rows) + row_padding * (len(rows) - 1) + extra
    canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    if header:
        ImageDraw.Draw(canvas).text((8, 8), header, fill=(0, 0, 0), font=_get_font(header_font_size))
    y = extra
    for row in rows:
        x = (max_w - row.size[0]) // 2
        canvas.paste(row, (x, y))
        y += row.size[1] + row_padding
    return canvas


# colors for classes 1, 2, ... (background skipped), cycled when there are more classes
_CLASS_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 128, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def overlay_multiclass(ct_pil: Image.Image, mask_np: np.ndarray, alpha: float = 0.45) -> Image.Image:
    out = ct_pil
    for c in np.unique(mask_np):
        c = int(c)
        if c == 0:
            continue
        color = _CLASS_COLORS[(c - 1) % len(_CLASS_COLORS)]
        out = overlay_mask(out, (mask_np == c).astype(np.uint8), color=color, alpha=alpha)
    return out


def hu_to_uint8(volume_np: np.ndarray, window_center: float = 40.0, window_width: float = 400.0) -> np.ndarray:
    """CT HU → uint8 for display (default: soft-tissue window)."""
    lo = window_center - window_width / 2
    hi = window_center + window_width / 2
    v = np.clip(volume_np, lo, hi)
    return ((v - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)


def pick_gt_slices(gt_mask: np.ndarray, max_slices: int | None = None) -> list[int]:
    """Axis-0 indices with GT foreground, evenly subsampled to max_slices."""
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    flat = gt_mask.reshape(gt_mask.shape[0], -1)
    cand = np.where(flat.sum(axis=1) > 0)[0]
    if cand.size == 0:
        return []
    if max_slices is None or cand.size <= max_slices:
        return [int(z) for z in cand]
    idx = np.linspace(0, cand.size - 1, max_slices).round().astype(int)
    return [int(cand[i]) for i in idx]


def case_dice(pred: np.ndarray, label: np.ndarray, num_classes: int) -> dict[int, float]:
    """Same "present in GT" rule as inference._compute_case_dice: NaN for classes absent from the GT."""
    out: dict[int, float] = {}
    for c in range(1, num_classes):
        p = pred == c
        lbl = label == c
        gt_vox = lbl.sum()
        if gt_vox == 0:
            out[c] = float("nan")
            continue
        s = p.sum() + gt_vox
        out[c] = float(2.0 * (p & lbl).sum() / s)
    return out


def save_seg_panel(
    volume_hu: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: str,
    header: str | None = None,
    max_slices: int | None = None,
    window_center: float = 40.0,
    window_width: float = 400.0,
) -> bool:
    """Inputs are (D, H, W) with axis 0 = slice; returns False when there are no GT slices."""
    z_list = pick_gt_slices(gt_mask, max_slices=max_slices)
    if not z_list:
        return False

    vol_u8 = hu_to_uint8(volume_hu, window_center, window_width)

    rows = []
    for z in z_list:
        ct_pil = ct_slice_to_pil(vol_u8[z])
        panels = [
            ct_pil,
            overlay_multiclass(ct_pil, gt_mask[z]),
            overlay_multiclass(ct_pil, pred_mask[z]),
        ]
        labels = [f"CT z={z}", "GT overlay", "Pred overlay"]
        rows.append(create_side_by_side(panels, labels=labels))

    panel = create_multi_row(rows, header=header)
    if panel is None:
        return False

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    panel.save(save_path)
    return True
