import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE = {}


def _get_font(size=28):
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            _FONT_CACHE[size] = ImageFont.load_default(size=size)
    return _FONT_CACHE[size]


def similarity_map_to_colored_image(similarity_map, vmin=0, vmax=1.0):
    if isinstance(similarity_map, torch.Tensor):
        similarity_map = similarity_map.cpu().numpy()

    normalized = np.clip((similarity_map - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap("magma")
    colored = cmap(normalized)
    return Image.fromarray((colored[..., :3] * 255).astype(np.uint8))


def overlay_heatmap(slice_img, mask, threshold=0.5, colormap=cv2.COLORMAP_JET, beta=0.5):
    slice_bgr = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)

    mask_clipped = np.clip(mask, threshold, 1.0)
    mask_norm = ((mask_clipped - threshold) / (1.0 - threshold + 1e-8) * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(mask_norm, colormap)
    heatmap[mask_norm == 0] = 0

    overlay = cv2.addWeighted(slice_bgr, 1 - beta, heatmap, beta, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def monai_volume_to_slices(volume, window_center=-600.0, window_width=1500.0):
    hu_volume = volume.float() * 1000.0
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    windowed = torch.clamp(hu_volume, img_min, img_max)
    return ((windowed - img_min) / (img_max - img_min) * 255.0).to(torch.uint8)


def load_raw_volume(path):
    from monai.transforms import LoadImage

    loader = LoadImage(image_only=False, ensure_channel_first=False)
    data, meta = loader(path)

    volume = torch.as_tensor(data).float().squeeze()
    affine = np.array(meta["affine"])
    orig_shape = volume.shape
    spacing = tuple(abs(float(affine[i, i])) for i in range(3))

    return volume, spacing, orig_shape, affine


def volume_to_slices(volume, window_center=-600.0, window_width=1500.0):
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    windowed = torch.clamp(volume.float(), img_min, img_max)
    slices = ((windowed - img_min) / (img_max - img_min) * 255.0).to(torch.uint8)
    return slices.permute(2, 0, 1)


def _bounds(src_len, dst_len):
    if dst_len >= src_len:
        dst_start = (dst_len - src_len) // 2
        dst_end = dst_start + src_len
        src_start = 0
        src_end = src_len
    else:
        src_start = (src_len - dst_len) // 2
        src_end = src_start + dst_len
        dst_start = 0
        dst_end = dst_len
    return src_start, src_end, dst_start, dst_end


def _get_monai_axis_mapping(orig_affine, monai_axcodes="IPL"):
    from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform

    orig_ornt = axcodes2ornt(aff2axcodes(orig_affine))
    monai_ornt = axcodes2ornt(tuple(monai_axcodes))
    fwd = ornt_transform(orig_ornt, monai_ornt)
    return fwd


def _monai_ipl_to_original(tensor, orig_shape):
    lps = tensor.permute(2, 1, 0)
    lps = torch.flip(lps, dims=[2])
    return lps


def similarity_map_to_original_space(
    sim_map,
    orig_shape,
    orig_spacing,
    orig_affine,
    target_spacing=(1.5, 0.75, 0.75),
    roi_size=(240, 480, 480),
    final_size=(240, 384, 384),
    monai_orientation="IPL",
):
    d_roi, h_roi, w_roi = roi_size
    fwd = _get_monai_axis_mapping(orig_affine, monai_orientation)

    monai_counts = []
    monai_spacings = []
    for monai_dim in range(3):
        orig_dim = int(fwd[monai_dim][0])
        monai_counts.append(orig_shape[orig_dim])
        monai_spacings.append(orig_spacing[orig_dim])

    if final_size is not None and tuple(final_size) != tuple(roi_size):
        sim_map = (
            F.interpolate(
                sim_map.float().unsqueeze(0).unsqueeze(0),
                size=(d_roi, h_roi, w_roi),
                mode="trilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    d_rs = int(monai_counts[0] * (monai_spacings[0] / target_spacing[0]))
    h_rs = int(monai_counts[1] * (monai_spacings[1] / target_spacing[1]))
    w_rs = int(monai_counts[2] * (monai_spacings[2] / target_spacing[2]))

    ds0, ds1, dd0, dd1 = _bounds(d_roi, d_rs)
    hs0, hs1, hd0, hd1 = _bounds(h_roi, h_rs)
    ws0, ws1, wd0, wd1 = _bounds(w_roi, w_rs)

    map_rs = torch.zeros((d_rs, h_rs, w_rs), dtype=sim_map.dtype, device=sim_map.device)
    map_rs[dd0:dd1, hd0:hd1, wd0:wd1] = sim_map[ds0:ds1, hs0:hs1, ws0:ws1]

    map_monai = (
        F.interpolate(
            map_rs.unsqueeze(0).unsqueeze(0),
            size=(monai_counts[0], monai_counts[1], monai_counts[2]),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
    )

    map_lps = _monai_ipl_to_original(map_monai, orig_shape)
    return map_lps.permute(2, 0, 1)


def save_similarity_visualization(volume_slices, sim_map, save_dir, finding_name, sim_prob, threshold=0.5):
    folder_name = f"{finding_name}_prob_{sim_prob:.2f}"
    base_dir = os.path.join(save_dir, folder_name)

    simmap_dir = os.path.join(base_dir, "similarity_map")
    simmask_dir = os.path.join(base_dir, f"similarity_mask_{threshold}")
    overlay_dir = os.path.join(base_dir, f"overlay_{threshold}")
    os.makedirs(simmap_dir, exist_ok=True)
    os.makedirs(simmask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    for z in range(volume_slices.shape[0]):
        slice_np = volume_slices[z].cpu().numpy()
        sim_np = sim_map[z].cpu().numpy() if isinstance(sim_map[z], torch.Tensor) else sim_map[z]

        sim_img = similarity_map_to_colored_image(sim_np, vmin=0, vmax=1.0)
        sim_img.save(os.path.join(simmap_dir, f"slice_{z}.png"))

        masked = sim_np * (sim_np >= threshold)
        masked_img = similarity_map_to_colored_image(masked, vmin=threshold, vmax=1.0)
        masked_img.save(os.path.join(simmask_dir, f"slice_{z}.png"))

        overlay = overlay_heatmap(slice_np, sim_np, threshold=threshold)
        Image.fromarray(overlay).save(os.path.join(overlay_dir, f"slice_{z}.png"))


def save_segmentation_visualization(
    volume_slices, sim_map, gt_mask, save_dir, finding_idx, finding_name, threshold=0.5
):
    folder_name = f"{finding_idx}_{finding_name[:60]}"
    base_dir = os.path.join(save_dir, folder_name)

    simmap_dir = os.path.join(base_dir, "similarity_map")
    gtmask_dir = os.path.join(base_dir, "gt_mask")
    simmask_dir = os.path.join(base_dir, f"similarity_mask_{threshold}")
    overlay_dir = os.path.join(base_dir, f"overlay_{threshold}")
    os.makedirs(simmap_dir, exist_ok=True)
    os.makedirs(gtmask_dir, exist_ok=True)
    os.makedirs(simmask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    for z in range(volume_slices.shape[0]):
        mask_np = gt_mask[z].cpu().numpy() if isinstance(gt_mask[z], torch.Tensor) else gt_mask[z]
        if not mask_np.any():
            continue

        slice_np = volume_slices[z].cpu().numpy()
        sim_np = sim_map[z].cpu().numpy() if isinstance(sim_map[z], torch.Tensor) else sim_map[z]

        similarity_map_to_colored_image(sim_np, vmin=0, vmax=1.0).save(os.path.join(simmap_dir, f"slice_{z}.png"))
        Image.fromarray((mask_np * 255).astype(np.uint8)).save(os.path.join(gtmask_dir, f"slice_{z}.png"))

        masked = sim_np * (sim_np >= threshold)
        similarity_map_to_colored_image(masked, vmin=threshold, vmax=1.0).save(
            os.path.join(simmask_dir, f"slice_{z}.png")
        )

        overlay = overlay_heatmap(slice_np, sim_np, threshold=threshold)
        Image.fromarray(overlay).save(os.path.join(overlay_dir, f"slice_{z}.png"))


def save_volume_slices(volume_slices, save_dir):
    slices_dir = os.path.join(save_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)
    for z in range(volume_slices.shape[0]):
        Image.fromarray(volume_slices[z].cpu().numpy()).save(os.path.join(slices_dir, f"slice_{z}.png"))


def _ct_slice_to_pil(slice_np):
    if isinstance(slice_np, torch.Tensor):
        slice_np = slice_np.cpu().numpy()
    return Image.fromarray(slice_np).convert("RGB")


def _overlay_mask(slice_img, mask_np, color=(0, 255, 0), alpha=0.45):
    if isinstance(mask_np, torch.Tensor):
        mask_np = mask_np.cpu().numpy()
    mask_bool = mask_np.astype(bool)
    if not mask_bool.any():
        return slice_img.copy()
    base = slice_img.copy().convert("RGBA")
    overlay = np.zeros((*mask_bool.shape, 4), dtype=np.uint8)
    overlay[mask_bool] = [*color, int(255 * alpha)]
    overlay_img = Image.fromarray(overlay, "RGBA").resize(base.size, Image.NEAREST)
    return Image.alpha_composite(base, overlay_img).convert("RGB")


def _sim_map_to_color_pil(sim_np, vmin=0.0, vmax=1.0, cmap_name="magma"):
    if isinstance(sim_np, torch.Tensor):
        sim_np = sim_np.cpu().numpy()
    normalized = np.clip((sim_np - vmin) / (vmax - vmin + 1e-8), 0, 1)
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(normalized)
    return Image.fromarray((colored[..., :3] * 255).astype(np.uint8))


def _overlay_heatmap_pil(slice_np, sim_np, threshold=0.0, beta=0.5):
    arr = overlay_heatmap(
        slice_np.cpu().numpy() if isinstance(slice_np, torch.Tensor) else slice_np,
        sim_np.cpu().numpy() if isinstance(sim_np, torch.Tensor) else sim_np,
        threshold=threshold,
        beta=beta,
    )
    return Image.fromarray(arr)


def _create_side_by_side(images, labels=None, padding=4, label_height=48, font_size=32):
    if not images:
        return None
    target_h = max(img.size[1] for img in images)
    resized = []
    for img in images:
        if img.size[1] != target_h:
            ratio = target_h / img.size[1]
            new_w = int(img.size[0] * ratio)
            img = img.resize((new_w, target_h), Image.BILINEAR)
        resized.append(img)
    total_w = sum(img.size[0] for img in resized) + padding * (len(resized) - 1)
    extra_h = label_height if labels else 0
    canvas = Image.new("RGB", (total_w, target_h + extra_h), (255, 255, 255))
    font = _get_font(font_size)
    draw = ImageDraw.Draw(canvas)
    x_offset = 0
    for i, img in enumerate(resized):
        if labels and i < len(labels):
            draw.text((x_offset + 4, 4), labels[i], fill=(0, 0, 0), font=font)
        canvas.paste(img, (x_offset, extra_h))
        x_offset += img.size[0] + padding
    return canvas


def _create_multi_row(rows, row_padding=6, header=None, header_height=60, header_font_size=40):
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    max_w = max(r.size[0] for r in rows)
    extra = header_height if header else 0
    total_h = sum(r.size[1] for r in rows) + row_padding * (len(rows) - 1) + extra
    canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    if header:
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), header, fill=(0, 0, 0), font=_get_font(header_font_size))
    y_offset = extra
    for row in rows:
        x_offset = (max_w - row.size[0]) // 2
        canvas.paste(row, (x_offset, y_offset))
        y_offset += row.size[1] + row_padding
    return canvas


def _pick_panel_slices(gt_mask, sim_map, max_slices=None):
    sim_np = sim_map.cpu().numpy() if isinstance(sim_map, torch.Tensor) else np.asarray(sim_map)

    if gt_mask is not None:
        gt_np = gt_mask.cpu().numpy() if isinstance(gt_mask, torch.Tensor) else np.asarray(gt_mask)
        candidates = np.where(gt_np.reshape(gt_np.shape[0], -1).sum(axis=1) > 0)[0]
        if candidates.size == 0:
            return []
        if max_slices is None or candidates.size <= max_slices:
            return [int(z) for z in candidates]
        idx = np.linspace(0, candidates.size - 1, max_slices).round().astype(int)
        return [int(candidates[i]) for i in idx]

    sim_per_slice = sim_np.reshape(sim_np.shape[0], -1).max(axis=1)
    n = max_slices if max_slices is not None else sim_np.shape[0]
    return sorted(int(z) for z in np.argsort(-sim_per_slice)[:n])


def save_similarity_panel(
    volume_slices,
    sim_map,
    save_path,
    finding_name,
    sim_prob,
    max_slices=None,
    gt_mask=None,
):
    z_list = _pick_panel_slices(gt_mask, sim_map, max_slices=max_slices)
    if not z_list:
        return

    has_gt = gt_mask is not None

    rows = []
    for z in z_list:
        slice_np = volume_slices[z]
        sim_np = sim_map[z]
        ct_pil = _ct_slice_to_pil(slice_np)

        panels = [ct_pil]
        labels = [f"CT z={z}"]

        if has_gt:
            gt_np = gt_mask[z].cpu().numpy() if isinstance(gt_mask[z], torch.Tensor) else gt_mask[z]
            panels.append(_overlay_mask(ct_pil, gt_np, color=(0, 255, 0)))
            labels.append("GT overlay")

        panels.append(_overlay_heatmap_pil(slice_np, sim_np, threshold=0.0))
        labels.append("Sim overlay")

        panels.append(_sim_map_to_color_pil(sim_np, vmin=0.0, vmax=1.0))
        labels.append("Sim map")

        rows.append(_create_side_by_side(panels, labels=labels, label_height=40, font_size=28))

    header = f"{finding_name}  |  prob={sim_prob:.3f}  |  slices={len(z_list)}"
    composite = _create_multi_row(rows, header=header)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    composite.save(save_path)
