import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from ..distribution_utils import compute_shard_indices, is_main_process
from .datasets.chestx_det import CHESTX_DET_EVAL_FINDINGS, load_chestx_det_json
from .segmentation_utils import polygon2mask
from .utils import interpolate_similarity_scores

_FONT_CACHE = {}


def _get_font(size=28):
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            _FONT_CACHE[size] = ImageFont.load_default(size=size)
    return _FONT_CACHE[size]


def draw_prob_box(image, prob, font_size=48, box_padding=8):
    img = image.copy().convert("RGBA")
    font = _get_font(font_size)

    text = f"{prob:.3f}"
    tmp_draw = ImageDraw.Draw(img)
    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    tx0, ty0, tx1, ty1 = bbox
    tw, th = tx1 - tx0, ty1 - ty0

    w, _ = img.size
    bw = tw + box_padding * 2
    bh = th + box_padding * 2
    box = [w - bw, 0, w, bh]

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(box, fill=(255, 255, 255, 200))
    img = Image.alpha_composite(img, overlay)

    ImageDraw.Draw(img).text(
        (box[0] + box_padding - tx0, box[1] + box_padding - ty0),
        text,
        fill=(0, 0, 0),
        font=font,
    )
    return img.convert("RGB")


def similarity_map_to_colored_image(similarity_map, vmin=0.0, vmax=1.0):
    if isinstance(similarity_map, torch.Tensor):
        similarity_map = similarity_map.cpu().numpy()
    similarity_map = similarity_map.squeeze()
    normalized = np.clip((similarity_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
    cmap = plt.get_cmap("magma")
    colored = cmap(normalized)
    return Image.fromarray((colored[..., :3] * 255).astype(np.uint8))


def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.4):
    img = image.copy().convert("RGBA")
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.squeeze().astype(np.float32)
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask > 0] = [*color, int(255 * alpha)]
    mask_img = Image.fromarray(overlay, "RGBA").resize(img.size, Image.NEAREST)
    return Image.alpha_composite(img, mask_img).convert("RGB")


def overlay_heatmap_on_image(image, sim_map, threshold=0.0, beta=0.5):
    import cv2

    if isinstance(sim_map, torch.Tensor):
        sim_map = sim_map.cpu().numpy()
    sim_map = sim_map.squeeze().astype(np.float32)

    img_rgb = np.array(image.convert("RGB"))
    sim_resized = cv2.resize(sim_map, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    sim_clipped = np.clip(sim_resized, threshold, 1.0)
    sim_norm = ((sim_clipped - threshold) / (1.0 - threshold + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(sim_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap[sim_norm == 0] = img_rgb[sim_norm == 0]
    blended = cv2.addWeighted(img_rgb, 1 - beta, heatmap, beta, 0)
    return Image.fromarray(blended)


def draw_mask_contour(image, mask, color=(0, 255, 0), width=3):
    import cv2

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask_bool = mask.squeeze().astype(np.uint8)
    if mask_bool.sum() == 0:
        return image.copy()
    img_rgb = np.array(image.convert("RGB"))
    mask_resized = cv2.resize(mask_bool, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_rgb, contours, -1, color, thickness=width)
    return Image.fromarray(img_rgb)


def draw_bboxes_on_image(image, bboxes, color=(0, 255, 0), width=3):
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for x_min, y_min, x_max, y_max in bboxes:
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
    return img


def create_side_by_side(images, labels=None, padding=4, label_height=80, font_size=60):
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


def create_multi_row(rows, row_padding=8):
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    max_w = max(r.size[0] for r in rows)
    total_h = sum(r.size[1] for r in rows) + row_padding * (len(rows) - 1)
    canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y_offset = 0
    for row in rows:
        x_offset = (max_w - row.size[0]) // 2
        canvas.paste(row, (x_offset, y_offset))
        y_offset += row.size[1] + row_padding
    return canvas


@torch.no_grad()
def _batch_compute_similarity(model, image_processor, tokenizer, image_pil_list, texts, batch_size):
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)

    sim_probs_all = []
    logit_probs_all = []
    components_all = []
    has_components = None

    for start in range(0, len(image_pil_list), batch_size):
        batch_imgs = image_pil_list[start : start + batch_size]
        batch_sizes = [(img.size[1], img.size[0]) for img in batch_imgs]

        processed = [np.array(image_processor(img)["pixel_values"]).squeeze(0) for img in batch_imgs]
        pixel_values = torch.FloatTensor(np.stack(processed)).to(model.device, model.dtype)

        output = model.compute_logits(pixel_values, [tokenized])
        sim_scores = output["similarity_scores"]
        logits = output["logits"]

        sim_scores = sim_scores.detach().float().cpu()
        logit_probs = logits.sigmoid().detach().float().cpu().numpy()

        raw_comp = output.get("sim_components")
        if has_components is None:
            has_components = raw_comp is not None
        comp_cpu = {k: v.detach().float().cpu() for k, v in raw_comp.items()} if raw_comp else None

        for i, (h, w) in enumerate(batch_sizes):
            num_texts = sim_scores.shape[1]
            per_text = []
            for j in range(num_texts):
                recon = interpolate_similarity_scores(sim_scores[i, j], (h, w), image_processor)
                per_text.append(recon.numpy().squeeze())
            sim_probs_all.append(np.stack(per_text))
            logit_probs_all.append(logit_probs[i])

            if comp_cpu:
                img_comp = {}
                for k, v in comp_cpu.items():
                    per_text_comp = []
                    for j in range(num_texts):
                        recon = interpolate_similarity_scores(v[i, j], (h, w), image_processor)
                        per_text_comp.append(recon.numpy().squeeze())
                    img_comp[k] = np.stack(per_text_comp)
                components_all.append(img_comp)

        del output, pixel_values

    return sim_probs_all, logit_probs_all, (components_all if has_components else None)


def _finding_to_variant_prompts(finding_name):
    f_lower = finding_name.lower().strip().replace("_", " ")
    if f_lower == "nodule mass":
        f_lower = "nodule and mass"
    return [
        f"There is {f_lower}.",
        f"There is no {f_lower}.",
        f"There is left {f_lower}.",
        f"There is right {f_lower}.",
    ]


PROMPT_LABELS = ["is", "is no", "is right", "is left"]


@torch.no_grad()
def visualize_chestx_det(
    model,
    image_processor,
    tokenizer,
    data_path,
    image_dir,
    save_dir,
    batch_size=16,
    max_samples_per_finding=50,
):
    data_list = load_chestx_det_json(data_path, image_dir)

    for finding in CHESTX_DET_EVAL_FINDINGS:
        pos_samples = []
        neg_samples = []

        for item in data_list:
            polygons = item["masks"].get(finding, [])

            if polygons:
                pos_samples.append((item["image_path"], polygons))
            else:
                neg_samples.append((item["image_path"], None))

        if max_samples_per_finding:
            pos_samples = pos_samples[:max_samples_per_finding]
            neg_samples = neg_samples[:max_samples_per_finding]

        all_samples = [(s, True) for s in pos_samples] + [(s, False) for s in neg_samples]
        if not all_samples:
            continue

        start_idx, end_idx = compute_shard_indices(all_samples)
        local_samples = all_samples[start_idx:end_idx]
        if not local_samples:
            continue

        finding_slug = finding.lower().replace(" ", "_")
        pos_dir = os.path.join(save_dir, finding_slug, "positive")
        neg_dir = os.path.join(save_dir, finding_slug, "negative")
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)

        texts = _finding_to_variant_prompts(finding)

        image_list = []
        meta_list = []
        for (image_path, polygons), is_positive in local_samples:
            image_list.append(Image.open(image_path).convert("RGB"))
            meta_list.append((image_path, polygons, is_positive))

        if is_main_process():
            print(
                f"[Vis ChestXDet] {finding}: {len(image_list)} images (pos={len(pos_samples)}, neg={len(neg_samples)})"
            )

        sim_probs, logit_probs, components_list = _batch_compute_similarity(
            model,
            image_processor,
            tokenizer,
            image_list,
            texts,
            batch_size,
        )

        it = range(len(image_list))
        if is_main_process():
            it = tqdm(it, desc=f"  Saving {finding}")

        for idx in it:
            img = image_list[idx]
            image_path, polygons, is_positive = meta_list[idx]
            h, w = img.size[1], img.size[0]
            fname = os.path.splitext(os.path.basename(image_path))[0]

            panels = [img]
            labels = ["Original"]

            gt_mask = None
            if is_positive and polygons:
                gt_mask = polygon2mask(polygons, h, w)
                panels.append(overlay_mask_on_image(img, gt_mask))
                labels.append(f"GT ({finding})")
            else:
                panels.append(img.copy())
                labels.append("No Mask")

            pos_sim = sim_probs[idx][0]
            overlay_panel = overlay_heatmap_on_image(img, pos_sim, threshold=0.0)
            if gt_mask is not None:
                overlay_panel = draw_mask_contour(overlay_panel, gt_mask, color=(0, 255, 0), width=3)
            panels.append(overlay_panel)
            labels.append("Sim overlay (pos)")

            for j in range(4):
                sim_img = similarity_map_to_colored_image(sim_probs[idx][j])
                sim_img = draw_prob_box(sim_img, float(logit_probs[idx][j]))
                panels.append(sim_img)
                labels.append(texts[j])

            row1 = create_side_by_side(panels, labels)

            if components_list:
                comp = components_list[idx]
                comp_panels = []
                comp_labels = []
                for key in comp:
                    comp_panels.append(similarity_map_to_colored_image(comp[key][0]))
                    comp_labels.append(key)
                row2 = create_side_by_side(comp_panels, comp_labels)
                composite = create_multi_row([row1, row2])
            else:
                composite = row1

            out_dir = pos_dir if is_positive else neg_dir
            composite.save(os.path.join(out_dir, f"{fname}.png"))

        del image_list, sim_probs, logit_probs
        torch.cuda.empty_cache()


@torch.no_grad()
def visualize_ms_cxr(
    model,
    image_processor,
    tokenizer,
    data,
    image_root_dir,
    save_dir,
    batch_size=16,
    max_samples=None,
):
    os.makedirs(save_dir, exist_ok=True)

    if max_samples and len(data) > max_samples:
        data = data[:max_samples]

    start_idx, end_idx = compute_shard_indices(data)
    local_data = data[start_idx:end_idx]
    if not local_data:
        return

    it = enumerate(local_data)
    if is_main_process():
        it = tqdm(it, total=len(local_data), desc="Vis MS-CXR")

    for local_idx, d in it:
        global_idx = start_idx + local_idx
        image_path = d["image"]
        image = Image.open(os.path.join(image_root_dir, image_path))
        text = d["det"][0]["name"]
        bbox = d["det"][0]["label"][0]

        sim_probs, logit_probs, components_list = _batch_compute_similarity(
            model,
            image_processor,
            tokenizer,
            [image],
            [text],
            batch_size=1,
        )
        prob_map = sim_probs[0][0]
        logit_prob = float(logit_probs[0][0])

        img_with_bbox = draw_bboxes_on_image(image, [bbox])
        sim_colored = similarity_map_to_colored_image(prob_map)
        sim_colored = draw_prob_box(sim_colored, logit_prob)

        sim_overlay = overlay_heatmap_on_image(image, prob_map, threshold=0.0)
        sim_overlay = draw_bboxes_on_image(sim_overlay, [bbox], color=(0, 255, 0), width=3)

        row1 = create_side_by_side(
            [image, img_with_bbox, sim_overlay, sim_colored],
            labels=["Original", "GT BBox", "Sim overlay", f"Sim: {text}"],
        )

        if components_list:
            comp = components_list[0]
            comp_panels = []
            comp_labels = []
            for key in comp:
                comp_panels.append(similarity_map_to_colored_image(comp[key][0]))
                comp_labels.append(key)
            row2 = create_side_by_side(comp_panels, comp_labels)
            composite = create_multi_row([row1, row2])
        else:
            composite = row1

        fname = os.path.splitext(os.path.basename(image_path))[0]
        composite.save(os.path.join(save_dir, f"{global_idx:04d}_{fname}.png"))
