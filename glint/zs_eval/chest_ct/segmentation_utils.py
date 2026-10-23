import os
import tempfile

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from common import save_json
from glint.visualization.chest_ct.load_dataset import load_rexgrounding_dataset
from glint.visualization.chest_ct.similarity_map import (
    load_raw_volume,
    monai_volume_to_slices,
    save_segmentation_visualization,
    save_similarity_panel,
    save_volume_slices,
    similarity_map_to_original_space,
    volume_to_slices,
)
from glint.zs_eval.distribution_utils import (
    compute_shard_indices,
    is_distributed,
    is_main_process,
)

from .utils import (
    _get_vision_spatial_config,
    model_inference_similarity_map,
)


def dice_score_binary(pred, target, eps=1e-6):
    pred = pred.bool()
    target = target.bool()
    intersection = (pred & target).sum()
    denom = pred.sum() + target.sum()
    return (2.0 * intersection + eps) / (denom + eps)


def _load_monai_volume(raw_path, monai_cfg, cache_name="ReXGroundingCT/val"):
    from glint.preprocess.chest_ct.monai.transform import build_dataset

    result = build_dataset([{"image": raw_path}], monai_cfg, cache_name)[0]
    volume = result["image"].squeeze(0)

    meta = result.get("image_meta_dict", {})
    orig_affine = meta.get("original_affine", meta.get("affine"))
    if orig_affine is not None:
        orig_affine = np.array(orig_affine)
    orig_shape = meta.get("spatial_shape")
    if orig_shape is not None:
        orig_shape = tuple(int(s) for s in orig_shape)

    orig_meta = {
        "affine": orig_affine,
        "shape": orig_shape,
        "spacing": tuple(abs(float(orig_affine[i, i])) for i in range(3)) if orig_affine is not None else None,
    }
    return volume, orig_meta


def _build_mask_transform(monai_cfg):
    from monai.transforms import (
        CenterSpatialCropd,
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        Orientationd,
        Resized,
        Spacingd,
        SpatialPadd,
    )

    return Compose(
        [
            LoadImaged(keys=["mask"], image_only=False),
            EnsureChannelFirstd(keys=["mask"]),
            Orientationd(keys=["mask"], axcodes=monai_cfg.orientation),
            Spacingd(keys=["mask"], pixdim=monai_cfg.target_spacing, mode="nearest"),
            CenterSpatialCropd(keys=["mask"], roi_size=monai_cfg.roi_size),
            SpatialPadd(keys=["mask"], spatial_size=monai_cfg.roi_size, mode="constant", constant_values=0),
            Resized(keys=["mask"], spatial_size=monai_cfg.final_size, mode="nearest"),
            EnsureTyped(keys=["mask"], dtype=torch.float32),
        ]
    )


def _load_masks_monai(mask_path, ct_affine, monai_cfg):
    mask_transform = _build_mask_transform(monai_cfg)

    mask_nii = nib.load(mask_path)
    raw_data = mask_nii.get_fdata()  # (F, H, W, D) or (H, W, D)

    masks = {}
    if raw_data.ndim == 3:
        masks[0] = _transform_single_mask(raw_data, ct_affine, mask_transform)
    else:
        for i in range(raw_data.shape[0]):
            masks[i] = _transform_single_mask(raw_data[i], ct_affine, mask_transform)

    return masks


def _transform_single_mask(mask_array, ct_affine, mask_transform):
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    try:
        mask_nii = nib.Nifti1Image(mask_array.astype(np.float32), ct_affine)
        nib.save(mask_nii, tmp.name)
        result = mask_transform({"mask": tmp.name})
        return result["mask"].squeeze(0) > 0.5
    finally:
        os.unlink(tmp.name)


@torch.no_grad()
def eval_segmentation_rexgrounding(
    model,
    tokenizer,
    data_root_dir,
    cfg,
    save_root_dir,
    monai_cfg=None,
    threshold_step=0.01,
):
    device = next(model.parameters()).device
    data_list = load_rexgrounding_dataset(data_root_dir, cfg, split="val")

    rex_cfg = cfg["dataset"]["ReXGroundingCT"]
    use_original_size = rex_cfg.get("use_original_size", True)
    viz_threshold = rex_cfg.get("visualization_threshold", 0.8)
    save_visualization = rex_cfg.get("save_visualization", True)
    save_panel = rex_cfg.get("save_panel", True)
    panel_max_slices = rex_cfg.get("panel_max_slices", None)
    viz_orientation = rex_cfg.get("visualization_orientation", None)  # None=original, "IPL" etc=MONAI space

    image_size, patch_size, input_slices, frame_length = _get_vision_spatial_config(model.config.vision_config)
    model_volume_shape = (input_slices, image_size, image_size)

    monai_cfg_dict = cfg.get("dataset", {}).get("monai_config", {})
    monai_target_spacing = tuple(monai_cfg_dict.get("target_spacing", [1.5, 0.75, 0.75]))
    monai_roi_size = tuple(monai_cfg_dict.get("roi_size", [240, 480, 480]))
    monai_final_size = tuple(monai_cfg_dict.get("final_size", [240, 384, 384]))
    monai_orientation = monai_cfg_dict.get("orientation", "IPL")

    if is_distributed():
        start_idx, end_idx = compute_shard_indices(data_list)
        data_list = data_list[start_idx:end_idx]

    n_steps = int(round(1.0 / threshold_step)) + 1
    thresholds = torch.linspace(0.0, 1.0, steps=n_steps, device=device)
    dice_sums = torch.zeros_like(thresholds)
    pair_count = 0

    finding_dice_sums = {}
    finding_counts = {}

    os.makedirs(save_root_dir, exist_ok=True)

    it = tqdm(data_list, desc="CT Segmentation") if is_main_process() else data_list
    for data in it:
        image_id = data["image_id"]
        raw_path = data["raw_path"]
        mask_path = data["mask_path"]

        if monai_cfg is not None:
            volume, orig_meta = _load_monai_volume(raw_path, monai_cfg)
        else:
            raise ValueError("MONAI config required for ReXGroundingCT eval")

        viz_in_monai = viz_orientation is not None

        if use_original_size:
            orig_spacing = orig_meta["spacing"]
            orig_shape = orig_meta["shape"]
            orig_affine = orig_meta["affine"]
            mask_nii = nib.load(mask_path)
            mask_all = torch.from_numpy(mask_nii.get_fdata())

        need_monai_masks = not use_original_size or (save_visualization and viz_in_monai)
        masks_monai = _load_masks_monai(mask_path, orig_meta["affine"], monai_cfg) if need_monai_masks else None

        if save_visualization:
            if viz_in_monai or not use_original_size:
                volume_slices = monai_volume_to_slices(volume)
            else:
                raw_vol, _, _, _ = load_raw_volume(raw_path)
                volume_slices = volume_to_slices(raw_vol)
                del raw_vol
            scan_dir = os.path.join(save_root_dir, image_id)
            save_volume_slices(volume_slices, scan_dir)

        for idx_str, finding_text in data["findings"].items():
            idx = int(idx_str)

            sim_prob, sim_map = model_inference_similarity_map(
                volume_data=volume,
                model=model,
                text=finding_text,
                tokenizer=tokenizer,
                target_shape=model_volume_shape,
            )

            if use_original_size:
                sim_map_eval = similarity_map_to_original_space(
                    sim_map.to(device),
                    orig_shape=orig_shape,
                    orig_spacing=orig_spacing,
                    orig_affine=orig_affine,
                    target_spacing=monai_target_spacing,
                    roi_size=monai_roi_size,
                    final_size=monai_final_size,
                    monai_orientation=monai_orientation,
                )
                mask_orig = mask_all[idx] if mask_all.ndim > 3 else mask_all
                mask_eval = (mask_orig.permute(2, 0, 1) > 0).to(device)
            else:
                sim_map_eval = sim_map.to(device)
                mask_eval = masks_monai[idx].to(device)

            finding_key = finding_text.strip()
            if finding_key not in finding_dice_sums:
                finding_dice_sums[finding_key] = torch.zeros_like(thresholds)
                finding_counts[finding_key] = 0

            for t_idx in range(len(thresholds)):
                pred = sim_map_eval >= thresholds[t_idx]
                dice = dice_score_binary(pred, mask_eval)
                dice_sums[t_idx] += dice
                finding_dice_sums[finding_key][t_idx] += dice

            pair_count += 1
            finding_counts[finding_key] += 1

            if save_visualization:
                if viz_in_monai:
                    sim_map_viz = sim_map.to(device)
                    mask_viz = masks_monai[idx].to(device)
                else:
                    sim_map_viz = sim_map_eval
                    mask_viz = mask_eval
                save_segmentation_visualization(
                    volume_slices=volume_slices,
                    sim_map=sim_map_viz,
                    gt_mask=mask_viz,
                    save_dir=scan_dir,
                    finding_idx=idx,
                    finding_name=finding_text,
                    threshold=viz_threshold,
                )

                if save_panel:
                    safe_name = finding_text.strip().replace("/", "_")[:80]
                    panel_path = os.path.join(
                        scan_dir,
                        f"panel_{idx:03d}_{safe_name}.png",
                    )
                    save_similarity_panel(
                        volume_slices=volume_slices,
                        sim_map=sim_map_viz,
                        save_path=panel_path,
                        finding_name=finding_text,
                        sim_prob=float(sim_prob),
                        max_slices=panel_max_slices,
                        gt_mask=mask_viz,
                    )

    if is_distributed():
        dist.reduce(dice_sums, dst=0, op=dist.ReduceOp.SUM)

        pair_count_t = torch.tensor([pair_count], device=device, dtype=torch.long)
        dist.reduce(pair_count_t, dst=0, op=dist.ReduceOp.SUM)
        pair_count = pair_count_t.item()

        all_finding = [None] * dist.get_world_size() if is_main_process() else None
        dist.gather_object(
            {"sums": {k: v.cpu() for k, v in finding_dice_sums.items()}, "counts": finding_counts},
            object_gather_list=all_finding,
            dst=0,
        )

        if not is_main_process():
            return None

        finding_dice_sums, finding_counts = {}, {}
        for gpu_data in all_finding:
            for key, tensor in gpu_data["sums"].items():
                if key in finding_dice_sums:
                    finding_dice_sums[key] += tensor.to(device)
                    finding_counts[key] += gpu_data["counts"][key]
                else:
                    finding_dice_sums[key] = tensor.clone().to(device)
                    finding_counts[key] = gpu_data["counts"][key]

    mean_dice = dice_sums / max(pair_count, 1)
    best_idx = int(mean_dice.argmax().item())
    best_threshold = float(thresholds[best_idx].item())
    best_dice = float(mean_dice[best_idx].item())

    result = {
        "mean_dice": best_dice,
        "best_threshold": best_threshold,
        "num_pairs": pair_count,
        "dice_by_threshold": {f"{t.item():.2f}": float(d.item()) for t, d in zip(thresholds, mean_dice)},
    }

    finding_results = {}
    for key in finding_dice_sums:
        f_dice = finding_dice_sums[key] / max(finding_counts[key], 1)
        f_best_idx = int(f_dice.argmax().item())
        finding_results[key] = {
            "dice": float(f_dice[f_best_idx].item()),
            "best_threshold": float(thresholds[f_best_idx].item()),
            "count": finding_counts[key],
        }
    result["per_finding"] = finding_results

    save_json(result, os.path.join(save_root_dir, "result.json"))
    print(f"Best threshold={best_threshold:.2f}, Mean Dice={best_dice:.4f}")

    result["main_metric"] = {"dice": best_dice}
    return result
