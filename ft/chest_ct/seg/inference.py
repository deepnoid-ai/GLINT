"""Sliding-window CT segmentation inference, inverted back to the original CT space for Dice."""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from tqdm import tqdm

from common import logger
from glint.zs_eval.distribution_utils import (
    compute_shard_indices,
    gather_results,
    is_distributed,
    is_main_process,
)


def build_inference_transforms(
    spacing: tuple[float, float, float],
    orientation: str = "RAS",
    a_min: float = -175.0,
    a_max: float = 250.0,
    b_min: float = 0.0,
    b_max: float = 1.0,
) -> Compose:
    """Must match the training cache transforms; MONAI keeps the metadata Invertd needs."""
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes=orientation),
            Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            ),
        ]
    )


def _case_id_from_path(image_path: str) -> str:
    return Path(image_path).stem.replace(".nii", "").replace("_0000", "")


def _align_axes_to(ref: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """Fix a NIfTI (H, W, D) vs pred (D, H, W) axis-order mismatch with one (2, 0, 1) transpose."""
    if arr.shape == ref.shape:
        return arr
    if arr.ndim == 3 and arr.transpose(2, 0, 1).shape == ref.shape:
        return arr.transpose(2, 0, 1)
    raise ValueError(f"Cannot align GT shape {arr.shape} to pred shape {ref.shape}")


def _load_raw_hu_volume(image_path: str) -> np.ndarray:
    """Load raw HU as an (X, Y, Z) array from a NIfTI file or a DICOM series dir."""
    if os.path.isdir(image_path):
        import SimpleITK as sitk

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(image_path))
        # sitk is (z, y, x); transpose to nib.load's (x, y, z)
        return sitk.GetArrayFromImage(reader.Execute()).astype(np.float32).transpose(2, 1, 0)
    return nib.load(image_path).get_fdata().astype(np.float32)


def reorient_to_display(
    volume: np.ndarray,
    affine: np.ndarray,
    target_axcodes: tuple[str, str, str] = ("I", "P", "L"),
) -> np.ndarray:
    """Reorient via the affine so panel axis 0 is always the axial slice."""
    orig_ornt = nib.orientations.io_orientation(affine)
    target_ornt = nib.orientations.axcodes2ornt(target_axcodes)
    transform = nib.orientations.ornt_transform(orig_ornt, target_ornt)
    return nib.orientations.apply_orientation(volume, transform)


@torch.no_grad()
def _predict_single_volume(
    model: torch.nn.Module,
    data_dict: dict,
    preprocessing: Compose,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    device: torch.device,
) -> tuple[str, np.ndarray, np.ndarray | None, np.ndarray]:
    """Returns case_id, pred (D, H, W) uint8, label or None, affine, all in original NIfTI space."""
    case_id = _case_id_from_path(data_dict["image"])

    transformed = preprocessing({"image": data_dict["image"]})
    image = transformed["image"].unsqueeze(0).to(device)  # (1, 1, D', H', W')

    def model_fn(x):
        return model(pixel_values=x)["output"]

    logits = sliding_window_inference(
        inputs=image,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model_fn,
        overlap=overlap,
        mode="gaussian",
    )  # (1, C, D', H', W')

    # Invertd: reverse Spacing + Orientation on logits (trilinear)
    transformed["pred"] = logits.squeeze(0).cpu()
    transformed["pred"].meta = transformed["image"].meta.copy()
    invert_transform = Invertd(
        keys="pred",
        transform=preprocessing,
        orig_keys="image",
        nearest_interp=False,
        to_tensor=True,
    )
    inverted = invert_transform(transformed)
    logits_original = inverted["pred"]  # (C, D, H, W) in original space

    seg = logits_original.argmax(dim=0)  # (D, H, W)
    pred_np = seg.numpy().astype(np.uint8)

    label_np: np.ndarray | None = None
    affine = np.eye(4)
    if data_dict.get("label") and os.path.exists(data_dict["label"]):
        nii = nib.load(data_dict["label"])
        label_np = nii.get_fdata().astype(np.uint8)
        affine = nii.affine
        label_np = _align_axes_to(pred_np, label_np)
    else:
        orig_affine = transformed["image"].meta.get("original_affine", None)
        if orig_affine is not None:
            affine = np.array(orig_affine)

    return case_id, pred_np, label_np, affine


def _compute_case_dice(pred_np: np.ndarray, label_np: np.ndarray, num_classes: int) -> dict[int, float]:
    """Per-class foreground Dice; NaN when the class is absent from the GT (MSD "present in GT" rule)."""
    out: dict[int, float] = {}
    for c in range(1, num_classes):
        p = pred_np == c
        l_ = label_np == c
        gt_vox = l_.sum()
        if gt_vox == 0:
            out[c] = float("nan")
            continue
        s = p.sum() + gt_vox
        out[c] = float(2.0 * (p & l_).sum() / s)
    return out


def _compute_case_region_dice(
    pred_np: np.ndarray,
    label_np: np.ndarray,
    regions: dict[str, list[int]],
) -> dict[str, float]:
    """Dice on merged {name: [class_ids]} regions (BraTS/KiTS convention); NaN when absent from the GT."""
    out: dict[str, float] = {}
    for name, class_ids in regions.items():
        p = np.isin(pred_np, list(class_ids))
        g = np.isin(label_np, list(class_ids))
        gt_vox = g.sum()
        if gt_vox == 0:
            out[name] = float("nan")
            continue
        s = p.sum() + gt_vox
        out[name] = float(2.0 * (p & g).sum() / s)
    return out


@torch.no_grad()
def run_ddp_sliding_window_inference(
    model: torch.nn.Module,
    data_dicts: list[dict],
    num_classes: int,
    spacing: tuple[float, float, float],
    roi_size: tuple[int, int, int],
    orientation: str = "RAS",
    a_min: float = -175.0,
    a_max: float = 250.0,
    b_min: float = 0.0,
    b_max: float = 1.0,
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    save_predictions_dir: str | None = None,
    save_panels_dir: str | None = None,
    panel_max_slices: int | None = None,
    panel_window_center: float = 40.0,
    panel_window_width: float = 400.0,
    regions: dict[str, list[int]] | None = None,
    device: torch.device | None = None,
) -> list[dict] | None:
    """Shards cases across ranks; returns the gathered per-case Dice on rank 0 and None elsewhere."""
    if device is None:
        device = next(model.parameters()).device

    start_idx, end_idx = compute_shard_indices(data_dicts)
    shard = data_dicts[start_idx:end_idx]

    preprocessing = build_inference_transforms(spacing, orientation, a_min, a_max, b_min, b_max)
    if save_predictions_dir:
        os.makedirs(save_predictions_dir, exist_ok=True)
    if save_panels_dir:
        os.makedirs(save_panels_dir, exist_ok=True)
        # Lazy import so this module stays importable without PIL/matplotlib
        from ft.chest_ct.seg.visualization.panel import save_seg_panel

    model.eval()
    per_case: list[dict] = []
    it = tqdm(shard, desc=f"[rank{int(os.environ.get('RANK', 0))}] sliding-window", disable=not is_main_process())
    for data_dict in it:
        case_id, pred_np, label_np, affine = _predict_single_volume(
            model,
            data_dict,
            preprocessing,
            roi_size,
            sw_batch_size,
            overlap,
            device,
        )

        if save_predictions_dir:
            nib.save(nib.Nifti1Image(pred_np, affine), os.path.join(save_predictions_dir, f"{case_id}_pred.nii.gz"))

        dice_per_class: dict[int, float] | None = None
        if label_np is not None:
            dice_per_class = _compute_case_dice(pred_np, label_np, num_classes)
            entry: dict = {"case_id": case_id, "dice_per_class": dice_per_class}
            if regions:
                entry["dice_per_region"] = _compute_case_region_dice(pred_np, label_np, regions)
            per_case.append(entry)

        if save_panels_dir and label_np is not None:
            raw_hu = _load_raw_hu_volume(data_dict["image"])
            raw_hu = _align_axes_to(pred_np, raw_hu)

            raw_hu_disp = reorient_to_display(raw_hu, affine)
            label_disp = reorient_to_display(label_np, affine)
            pred_disp = reorient_to_display(pred_np, affine)

            dice_str = ", ".join(f"c{c}={v:.3f}" for c, v in dice_per_class.items() if not np.isnan(v))
            save_seg_panel(
                volume_hu=raw_hu_disp,
                gt_mask=label_disp,
                pred_mask=pred_disp,
                save_path=os.path.join(save_panels_dir, f"{case_id}.png"),
                header=f"{case_id}  |  {dice_str}",
                max_slices=panel_max_slices,
                window_center=panel_window_center,
                window_width=panel_window_width,
            )

    if is_distributed():
        per_case = gather_results(per_case)

    return per_case if is_main_process() else None


@torch.no_grad()
def run_sliding_window_inference(
    model: torch.nn.Module,
    data_dicts: list[dict],
    spacing: tuple[float, float, float],
    orientation: str = "RAS",
    roi_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    output_dir: str | None = None,
    a_min: float = -175.0,
    a_max: float = 250.0,
    b_min: float = 0.0,
    b_max: float = 1.0,
    device: torch.device | None = None,
) -> list[dict]:
    if device is None:
        device = next(model.parameters()).device

    preprocessing = build_inference_transforms(spacing, orientation, a_min, a_max, b_min, b_max)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model.eval()
    results: list[dict] = []
    for data_dict in tqdm(data_dicts, desc="Inference"):
        case_id, pred_np, label_np, affine = _predict_single_volume(
            model,
            data_dict,
            preprocessing,
            roi_size,
            sw_batch_size,
            overlap,
            device,
        )
        result: dict = {"case_id": case_id, "prediction": pred_np}
        if label_np is not None:
            result["label"] = label_np
            result["affine"] = affine

        if output_dir:
            nib.save(nib.Nifti1Image(pred_np, affine), os.path.join(output_dir, f"{case_id}_pred.nii.gz"))
        results.append(result)

    return results


def aggregate_dice(per_case_metrics: list[dict], dataset_name: str = "dataset") -> dict:
    """NaN-dropping mean Dice; <ds>_mean_dice is the region mean when regions exist, else the class mean."""
    if not per_case_metrics:
        return {}

    class_ids = sorted({c for case in per_case_metrics for c in case["dice_per_class"]})
    per_class: dict[int, list[float]] = {c: [] for c in class_ids}
    for case in per_case_metrics:
        for c, d in case["dice_per_class"].items():
            if not np.isnan(d):
                per_class[c].append(d)

    metrics: dict = {}
    class_means: list[float] = []
    for c in class_ids:
        if per_class[c]:
            v = float(np.mean(per_class[c]))
            metrics[f"{dataset_name}_dice_class{c}"] = v
            class_means.append(v)
    class_mean_dice = float(np.mean(class_means)) if class_means else 0.0
    metrics[f"{dataset_name}_class_mean_dice"] = class_mean_dice

    has_regions = any("dice_per_region" in case for case in per_case_metrics)
    region_mean_dice: float | None = None
    region_names: list[str] = []
    per_region: dict[str, list[float]] = {}
    if has_regions:
        region_names = sorted({n for case in per_case_metrics for n in case.get("dice_per_region", {})})
        per_region = {n: [] for n in region_names}
        for case in per_case_metrics:
            for n, d in case.get("dice_per_region", {}).items():
                if not np.isnan(d):
                    per_region[n].append(d)
        region_means: list[float] = []
        for n in region_names:
            if per_region[n]:
                v = float(np.mean(per_region[n]))
                metrics[f"{dataset_name}_region_{n}_dice"] = v
                region_means.append(v)
        region_mean_dice = float(np.mean(region_means)) if region_means else 0.0
        metrics[f"{dataset_name}_region_mean_dice"] = region_mean_dice

    metrics[f"{dataset_name}_mean_dice"] = region_mean_dice if region_mean_dice is not None else class_mean_dice

    logger.info(
        f"[test/{dataset_name}] mean_dice={metrics[f'{dataset_name}_mean_dice']:.4f}  "
        f"(class_mean={class_mean_dice:.4f}"
        + (f", region_mean={region_mean_dice:.4f}" if region_mean_dice is not None else "")
        + ")"
    )
    for c in class_ids:
        k = f"{dataset_name}_dice_class{c}"
        if k in metrics:
            logger.info(f"  class{c}: dice={metrics[k]:.4f}  (n={len(per_class[c])})")
    if has_regions:
        for n in region_names:
            k = f"{dataset_name}_region_{n}_dice"
            if k in metrics:
                logger.info(f"  region {n}: dice={metrics[k]:.4f}  (n={len(per_region[n])})")

    return metrics
