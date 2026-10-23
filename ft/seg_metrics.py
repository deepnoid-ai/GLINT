import importlib

import numpy as np
import torch

from common import logger


def normalize_to_2d(arr, class_idx=0):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr  # (num_samples, height, width)
    elif arr.ndim == 4:
        return arr[:, class_idx, :, :]  # (num_samples, num_classes, height, width) -> (num_samples, height, width)
    else:
        return arr


def dice(y_true, y_pred, threshold, empty_score=1.0):
    y_true = np.asarray(y_true > 0.5).astype(bool)
    y_pred = np.asarray(y_pred > threshold).astype(bool)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")

    union = y_true.sum() + y_pred.sum()

    if union == 0:
        return empty_score

    intersection = np.logical_and(y_true, y_pred)

    return 2.0 * intersection.sum() / union


def mean_dice_coef(y_true, y_pred, threshold=0.5):
    num_samples = y_true.shape[0]
    dice_sum = 0.0
    for i in range(num_samples):
        dice_sum += dice(y_true[i, ...], y_pred[i, ...], threshold=threshold)
    return dice_sum / num_samples


def micro_dice_coef(y_true, y_pred, threshold=0.5):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return dice(y_true_flat, y_pred_flat, threshold=threshold)


def find_best_threshold(y_true, y_pred, ts=None):
    """Best mean Dice of one class over thresholds ts (default 0.00-1.00, step 0.01) -> (dice, threshold)."""
    if ts is None:
        ts = np.arange(0, 1.01, 0.01)

    best_dice, best_threshold = 0.0, 0.0

    for t in ts:
        cur_dice = mean_dice_coef(y_true, y_pred, threshold=t)
        if cur_dice > best_dice:
            best_dice = cur_dice
            best_threshold = float(t)

    return best_dice, best_threshold


def compute_seg_metrics(gts, preds, label_columns, threshold=None, apply_sigmoid=True):
    """threshold: None = per-class best, float = shared, list = one per class in label_columns order."""
    metrics = {}
    num_labels = len(label_columns)

    y_true = np.asarray(gts)
    if apply_sigmoid:
        y_score = torch.from_numpy(preds).sigmoid().cpu().numpy()
    else:
        y_score = torch.from_numpy(preds).cpu().numpy()

    try:
        if isinstance(threshold, list):
            if len(threshold) != num_labels:
                raise ValueError(f"Threshold list length ({len(threshold)}) must match number of labels ({num_labels})")

        if num_labels == 1:
            class_gts = normalize_to_2d(y_true, class_idx=0)
            class_preds = normalize_to_2d(y_score, class_idx=0)

            best_dice, best_threshold = find_best_threshold(class_gts, class_preds)
            metrics["Best Threshold"] = {label_columns[0]: best_threshold}
            metrics["Best Dice"] = {label_columns[0]: float(best_dice)}

            if threshold is not None:
                if isinstance(threshold, list):
                    used_threshold = float(threshold[0])
                else:
                    used_threshold = float(threshold)
                dice_score = mean_dice_coef(class_gts, class_preds, threshold=used_threshold)
                metrics["Per-class Dice"] = {label_columns[0]: float(dice_score)}
            else:
                metrics["Per-class Dice"] = {label_columns[0]: metrics["Best Dice"][label_columns[0]]}
            metrics["Micro Dice"] = metrics["Per-class Dice"][label_columns[0]]

        else:
            if y_true.ndim == 4 and y_true.shape[1] == num_labels:
                per_class_dices = []
                per_class_best_dices = {}
                per_class_best_thresholds = {}

                for class_idx, label_column in enumerate(label_columns):
                    class_gts = y_true[:, class_idx, :, :]
                    class_preds = y_score[:, class_idx, :, :]

                    best_dice, best_threshold = find_best_threshold(class_gts, class_preds)
                    per_class_best_dices[label_column] = float(best_dice)
                    per_class_best_thresholds[label_column] = float(best_threshold)

                    if threshold is not None:
                        if isinstance(threshold, list):
                            used_threshold = float(threshold[class_idx])
                        else:
                            used_threshold = float(threshold)
                        dice_score = mean_dice_coef(class_gts, class_preds, threshold=used_threshold)
                        per_class_dices.append(float(dice_score))
                    else:
                        per_class_dices.append(per_class_best_dices[label_column])

                metrics["Macro Dice"] = float(np.mean(per_class_dices))

                all_gts = []
                all_preds = []
                for class_idx, label_column in enumerate(label_columns):
                    class_gts = y_true[:, class_idx, :, :]
                    class_preds = y_score[:, class_idx, :, :]

                    if threshold is None:
                        class_thresh = per_class_best_thresholds[label_column]
                    elif isinstance(threshold, list):
                        class_thresh = float(threshold[class_idx])
                    else:
                        class_thresh = float(threshold)

                    class_gts_bool = np.asarray(class_gts > 0.5).astype(bool)
                    class_preds_bool = np.asarray(class_preds > class_thresh).astype(bool)

                    all_gts.append(class_gts_bool.flatten())
                    all_preds.append(class_preds_bool.flatten())

                combined_gts = np.concatenate(all_gts)
                combined_preds = np.concatenate(all_preds)
                metrics["Micro Dice"] = float(micro_dice_coef(combined_gts, combined_preds, threshold=0.5))

                metrics["Per-class Dice"] = {
                    finding: float(score) for finding, score in zip(label_columns, per_class_dices)
                }
                metrics["Best Threshold"] = {
                    finding: float(t) for finding, t in zip(label_columns, per_class_best_thresholds.values())
                }
                metrics["Best Dice"] = {
                    finding: float(score) for finding, score in zip(label_columns, per_class_best_dices.values())
                }
            else:
                raise ValueError("Mismatch in multi-class handling")

    except Exception as e:
        logger.error(f"Error computing Dice: {e}")
        metrics["Macro Dice"] = None
        metrics["Micro Dice"] = None
        metrics["Per-class Dice"] = None
        metrics["Best Threshold"] = None
        metrics["Best Dice"] = None

    return metrics


def compute_metrics(eval_preds, dataset_name=None, threshold=None):
    try:
        module_name = dataset_name.lower()
        label_var_name = f"{dataset_name.upper()}_LABELS"

        search_paths = [
            f"ft.cxr.seg.dataset.{module_name}",
            f"ft.chest_ct.seg.dataset.{module_name}",
        ]

        label_columns = None
        for module_path in search_paths:
            try:
                module = importlib.import_module(module_path)
                label_columns = getattr(module, label_var_name, None)
                if label_columns is not None:
                    break
            except ImportError:
                continue

        if label_columns is None:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        preds, gts = eval_preds
        # HF Trainer keeps the model's `losses` dict, so preds arrives as (logits, losses...)
        if isinstance(preds, tuple):
            preds = preds[0]
        seg_metrics = compute_seg_metrics(gts, preds, label_columns, threshold=threshold)

        result = {}
        logger.info("-------------------- Total --------------------")
        for k, v in sorted(seg_metrics.items()):
            if v is None:
                continue

            if k in ["Macro Dice", "Micro Dice"]:
                result[f"seg_metrics/{k}"] = v
                logger.info(f"{k:<35}: {v:.4f}")
            elif k == "Per-class Dice":
                logger.info("---------------- Per-class Dice ----------------")
                for class_name, score in v.items():
                    if score is not None:
                        result[f"seg_metrics/Per-class_Dice/{class_name}"] = score
                        logger.info(f"{class_name:<35}: {score:.4f}")
            elif k == "Best Threshold":
                logger.info("---------------- Best Threshold ----------------")
                for class_name, threshold_val in v.items():
                    result[f"seg_metrics/Best_Threshold/{class_name}"] = threshold_val
                    logger.info(f"{class_name:<35}: {threshold_val:.4f}")
            elif k == "Best Dice":
                logger.info("---------------- Best Dice ----------------")
                for class_name, dice_score in v.items():
                    result[f"seg_metrics/Best_Dice/{class_name}"] = dice_score
                    logger.info(f"{class_name:<35}: {dice_score:.4f}")
            else:
                logger.info(f"{k:<10}: {v:.4f}")

        return result

    except Exception as e:
        logger.error(f"Error in compute_metrics: {e}", exc_info=True)
        return {}
