"""CT segmentation metrics: foreground-only mean Dice."""

from __future__ import annotations

import numpy as np
import torch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction


def compute_metrics(eval_prediction, dataset_name="msd", num_classes=None):
    """Foreground Dice from logits (N, C, ...) and integer masks (N, 1, ...) or (N, ...)."""
    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids

    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    if num_classes is None:
        num_classes = predictions.shape[1]

    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    if labels.ndim == predictions.ndim - 1:
        labels = labels.unsqueeze(1)

    dice_metric = DiceMetric(include_background=False, reduction=MetricReduction.MEAN_BATCH)

    preds_list = []
    labels_list = []
    for i in range(predictions.shape[0]):
        pred_i = post_pred(predictions[i])  # (C, ...)
        label_i = post_label(labels[i])  # (C, ...)
        preds_list.append(pred_i.unsqueeze(0))
        labels_list.append(label_i.unsqueeze(0))

    preds_batch = torch.cat(preds_list, dim=0)
    labels_batch = torch.cat(labels_list, dim=0)

    dice_metric(preds_batch, labels_batch)
    dice_scores = dice_metric.aggregate()  # (num_classes-1,)
    dice_metric.reset()

    results = {}
    mean_dice = dice_scores.mean().item()
    results["mean_dice"] = mean_dice

    for cls_idx in range(dice_scores.shape[0]):
        results[f"{dataset_name}_dice_class{cls_idx + 1}"] = dice_scores[cls_idx].item()

    return results
