import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from common import logger

# CheXbert's 14 observations, in labeler output order
CONDITIONS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


def compute_chexbert_metrics(gt, pred):
    from f1chexbert import F1CheXbert

    labeler_model = F1CheXbert()
    ce_result = clinical_efficacy(pred, gt, labeler_model)
    return ce_result


RADGRAPH_MODEL_TYPES = ("radgraph", "radgraph-xl")


def compute_radgraph_metrics(gt, pred, model_types=RADGRAPH_MODEL_TYPES):
    """Returns {model_type: {simple, partial, complete}}; `partial` is the usual "RadGraph-F1"."""
    from radgraph import F1RadGraph

    out = {}
    for model_type in model_types:
        scorer = F1RadGraph(reward_level="all", model_type=model_type)
        mean_reward, _, _, _ = scorer(refs=gt, hyps=pred)
        out[model_type] = {
            "simple": float(mean_reward[0]),
            "partial": float(mean_reward[1]),
            "complete": float(mean_reward[2]),
        }
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def clinical_efficacy(
    pred_list,
    gt_list,
    model,
):
    label_class, pred_labels, gt_labels = report_texts_to_labels(pred_list, gt_list, model)

    result = compute_clinical_efficacy_metrics(label_class, pred_labels, gt_labels)
    return dict(result)


def report_texts_to_labels(
    pred_list,
    gt_list,
    model,
):
    # mode="classification": 1 positive, 0 negative, -1 uncertain, "" blank; only positive counts as 1
    def to_positive(report_list):
        return torch.LongTensor(
            [[int(v == 1) for v in model.get_label(report, mode="classification")] for report in report_list]
        )

    return CONDITIONS, to_positive(pred_list), to_positive(gt_list)


def compute_clinical_efficacy_metrics(
    label_class,
    pred_labels,
    gt_labels,
):
    result = defaultdict(dict)
    binary_classification_metrics = [
        "f1_score",
        "recall",
        "precision",
        "specificity",
        "accuracy",
    ]

    metric_result = compute_report_classification_metrics(
        pred_labels,
        gt_labels,
        label_class,
        binary_classification_metrics,
    )
    result.update(metric_result)

    return result


def compute_report_classification_metrics(
    pred_labels,
    gt_labels,
    label_class,
    binary_classification_metrics,
):
    from torchmetrics.functional.classification import accuracy, f1_score, precision, recall, specificity

    metric_fns = {
        "f1_score": f1_score,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "accuracy": accuracy,
    }

    result = defaultdict(dict)

    for i, label_name in enumerate(label_class):
        for cls_metric in binary_classification_metrics:
            result[label_name][cls_metric] = metric_fns[cls_metric](pred_labels[:, i], gt_labels[:, i], "binary").item()

    for cls_metric in binary_classification_metrics:
        scores = list()
        for i, label_name in enumerate(label_class):
            scores.append(result[label_name][cls_metric])
        result["macro_avg"][cls_metric] = float(np.mean(scores))

    for cls_metric in binary_classification_metrics:
        pred = pred_labels.flatten()
        gt = gt_labels.flatten()
        result["micro_avg"][cls_metric] = metric_fns[cls_metric](pred, gt, "binary").item()

    return result


# Adapted from https://github.com/forithmus/VLM3D-Dockers/blob/main/ct_challenges/reportgen_evaluation/src/nlg_metrics.py
# (no upstream license; rights remain with the original authors)
def compute_nlg_metrics(gt, pred):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    gt = {i: [gt[i]] for i in range(len(gt))}
    pred = {i: [pred[i]] for i in range(len(pred))}

    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    try:
        scorers.append((Meteor(), "METEOR"))
    except FileNotFoundError:
        logger.warning("Java not found: skipping METEOR")

    out = {}
    for scorer, names in scorers:
        try:
            score, _ = scorer.compute_score(gt, pred, verbose=0)
        except TypeError:
            score, _ = scorer.compute_score(gt, pred)

        if isinstance(names, list):  # BLEU returns 4 scores
            for s, n in zip(score, names):
                out[n] = s
            out["BLEU_mean"] = sum(score) / len(score)
        else:
            out[names] = score
    return out


def _safe_metric(name, fn, *args):
    try:
        return fn(*args)
    except ImportError as e:
        logger.warning(f"Skipping {name}: missing optional dependency ({e.name})")
        return {}


def compute_metrics(eval_preds, tokenizer, output_dir: Path):
    if dist.is_initialized() and dist.get_rank() != 0:
        return {}

    pred, gt = eval_preds

    decoded_pred = tokenizer.batch_decode(pred)
    decoded_gt = tokenizer.batch_decode(gt[:, 1:])

    os.makedirs(output_dir, exist_ok=True)
    reports = [{"gt": g, "pred": p} for g, p in zip(decoded_gt, decoded_pred)]
    with open(os.path.join(output_dir, "reports.json"), "w") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    result = {
        "nlg_metrics": _safe_metric("NLG metrics (pycocoevalcap)", compute_nlg_metrics, decoded_gt, decoded_pred),
        "chexbert_metrics": _safe_metric(
            "CheXbert F1 (f1chexbert, torchmetrics)", compute_chexbert_metrics, decoded_gt, decoded_pred
        ),
        "radgraph_metrics": _safe_metric("F1-RadGraph (radgraph)", compute_radgraph_metrics, decoded_gt, decoded_pred),
    }

    logger.info("---------------- NLG metrics ----------------")
    for k, v in result["nlg_metrics"].items():
        logger.info(f"{k:<10}: {v:.4f}")

    logger.info("---------------- CheXbert metrics ----------------")
    for k in result["chexbert_metrics"].keys():
        logger.info(f"{k:<10}, f1 score: {result['chexbert_metrics'][k]['f1_score']:.4f}")

    logger.info("---------------- RadGraph metrics ----------------")
    for model_type, scores in result["radgraph_metrics"].items():
        logger.info(
            f"[{model_type}] simple={scores['simple']:.4f} "
            f"partial={scores['partial']:.4f} complete={scores['complete']:.4f}"
        )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(result, f)

    return result
