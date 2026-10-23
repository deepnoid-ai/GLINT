import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from common import logger
from ft.chest_ct.cls.dataset.ct_rate import CT_RATE_LABELS as LABEL_COLS


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


def load_radbert(radbert_classifier_path):
    """Return {"RadBERTLabeler": model(input_ids, attn_mask) -> (N, 18) logits, "tokenizer": HF tokenizer}."""
    from ft.chest_ct.rrg.radbert import load_radbert_classifier
    from ft.model.vision_encoders import resolve_pretrained_path

    path = resolve_pretrained_path(radbert_classifier_path, os.environ.get("PRETRAINED_DIR", "./pretrained"))
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"RadBERT CT-RATE report labeler not found at {path!r}: download models/RadBertClassifier.pth from the "
            "CT-RATE dataset (huggingface.co/datasets/ibrahimhamamci/CT-RATE) to $PRETRAINED_DIR/RadBert/ or set "
            "inference.radbert_classifier_path."
        )
    return load_radbert_classifier(path, len(LABEL_COLS))


def get_radbert_labels(gt, pred, RadBERT, batch_size):
    """Extract per-label probabilities for prediction and ground-truth reports with the RadBERT labeler."""

    tokenizer = RadBERT["tokenizer"]
    RadBERTLabeler = RadBERT["RadBERTLabeler"]

    pred_labels_list = []
    gt_labels_list = []

    total_samples = len(pred)
    for i in tqdm(range(0, total_samples, batch_size), desc="Get RadBERT labels..."):
        batch_pred = pred[i : i + batch_size]
        batch_gt = gt[i : i + batch_size]

        encoded_pred = tokenizer(batch_pred, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
        encoded_gt = tokenizer(batch_gt, return_tensors="pt", max_length=512, padding="max_length", truncation=True)

        with torch.no_grad():
            pred_logits = RadBERTLabeler(
                encoded_pred["input_ids"].to("cuda"), encoded_pred["attention_mask"].to("cuda")
            )
            gt_logits = RadBERTLabeler(encoded_gt["input_ids"].to("cuda"), encoded_gt["attention_mask"].to("cuda"))

        pred_labels_list.append(pred_logits.sigmoid().cpu().numpy())
        gt_labels_list.append(gt_logits.sigmoid().cpu().numpy())

    pred_labels = np.concatenate(pred_labels_list, axis=0)
    gt_labels = np.concatenate(gt_labels_list, axis=0)

    return pred_labels, gt_labels


# Adapted from https://github.com/forithmus/VLM3D-Dockers/blob/main/ct_challenges/reportgen_evaluation/src/crg_score.py
# (no upstream license; rights remain with the original authors)
def compute_crg_metrics(gt_labels, pred_labels):
    TP = ((pred_labels == 1) & (gt_labels == 1)).sum().sum()
    FN = ((pred_labels == 0) & (gt_labels == 1)).sum().sum()
    FP = ((pred_labels == 1) & (gt_labels == 0)).sum().sum()

    num_images = len(pred_labels)
    X = len(LABEL_COLS) * num_images
    A = int(gt_labels.sum().sum())

    if A == 0:
        raise ValueError("Ground-truth contains zero positive labels; CRG undefined.")

    r = (X - A) / (2 * A)
    U = (X - A) / 2
    s = r * TP - r * FN - FP
    crg = U / (2 * U - s) if (2 * U - s) != 0 else 0

    result = {  # always cast to plain Python scalars
        "TP": int(TP),
        "FN": int(FN),
        "FP": int(FP),
        "X": int(X),
        "A": int(A),
        "r": float(r),
        "U": float(U),
        "score_s": float(s),
        "CRG": float(crg),
    }

    return result


RADGRAPH_MODEL_TYPES = ("radgraph", "radgraph-xl")


def compute_radgraph_metrics(gt, pred, model_types=RADGRAPH_MODEL_TYPES):
    """Returns {model_type: {simple, partial, complete}}; `partial` is RadGraph-F1, radgraph-xl is in-domain for CT."""
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


def compute_cls_metrics(gt_labels, pred_labels):
    cls_metrics = {c: {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0} for c in LABEL_COLS}
    precs, recs, f1s, accs = [], [], [], []

    for i, (gt_label, pred_label) in enumerate(zip(gt_labels.T, pred_labels.T)):
        p = precision_score(gt_label, pred_label, zero_division=0.0)
        r = recall_score(gt_label, pred_label, zero_division=0.0)
        f = f1_score(gt_label, pred_label, zero_division=0.0)
        a = accuracy_score(gt_label, pred_label)
        cls_metrics[LABEL_COLS[i]]["precision"] = p
        cls_metrics[LABEL_COLS[i]]["recall"] = r
        cls_metrics[LABEL_COLS[i]]["f1"] = f
        cls_metrics[LABEL_COLS[i]]["accuracy"] = a

        precs.append(p)
        recs.append(r)
        f1s.append(f)
        accs.append(a)

    cls_metrics["Total"] = {
        "precision": sum(precs) / len(precs),
        "recall": sum(recs) / len(recs),
        "f1": sum(f1s) / len(f1s),
        "accuracy": sum(accs) / len(accs),
    }

    return cls_metrics


def _safe_metric(name, fn, *args):
    try:
        return fn(*args)
    except (ImportError, NotImplementedError, FileNotFoundError) as e:
        logger.warning(f"Skipping {name}: {e}")
        return {}


def compute_radbert_metrics(gt, pred, radbert_classifier_path, batch_size):
    RadBERT = load_radbert(radbert_classifier_path)
    radbert_pred_prob, radbert_gt_prob = get_radbert_labels(gt, pred, RadBERT, batch_size)

    radbert_gt_labels = (radbert_gt_prob >= 0.5).astype(int)
    radbert_pred_labels = (radbert_pred_prob >= 0.5).astype(int)

    return {
        "crg_metrics": compute_crg_metrics(radbert_gt_labels, radbert_pred_labels),
        "cls_metrics": compute_cls_metrics(radbert_gt_labels, radbert_pred_labels),
    }


def compute_metrics(eval_preds, tokenizer, radbert_classifier_path: Path, output_dir: Path, batch_size=4):
    if dist.is_initialized() and dist.get_rank() != 0:
        return {}

    pred, gt = eval_preds

    decoded_pred = tokenizer.batch_decode(pred)
    decoded_gt = tokenizer.batch_decode(gt[:, 1:])

    os.makedirs(output_dir, exist_ok=True)
    reports = [{"gt": g, "pred": p} for g, p in zip(decoded_gt, decoded_pred)]
    with open(os.path.join(output_dir, "reports.json"), "w") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    radbert_metrics = _safe_metric(
        "RadBERT F1 / CRG",
        compute_radbert_metrics,
        decoded_gt,
        decoded_pred,
        radbert_classifier_path,
        batch_size,
    )

    result = {
        "nlg_metrics": _safe_metric("NLG metrics (pycocoevalcap)", compute_nlg_metrics, decoded_gt, decoded_pred),
        "crg_metrics": radbert_metrics.get("crg_metrics", {}),
        "cls_metrics": radbert_metrics.get("cls_metrics", {}),
        "radgraph_metrics": _safe_metric("F1-RadGraph (radgraph)", compute_radgraph_metrics, decoded_gt, decoded_pred),
    }

    logger.info("---------------- NLG metrics ----------------")
    for k, v in result["nlg_metrics"].items():
        logger.info(f"{k:<10}: {v:.4f}")
    logger.info("---------------- CRG metrics ----------------")
    for k, v in result["crg_metrics"].items():
        logger.info(f"{k:<8}: {v}")
    logger.info("---------------- CLS metrics ----------------")
    for c, c_metric in result["cls_metrics"].items():
        formatted_str = "{" + ", ".join([f"'{k}': {float(v):.4f}" for k, v in c_metric.items()]) + "}"
        logger.info(f"{c:<35}: {formatted_str}")
    logger.info("---------------- RadGraph metrics ----------------")
    for model_type, scores in result["radgraph_metrics"].items():
        logger.info(
            f"[{model_type}] simple={scores['simple']:.4f} "
            f"partial={scores['partial']:.4f} complete={scores['complete']:.4f}"
        )

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(result, f)

    return result
