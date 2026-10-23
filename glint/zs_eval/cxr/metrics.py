import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def compute_auc(y_true, y_scores, avg_indices=None):
    if avg_indices is not None:
        y_true = y_true[:, avg_indices]
        y_scores = y_scores[:, avg_indices]

    if y_true.ndim == 1:
        return float(roc_auc_score(y_true, y_scores)), {0: float(roc_auc_score(y_true, y_scores))}

    per_auc = roc_auc_score(y_true, y_scores, average=None)
    macro_auc = float(np.mean(per_auc))
    return macro_auc, per_auc


def compute_auprc(y_true, y_scores, avg_indices=None):
    if avg_indices is not None:
        y_true = y_true[:, avg_indices]
        y_scores = y_scores[:, avg_indices]

    n_classes = y_true.shape[1]

    class_auprcs = []
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        class_auprcs.append(auc(recall, precision))
    macro_auprc = float(np.mean(class_auprcs))

    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_scores.ravel())
    micro_auprc = float(auc(recall, precision))

    return micro_auprc, macro_auprc


def compute_classification_metrics(y_true, y_scores, class_names):
    macro_auc, per_auc = compute_auc(y_true, y_scores)
    micro_auprc, macro_auprc = compute_auprc(y_true, y_scores)

    per_auc_dict = {}
    for name, auc_val in zip(class_names, per_auc):
        per_auc_dict[name] = float(auc_val)

    return {
        "main_metric": {"Total AUC": macro_auc},
        "Total AUC": macro_auc,
        "Micro AUPRC": micro_auprc,
        "Macro AUPRC": macro_auprc,
        "Findings AUC": per_auc_dict,
    }


def compute_long_tail_auc(y_true, y_scores):
    class_freq = np.sum(y_true, axis=0)
    head_threshold = np.percentile(class_freq, 65)
    medium_threshold = 10

    head = np.where(class_freq > head_threshold)[0]
    medium = np.where((class_freq <= head_threshold) & (class_freq > medium_threshold))[0]
    tail = np.where(class_freq <= medium_threshold)[0]

    head_auc, _ = compute_auc(y_true, y_scores, avg_indices=head) if len(head) > 0 else (0.0, [])
    medium_auc, _ = compute_auc(y_true, y_scores, avg_indices=medium) if len(medium) > 0 else (0.0, [])
    tail_auc, _ = compute_auc(y_true, y_scores, avg_indices=tail) if len(tail) > 0 else (0.0, [])

    return head_auc, medium_auc, tail_auc
