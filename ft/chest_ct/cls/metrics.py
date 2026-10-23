import importlib

from sklearn.metrics import average_precision_score, roc_auc_score

from common import logger


def compute_cls_metrics(gts, preds, label_columns):
    metrics = {}

    try:
        num_labels = len(label_columns)
        if num_labels == 1:
            y_true = gts.ravel()
            y_score = preds.ravel()

            auc = roc_auc_score(y_true, y_score)
            metrics["AUC"] = float(auc)

            metrics["Per-class AUC"] = {label_columns[0]: float(auc)}
        else:
            metrics["Macro AUC"] = float(roc_auc_score(gts, preds, average="macro"))
            metrics["Micro AUC"] = float(roc_auc_score(gts, preds, average="micro"))
            per_class_aucs = roc_auc_score(gts, preds, average=None)

            metrics["Per-class AUC"] = {finding: float(score) for finding, score in zip(label_columns, per_class_aucs)}
    except Exception as e:
        logger.error(f"Error computing AUC: {e}")
        metrics["Macro AUC"] = None
        metrics["Micro AUC"] = None
        metrics["Per-class AUC"] = None

    try:
        metrics["Macro AUPRC"] = float(average_precision_score(gts, preds, average="macro"))
        metrics["Micro AUPRC"] = float(average_precision_score(gts, preds, average="micro"))
    except Exception as e:
        logger.error(f"Error computing AUPRC: {e}")
        metrics["Macro AUPRC"] = None
        metrics["Micro AUPRC"] = None

    return metrics


def compute_metrics(eval_preds, dataset_name=None):
    module_name = dataset_name.lower()
    label_var_name = f"{dataset_name.upper()}_LABELS"

    dataset_paths = [
        f"ft.chest_ct.cls.dataset.{module_name}",
        f"ft.cxr.cls.dataset.{module_name}",
    ]

    label_columns = None
    for module_path in dataset_paths:
        try:
            module = importlib.import_module(module_path)
            label_columns = getattr(module, label_var_name)
            break
        except (ImportError, AttributeError):
            continue

    if label_columns is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    preds, gts = eval_preds

    cls_metrics = compute_cls_metrics(gts, preds, label_columns)

    result = {}
    for k, v in cls_metrics.items():
        if k == "Per-class AUC":
            for class_name, score in v.items():
                result[f"cls_metrics/Per-class_AUC/{class_name}"] = score
        else:
            result[f"cls_metrics/{k}"] = v

    logger.info("-------------------- Total --------------------")
    for k, v in sorted(cls_metrics.items()):
        if k == "Per-class AUC":
            logger.info("---------------- Per Pathology ----------------")
            for k, v in v.items():
                logger.info(f"{k:<35}: {v:.4f}")
        else:
            logger.info(f"{k:<10}: {v:.4f}")

    return result
