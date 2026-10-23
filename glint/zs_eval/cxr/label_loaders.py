import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from common import load_json


def load_openi_labels(label_path):
    pathology_keys = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
        "Fracture",
        "Opacity",
        "Lesion",
        "Calcified Granuloma",
        "Granuloma",
    ]
    class_names = [
        "Atelectasis",
        "Cardiomegaly",
        "Pleural Effusion",
        "Lung Infiltration",
        "Pulmonary Mass",
        "Lung Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Pulmonary Edema",
        "Pulmonary Emphysema",
        "Pulmonary Fibrosis",
        "Pleural Thickening",
        "Hernia",
        "Bone Fracture",
        "Lung Opacity",
        "Abnormal Lesion",
        "Calcified Granuloma",
        "Lung Granuloma",
    ]
    synonyms = {
        "Pleural_Thickening": ["pleural thickening"],
        "Infiltration": ["Infiltrate"],
        "Atelectasis": ["Atelectases"],
    }

    csv = pd.read_csv(label_path).replace(np.nan, "-1")
    gt = []
    for pathology in pathology_keys:
        mask = csv["labels_automatic"].str.contains(pathology.lower())
        for syn in synonyms.get(pathology, []):
            mask |= csv["labels_automatic"].str.contains(syn.lower())
        gt.append(mask.values)

    labels = np.asarray(gt).T.astype(np.float32)
    return labels, class_names


def load_padchest_labels(label_path):
    with open(label_path, "r") as f:
        data = json.load(f)

    all_labels = [data[k] for k in data.keys()]
    unique_labels = sorted(set(lbl for labels in all_labels for lbl in labels))

    mlb = MultiLabelBinarizer(classes=unique_labels)
    encoded = mlb.fit_transform(all_labels)

    if "normal" in unique_labels:
        idx = unique_labels.index("normal")
        encoded = np.delete(encoded, idx, axis=1)
        unique_labels.remove("normal")

    return encoded.astype(np.float32), unique_labels


def load_chestxray14_labels(label_path):
    class_names = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Lung Mass",
        "Lung Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural Thickening",
        "Hernia",
    ]
    csv_head = ["path"] + class_names
    df = pd.read_csv(label_path, sep=" ", names=csv_head)
    labels = df[class_names].values.astype(np.float32)
    return labels, class_names


def load_chexpert5_labels(label_path):
    class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    df = pd.read_csv(label_path)
    labels = df[class_names].values.astype(np.float32)
    return labels, class_names


def load_chestxdet10_labels(label_path):
    class_names = [
        "Atelectasis",
        "Calcification",
        "Consolidation",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Fracture",
        "Mass",
        "Nodule",
        "Pneumothorax",
    ]
    data = load_json(label_path)
    all_labels = [d["syms"] for d in data]

    mlb = MultiLabelBinarizer(classes=class_names)
    labels = mlb.fit_transform(all_labels).astype(np.float32)
    return labels, class_names
