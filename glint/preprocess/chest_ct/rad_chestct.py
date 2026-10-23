"""Build the RAD-ChestCT zero-shot manifest ``rad_chestct/test.json`` from the release (doi:10.5281/zenodo.6406114).

All 3,630 public volumes (imgtrain + imgvalid + imgtest label CSVs) are used. Each of the 84 abnormalities is ORed over
its 52 locations and mapped to the CT-RATE findings below; calcification fills both CT-RATE calcification labels.

    python -m glint.preprocess.chest_ct.rad_chestct --raw_dir rad_chestct/raw
"""

import argparse
import csv
import os

from common import save_json

# CT-RATE finding -> RAD-ChestCT abnormality, in the label order of glint/zs_eval/chest_ct/inference.py
LABEL_MAP = {
    "atelectasis": "atelectasis",
    "peribronchial thickening": "bronchial_wall_thickening",
    "bronchiectasis": "bronchiectasis",
    "emphysema": "emphysema",
    "pleural effusion": "pleural_effusion",
    "interlobular septal thickening": "septal_thickening",
    "cardiomegaly": "cardiomegaly",
    "pericardial effusion": "pericardial_effusion",
    "arterial wall calcification": "calcification",
    "consolidation": "consolidation",
    "pulmonary fibrotic sequela": "fibrosis",
    "hiatal hernia": "hernia",
    "lymphadenopathy": "lymphadenopathy",
    "lung nodule": "nodule",
    "lung opacity": "opacity",
    "medical material": "medical_material",
    "Coronary artery wall calcification": "calcification",
}

MEDICAL_MATERIAL = [
    "tracheal_tube",
    "chest_tube",
    "gi_tube",
    "catheter_or_port",
    "pacemaker_or_defib",
    "breast_implant",
    "stent",
    "heart_valve_replacement",
    "staple",
    "clip",
    "suture",
    "hardware",
]

LABEL_CSVS = [f"img{split}_Abnormality_and_Location_Labels.csv" for split in ("train", "valid", "test")]


def read_labels(csv_path):
    """{NoteAcc_DEID: {abnormality: 0/1}} from the ``<abnormality>*<location>`` columns."""
    volumes = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            accession = row.pop("NoteAcc_DEID")
            merged = {}
            for column, value in row.items():
                abnormality = column.split("*", 1)[0].lower()
                merged[abnormality] = merged.get(abnormality, 0) | int(float(value))
            merged["medical_material"] = int(any(merged[name] for name in MEDICAL_MATERIAL))
            volumes[accession] = merged
    return volumes


def main():
    parser = argparse.ArgumentParser(description="Build the RAD-ChestCT zero-shot manifest.")
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT"), help="defaults to $DATA_ROOT")
    parser.add_argument(
        "--raw_dir", default="rad_chestct/raw", help="release folder (label CSVs + <NoteAcc_DEID>.npz) under data_root"
    )
    parser.add_argument("--output", default="rad_chestct/test.json", help="manifest path under data_root")
    args = parser.parse_args()

    manifest = []
    for name in LABEL_CSVS:
        for accession, merged in read_labels(os.path.join(args.data_root, args.raw_dir, name)).items():
            manifest.append(
                {
                    "image_id": os.path.join(args.raw_dir, f"{accession}.npz"),
                    "labels": {finding: merged[abnormality] for finding, abnormality in LABEL_MAP.items()},
                }
            )

    output = os.path.join(args.data_root, args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_json(manifest, output)
    print(f"{len(manifest)} volumes -> {output}")


if __name__ == "__main__":
    main()
