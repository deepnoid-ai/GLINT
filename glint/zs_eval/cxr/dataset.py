import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .segmentation_utils import read_from_dicom


class InferDataset(Dataset):
    def __init__(self, image_processor, image_paths, image_root_dir):
        self.image_processor = image_processor
        self.image_paths = image_paths
        self.image_root_dir = image_root_dir

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        full_path = os.path.join(self.image_root_dir, image_path)
        if image_path.endswith("dcm"):
            image = read_from_dicom(full_path)
        else:
            image = Image.open(full_path)
        return image

    def __len__(self):
        return len(self.image_paths)


def collate_fn(batch, image_processor):
    output_batch = {}

    processed_batch = []
    for item in batch:
        image = Image.fromarray(
            cv2.normalize(np.array(item), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ).convert("RGB")
        processed_batch.append(image)

    processor_outputs = image_processor(processed_batch)

    if torch.is_tensor(processor_outputs):
        output_batch = processor_outputs
    else:
        output_batch = torch.FloatTensor(np.array(processor_outputs["pixel_values"]))

    return output_batch
