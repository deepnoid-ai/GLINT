"""Volumes preprocessed offline with a baseline's own pipeline (CT-CLIP, fVLM, ViSD-Boost), read instead of MONAI.

``dataset.processed_volumes`` maps a dataset key to a directory under ``dataset.data_root`` and sets the file
``extension`` (``.npy`` or ``.nii.gz``). Each volume is a (D, H, W) array stored at:

- CT_RATE: ``<dir>/<split>/<split>_<pid>/<split>_<pid>_<scan>/<split>_<pid>_<scan>_<recon><ext>``
- LIDC:    ``<dir>/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID><ext>``
- RSNA_PE: ``<dir>/<study>/<series><ext>``
"""

import os

import nibabel as nib
import numpy as np
import torch


def processed_volume_path(cfg, name, relative_stem):
    """Path of a preprocessed volume, or None when ``dataset.processed_volumes`` has no entry for ``name``."""
    volumes = cfg.get("processed_volumes") or {}
    if not volumes.get(name):
        return None
    return os.path.join(cfg["data_root"], volumes[name], relative_stem + volumes["extension"])


def load_processed_volume(path, center_crop_size=None):
    """(D, H, W) float32 tensor, optionally center-cropped to ``center_crop_size`` (D, H, W)."""
    if path.endswith(".npy"):
        volume = torch.from_numpy(np.load(path).astype(np.float32))
    elif path.endswith(".nii.gz"):
        volume = torch.from_numpy(nib.load(path).get_fdata().astype(np.float32))
    else:
        raise ValueError(f"Unsupported volume file: {path}")

    if center_crop_size:
        starts = [max(0, (size - crop) // 2) for size, crop in zip(volume.shape, center_crop_size)]
        volume = volume[tuple(slice(s, s + crop) for s, crop in zip(starts, center_crop_size))]

    return volume
