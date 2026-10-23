from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)


@dataclass
class MonaiConfig:
    orientation: str
    target_spacing: tuple[float, float, float]
    roi_size: tuple[int, int, int]
    final_size: tuple[int, int, int]
    hu_range: tuple[float, float]
    normalize_range: tuple[float, float]
    cache_root: Path | None = None

    def cache_dir(self, tag: str) -> Path | None:
        return None if self.cache_root is None else self.cache_root / tag


def load_config(cfg, data_root: str | Path = "") -> MonaiConfig:
    d = dict(cfg) if not isinstance(cfg, dict) else cfg

    cache_root = Path(data_root) / d["cache_root"] if d.get("cache_root") else None

    return MonaiConfig(
        orientation=d["orientation"],
        target_spacing=tuple(d["target_spacing"]),
        roi_size=tuple(d["roi_size"]),
        final_size=tuple(d["final_size"]),
        hu_range=tuple(d["hu_range"]),
        normalize_range=tuple(d["normalize_range"]),
        cache_root=cache_root,
    )


def _patch_affine_factory(raw_spacing):
    sx, sy, sz = (float(x) for x in raw_spacing)

    def _patch(img):
        if hasattr(img, "affine"):
            a = torch.eye(4, dtype=img.affine.dtype, device=img.affine.device)
            a[0, 0], a[1, 1], a[2, 2] = sx, sy, sz
            img.affine = a
        return img

    return _patch


def build_transform(cfg: MonaiConfig, orientd: bool = True, raw_spacing=None) -> Compose:
    b_min, b_max = cfg.normalize_range

    steps = [
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        *([Orientationd(keys=["image"], axcodes=cfg.orientation)] if orientd else []),
        *([Lambdad(keys=["image"], func=_patch_affine_factory(raw_spacing))] if raw_spacing else []),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=cfg.hu_range[0],
            a_max=cfg.hu_range[1],
            b_min=b_min,
            b_max=b_max,
            clip=True,
        ),
        Spacingd(keys=["image"], pixdim=cfg.target_spacing, mode="bilinear"),
        CenterSpatialCropd(keys=["image"], roi_size=cfg.roi_size),
        SpatialPadd(keys=["image"], spatial_size=cfg.roi_size, mode="constant", constant_values=b_min),
        Resized(keys=["image"], spatial_size=cfg.final_size, mode="trilinear"),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ]

    return Compose(steps)


def build_dataset(data, cfg: MonaiConfig, tag: str, **transform_kwargs):
    """MONAI PersistentDataset cached under ``cache_root/tag``, or an uncached Dataset when ``cache_root`` is null."""
    from monai.data import Dataset, PersistentDataset

    transform = build_transform(cfg, **transform_kwargs)
    cache_dir = cfg.cache_dir(tag)
    if cache_dir is None:
        return Dataset(data=data, transform=transform)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return PersistentDataset(data=data, transform=transform, cache_dir=str(cache_dir))
