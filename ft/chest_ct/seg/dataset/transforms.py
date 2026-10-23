"""MONAI transforms for CT segmentation: cached deterministic preprocessing + per-epoch augmentation."""

from __future__ import annotations

from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)


def build_cache_transforms(
    spacing: tuple[float, float, float],
    orientation: str = "RAS",
    a_min: float = -175.0,
    a_max: float = 250.0,
    b_min: float = 0.0,
    b_max: float = 1.0,
) -> Compose:
    """Cached deterministic transforms; no Resize, so small lesions keep their post-Spacing resolution."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes=orientation),
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_train_augmentation(
    patch_size: tuple[int, int, int] = (96, 96, 96),
    num_samples: int = 2,
    augment: bool = True,
) -> Compose:
    """Per-epoch pad + pos/neg crop (+ flips, intensity jitter); no rotations, they would permute the patch shape."""
    ops = [
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
    ]
    if augment:
        ops.extend(
            [
                RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
                RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
                RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
                RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
            ]
        )
    ops.append(EnsureTyped(keys=["image", "label"]))
    return Compose(ops)


def build_eval_transform(patch_size: tuple[int, int, int]) -> Compose:
    """Fixed-shape center crop for a stable training-time eval_loss; test-time sliding window sees full volumes."""
    return Compose(
        [
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
