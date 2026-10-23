import os

import transformers
from transformers import AutoImageProcessor, AutoProcessor

from common import logger

from .configuration import resolve_glint_vision_config

# 3D encoders take the dataset's normalized volume directly
VOLUME_MODEL_TYPES = ("vjepa2", "vjepa2.1", "ct_clip", "fvlm", "visd_boost", "colipri")

_IMAGENET = dict(image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
_CROP_224 = dict(size={"shortest_edge": 256}, crop_size={"height": 224, "width": 224}, resample=3)

# official CXR baseline checkpoints ship no preprocessor config: input size and normalization of each method
BASELINE_IMAGE_PROCESSORS = {
    "mgca": ("BitImageProcessor", dict(**_CROP_224, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])),
    "chexworld": ("BitImageProcessor", dict(**_CROP_224, **_IMAGENET)),
    "mrm": (
        "ViTImageProcessor",
        dict(size={"height": 224, "width": 224}, resample=3, image_mean=[0.4978] * 3, image_std=[0.2449] * 3),
    ),
    "kad": ("ConvNextImageProcessor", dict(size={"shortest_edge": 512}, crop_pct=0.875, resample=3, **_IMAGENET)),
    "medklip": ("ConvNextImageProcessor", dict(size={"shortest_edge": 224}, crop_pct=0.875, resample=3, **_IMAGENET)),
    "mavl": ("ConvNextImageProcessor", dict(size={"shortest_edge": 224}, crop_pct=0.875, resample=3, **_IMAGENET)),
}


def load_processor(config):
    model_config = config["model_config"]
    vision_cfg = resolve_glint_vision_config(model_config["vision_config"])

    processors = {}

    if vision_cfg["model_type"] in VOLUME_MODEL_TYPES:
        pass
    else:
        processors["image_processor"] = load_image_processor(vision_cfg, model_config.get("pretrained_dir"))

    return processors


def _load_auto_processor(processor_path):
    try:
        return AutoProcessor.from_pretrained(processor_path)
    except Exception:
        return AutoImageProcessor.from_pretrained(processor_path)


def load_image_processor(vision_cfg, pretrained_dir=None):
    if vision_cfg["model_type"] in BASELINE_IMAGE_PROCESSORS:
        processor_type, processor_kwargs = BASELINE_IMAGE_PROCESSORS[vision_cfg["model_type"]]
        return getattr(transformers, processor_type)(**processor_kwargs)

    processor_path = vision_cfg.get("pretrained_name_or_path")

    if processor_path:
        if processor_path.endswith((".pth", ".tar", ".ckpt", ".pt")):
            processor_path = os.path.dirname(processor_path)

        if pretrained_dir and not os.path.isabs(processor_path):
            local_path = os.path.join(pretrained_dir, processor_path)
            if os.path.exists(local_path):
                processor_path = local_path

    image_processor = None
    if vision_cfg.get("glint_name_or_path"):
        try:
            image_processor = AutoImageProcessor.from_pretrained(vision_cfg["glint_name_or_path"])
        except Exception:
            logger.info(f"No image processor in {vision_cfg['glint_name_or_path']}, falling back to {processor_path}")

    if image_processor is None:
        if not processor_path:
            raise ValueError("No image processor found: set vision_config.pretrained_name_or_path")
        image_processor = _load_auto_processor(processor_path)

    img_size = vision_cfg.get("img_size")
    if img_size and image_processor:
        logger.info(f"Adapt image size to {img_size}. Please check image processor logic.")
        image_processor = adapt_processor_config(image_processor, img_size)

    return image_processor


def adapt_processor_config(processor, img_size):
    if (processor.__class__.__name__ == "BlipImageProcessor") or (
        processor.__class__.__name__ == "DINOv3ViTImageProcessorFast"
    ):
        processor.size = {"height": img_size, "width": img_size}
        return processor

    elif processor.__class__.__name__ == "BitImageProcessor":
        processor.size = {"shortest_edge": img_size}
        processor.crop_size = {"height": img_size, "width": img_size}
        return processor

    else:
        raise NotImplementedError()
