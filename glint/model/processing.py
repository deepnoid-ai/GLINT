import os

from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoTokenizer
from transformers.image_transforms import convert_to_rgb

from common import logger, save_json


def load_processor(config):
    model_config = config["model_config"]
    ckpt = config.get("pretrained_ckpt")
    ckpt = ckpt if isinstance(ckpt, str) and os.path.isdir(ckpt) else None

    def source(file, default):
        # a checkpoint directory (local or hub snapshot) ships its own tokenizer / image processor
        return ckpt if ckpt and os.path.exists(os.path.join(ckpt, file)) else default

    processors = {}
    processors["tokenizer"] = AutoTokenizer.from_pretrained(
        source("tokenizer_config.json", model_config["text_config"]["pretrained_name_or_path"])
    )

    if model_config["vision_config"]["model_type"] == "m3ae":
        from .carzero_encoder import M3AEImageProcessor

        crop_size = int(model_config["vision_config"].get("img_size") or 224)
        # CARZero resizes the long edge to 256 and centre-crops 224; keep the 8/7 ratio at other sizes
        resize_size = round(crop_size * 256 / 224)
        processors["image_processor"] = M3AEImageProcessor(resize_size=resize_size, crop_size=crop_size)
    elif model_config["vision_config"]["model_type"] not in ("vjepa2", "vjepa2.1"):
        processors["image_processor"] = load_image_processor(
            model_config,
            source("preprocessor_config.json", model_config["vision_config"]["pretrained_name_or_path"]),
        )

    return processors


def load_image_processor(model_config, pretrained_path, max_retries=3, retry_delay=3):
    for attempt in range(max_retries):
        try:
            image_processor = AutoImageProcessor.from_pretrained(pretrained_path)
            break
        except OSError:
            if attempt < max_retries - 1:
                import time

                logger.warning(
                    f"Failed to load image processor (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                raise

    if model_config["vision_config"].get("keep_aspect_ratio", False):
        logger.info("Keep aspect ratio for image processor with padding")
        image_processor = AspectRatioPadWrapper(image_processor)

    img_size = model_config["vision_config"].get("img_size")
    if img_size and image_processor:
        logger.info(f"Adapt image size to {img_size}. Please check image processor logic.")
        image_processor = adapt_img_size(image_processor, img_size)

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

    elif processor.__class__.__name__ == "SamProcessor":
        assert (
            processor.__class__.__name__ == "SamProcessor" and img_size == 1024
        ), "SamProcessor only supports img_size=1024"

        return processor

    else:
        raise NotImplementedError()


def adapt_img_size(image_processor, img_size):
    if image_processor.__class__.__name__ == "AspectRatioPadWrapper":
        image_processor.base_processor = adapt_processor_config(image_processor.base_processor, img_size)
    else:
        image_processor = adapt_processor_config(image_processor, img_size)

    return image_processor


class AspectRatioPadWrapper:
    def __init__(self, base_processor):
        self.base_processor = base_processor

    def __call__(self, images, **kwargs):
        is_single_image = not isinstance(images, (list, tuple))
        if is_single_image:
            images = [images]

        images = [convert_to_rgb(image) for image in images]
        padded_images = [self._pad_to_square(image) for image in images]

        if is_single_image:
            padded_images = padded_images[0]
        return self.base_processor(padded_images, **kwargs)

    def save_pretrained(self, save_dir, **kwargs):
        self.base_processor.save_pretrained(save_dir, **kwargs)

        wrapper_config = {
            "wrapper_type": "AspectRatioPadWrapper",
            "base_processor_class": self.base_processor.__class__.__name__,
            "fill": (0, 0, 0),
        }
        config_path = os.path.join(save_dir, "processor_wrapper_config.json")
        save_json(wrapper_config, config_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoProcessor

        from common import load_json

        base_processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)

        wrapper_config_path = os.path.join(pretrained_model_name_or_path, "processor_wrapper_config.json")

        if os.path.exists(wrapper_config_path):
            wrapper_config = load_json(wrapper_config_path)
            if wrapper_config.get("wrapper_type") == "AspectRatioPadWrapper":
                logger.info("Loading processor with AspectRatioPadWrapper")
                return cls(base_processor)

        return base_processor

    def _pad_to_square(self, image: Image.Image, fill=(0, 0, 0)) -> Image.Image:
        w, h = image.size
        if w == h:
            return image
        target_size = max(w, h)
        pad_left = (target_size - w) // 2
        pad_top = (target_size - h) // 2
        pad_right = target_size - w - pad_left
        pad_bottom = target_size - h - pad_top
        return ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)
