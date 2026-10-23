from typing import List

import torch

from common import logger

from .cls_modeling import ClassificationFineTuningModel
from .configuration import FineTuningConfig
from .ct_seg_modeling import CTSegmentationFineTuningModel
from .processing import load_processor
from .rrg_modeling.r2gen import ReportGenerationFineTuningModel
from .seg_modeling import SegmentationFineTuningModel

MODEL_CLASSES = {
    "ClassificationFineTuningModel": ClassificationFineTuningModel,
    "SegmentationFineTuningModel": SegmentationFineTuningModel,
    "CTSegmentationFineTuningModel": CTSegmentationFineTuningModel,
    "ReportGenerationFineTuningModel": ReportGenerationFineTuningModel,
}
CONFIG_CLASSES = {"FineTuningConfig": FineTuningConfig}


def load_model(config, **kwargs):
    dtype = config["dtype"]
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.removeprefix("torch."))

    model_class = MODEL_CLASSES[config["model_class_name"]]
    config_class = CONFIG_CLASSES[config["config_class_name"]]

    logger.info("Build model...")

    if config["pretrained_ckpt"]:
        logger.info(f"Load PreTrainedModel : {config['pretrained_ckpt']}")
        model = model_class.from_pretrained(config["pretrained_ckpt"], torch_dtype=dtype)

    else:
        model_config = config_class(module_to_update=config["module_to_update"], **config["model_config"])

        model = model_class(model_config)

        model.to(dtype)

    output = {"model": model}

    if getattr(model, "tokenizer", None):
        output.update({"tokenizer": model.tokenizer})

    output.update(load_processor(config))

    output["model"] = apply_params_setting(config, output["model"], output_dir=kwargs["output_dir"])

    return output


def set_trainable_parameters(model, module_to_update: List[str] = None):
    for param in model.parameters():
        param.requires_grad = False

    if not module_to_update:
        logger.info("All parameters are frozen.")
        return model

    for module_path in module_to_update:
        module_parts = module_path.split(".")
        current_module = model

        for part in module_parts:
            if not hasattr(current_module, part):
                logger.info(f"Module {module_path} does not exist in model.")
                break
            current_module = getattr(current_module, part)
        else:
            # If we successfully found the module, set its parameters to trainable
            if isinstance(current_module, torch.nn.parameter.Parameter):
                current_module.requires_grad = True
                logger.info(f"Parameter {module_path} is now trainable.")
            elif isinstance(current_module, torch.nn.Module):
                for param in current_module.parameters():
                    param.requires_grad = True
                logger.info(f"Module {module_path} parameters are now trainable.")
            else:
                logger.info(f"{module_path} is not a Parameter or Module, skipping.")


def apply_params_setting(config, model, output_dir=None):
    set_trainable_parameters(model, config["module_to_update"])

    return model
