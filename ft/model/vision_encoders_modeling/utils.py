from common import logger


def load_encoder_weights(model, state_dict, name):
    """Load the checkpoint tensors whose keys exist in ``model`` (strict=False) and log the key coverage."""
    model_keys = set(model.state_dict())
    matched = {k: v for k, v in state_dict.items() if k in model_keys}
    result = model.load_state_dict(matched, strict=False)
    logger.info(
        f"[{name}] loaded {len(matched)} tensors, missing {len(result.missing_keys)}, "
        f"ignored {len(state_dict) - len(matched)} non-encoder checkpoint tensors"
    )
    if result.missing_keys:
        logger.warning(f"[{name}] missing keys: {result.missing_keys}")
    return result
