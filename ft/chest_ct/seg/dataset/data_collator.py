import torch


def collate_fn(batch):
    """Flatten the per-volume patch lists from RandCropByPosNegLabeld, then stack."""
    flat = []
    for item in batch:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    outputs = {}
    outputs["pixel_values"] = torch.stack([item["pixel_values"] for item in flat])
    outputs["masks"] = torch.stack([item["masks"] for item in flat])

    return outputs
