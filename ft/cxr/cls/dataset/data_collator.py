import torch


def collate_fn(batch):
    outputs = {}

    outputs["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
    outputs["labels"] = torch.stack([item["labels"] for item in batch])
    return outputs
