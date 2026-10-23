import numpy as np
import torch


def collate_fn(batch, tokenizer):
    outputs = {}

    targets = np.zeros((len(batch), tokenizer.max_token), dtype=int)
    targets_masks = np.zeros((len(batch), tokenizer.max_token), dtype=int)

    pixel_values = []

    for i, item in enumerate(batch):
        pixel_values.append(item["pixel_values"])

        report_id = tokenizer(item["report"])
        targets[i, : len(report_id)] = report_id

        report_mask = [1] * len(report_id)
        targets_masks[i, : len(report_mask)] = report_mask

    outputs["pixel_values"] = torch.stack(pixel_values)
    outputs["reports"] = torch.LongTensor(targets)
    outputs["reports_masks"] = torch.FloatTensor(targets_masks)

    return outputs
