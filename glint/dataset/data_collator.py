import torch


def collate_fn(batch, tokenizer):
    outputs = {"pixel_values": torch.stack([item["pixel_values"] for item in batch])}

    if "cached_teacher_features" in batch[0]:
        outputs["cached_teacher_features"] = {
            layer_idx: torch.stack([item["cached_teacher_features"][layer_idx] for item in batch])
            for layer_idx in batch[0]["cached_teacher_features"]
        }

    if "sentences" in batch[0]:
        outputs["enc_sents"] = [
            tokenizer(item["sentences"], padding=True, truncation=True, return_tensors="pt") for item in batch
        ]
    return outputs
