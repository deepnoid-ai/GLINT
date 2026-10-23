import torch


def multi_positive_nce_loss(
    logits,
    group_map,
    temperature,
    additional_neg_mask=None,
):
    scaled_logits = torch.exp(logits / temperature)
    np_total = group_map.size(0)
    pos_logits = scaled_logits[torch.arange(np_total), group_map]

    row_loss = get_row_loss(
        scaled_logits[:np_total],
        pos_logits,
    )

    neg_mask = torch.ones_like(scaled_logits)

    if additional_neg_mask is not None:
        neg_mask[-len(additional_neg_mask) :, :] = additional_neg_mask

    neg_mask[torch.arange(np_total), group_map] = 0

    column_loss = get_col_loss(
        scaled_logits,
        pos_logits,
        neg_mask,
        group_map,
    )

    losses = {}
    losses["per_text_loss"] = row_loss.mean()
    losses["per_image_loss"] = column_loss.mean()

    losses["mpnce_loss"] = (losses["per_text_loss"] + losses["per_image_loss"]) / 2

    losses["loss"] = losses["mpnce_loss"]

    return losses


def multi_positive_nce_local_loss(
    logits_per_text,
    logits_per_image,
    group_map,
    local_group_map,
    local_group_map_per_image,
    temperature,
    local_rank,
    B_local,
    additional_neg_mask=None,
):
    logits_per_text = torch.exp(logits_per_text / temperature)
    pos_logits_per_text = logits_per_text[torch.arange(logits_per_text.size(0)), local_group_map]

    row_loss = get_row_loss(
        logits_per_text,
        pos_logits_per_text,
    )

    logits_per_image = torch.exp(logits_per_image / temperature)

    local_start_idx = torch.where(group_map == local_rank * B_local)[0][0]
    local_end_idx = local_start_idx + len(local_group_map_per_image)

    pos_text_indices = torch.LongTensor(range(local_start_idx, local_end_idx))

    pos_logits_per_image = logits_per_image[pos_text_indices, local_group_map_per_image]

    neg_mask = torch.ones_like(logits_per_image)

    if additional_neg_mask is not None:
        neg_mask[-len(additional_neg_mask) :, :] = additional_neg_mask

    neg_mask[pos_text_indices, local_group_map_per_image] = 0

    column_loss = get_col_loss(
        logits_per_image,
        pos_logits_per_image,
        neg_mask,
        local_group_map_per_image,
    )

    losses = {}
    losses["per_text_loss"] = row_loss.mean()
    losses["per_image_loss"] = column_loss.mean()

    losses["mpnce_loss"] = (losses["per_text_loss"] + losses["per_image_loss"]) / 2

    losses["loss"] = losses["mpnce_loss"]
    return losses


def get_row_loss(
    logits: torch.Tensor,
    pos_logits: torch.Tensor,
    eps: float = 1e-8,
):
    row_sum_logits = logits.sum(dim=1)
    p_row = pos_logits / (row_sum_logits + eps)

    return -torch.log(p_row + eps)


def get_col_loss(
    logits: torch.Tensor,
    pos_logits: torch.Tensor,
    neg_mask: torch.Tensor,
    group_map: torch.Tensor,
    eps: float = 1e-8,
):
    neg_logits = logits * neg_mask
    sum_neg_logits = neg_logits.sum(dim=0)
    sum_neg_logits = sum_neg_logits[group_map]
    p_column = pos_logits / (pos_logits + sum_neg_logits + eps)

    return -torch.log(p_column + eps)
