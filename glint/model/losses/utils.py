import torch
import torch.distributed as dist


def pad_and_gather(tensor):
    local_size = torch.tensor(tensor.size(), device=tensor.device)

    all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)

    max_size = torch.stack(all_sizes).max(dim=0)[0]

    padded_tensor = torch.zeros(max_size.tolist(), device=tensor.device)
    padded_tensor[: local_size[0]] = tensor

    gathered_tensors = dist.nn.all_gather(padded_tensor)

    gathered_tensors = [g[: s[0]] for g, s in zip(gathered_tensors, all_sizes)]

    gathered_tensors = torch.cat(gathered_tensors, dim=0)

    return gathered_tensors


def pad_and_gather_2nd_dim(tensor, mask):
    local_size = torch.tensor(tensor.size(), device=tensor.device)

    if local_size[1] == 0:
        local_size = torch.tensor([local_size[0], 1, local_size[2]], device=tensor.device)
        tensor = torch.zeros(
            (local_size[0], 1, local_size[2]),
            device=tensor.device,
            requires_grad=True,
        )
        mask = torch.zeros((local_size[0], 1), device=mask.device, dtype=torch.bool)

    all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)

    max_len = max([i[1].item() for i in all_sizes])

    padded_tensor = torch.zeros((local_size[0], max_len, local_size[2]), device=tensor.device)
    padded_tensor[:, : local_size[1]] = tensor

    padded_mask = torch.zeros((local_size[0], max_len), device=mask.device)
    padded_mask[:, : local_size[1]] = mask

    gathered_tensors = dist.nn.all_gather(padded_tensor)
    gathered_masks = dist.nn.all_gather(padded_mask)

    gathered_tensors = torch.cat(gathered_tensors, dim=0)
    gathered_masks = torch.cat(gathered_masks, dim=0)

    return gathered_tensors, gathered_masks
