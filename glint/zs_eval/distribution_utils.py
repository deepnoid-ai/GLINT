from functools import partial

import torch
from torch.utils.data import DataLoader


def is_distributed():
    return hasattr(torch.distributed, "is_initialized") and torch.distributed.is_initialized()


def get_world_size():
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


def get_rank():
    if is_distributed():
        return torch.distributed.get_rank()
    return 0


def is_main_process():
    return get_rank() == 0


def compute_shard_indices(seq):
    if not is_distributed():
        return 0, len(seq)

    rank = get_rank()
    world = get_world_size()
    total = len(seq)

    base = total // world
    rem = total % world
    start_idx = rank * base + min(rank, rem)
    end_idx = start_idx + base + (1 if rank < rem else 0)
    return start_idx, end_idx


def create_multi_gpu_dataloader(dataset, batch_size, num_workers, collate_fn, image_processor):
    if is_distributed():
        start_idx, end_idx = compute_shard_indices(dataset)

        from torch.utils.data import Subset

        subset = Subset(dataset, range(start_idx, end_idx))

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, image_processor=image_processor),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, image_processor=image_processor),
        )
    return loader


def gather_results(results):
    if not is_distributed():
        return results

    world = get_world_size()

    if is_main_process():
        gathered_results = [None] * world
        torch.distributed.gather_object(results, object_gather_list=gathered_results, dst=0)

        if isinstance(results, torch.Tensor):
            return torch.cat(gathered_results, dim=0)
        elif isinstance(results, list):
            combined = []
            for result_list in gathered_results:
                combined.extend(result_list)
            return combined
        else:
            return gathered_results
    else:
        torch.distributed.gather_object(results, dst=0)
        return None
