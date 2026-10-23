"""Pre-compute and cache DinoV3 ViT-7B teacher hidden states for CXR DFR training."""

import argparse
import os

import cv2
import numpy as np
import PIL
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from common import Config, logger


def parse_args():
    parser = argparse.ArgumentParser(description="Build CXR DFR teacher feature cache")
    parser.add_argument(
        "--cfg-path",
        default=os.path.join(os.path.dirname(__file__), "../../config.yaml"),
        help="path to configuration file.",
    )
    parser.add_argument(
        "--add_cfg_list",
        default=[],
        type=str,
        nargs="+",
        help="List of YAML config files to merge.",
    )
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Directory to save cached teacher features.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "eval"],
        help="Dataset splits to cache.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--cache_layers",
        type=int,
        nargs="+",
        default=[36],
        help="1-based teacher layer indices to cache. Default [36] matches the released CXR model.",
    )
    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return rank, world_size, device
    else:
        return 0, 1, torch.device("cuda")


def build_teacher_model(model_config):
    from glint.model.configuration import DFRConfig
    from glint.model.vision_encoders import build_vision_encoder

    dfr_dict = dict(model_config["dfr_config"])

    pretrained_dir = model_config.get("pretrained_dir")
    if pretrained_dir and "vfm_teacher_config" in dfr_dict:
        dfr_dict["vfm_teacher_config"].setdefault("pretrained_dir", pretrained_dir)

    dfr_config = DFRConfig.from_exp_config(dfr_dict)

    model = build_vision_encoder(dfr_config.vfm_teacher_config)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_image_processor(model_config):
    from glint.model.processing import load_image_processor

    return load_image_processor(model_config, model_config["vision_config"]["pretrained_name_or_path"])


def get_teacher_num_blocks(teacher):
    if hasattr(teacher, "config") and getattr(teacher.config, "num_hidden_layers", None):
        return teacher.config.num_hidden_layers
    if hasattr(teacher, "encoder") and hasattr(teacher.encoder, "blocks"):
        return len(teacher.encoder.blocks)
    raise ValueError("Cannot determine number of blocks from teacher model")


def get_image_id(data_instance):
    return os.path.splitext(os.path.basename(data_instance["image"]))[0]


class CXRCacheDataset(Dataset):
    def __init__(self, data_list, cache_dir, image_processor):
        self.data_list = data_list
        self.cache_dir = cache_dir
        self.image_processor = image_processor

        self.indices = []
        for idx in range(len(data_list)):
            image_id = get_image_id(data_list[idx])
            if not os.path.exists(os.path.join(cache_dir, f"{image_id}.safetensors")):
                self.indices.append(idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        data_instance = self.data_list[idx]

        image = PIL.Image.open(data_instance["image"])

        if "I" in image.mode:
            image = PIL.Image.fromarray(
                cv2.normalize(
                    np.array(image),
                    None,
                    0,
                    255,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U,
                )
            )

        image = image.convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0]

        return {"pixel_values": pixel_values, "image_id": get_image_id(data_instance)}


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()

    cfg = Config(args).config
    model_config = cfg["model"]["model_config"]
    dataset_cfg = cfg["dataset"]

    if rank == 0:
        logger.info("Building teacher model...")
    teacher = build_teacher_model(model_config)
    teacher = teacher.to(device)

    image_processor = build_image_processor(model_config)

    num_blocks = get_teacher_num_blocks(teacher)
    cache_layers = args.cache_layers
    invalid = [i for i in cache_layers if not (1 <= i <= num_blocks)]
    if invalid:
        raise ValueError(f"cache_layers {invalid} out of range (teacher has {num_blocks} blocks, 1-based)")
    if rank == 0:
        logger.info(f"Teacher has {num_blocks} blocks, caching layers: {cache_layers}")
        logger.info(f"World size: {world_size}, batch_size per GPU: {args.batch_size}")

    from glint.dataset.cxr.mimic_cxr import load_mimic_cxr_dataset as input_json_file_load

    if rank == 0:
        os.makedirs(args.cache_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    for split in args.splits:
        dataset_names = dataset_cfg.get(split, [])
        if not dataset_names:
            if rank == 0:
                logger.warning(f"No datasets for split '{split}', skipping.")
            continue

        for dataset_name in dataset_names:
            data_list = input_json_file_load(dataset_name, split, dataset_cfg)
            if rank == 0:
                logger.info(f"Caching {dataset_name}/{split}: {len(data_list)} samples → {args.cache_dir}")

            ds = CXRCacheDataset(data_list, args.cache_dir, image_processor)
            if rank == 0:
                logger.info(f"  {len(ds)} samples remaining (skipping already cached)")

            if len(ds) == 0:
                continue

            if world_size > 1:
                sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
            else:
                sampler = None

            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            pbar = tqdm(loader, desc=f"[rank {rank}] {dataset_name}/{split}", disable=(rank != 0))
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(device)
                image_ids = batch["image_id"]

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    teacher_outputs = teacher(pixel_values, output_hidden_states=True)

                extracted = {
                    layer_idx: teacher_outputs.hidden_states[layer_idx].to("cpu", dtype=torch.bfloat16)
                    for layer_idx in cache_layers
                }
                del teacher_outputs

                for b in range(len(image_ids)):
                    cache_path = os.path.join(args.cache_dir, f"{image_ids[b]}.safetensors")
                    if os.path.exists(cache_path):
                        continue
                    cached = {str(layer_idx): extracted[layer_idx][b].contiguous() for layer_idx in cache_layers}
                    save_file(cached, cache_path)
                del extracted

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        logger.info(f"Cache complete: {args.cache_dir}")


if __name__ == "__main__":
    main()
