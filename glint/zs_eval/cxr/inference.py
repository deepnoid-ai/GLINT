import os
import traceback

import pandas as pd
import torch

from common import load_json, logger, save_json

from ..distribution_utils import is_main_process
from .classification_utils import eval_classification
from .grounding_utils import chestXDet10_eval_grounding, eval_refer_grounding
from .segmentation_utils import eval_segmentation_chestx_det, eval_segmentation_rsna_medklip, eval_segmentation_siim
from .visualization_utils import visualize_chestx_det, visualize_ms_cxr


def resolve_dataset_paths(dataset_paths_config, data_root_dir):
    resolved = {}
    for dataset_name, paths in dataset_paths_config.items():
        entry = {key: os.path.join(data_root_dir, val) for key, val in paths.items() if val is not None}
        # manifest image paths are relative to DATA_ROOT (RadZero layout) unless image_root is given
        entry.setdefault("image_root", data_root_dir)
        resolved[dataset_name] = entry
    return resolved


class Inference:
    def __init__(
        self,
        cls_dataset,
        det_dataset,
        seg_dataset,
        dataset_paths,
        batch_size,
        num_workers,
    ):
        self.dataset_paths = dataset_paths
        self.cls_dataset = cls_dataset
        self.det_dataset = det_dataset
        self.seg_dataset = seg_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    @torch.no_grad()
    def classification(self, model, image_processor, tokenizer, save_root_dir):
        try:
            os.makedirs(save_root_dir, exist_ok=True)

            image_paths, text_paths = [], []
            for cls_d in self.cls_dataset:
                cls_d_dirs = self.dataset_paths[cls_d]
                image_paths.append(cls_d_dirs["image_path"])
                text_paths.append(cls_d_dirs["text_path"])

            performances = eval_classification(
                model,
                image_processor,
                tokenizer,
                self.cls_dataset,
                self.batch_size,
                self.num_workers,
                save_root_dir,
                image_paths,
                text_paths,
                self.dataset_paths,
            )

            if is_main_process() and performances:
                logger.info(f"[zs_eval] classification metrics: {performances}")
                save_json(performances, os.path.join(save_root_dir, "result.json"))

        except Exception as e:
            print(f"Error in zero shot classification: {e}\n{traceback.format_exc()}")

    @torch.no_grad()
    def grounding(self, model, image_processor, tokenizer, save_root_dir):
        try:
            os.makedirs(save_root_dir, exist_ok=True)

            result = {}
            for det_d in self.det_dataset:
                det_d_dirs = self.dataset_paths[det_d]

                if det_d == "ChestXDet10":
                    chestxdet10_result = chestXDet10_eval_grounding(
                        model,
                        image_processor,
                        tokenizer,
                        self.batch_size,
                        self.num_workers,
                        det_d_dirs["image_path"],
                        det_d_dirs["text_path"],
                        det_d_dirs["label_path"],
                        det_d_dirs["image_root"],
                    )

                    if is_main_process() and chestxdet10_result:
                        result["ChestXDet10"] = chestxdet10_result

                elif det_d == "MS-CXR":
                    data = load_json(det_d_dirs["data_path"])

                    ms_cxr_result = eval_refer_grounding(
                        model,
                        image_processor,
                        tokenizer,
                        data,
                        det_d_dirs["image_root"],
                    )

                    if is_main_process() and ms_cxr_result:
                        result["MS-CXR"] = ms_cxr_result

            if is_main_process() and result:
                logger.info(f"[zs_eval] grounding metrics: {result}")
                save_json(result, os.path.join(save_root_dir, "result.json"))

        except Exception as e:
            print(f"Error in zero shot grounding: {e}\n{traceback.format_exc()}")

    @torch.no_grad()
    def segmentation(
        self,
        model,
        image_processor,
        tokenizer,
        save_root_dir,
        compute_pixel_level_auroc,
        threshold_step=0.01,
    ):
        try:
            os.makedirs(save_root_dir, exist_ok=True)

            result = {}
            for seg_d in self.seg_dataset:
                seg_d_dirs = self.dataset_paths[seg_d]

                if seg_d == "SIIM":
                    data = pd.read_csv(seg_d_dirs["data_path"])
                    text = "There is pneumothorax."

                    siim_result = eval_segmentation_siim(
                        model,
                        image_processor,
                        tokenizer,
                        data,
                        text,
                        seg_d_dirs["image_root"],
                        compute_pixel_level_auroc,
                        threshold_step=threshold_step,
                    )
                    if is_main_process() and siim_result:
                        result["SIIM"] = siim_result

                elif seg_d == "RSNA":
                    medklip_data = pd.read_csv(seg_d_dirs["data_path"])
                    text = "There is pneumonia."

                    rsna_result = eval_segmentation_rsna_medklip(
                        model,
                        image_processor,
                        tokenizer,
                        medklip_data,
                        text,
                        seg_d_dirs["image_root"],
                        compute_pixel_level_auroc,
                        threshold_step=threshold_step,
                    )

                    if is_main_process() and rsna_result:
                        result["RSNA"] = rsna_result

                elif seg_d == "ChestXDet":
                    chestx_det_result = eval_segmentation_chestx_det(
                        model,
                        image_processor,
                        tokenizer,
                        seg_d_dirs["data_path"],
                        seg_d_dirs.get("image_dir"),
                        compute_pixel_level_auroc,
                        threshold_step=threshold_step,
                    )
                    if is_main_process() and chestx_det_result:
                        result["ChestXDet"] = chestx_det_result

            if is_main_process() and result:
                logger.info(f"[zs_eval] segmentation metrics: {result}")
                save_json(result, os.path.join(save_root_dir, "result.json"))

        except Exception as e:
            print(f"Error in zero shot segmentation: {e}\n{traceback.format_exc()}")

    @torch.no_grad()
    def visualization(
        self,
        model,
        image_processor,
        tokenizer,
        save_root_dir,
        vis_dataset=None,
        max_samples_per_finding=None,
    ):
        try:
            if vis_dataset is None:
                vis_dataset = list(set((self.seg_dataset or []) + (self.det_dataset or [])))

            os.makedirs(save_root_dir, exist_ok=True)

            for ds_name in vis_dataset:
                ds_dirs = self.dataset_paths.get(ds_name)
                if ds_dirs is None:
                    if is_main_process():
                        print(f"Visualization: dataset paths not found for {ds_name}, skipping")
                    continue

                if ds_name == "ChestXDet":
                    visualize_chestx_det(
                        model=model,
                        image_processor=image_processor,
                        tokenizer=tokenizer,
                        data_path=ds_dirs["data_path"],
                        image_dir=ds_dirs.get("image_dir"),
                        save_dir=os.path.join(save_root_dir, "ChestXDet"),
                        batch_size=self.batch_size,
                        max_samples_per_finding=max_samples_per_finding,
                    )

                elif ds_name == "MS-CXR":
                    data = load_json(ds_dirs["data_path"])
                    visualize_ms_cxr(
                        model=model,
                        image_processor=image_processor,
                        tokenizer=tokenizer,
                        data=data,
                        image_root_dir=ds_dirs["image_root"],
                        save_dir=os.path.join(save_root_dir, "MS-CXR"),
                        batch_size=self.batch_size,
                        max_samples=max_samples_per_finding,
                    )

                else:
                    if is_main_process():
                        print(f"Visualization: not implemented for {ds_name}, skipping")

        except Exception as e:
            print(f"Error in visualization: {e}\n{traceback.format_exc()}")
