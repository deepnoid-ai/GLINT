# Pretraining & zero-shot evaluation

- Configs are composed with `--add_cfg_list` (later files win).
- Data paths are relative to `DATA_ROOT`; outputs go to `$OUTPUT_ROOT/glint/<user>/<name>/`.
- W&B is off by default; enable it with `train.report_to: wandb`.
- `--ckpt` takes a HF repo id or a local checkpoint.
- Zero-shot metrics: Macro AUROC from σ(ℓ/τ_l) for classification, Dice at the best threshold on a 0.01 grid for
  segmentation. Each run also saves prediction CSVs and similarity-map panels.
- Pretraining manifests are JSON lists per split, `{"image_path", "report_decomposition"}`, built from the official
  reports with [`preprocess/report_decomposition/`](preprocess/report_decomposition/README.md). Records with an empty
  decomposition are skipped.
- Paper runs use 8 GPUs. Keep GPUs × `per_device_train_batch_size` when changing `--nproc_per_node`.

## CXR

### Zero-shot evaluation

```bash
# cls OpenI/PadChest/ChestXray14/CheXpert/ChestX-Det10 · grounding ChestX-Det10/MS-CXR · seg SIIM/ChestX-Det
torchrun --nproc_per_node=8 -m glint.run \
    --add_cfg_list cxr_data cxr_model amp_bf16 cxr_eval \
    --train False --inference True --ckpt Deepnoid/GLINT-CXR --user $USER --name glint_cxr
```

- **Evaluation sets:** protocol, test sets and class prompts follow [RadZero](https://github.com/deepnoid-ai/RadZero).
  Copy its evaluation sets next to each dataset's images:
  `hf download Deepnoid/RadZero --include "data/*" --local-dir . && cp -r data/* $DATA_ROOT/`
- **ChestX-Det (seg):** the official release (`ChestX_Det_{train,test}.json` + images) in `chestx_det/raw/`.

### Pretraining

```bash
torchrun --nproc_per_node=8 -m glint.run \
    --add_cfg_list cxr_data cxr_model amp_bf16 cxr_train \
    --train True --inference False --user $USER --name glint_cxr
```

- **Data:** `mimic_cxr/{train,validate,test}.json`, official splits minus the MS-CXR test images.
- **Base weights:** [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m),
  teacher `dinov3-vit7b16` (both gated).
- **Batch:** 16 per GPU (effective 128).
- **Alternative initializations (appendix):**
  - CARZero M3AE ViT-B/16: `vision_config.model_type: m3ae`, `pretrained_name_or_path: CARZero/VITB-16-M3AE_last.ckpt`
    (needs `timm`).
  - Franca ViT-B/14: `model_type: dinov2`, after
    `python -m glint.model.convert_franca_to_hf_dinov2 --ckpt url --out $PRETRAINED_DIR/franca-vitb14-in21k`.
- **Cached teacher (optional, ~12 TB):** build it once, then add `cxr_dfr_cached` to the training command.

```bash
torchrun --nproc_per_node=8 -m glint.model.dfr.build_teacher_cache_cxr \
    --add_cfg_list cxr_data cxr_model --cache_dir ${DFR_CACHE_DIR}/cxr_dinov3_7B
```

## CT

### Zero-shot evaluation

```bash
# cls CT-RATE/RAD-ChestCT · free-text seg ReXGroundingCT
torchrun --nproc_per_node=8 -m glint.run \
    --add_cfg_list ct_data ct_model amp_bf16 ct_eval \
    --train False --inference True --ckpt Deepnoid/GLINT-CT --user $USER --name glint_ct
```

- **CT-RATE:** `ct_rate/test.json` lists the volumes of
  [`data/splits/ct_rate/test.txt`](https://huggingface.co/Deepnoid/GLINT-CT/tree/main/data/splits/ct_rate) as
  `{"image_id", "labels"}` (18 labels from `multi_abnormality_labels/valid_predicted_labels.csv`); volumes are read from
  `CT-RATE/dataset/valid_fixed/`.
- **RAD-ChestCT:** the [Zenodo release](https://zenodo.org/records/6406114) (restricted access) in `rad_chestct/raw/`,
  then `python -m glint.preprocess.chest_ct.rad_chestct`.
- **ReXGroundingCT:** the base `dataset.json` + `segmentations/` in `rexgroundingct/`; the 50-case validation split is
  used.
- **Preprocessing:** on the fly, resized to 240 × 384 × 384; set `dataset.monai_config.cache_root` to cache the volumes
  (~140 MB each).

### Pretraining

```bash
torchrun --nproc_per_node=8 -m glint.run \
    --add_cfg_list ct_data ct_model amp_bf16 ct_train \
    --train True --inference False --user $USER --name glint_ct
```

- **Data:** `ct_rate/{train,val,test}.json` (records also carry `image_id`, `labels`), the
  [`data/splits/ct_rate/`](https://huggingface.co/Deepnoid/GLINT-CT/tree/main/data/splits/ct_rate) lists minus
  `pretrain_exclusions.txt`.
- **Base weights:** V-JEPA 2.1 `vjepa2_1_vitl_dist_vitG_384.pt`, teacher `vjepa2_1_vitG_384.pt`
  ([facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)), in `$PRETRAINED_DIR/vjepa2_1/`.
- **Batch:** 4 per GPU (effective 32).
- **Disk:** preprocessed volumes are cached in `DATA_ROOT/monai_cache/` (~6.5 TB); move it with
  `dataset.monai_config.cache_root`.
- **Cached teacher (optional, ~3.6 TB):** build it once, then add `ct_dfr_cached` to the training command.

```bash
torchrun --nproc_per_node=8 -m glint.model.dfr.build_teacher_cache_ct \
    --add_cfg_list ct_data ct_model --cache_dir ${DFR_CACHE_DIR}/ct_vjepa21G
```
