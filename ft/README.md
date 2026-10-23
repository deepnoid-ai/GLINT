# Downstream fine-tuning

- The GLINT encoder is frozen and loaded with
  `vision_config: {model_type: glint, pretrained_name_or_path: Deepnoid/GLINT-CXR}` (or a local checkpoint).
- Baselines run through the same task code and protocol; only the encoder config changes.
- Install the `ft` extra (`uv sync --extra ft`) for baselines, report-generation metrics and DICOM input of CT seg.
- Hyperparameters follow Table 7 of the paper. Keep the effective batch (GPUs × per-device batch × accumulation) when
  changing the GPU count.
- Manifests are JSON lists per split with paths relative to `DATA_ROOT` (readers in each task's `dataset/`). Split ID
  lists ship with [GLINT-CXR](https://huggingface.co/Deepnoid/GLINT-CXR/tree/main/data) and
  [GLINT-CT](https://huggingface.co/Deepnoid/GLINT-CT/tree/main/data).
- Baseline weights go under `$PRETRAINED_DIR` at the paths below; Hugging Face repos are fetched automatically.
  Other checkpoints are pickles loaded with `weights_only=False`, so take them only from the official sources.

## CXR

```bash
# GLINT: classification ({nih,vindr,rsna,siim}_data), segmentation ({chestx_det,siim}_data), report generation
python -m ft.cxr.cls.run --add_cfg_list cxr_model glint_cxr nih_data --user $USER --name glint_nih
python -m ft.cxr.seg.run --add_cfg_list cxr_ft_seg glint_cxr chestx_det_data --user $USER --name glint_cxdet
torchrun --nproc_per_node=2 -m ft.cxr.rrg.run --add_cfg_list model glint_cxr mimic_cxr --user $USER --name glint_mimic

# Baselines: B = dinov3_l, mgca, kad, medklip, mrm, mavl, chexworld or raddino
B=mgca
python -m ft.cxr.cls.run --add_cfg_list cxr_model $B nih_data --user $USER --name ${B}_nih
python -m ft.cxr.seg.run --add_cfg_list cxr_ft_seg $B chestx_det_data --user $USER --name ${B}_cxdet
torchrun --nproc_per_node=4 -m ft.cxr.rrg.run --add_cfg_list model $B mimic_cxr --user $USER --name ${B}_mimic  # dinov3_l: 2 GPUs
```

- **Report-generation batch (effective 64):** `glint_cxr` 2 GPUs × 32; `dinov3_l` 2 GPUs × 16 × 2 accumulation; other
  baselines 4 GPUs × 16.
- **Report-generation metrics:** METEOR needs Java; F1CheXbert / F1RadGraph fetch their weights on first use. Missing
  metric packages are skipped with a warning.

| Task | Manifest fields |
|---|---|
| cls (NIH / VinDr / RSNA-PN) | `image_id`, `labels` |
| seg (ChestX-Det) | `image_id`, `labels`, `masks` |
| cls / seg (SIIM) | split lists of `SOPInstanceUID`s; images / masks as `siim_acr/png_{images,masks}/<SOPInstanceUID>.png` ([BenchX](https://github.com/yangzhou12/BenchX) preprocessing); the classification label is a non-empty mask |
| rrg (MIMIC-CXR) | `dicom_id`, `study_id`, `subject_id`, `findings` (official splits; training uses 10% of train, seed 42) |

| Baseline | `model_type` | Config | Official source | Path under `$PRETRAINED_DIR` |
|---|---|---|---|---|
| DINOv3 ViT-L/16 | `dinov3_vit` | `dinov3_l` | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (gated) | — |
| MGCA | `mgca` | `mgca` | [HKU-MedAI/MGCA](https://github.com/HKU-MedAI/MGCA) | `MGCA/vit_base.ckpt` |
| KAD-512 | `kad` | `kad` | [xiaoman-zhang/KAD](https://github.com/xiaoman-zhang/KAD) (KAD_512) | `KAD/best_valid.pt` |
| MedKLIP | `medklip` | `medklip` | [MediaBrain-SJTU/MedKLIP](https://github.com/MediaBrain-SJTU/MedKLIP) | `MedKLIP/checkpoint_final.pth` |
| MRM | `mrm` | `mrm` | [RL4M/MRM-pytorch](https://github.com/RL4M/MRM-pytorch) | `MRM/MRM.pth` |
| MAVL | `mavl` | `mavl` | [hieuvmphan/CVPR2024_MAVL](https://github.com/hieuvmphan/CVPR2024_MAVL) | `MAVL/checkpoint_full_40.pth` |
| CheXWorld | `chexworld` | `chexworld` | [LeapLabTHU/CheXWorld](https://github.com/LeapLabTHU/CheXWorld) | `CheXWorld/chexworld_pretrained.tar` |
| RAD-DINO | `dinov2` | `raddino` | [microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino) | — |

MGCA, MRM and CheXWorld need `timm` (`ft` extra).

## CT

```bash
# GLINT: classification (CT-RATE / LIDC-IDRI / RSNA-PE), report generation (CT-RATE),
# segmentation (glint_ct_seg_freeze_{lung_msd,nsclc})
torchrun --nproc_per_node=8 -m ft.chest_ct.cls.run \
    --add_cfg_list model chest_ct_monai glint_ct glint_ct_ft_cls_all --user $USER --name glint_ct_cls
torchrun --nproc_per_node=4 -m ft.chest_ct.rrg.run \
    --add_cfg_list ct_rate_ft_data chest_ct_monai model glint_ct --user $USER --name glint_ct_rrg
torchrun --nproc_per_node=8 -m ft.chest_ct.seg.run \
    --add_cfg_list encoder_glint_ct data_config_glint_ct glint_ct_seg_freeze_lung_msd --user $USER --name glint_msd

# Baselines, V-JEPA 2.1: same inputs as GLINT
torchrun --nproc_per_node=8 -m ft.chest_ct.cls.run \
    --add_cfg_list model chest_ct_monai vjepa21 glint_ct_ft_cls_all --user $USER --name vjepa21_ct_cls
torchrun --nproc_per_node=4 -m ft.chest_ct.rrg.run \
    --add_cfg_list ct_rate_ft_data chest_ct_monai model vjepa21 --user $USER --name vjepa21_ct_rrg
torchrun --nproc_per_node=8 -m ft.chest_ct.seg.run \
    --add_cfg_list encoder_vjepa21 data_config_glint_ct glint_ct_seg_freeze_lung_msd --user $USER --name vjepa21_msd

# Baselines with their own inputs: B = ct_clip, fvlm, visd_boost or colipri (seg: ct_clip or colipri)
B=ct_clip
torchrun --nproc_per_node=8 -m ft.chest_ct.cls.run \
    --add_cfg_list model $B glint_ct_ft_cls_all --user $USER --name ${B}_ct_cls
torchrun --nproc_per_node=4 -m ft.chest_ct.rrg.run \
    --add_cfg_list ct_rate_ft_data model $B --user $USER --name ${B}_ct_rrg
torchrun --nproc_per_node=8 -m ft.chest_ct.seg.run \
    --add_cfg_list encoder_$B data_config_$B glint_ct_seg_freeze_lung_msd --user $USER --name ${B}_msd
```

- **Batch:** classification 8 GPUs × 16 = 128 (CT-RATE) / 8 × 4 = 32 (LIDC-IDRI, RSNA-PE); report generation
  4 × 16 = 64, for every encoder.
- **Not in Table 7:** 200 warm-up steps for classification; for segmentation, weight decay 1e-5, 5% warm-up and 2 random
  96³ crops per volume (8 volumes per step).
- **Segmentation baselines:** V-JEPA 2.1, CT-CLIP and COLIPRI, as in the paper.
- **RadBERT F1 / CRG:** need `models/RadBertClassifier.pth` from
  [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/tree/main/models) in `$PRETRAINED_DIR/RadBert/`.

| Task | Manifest fields |
|---|---|
| cls / rrg (CT-RATE) | `image_id`, `labels`, `Findings_EN`, `Impressions_EN`, `report_decomposition` |
| cls (LIDC-IDRI) | `metadata.{PatientID,StudyInstanceUID,SeriesInstanceUID}`, `labels.nodules` |
| cls (RSNA-PE) | `image_id`, `labels.pulmonary_embolism` |
| seg (MSD / NSCLC) | datalist of `{image, label}`; builders in `ft/chest_ct/seg/preprocessing/` (NSCLC DICOM: `uv run ft/chest_ct/seg/preprocessing/convert_radiomics.py`, own env with `pydicom<3.0`) |

| Baseline | `model_type` | Config | Official source | Path under `$PRETRAINED_DIR` |
|---|---|---|---|---|
| V-JEPA 2.1 ViT-L | `vjepa2.1` | `vjepa21`, seg `encoder_vjepa21` | [vjepa2_1_vitl_dist_vitG_384.pt](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt) ([repo](https://github.com/facebookresearch/vjepa2)) | `vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt` |
| CT-CLIP | `ct_clip` | `ct_clip`, seg `encoder_ct_clip` | [CT-CLIP_v2.pt](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/models/CT-CLIP-Related/CT-CLIP_v2.pt) ([repo](https://github.com/ibrahimethemhamamci/CT-CLIP)) | `CT-CLIP/CT-CLIP_v2.pt` |
| fVLM | `fvlm` | `fvlm` | [alibaba-damo-academy/fvlm](https://github.com/alibaba-damo-academy/fvlm) | `fVLM/model.pth` |
| ViSD-Boost | `visd_boost` | `visd_boost` | `checkpoint.pth`, `checkpoint_vqvae.pth` in the [repo](https://github.com/alibaba-damo-academy/ViSD-Boost) root | `ViSD-Boost/checkpoint.pth`, `ViSD-Boost/checkpoint_vqvae.pth` |
| COLIPRI | `colipri` | `colipri`, seg `encoder_colipri` | [microsoft/colipri](https://huggingface.co/microsoft/colipri) `model.safetensors` | `colipri/model.safetensors` |

Baseline inputs (`ft` extra: `timm` and `dynamic-network-architectures` for COLIPRI, `vector-quantize-pytorch==1.1.2`
for CT-CLIP):

- **COLIPRI:** read through MONAI with its pretraining geometry (RAS, 2 mm, 160³; `colipri.yaml`).
- **CT-CLIP, fVLM, ViSD-Boost (cls / rrg):** volumes prepared offline with each method's pipeline, set in
  `dataset.processed_volumes` (layout in `ft/chest_ct/processed_volumes.py`):
  - CT-CLIP: `nii_img_to_tensor` of [scripts/data_inference_nii.py](https://github.com/ibrahimethemhamamci/CT-CLIP/blob/main/scripts/data_inference_nii.py)
    (0.75 × 0.75 × 1.5 mm, HU clipped to [-1000, 1000] / 1000, center crop / pad to 480 × 480 × 240), saved as
    (240, 480, 480) `.npy` under `ct_clip_volumes/{ct_rate,lidc_idri,rsna_pe}/`.
  - fVLM, ViSD-Boost: fVLM [data/](https://github.com/alibaba-damo-academy/fvlm/tree/main/data) steps
    (`fix_data` → `generate_mask` (TotalSegmentator) → `resize` (1 × 1 × 3 mm) → `preprocess` (HU [-1150, 350] → [0, 1],
    ROI crop, pad to 112 × 256 × 352)), saved as `.nii.gz` under `fvlm_volumes/{ct_rate,lidc_idri,rsna_pe}/` and
    center-cropped to (112, 256, 352) when loaded.
