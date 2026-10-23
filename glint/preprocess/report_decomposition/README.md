# Report decomposition

- Splits each radiology report into per-finding sentences (`report_decomposition`) with `gpt-oss-120b` on vLLM
  (4 GPUs, tensor parallel).
- Has its own environment: run everything from this directory after `uv sync`.

```bash
python run.py --add_cfg_list decomposition mimic     # MIMIC-CXR
python run.py --add_cfg_list decomposition ct_rate   # CT-RATE
```

**Input:** `DATA_ROOT/<original_dir>/<split>.json`, one record per image with the report sections and the fields the
training loaders need (copied through).

| Config | Input | Report keys | Pass-through fields used by training |
|---|---|---|---|
| `mimic` | `mimic_cxr/reports/{train,validate,test}.json` (official MIMIC-CXR splits) | `findings`, `impression` | `image_path` (relative to `DATA_ROOT`) |
| `ct_rate` | `ct_rate/reports/{train,val,test}.json` (GLINT-CT `data/splits/ct_rate/`) | `Findings_EN`, `Impressions_EN` | `image_path`, `image_id`, `labels` |

**Output:** `DATA_ROOT/<save_dir>/<split>.json`

- The input records plus `report_decomposition` (list of sentences).
- That is, `mimic_cxr/{train,validate,test}.json` and `ct_rate/{train,val,test}.json`, read by
  `glint/configs/{cxr,ct}_data.yaml` and `ft/`.
- Identical reports are decomposed once.
- Intermediate and failed batches go to `<save_dir>/intermediate/<split>/`; the resolved config and prompt to
  `<save_dir>/snapshot/`.
