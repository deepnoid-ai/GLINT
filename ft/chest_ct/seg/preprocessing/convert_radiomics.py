# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydicom<3.0", "rt-utils", "simpleitk"]
# ///
"""Convert NSCLC-Radiomics (TCIA) DICOM CT + RTSTRUCT into per-patient NIfTI masks."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("nsclc_pp")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unnamed"


def patient_short(pid: str) -> str:
    m = re.match(r"LUNG1-(\d+)", pid)
    return f"lung_{m.group(1)}" if m else pid.lower().replace("-", "_")


def read_ct(series_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not files:
        raise FileNotFoundError(f"no DICOM files in {series_dir}")
    reader.SetFileNames(files)
    return reader.Execute()


def write_nifti(img: sitk.Image, out: Path) -> None:
    sitk.WriteImage(img, str(out), useCompression=True)


def mask_to_image(mask_yxz: np.ndarray, ct: sitk.Image) -> sitk.Image:
    # rt_utils returns (rows, cols, slices) = (Y, X, Z); sitk numpy is (Z, Y, X)
    arr = np.transpose(mask_yxz, (2, 0, 1)).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ct)
    return img


_GTV_PAT = re.compile(r"(?i)gtv")
_PRIMARY_PAT = re.compile(r"(?i)^gtv-?1$")
_LUNG_NAMES = {"lung-left", "lung-right", "lungs-total"}


def build_2class_label(pdir: Path) -> dict | None:
    """Write mask_2class.nii.gz (0=bg, 1=primary GTV-1, 2=other gtv*); None if the case has no GTV-1."""
    info_path = pdir / "info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    rois = info.get("rois", [])

    def _files_matching(pred):
        return [pdir / r["file"] for r in rois if pred(r["name"])]

    primary_files = _files_matching(lambda n: _PRIMARY_PAT.search(n))
    meta_files = _files_matching(lambda n: _GTV_PAT.search(n) and not _PRIMARY_PAT.search(n))
    if not primary_files:
        return None

    def _union(files: list[Path]) -> tuple[np.ndarray, sitk.Image] | None:
        merged: np.ndarray | None = None
        ref: sitk.Image | None = None
        for fp in files:
            img = sitk.ReadImage(str(fp))
            if ref is None:
                ref = img
            arr = sitk.GetArrayFromImage(img).astype(bool)
            merged = arr if merged is None else (merged | arr)
        return (merged, ref) if merged is not None else None

    primary = _union(primary_files)
    if primary is None:
        return None
    primary_arr, ref = primary
    meta_pair = _union(meta_files)

    label = np.zeros(primary_arr.shape, dtype=np.uint8)
    if meta_pair is not None:
        label[meta_pair[0]] = 2
    label[primary_arr] = 1  # primary wins on overlap

    out_img = sitk.GetImageFromArray(label)
    out_img.CopyInformation(ref)
    write_nifti(out_img, pdir / "mask_2class.nii.gz")

    voxels = {
        "bg": int((label == 0).sum()),
        "primary": int((label == 1).sum()),
        "meta": int((label == 2).sum()),
    }
    info["multiclass_2"] = {
        "file": "mask_2class.nii.gz",
        "classes": {"0": "bg", "1": "primary", "2": "meta"},
        "primary_from": [r["name"] for r in rois if _PRIMARY_PAT.search(r["name"])],
        "meta_from": [r["name"] for r in rois if _GTV_PAT.search(r["name"]) and not _PRIMARY_PAT.search(r["name"])],
        "voxels": voxels,
        "has_meta": meta_pair is not None,
    }
    info_path.write_text(json.dumps(info, indent=2))
    return voxels


def build_3class_label(pdir: Path) -> dict | None:
    """Write mask_3class.nii.gz (0=bg, 1=lung, 2=primary GTV-1, 3=other gtv*); None if the case has no GTV-1."""
    info_path = pdir / "info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    rois = info.get("rois", [])

    def _files_matching(pred):
        return [pdir / r["file"] for r in rois if pred(r["name"])]

    primary_files = _files_matching(lambda n: _PRIMARY_PAT.search(n))
    meta_files = _files_matching(lambda n: _GTV_PAT.search(n) and not _PRIMARY_PAT.search(n))
    lung_files = _files_matching(lambda n: n.lower().replace("_", "-") in _LUNG_NAMES)
    if not primary_files:
        return None

    def _union(files: list[Path]) -> tuple[np.ndarray, sitk.Image] | None:
        merged: np.ndarray | None = None
        ref: sitk.Image | None = None
        for fp in files:
            img = sitk.ReadImage(str(fp))
            if ref is None:
                ref = img
            arr = sitk.GetArrayFromImage(img).astype(bool)
            merged = arr if merged is None else (merged | arr)
        return (merged, ref) if merged is not None else None

    primary = _union(primary_files)
    if primary is None:
        return None
    primary_arr, ref = primary
    meta_pair = _union(meta_files)
    lung_pair = _union(lung_files)

    label = np.zeros(primary_arr.shape, dtype=np.uint8)
    # write order sets overlap priority: primary > meta > lung
    if lung_pair is not None:
        label[lung_pair[0]] = 1
    if meta_pair is not None:
        label[meta_pair[0]] = 3
    label[primary_arr] = 2

    out_img = sitk.GetImageFromArray(label)
    out_img.CopyInformation(ref)
    write_nifti(out_img, pdir / "mask_3class.nii.gz")

    voxels = {
        "bg": int((label == 0).sum()),
        "lung": int((label == 1).sum()),
        "primary": int((label == 2).sum()),
        "meta": int((label == 3).sum()),
    }
    info["multiclass_3"] = {
        "file": "mask_3class.nii.gz",
        "classes": {"0": "bg", "1": "lung", "2": "primary", "3": "meta"},
        "primary_from": [r["name"] for r in rois if _PRIMARY_PAT.search(r["name"])],
        "meta_from": [r["name"] for r in rois if _GTV_PAT.search(r["name"]) and not _PRIMARY_PAT.search(r["name"])],
        "lung_from": [r["name"] for r in rois if r["name"].lower().replace("_", "-") in _LUNG_NAMES],
        "voxels": voxels,
        "has_lung": lung_pair is not None,
        "has_meta": meta_pair is not None,
    }
    info_path.write_text(json.dumps(info, indent=2))
    return voxels


def merge_tumor_mask(pdir: Path, ct: sitk.Image | None = None) -> int:
    """OR-merge every gtv* ROI mask into mask_tumor.nii.gz and return its voxel count."""
    info_path = pdir / "info.json"
    if not info_path.exists():
        return 0
    info = json.loads(info_path.read_text())
    gtv_files = [pdir / r["file"] for r in info.get("rois", []) if _GTV_PAT.search(r["name"])]
    if not gtv_files:
        return 0
    merged: np.ndarray | None = None
    ref: sitk.Image | None = ct
    for fp in gtv_files:
        m_img = sitk.ReadImage(str(fp))
        if ref is None:
            ref = m_img
        arr = sitk.GetArrayFromImage(m_img).astype(bool)
        merged = arr if merged is None else (merged | arr)
    if merged is None or ref is None:
        return 0
    out_img = sitk.GetImageFromArray(merged.astype(np.uint8))
    out_img.CopyInformation(ref)
    write_nifti(out_img, pdir / "mask_tumor.nii.gz")
    voxels = int(merged.sum())
    info["tumor_mask"] = {
        "file": "mask_tumor.nii.gz",
        "merged_from": [r["name"] for r in info["rois"] if _GTV_PAT.search(r["name"])],
        "voxels": voxels,
    }
    info_path.write_text(json.dumps(info, indent=2))
    return voxels


def process_patient(
    pid: str, series_by_mod: dict, raw_root: Path, out_root: Path, log: logging.Logger
) -> tuple[str, str]:
    short = patient_short(pid)
    pdir = out_root / short
    if (pdir / ".done").exists():
        return pid, "skip"
    pdir.mkdir(parents=True, exist_ok=True)

    ct_meta = series_by_mod.get("CT")
    rt_meta = series_by_mod.get("RTSTRUCT")
    if ct_meta is None:
        return pid, "fail:no_ct"
    if rt_meta is None:
        return pid, "fail:no_rtstruct"

    ct_dir = raw_root / pid / ct_meta["StudyInstanceUID"] / ct_meta["SeriesInstanceUID"]
    rt_dir = raw_root / pid / rt_meta["StudyInstanceUID"] / rt_meta["SeriesInstanceUID"]
    rt_files = sorted(p for p in rt_dir.iterdir() if p.suffix == ".dcm")
    if not rt_files:
        return pid, "fail:no_rt_dcm"

    try:
        ct = read_ct(ct_dir)
        write_nifti(ct, pdir / "CT.nii.gz")

        rs = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(rt_files[0]))
        rois: list[dict] = []
        for name in rs.get_roi_names():
            try:
                mask = rs.get_roi_mask_by_name(name)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s ROI %s skipped: %s", pid, name, exc)
                continue
            voxels = int(mask.sum())
            if voxels == 0:
                log.warning("%s ROI %s empty", pid, name)
                continue
            fname = f"mask_{slug(name)}.nii.gz"
            write_nifti(mask_to_image(mask, ct), pdir / fname)
            rois.append({"name": name, "file": fname, "voxels": voxels})

        info = {
            "patient_id": pid,
            "ct": {
                "file": "CT.nii.gz",
                "series_uid": ct_meta["SeriesInstanceUID"],
                "image_count": ct_meta["ImageCount"],
                "size_xyz": list(ct.GetSize()),
                "spacing_xyz": [float(x) for x in ct.GetSpacing()],
                "origin_xyz": [float(x) for x in ct.GetOrigin()],
                "direction": [float(x) for x in ct.GetDirection()],
            },
            "rtstruct_series_uid": rt_meta["SeriesInstanceUID"],
            "rois": rois,
        }
        (pdir / "info.json").write_text(json.dumps(info, indent=2))
        merge_tumor_mask(pdir, ct=ct)
        (pdir / ".done").write_text(f"{time.time():.0f}\n")
        return pid, "ok"
    except Exception as exc:  # noqa: BLE001
        log.error("FAILED %s: %s", pid, exc, exc_info=True)
        return pid, f"fail:{type(exc).__name__}:{exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--raw_root", type=Path, default=None, help="raw TCIA download (required for RTSTRUCT extraction)")
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: process first N patients")
    ap.add_argument("--patient", type=str, default="", help="debug: process this PatientID only")
    ap.add_argument(
        "--merge_tumor_only",
        action="store_true",
        help="skip RTSTRUCT extraction; just (re)build mask_tumor.nii.gz from existing per-ROI masks",
    )
    ap.add_argument(
        "--build_3class_only",
        action="store_true",
        help="skip RTSTRUCT extraction; just (re)build mask_3class.nii.gz (bg/lung/primary/meta)",
    )
    ap.add_argument(
        "--build_2class_only",
        action="store_true",
        help="skip RTSTRUCT extraction; just (re)build mask_2class.nii.gz (bg/primary/meta, no lung)",
    )
    args = ap.parse_args()

    if args.build_2class_only:
        log = setup_logger(args.out_root / "_preprocess.log")
        pdirs = sorted(p for p in args.out_root.glob("lung_*") if p.is_dir())
        if args.limit:
            pdirs = pdirs[: args.limit]
        log.info("2class-only: %d patients", len(pdirs))
        included, dropped = [], []
        for pdir in pdirs:
            v = build_2class_label(pdir)
            if v is None:
                dropped.append(pdir.name)
            else:
                included.append({"id": pdir.name, "meta": v["meta"] > 0})
        log.info("2class-only done: %d included, %d dropped (no GTV-1)", len(included), len(dropped))
        with_meta = sum(1 for x in included if x["meta"])
        log.info("  has meta: %d/%d", with_meta, len(included))
        if dropped:
            log.warning("  dropped (no GTV-1): %s", dropped)
        return 0

    if args.build_3class_only:
        log = setup_logger(args.out_root / "_preprocess.log")
        pdirs = sorted(p for p in args.out_root.glob("lung_*") if p.is_dir())
        if args.limit:
            pdirs = pdirs[: args.limit]
        log.info("3class-only: %d patients", len(pdirs))
        included, dropped = [], []
        for pdir in pdirs:
            v = build_3class_label(pdir)
            if v is None:
                dropped.append(pdir.name)
            else:
                has_lung = v["lung"] > 0
                has_meta = v["meta"] > 0
                included.append({"id": pdir.name, "lung": has_lung, "meta": has_meta})
        log.info("3class-only done: %d included, %d dropped (no GTV-1)", len(included), len(dropped))
        with_lung = sum(1 for x in included if x["lung"])
        with_meta = sum(1 for x in included if x["meta"])
        log.info("  has lung: %d/%d  has meta: %d/%d", with_lung, len(included), with_meta, len(included))
        if dropped:
            log.warning("  dropped (no GTV-1): %s", dropped)
        return 0

    if args.merge_tumor_only:
        log = setup_logger(args.out_root / "_preprocess.log")
        pdirs = sorted(p for p in args.out_root.glob("lung_*") if p.is_dir())
        if args.limit:
            pdirs = pdirs[: args.limit]
        log.info("merge-only: %d patients", len(pdirs))
        n_with, n_without = 0, 0
        for pdir in pdirs:
            v = merge_tumor_mask(pdir)
            if v > 0:
                n_with += 1
            else:
                n_without += 1
                log.warning("%s: no GTV ROIs to merge", pdir.name)
        log.info("merge-only done: %d with tumor, %d without", n_with, n_without)
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)
    log = setup_logger(args.out_root / "_preprocess.log")

    if args.raw_root is None:
        ap.error("--raw_root is required for RTSTRUCT extraction")
    meta_path = args.raw_root / "metadata.json"
    series = json.loads(meta_path.read_text())
    by_pid: dict[str, dict] = {}
    for s in series:
        by_pid.setdefault(s["PatientID"], {})[s["Modality"]] = s
    pids = sorted(by_pid)
    if args.patient:
        pids = [p for p in pids if p == args.patient]
    if args.limit:
        pids = pids[: args.limit]
    log.info("plan: %d patients, workers=%d", len(pids), args.workers)

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "fail": 0}
    failed: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_patient, p, by_pid[p], args.raw_root, args.out_root, log) for p in pids]
        for i, f in enumerate(cf.as_completed(futs), 1):
            pid, status = f.result()
            key = "ok" if status == "ok" else "skip" if status == "skip" else "fail"
            counts[key] += 1
            if key == "fail":
                failed.append(f"{pid}\t{status}")
            if i % 5 == 0 or i == len(pids):
                el = time.time() - t0
                eta = (len(pids) - i) * el / max(i, 1)
                log.info(
                    "progress %d/%d ok=%d skip=%d fail=%d rate=%.2f/s eta=%.0fs",
                    i,
                    len(pids),
                    counts["ok"],
                    counts["skip"],
                    counts["fail"],
                    i / max(el, 1),
                    eta,
                )

    if failed:
        (args.out_root / "_failed.txt").write_text("\n".join(failed) + "\n")
        log.warning("wrote _failed.txt with %d entries", len(failed))
    log.info("done %s in %.0fs", counts, time.time() - t0)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
