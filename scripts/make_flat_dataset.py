#!/usr/bin/env python
"""Build contrast-normalised images (filseg.preproc.flat3) and the YOLO dataset data/yolo/v3_flat.

Processed images: data/proc/flat3/{train_images,test_images}/<stem>.jpeg (JPEG q95, 4:4:4, no chroma
subsampling because the three channels carry different signals). The YOLO dataset mirrors data/yolo/v1:
same per-view file names (symlinks to the processed images) and the same label directories.
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from filseg.preproc import flat3  # noqa: E402

RAW = ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026"
OUT = ROOT / "data/proc/flat3"
PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]


def _one(args):
    src, dst = args
    if not dst.exists():
        cv2.imwrite(str(dst), flat3(cv2.imread(str(src))), PARAMS)


def main() -> None:
    jobs = []
    for sub, d in (("train", "train_images"), ("test", "test_images")):
        (OUT / d).mkdir(parents=True, exist_ok=True)
        jobs += [(p, OUT / d / p.name) for p in sorted((RAW / sub / d).glob("*.jpeg"))]
    with Pool(12) as pool:
        pool.map(_one, jobs, chunksize=4)
    v1, v3 = ROOT / "data/yolo/v1", ROOT / "data/yolo/v3_flat"
    for split in ("train", "val"):
        (v3 / "images" / split).mkdir(parents=True, exist_ok=True)
        (v3 / "labels").mkdir(parents=True, exist_ok=True)
        lab = v3 / "labels" / split
        if not lab.exists():
            lab.symlink_to(v1 / "labels" / split, target_is_directory=True)
        for f in (v1 / "images" / split).iterdir():
            stem = Path(os.readlink(f)).name
            dst = v3 / "images" / split / f.name
            if not dst.is_symlink():
                dst.symlink_to(OUT / "train_images" / stem)
    (v3 / "data.yaml").write_text(f"path: {v3}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    print("done", len(jobs), "images")


if __name__ == "__main__":
    main()
