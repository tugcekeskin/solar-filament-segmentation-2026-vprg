#!/usr/bin/env python
"""End-to-end inference: images dir -> submission.csv (the public-0.40 pipeline).

Stages: (1) YOLO11m-seg @1536 soft-mask prediction x3 (id / h-flip / v-flip) -> (2) instance fusion
-> (3) U-Net + SAM-LoRA crop refiners -> (4) 50/50 soft-mask blend -> (5) calibrated post-processing,
overlap resolution, CSV writing and validation. Intermediate caches are kept in --workdir.

Example:
    .venv/bin/python scripts/run_pipeline.py --images data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images --out submission.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # every file the pipeline loads is in the repository

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from filseg.ensemble import ensemble_caches  # noqa: E402
from filseg.infer import SOFT_COLUMN, load_cache, predict_raw, save_cache, unpack_soft  # noqa: E402
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows  # noqa: E402
from filseg.refine import load_model, refine_stem  # noqa: E402
from filseg.rle import H, W, encode_mask  # noqa: E402
from filseg.submission import validate_submission, write_submission  # noqa: E402

DEFAULTS = {
    "yolo": "weights/yolo11m_seg_1536.pt",
    "unet": "weights/unet_refiner.pt",
    "sam": "weights/sam_lora_refiner.pt",
    "calib": "configs/calib.json",
}


def log(msg: str) -> None:
    print(f"[pipeline {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def refine_cache(raw: pd.DataFrame, weights: str, images: dict[str, str], batch: int, device: str,
                 min_score: float = 0.1) -> pd.DataFrame:
    """Same logic as scripts/refine_cache.py, but reading images from an arbitrary directory."""
    raw = raw[raw["score"] >= min_score]
    model = load_model(weights, device)
    rows = []
    for stem, g in raw.groupby("stem", sort=True):
        img = cv2.imread(images[stem], cv2.IMREAD_GRAYSCALE)
        new = refine_stem(model, img, g[SOFT_COLUMN].tolist(), batch, device)
        for sc, nt in zip(g["score"], new):
            if nt is None:
                continue
            y0, x0, crop = unpack_soft(nt)
            full = np.zeros((H, W), np.uint8)
            full[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop >= 128
            area = int(full.sum())
            if area == 0:
                continue
            rows.append((stem, float(sc), encode_mask(full), area, nt))
    del model
    import torch

    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["stem", "score", "rle", "area", SOFT_COLUMN])


def blend(a_path: Path, b_path: Path, out: Path) -> None:
    """50/50 soft-mask blend via scripts/blend_caches.py (unchanged)."""
    import blend_caches

    old = sys.argv
    sys.argv = ["blend_caches.py", str(a_path), str(b_path), str(out)]
    try:
        blend_caches.main()
    finally:
        sys.argv = old


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="directory of *.jpeg test images")
    p.add_argument("--out", required=True, help="output submission.csv")
    p.add_argument("--workdir", help="where intermediate caches go (default: <out dir>/pipeline_work)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--yolo", default=DEFAULTS["yolo"])
    p.add_argument("--unet", default=DEFAULTS["unet"])
    p.add_argument("--sam", default=DEFAULTS["sam"])
    p.add_argument("--calib", default=DEFAULTS["calib"], help="calib.json holding best_params")
    p.add_argument("--imgsz", type=int, default=1536)
    p.add_argument("--batch", type=int, default=2, help="YOLO batch size")
    p.add_argument("--workers", type=int, default=8, help="processes for instance fusion")
    p.add_argument("--force", action="store_true", help="recompute stages even if their cache exists")
    a = p.parse_args()

    out = Path(a.out)
    wd = Path(a.workdir) if a.workdir else out.parent / "pipeline_work"
    wd.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(str(q) for q in Path(a.images).glob("*.jpeg"))
    if not paths:
        sys.exit(f"no *.jpeg images in {a.images}")
    images = {Path(q).stem: q for q in paths}
    log(f"{len(paths)} images from {a.images}; workdir {wd}")
    t0 = time.time()

    def stage(name: str):
        f = wd / name
        return f, (f.exists() and not a.force)

    # 1. soft-mask prediction, no flip / h / v
    flips = {"": "raw.pkl", "h": "raw_h.pkl", "v": "raw_v.pkl"}
    for flip, name in flips.items():
        f, done = stage(name)
        if done:
            log(f"skip {name}")
            continue
        df = predict_raw(a.yolo, paths, a.imgsz, 0.01, 0.7, 100, a.batch, soft=True, flip=flip,
                         device=a.device if a.device != "cuda" else None)
        save_cache(df, str(f))
        log(f"predict flip={flip or 'none'}: {len(df)} instances")

    # 2. instance fusion
    f_fused, done = stage("flip3.pkl")
    if not done:
        fused = ensemble_caches([load_cache(str(wd / n)) for n in flips.values()], 0.5, 1, a.workers)
        save_cache(fused, str(f_fused))
        log(f"fused: {len(fused)} instances")

    # 3. refiners
    fused = None
    for key, weights, batch in (("unet", a.unet, 64), ("sam", a.sam, 2)):
        f, done = stage(f"{key}.pkl")
        if done:
            continue
        fused = load_cache(str(f_fused)) if fused is None else fused
        df = refine_cache(fused, weights, images, batch, a.device)
        save_cache(df, str(f))
        log(f"refined ({key}): {len(df)} instances")

    # 4. blend (SAM first, U-Net second, w=0.5)
    f_blend, done = stage("blend.pkl")
    if not done:
        blend(wd / "sam.pkl", wd / "unet.pkl", f_blend)

    # 5. post-process -> submission
    bp = json.load(open(a.calib))["best_params"]
    params = PPParams(bp["conf"], bp["min_area"], bp["max_det"], bp["nms"], bp.get("erode", 0),
                      bp.get("erode_shape", "square"), bp.get("mask_thr"))
    log(f"params {params}")
    stems = sorted(images)
    raw = load_cache(str(f_blend))
    soft = group_soft_by_stem(raw) if "soft" in raw.columns else None
    sub = write_submission(postprocess_to_rows(group_by_stem(raw), params, stems, soft), str(out))
    rep = validate_submission(str(out), set(stems))
    log(f"validator ok={rep.ok} rows={rep.n_rows} stems={rep.n_stems}/{len(stems)} "
        f"overlap_pairs={rep.n_overlap_pairs} errors={rep.errors[:5]} warnings={rep.warnings}")
    log(f"wrote {out} ({len(sub)} rows) in {time.time() - t0:.0f}s")
    if not rep.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
