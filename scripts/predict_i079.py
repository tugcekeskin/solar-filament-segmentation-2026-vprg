#!/usr/bin/env python
"""Run the external i079 detector + crop refiner and write a raw prediction cache.

--mode full:        detector (score >= --det-conf, default 0.3) + refiner/TTA/merge pipeline;
                    additionally writes <out>.native070.pkl (their default 0.70 pipeline).
--mode refine-only: refine the instances of --proposals (score >= --min-score) with the refiner.
Weights were trained on all train images (val stems included): val results are IN-SAMPLE.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from filseg import ext_i079 as E  # noqa: E402
from filseg.infer import RAW_COLUMNS, SOFT_COLUMN, save_cache  # noqa: E402
from filseg.rle import decode_rle, encode_mask  # noqa: E402

IMG_DIR = ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images"
TEST_DIR = ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images"


def stems_for(subset: str) -> tuple[list[str], Path]:
    if subset == "val":
        return sorted(json.load(open(ROOT / "data/splits/split_v1.json"))["val"]), IMG_DIR
    d = next((p for p in [TEST_DIR] if p.is_dir()), None)
    if d is None:
        raise SystemExit("test image dir not found")
    return sorted(p.stem for p in d.glob("*.jpeg")), d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", choices=["val", "test"], required=True)
    ap.add_argument("--out-cache", required=True)
    ap.add_argument("--mode", choices=["full", "refine-only"], default="full")
    ap.add_argument("--proposals")
    ap.add_argument("--det-conf", type=float, default=0.3)
    ap.add_argument("--min-score", type=float, default=0.1)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "refine-only" and not a.proposals:
        ap.error("--proposals required for refine-only")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stems, d = stems_for(a.subset)
    if a.limit:
        stems = stems[: a.limit]
    refiner = E.load_refiner(ROOT / E.REFINER_PATH, dev)
    detector = E.load_detector(ROOT / E.DETECTOR_PATH, dev) if a.mode == "full" else None
    prop = None
    if a.mode == "refine-only":
        raw = pd.read_pickle(a.proposals)
        prop = {k: g for k, g in raw[raw.score >= a.min_score].groupby("stem", sort=False)}
    rows, rows070 = [], []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        img = cv2.imread(str(d / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (2048, 2048):
            raise SystemExit(f"bad image {stem}")
        if a.mode == "full":
            masks, scores = E.detect(detector, img, dev, a.det_conf)
            rows += E.rows_from_instances(stem, E.infer_full(refiner, img, masks, scores, dev, a.det_conf, not a.no_tta))
            rows070 += E.rows_from_instances(stem, E.infer_full(refiner, img, masks, scores, dev, E.DEFAULT_SCORE_THRESHOLD, not a.no_tta))
        else:
            g = prop.get(stem)
            if g is None:
                continue
            ms = [decode_rle(r).astype(bool) for r in g.rle]
            ok = [j for j, m in enumerate(ms) if m.any()]
            res = E.refine_proposals(refiner, img, [ms[j] for j in ok], [g[SOFT_COLUMN].iloc[j] for j in ok], dev, not a.no_tta)
            for j, (m, sf) in zip(ok, res):
                rows.append((stem, float(g.score.iloc[j]), encode_mask(m.astype(np.uint8)), int(m.sum()), sf))
        if i % 10 == 0 or i == len(stems):
            print(f"{i}/{len(stems)} {(time.time() - t0) / i:.2f}s/img rows={len(rows)} "
                  f"gpu_peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GB", flush=True)
    cols = RAW_COLUMNS + [SOFT_COLUMN]
    Path(a.out_cache).parent.mkdir(parents=True, exist_ok=True)
    save_cache(pd.DataFrame(rows, columns=cols), a.out_cache)
    if a.mode == "full":
        save_cache(pd.DataFrame(rows070, columns=cols), str(a.out_cache) + ".native070.pkl")
    print(f"wrote {a.out_cache}: {len(rows)} instances, {(time.time() - t0) / len(stems):.2f}s/img")


if __name__ == "__main__":
    main()
