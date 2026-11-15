#!/usr/bin/env python
"""Grid-evaluate fragment merging on a val cache with fixed calibrated post-processing.

Example: scripts/eval_fragmerge.py --cache artifacts/y11m_1536_flip_ref/val_raw.pkl \
             --calib artifacts/y11m_1536_flip_ref/calib.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from filseg.fragmerge import merge_stem  # noqa: E402
from filseg.gt import build_gt_df, view_stem  # noqa: E402
from filseg.infer import load_cache  # noqa: E402
from filseg.metrics import fast_evaluate  # noqa: E402
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows  # noqa: E402

GT = "data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def _merge(args):
    stem, rles, radius, ang, close = args
    return stem, merge_stem(rles, radius, ang, 3, close)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    stems = sorted(json.load(open(ROOT / "data/splits/split_v1.json"))["val"])
    gt = build_gt_df(str(ROOT / GT), set(stems))
    raw = load_cache(a.cache)
    q = json.load(open(a.calib))["best_params"]
    params = PPParams(q["conf"], q["min_area"], q["max_det"], q["nms"], q["erode"], q["erode_shape"], q["mask_thr"])
    rows = postprocess_to_rows(group_by_stem(raw), params, stems, group_soft_by_stem(raw))
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df["stem"] = df["filament_id"].str.rsplit("_", n=1).str[0]
    by = {s: g["segmentation_rle"].tolist() for s, g in df.groupby("stem")}
    even = set(stems[0::2])
    gt_stem = gt["filament_id"].str.split("_", n=1).str[0].map(view_stem)

    def score(d: pd.DataFrame, tag: str) -> None:
        r = fast_evaluate(gt, d)
        st = d["filament_id"].str.rsplit("_", n=1).str[0]
        re = fast_evaluate(gt[gt_stem.isin(even)], d[st.isin(even)])
        ro = fast_evaluate(gt[~gt_stem.isin(even)], d[~st.isin(even)])
        print(f"{tag:32s} PQ={r.pq:.4f} (even {re.pq:.4f} odd {ro.pq:.4f}) SQ={r.sq:.4f} RQ={r.rq:.4f} "
              f"TP/FP/FN={r.tp}/{r.fp}/{r.fn} n={len(d)}", flush=True)

    score(df[["filament_id", "segmentation_rle"]], "no merge")
    with ProcessPoolExecutor(a.workers) as ex:
        for radius, ang, close in itertools.product([3, 6, 10, 16], [180.0, 30.0], [False, True]):
            out = []
            for stem, rl in ex.map(_merge, [(s, r, radius, ang, close) for s, r in by.items()]):
                out += [(f"{stem}_{i}", x) for i, x in enumerate(rl)]
            score(pd.DataFrame(out, columns=["filament_id", "segmentation_rle"]), f"r={radius} ang<={ang:.0f} close={int(close)}")


if __name__ == "__main__":
    main()
