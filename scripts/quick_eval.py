#!/usr/bin/env python
"""Fast val PQ for ablations: small fixed post-processing grid on a raw cache (no flip TTA, no full calibration).

Usage: quick_eval.py --cache artifacts/abl/<run>/val_raw.pkl --name <run> --out runs/ablation/results.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from filseg.gt import build_gt_df, view_stem  # noqa: E402
from filseg.infer import load_cache  # noqa: E402
from filseg.metrics import fast_evaluate  # noqa: E402
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows  # noqa: E402

GT = "data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
GRID = list(itertools.product([0.2, 0.25, 0.3, 0.35], [None, 0.6], [(1, "square"), (1, "cross"), (2, "cross")]))
_G = {}


def _one(cfg):
    conf, thr, (k, shape) = cfg
    p = PPParams(conf, 0, 20, 0.3, k, shape, thr)
    out = {}
    for name, stems in (("all", _G["stems"]), ("even", _G["stems"][0::2]), ("odd", _G["stems"][1::2])):
        gt = _G["gt"][_G["gt_stem"].isin(stems)]
        r = fast_evaluate(gt, pd.DataFrame(postprocess_to_rows(_G["g"], p, stems, _G["s"]),
                                           columns=["filament_id", "segmentation_rle"]))
        out[name] = r
    a = out["all"]
    return dict(conf=conf, mask_thr=thr, erode=k, shape=shape, pq=a.pq, sq=a.sq, rq=a.rq, tp=a.tp, fp=a.fp, fn=a.fn,
                pq_even=out["even"].pq, pq_odd=out["odd"].pq)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", default="runs/ablation/results.jsonl")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    stems = sorted(json.load(open("data/splits/split_v1.json"))["val"])
    gt = build_gt_df(GT, set(stems))
    raw = load_cache(a.cache)
    _G.update(stems=stems, gt=gt, gt_stem=gt["filament_id"].str.split("_", n=1).str[0].map(view_stem),
              g=group_by_stem(raw), s=group_soft_by_stem(raw) if "soft" in raw.columns else None)
    grid = [c for c in GRID if c[1] is None or _G["s"] is not None]
    with Pool(a.workers) as pool:
        res = pool.map(_one, grid)
    df = pd.DataFrame(res).sort_values("pq", ascending=False)
    best = df.iloc[0].to_dict()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps({"run": a.name, **best}) + "\n")
    print(f"RESULT {a.name} PQ={best['pq']:.4f} (even {best['pq_even']:.4f} odd {best['pq_odd']:.4f}) "
          f"SQ={best['sq']:.4f} RQ={best['rq']:.4f} TP/FP/FN={best['tp']}/{best['fp']}/{best['fn']} "
          f"conf={best['conf']} thr={best['mask_thr']} erode={best['erode']} {best['shape']}")


if __name__ == "__main__":
    main()
