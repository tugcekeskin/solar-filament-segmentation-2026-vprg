#!/usr/bin/env python
"""Grid-search post-processing params on the val cache against val GT (organizer PQ)."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from filseg.gt import build_gt_df, view_stem  # noqa: E402
from filseg.infer import load_cache  # noqa: E402
from filseg.metrics import fast_evaluate  # noqa: E402
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows, transform_grouped  # noqa: E402

DEFAULT_GT = "data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CONFS = [round(0.1 + 0.05 * i, 2) for i in range(11)]  # 0.1 .. 0.6
MIN_AREAS = [0, 100]
MAX_DETS = [20, 100]
NMS = [None, 0.3, 0.5]
MASK_THRS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
DEFAULT = PPParams(conf=0.25, min_area=0, max_det=100, nms=None)
KEYS = ["conf", "min_area", "max_det", "nms", "erode", "erode_shape", "mask_thr"]

_G = {}


def variants(has_soft: bool) -> list[tuple[str, int, str, float | None]]:
    """(category, erode, erode_shape, mask_thr) preprocessing variants."""
    v = [("baseline", 0, "square", None)]
    for k, shape in [(1, "square"), (1, "cross"), (2, "square"), (2, "cross"), (2, "ellipse")]:
        v.append(("erode", k, shape, None))
    if has_soft:
        for t in MASK_THRS:
            v.append(("soft", 0, "square", t))
        for t in MASK_THRS:
            for shape in ("square", "cross"):
                v.append(("combined", 1, shape, t))
    return v


def _eval_variant(var) -> list[dict]:
    cat, k, shape, thr = var
    grouped = transform_grouped(_G["grouped"], _G["soft"], k, shape, thr, min_score=min(CONFS))
    out = []
    for c, m, d, n in itertools.product(CONFS, MIN_AREAS, MAX_DETS, NMS):
        params = PPParams(c, m, d, n)  # transform already applied: erode/mask_thr recorded below
        rows = postprocess_to_rows(grouped, params, _G["stems"])
        pred = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        r = fast_evaluate(_G["gt"], pred)
        out.append({**params.to_dict(), "erode": k, "erode_shape": shape, "mask_thr": thr, "category": cat,
                    "pq": r.pq, "sq": r.sq, "rq": r.rq, "tp": r.tp, "fp": r.fp, "fn": r.fn,
                    "splits": r.splits, "merges": r.merges, "dup_hits": r.dup_hits, "n_pred": len(pred)})
    return out


def _half_pq(row, has_soft_unused=None) -> dict:
    """PQ of one config on even/odd (sorted order) val stems, for robustness checks."""
    thr = None if pd.isna(row["mask_thr"]) else float(row["mask_thr"])
    nms = None if pd.isna(row["nms"]) else float(row["nms"])
    params = PPParams(float(row["conf"]), int(row["min_area"]), int(row["max_det"]), nms)
    grouped = transform_grouped(_G["grouped"], _G["soft"], int(row["erode"]), row["erode_shape"], thr, params.conf)
    out = {}
    for name, sub in (("even", _G["stems"][0::2]), ("odd", _G["stems"][1::2])):
        pred = pd.DataFrame(postprocess_to_rows(grouped, params, sub), columns=["filament_id", "segmentation_rle"])
        gt = _G["gt"]
        gt = gt[gt["filament_id"].str.split("_", n=1).str[0].map(view_stem).isin(sub)]
        r = fast_evaluate(gt, pred)
        out[name] = {k: float(getattr(r, k)) for k in ("pq", "sq", "rq", "tp", "fp", "fn")}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gt-json", default=DEFAULT_GT)
    p.add_argument("--split", default="data/splits/split_v1.json")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    stems = sorted(json.load(open(a.split))["val"])
    _G["stems"] = stems
    _G["gt"] = build_gt_df(a.gt_json, set(stems))
    raw = load_cache(a.cache)
    has_soft = "soft" in raw.columns
    _G["grouped"] = group_by_stem(raw)
    _G["soft"] = group_soft_by_stem(raw) if has_soft else None
    vs = variants(has_soft)
    t = time.time()
    with Pool(a.workers) as pool:  # fork: workers inherit _G
        res = [r for chunk in pool.imap_unordered(_eval_variant, vs, chunksize=1) for r in chunk]
    df = pd.DataFrame(res).sort_values("pq", ascending=False, kind="stable").reset_index(drop=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "calib_grid.csv", index=False)

    def clean(row):
        return {"conf": float(row["conf"]), "min_area": int(row["min_area"]), "max_det": int(row["max_det"]),
                "nms": None if pd.isna(row["nms"]) else float(row["nms"]), "erode": int(row["erode"]),
                "erode_shape": str(row["erode_shape"]), "mask_thr": None if pd.isna(row["mask_thr"]) else float(row["mask_thr"])}

    def metrics(row):
        return {k: float(row[k]) for k in ("pq", "sq", "rq", "tp", "fp", "fn", "splits", "merges", "dup_hits")}

    best = df.iloc[0]
    per_cat = {c: df[df.category == c].iloc[0] for c in ("baseline", "erode", "soft", "combined") if (df.category == c).any()}
    d0 = df[(df.conf == 0.3) & (df.min_area == 0) & (df.max_det == 20) & (df.nms == 0.3) & (df.category == "baseline")].iloc[0]
    report = {"best_params": clean(best), "best_metrics": metrics(best), "best_category": best["category"],
              "default_params": clean(d0), "default_metrics": metrics(d0),
              "per_category": {c: {"params": clean(r), "metrics": metrics(r), "halves": _half_pq(r)} for c, r in per_cat.items()},
              "n_configs": len(df), "grid_csv": "calib_grid.csv"}
    (out / "calib.json").write_text(json.dumps(report, indent=1))
    print(f"{len(df)} configs in {time.time() - t:.0f}s")
    pd.set_option("display.width", 220)
    print(df.head(10).drop(columns=["splits", "merges", "dup_hits"]).to_string(float_format=lambda x: f"{x:.4f}"))
    for c, r in report["per_category"].items():
        m = r["metrics"]
        print(f"{c:9s} {r['params']}: PQ={m['pq']:.4f} SQ={m['sq']:.4f} RQ={m['rq']:.4f} TP={int(m['tp'])} FP={int(m['fp'])} "
              f"FN={int(m['fn'])} | even PQ={r['halves']['even']['pq']:.4f} odd PQ={r['halves']['odd']['pq']:.4f}")
    print(f"BEST {report['best_params']}: PQ={best.pq:.4f}")


if __name__ == "__main__":
    main()
