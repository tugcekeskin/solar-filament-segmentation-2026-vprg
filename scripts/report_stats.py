#!/usr/bin/env python
"""Report statistics: validation PQ of each pipeline stage with image-level bootstrap 95% CIs, paired
bootstrap of the gain of the final pipeline over each stage, and per-instrument PQ with CIs.

The bootstrap resamples validation images (all annotator views of an image together), recomputing pooled
PQ = sum IoU_TP / (TP + FP/2 + FN/2) from per-view numerators and denominators.
Usage: report_stats.py [--boot 2000 --out reports/report_stats.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_diag import GT_JSON, iou  # noqa: E402
from filseg.gt import build_gt_df  # noqa: E402
from filseg.infer import load_cache  # noqa: E402
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows  # noqa: E402
from filseg.rle import to_rle_obj  # noqa: E402

STAGES = [  # name, cache, calib (post-processing calibrated for that stage on the same fold)
    ("YOLO11m-seg 1536, single pass", "artifacts/y11m_1536/val_raw.pkl", "artifacts/y11m_1536/calib.json"),
    ("+ flip TTA (3 passes, fused)", "artifacts/ctx/val_flip3_orig.pkl", "artifacts/y11m_1536_flip/calib.json"),
    ("+ U-Net crop refiner", "artifacts/y11m_1536_flip_ref/val_raw.pkl", "artifacts/y11m_1536_flip_ref/calib.json"),
    ("+ SAM-LoRA / U-Net blend (final)", "artifacts/y11m_1536_flip_blend/val_raw.pkl",
     "artifacts/y11m_1536_flip_blend/calib.json"),
]


def per_view(cache: str, calib: str, stems: list, views: dict) -> dict:
    p = json.load(open(calib))["best_params"]
    raw = load_cache(cache)
    rows = postprocess_to_rows(group_by_stem(raw), PPParams(**p), stems, group_soft_by_stem(raw))
    pred = defaultdict(list)
    for k, r in rows:
        pred[k.rsplit("_", 1)[0]].append(to_rle_obj(r))
    out = {}  # stem -> [N, D, TP, FP, FN]
    for st, vs in views.items():
        P = pred.get(st, [])
        acc = np.zeros(5)
        for Gs in vs.values():
            M = iou(Gs, P) if P else np.zeros((len(Gs), 0))
            hit = M > 0.5
            tp, fn = int(hit.sum()), int((~hit.any(1)).sum())
            fp = int((~hit.any(0)).sum()) if P else 0
            acc += [M[hit].sum(), tp + 0.5 * fp + 0.5 * fn, tp, fp, fn]
        out[st] = acc
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="reports/report_stats.json")
    a = ap.parse_args()
    stems = sorted(json.load(open("data/splits/split_v1.json"))["val"])
    gt = build_gt_df(GT_JSON, set(stems))
    fid = gt["filament_id"].str.rsplit("_", n=1).str[0]
    views = defaultdict(lambda: defaultdict(list))
    for v, r in zip(fid, gt["segmentation_rle"]):
        views[v.split("-", 1)[1]][v].append(to_rle_obj(r))
    keys = sorted(views)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(keys), size=(a.boot, len(keys)))
    res = {"n_images": len(keys), "boot": a.boot, "stages": []}
    mats = []
    for name, cache, calib in STAGES:
        pv = per_view(cache, calib, stems, views)
        X = np.stack([pv[k] for k in keys])  # images x 5
        mats.append(X)
        pq = X[:, 0].sum() / X[:, 1].sum()
        bs = X[idx, 0].sum(1) / X[idx, 1].sum(1)
        tp, fp, fn = X[:, 2:].sum(0)
        res["stages"].append(dict(name=name, pq=pq, ci=[*np.percentile(bs, [2.5, 97.5])], sd=float(bs.std()),
                                  tp=int(tp), fp=int(fp), fn=int(fn)))
        print(f"{name:40s} PQ {pq:.4f} [{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}] "
              f"TP {tp:.0f} FP {fp:.0f} FN {fn:.0f}", flush=True)
    F = mats[-1]
    fb = F[idx, 0].sum(1) / F[idx, 1].sum(1)
    for s, X in zip(res["stages"][:-1], mats[:-1]):
        d = fb - X[idx, 0].sum(1) / X[idx, 1].sum(1)
        s["gain_of_final"] = dict(mean=float(d.mean()), ci=[*np.percentile(d, [2.5, 97.5])], p_le0=float((d <= 0).mean()))
        print(f"final - {s['name']}: {d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
    inst = defaultdict(list)
    for i, k in enumerate(keys):
        inst[k[-2:]].append(i)
    res["instruments"] = {}
    for ins, ii in sorted(inst.items()):
        X = F[ii]
        b = rng.integers(0, len(ii), size=(a.boot, len(ii)))
        bs = X[b, 0].sum(1) / X[b, 1].sum(1)
        res["instruments"][ins] = dict(images=len(ii), gt=int(X[:, 2].sum() + X[:, 4].sum()),
                                       pq=X[:, 0].sum() / X[:, 1].sum(), ci=[*np.percentile(bs, [2.5, 97.5])])
        print(f"{ins}: n={len(ii)} PQ {res['instruments'][ins]['pq']:.3f} "
              f"[{res['instruments'][ins]['ci'][0]:.3f}, {res['instruments'][ins]['ci'][1]:.3f}]")
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
