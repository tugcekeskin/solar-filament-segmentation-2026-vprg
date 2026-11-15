"""Per-candidate features (+ labels on train/val GT) for a second-stage candidate rescorer.

Candidates are the rows of a fused raw cache (score >= --min-score). Each row is keyed by
``stem|round(score, 6)``, the same key ``blend_caches.py`` uses, so the output can be joined to a
refined cache of the same raw source.

Features
  own:     score, area, soft-mask sharpness, shape (min-area-rect elongation, fill), radial position,
           darkness contrast vs a surrounding ring (filaments are dark), rank / crowding in the image,
           overlap with higher-scored candidates of the same cache;
  agree_*: for every --agree cache (other flips / other models): best IoU with any of its candidates
           and that candidate's score (0 when absent).
Label (only when the stem has GT): fraction of the stem's annotator views in which some GT instance
has IoU > 0.5 with the candidate (the organizer's TP rule), plus the mean best IoU over views.

Usage: rescore_features.py --cache RAW.pkl --out FEAT.pkl [--agree name=path ...] [--test]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.infer import SOFT_COLUMN, load_cache, unpack_soft  # noqa: E402
from filseg.preproc import disk_mask  # noqa: E402
from filseg.refine import IMG_DIR, IMG_DIR_TEST  # noqa: E402
from filseg.rle import H, W, to_rle_obj  # noqa: E402

GT_JSON = "data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def key(df: pd.DataFrame) -> pd.Series:
    return df["stem"] + "|" + df["score"].round(6).astype(str)


def iou_mat(a: list, b: list) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou(a, b, [0] * len(b)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--agree", nargs="*", default=[])
    p.add_argument("--min-score", type=float, default=0.01)
    p.add_argument("--test", action="store_true", help="no labels; images from the test dir")
    a = p.parse_args()

    raw = load_cache(a.cache)
    raw = raw[raw["score"] >= a.min_score].reset_index(drop=True)
    agree = {}
    for s in a.agree:
        name, path = s.split("=", 1)
        d = load_cache(path)
        agree[name] = {st: g for st, g in d[d["score"] >= a.min_score].groupby("stem")}

    gt_by_stem = {}
    if not a.test:
        from filseg.gt import build_gt_df

        gt = build_gt_df(GT_JSON, set(raw["stem"]))
        fid = gt["filament_id"].str.rsplit("_", n=1).str[0]
        gt["view"] = fid
        gt["stem"] = fid.str.split("-", n=1).str[1]
        gt_by_stem = {st: g for st, g in gt.groupby("stem")}

    rows = []
    for stem, g in raw.groupby("stem", sort=True):
        g = g.sort_values("score", ascending=False)
        path = Path(IMG_DIR_TEST if a.test else IMG_DIR) / f"{stem}.jpeg"
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        dm = disk_mask(img)
        ys, xs = np.nonzero(dm)
        cy, cx = ys.mean(), xs.mean()
        R = np.sqrt(dm.sum() / np.pi)
        disk_mean, disk_std = float(img[dm > 0].mean()), float(img[dm > 0].std()) + 1e-6
        P = [to_rle_obj(r) for r in g["rle"]]
        self_iou = iou_mat(P, P)
        ag = {n: iou_and_score(P, d.get(stem)) for n, d in agree.items()}
        lab = labels(P, gt_by_stem.get(stem)) if gt_by_stem else None
        n25 = int((g["score"] >= 0.25).sum())
        for i, (sc, area, soft) in enumerate(zip(g["score"], g["area"], g[SOFT_COLUMN])):
            y0, x0, crop = unpack_soft(soft)
            pr = crop.astype(np.float32) / 255.0
            m = pr >= 0.5
            f = {"stem": stem, "key": f"{stem}|{round(float(sc), 6)}", "score": float(sc), "area": float(area),
                 "rank": i, "n_cand25": n25, "instr": stem[-2:]}
            f["soft_mean"] = float(pr[m].mean()) if m.any() else 0.0
            f["soft_hi"] = float((pr >= 0.8).sum() / max(m.sum(), 1))
            f["soft_lo"] = float((pr >= 0.2).sum() / max(m.sum(), 1))
            mm = m.astype(np.uint8)
            cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            f["n_parts"] = len(cnts)
            if cnts:
                (_, _), (rw, rh), _ = cv2.minAreaRect(np.concatenate(cnts))
                lo, hi = sorted((max(rw, 1.0), max(rh, 1.0)))
                f["elong"], f["length"], f["fill"] = hi / lo, hi, float(m.sum()) / (lo * hi)
            else:
                f["elong"] = f["length"] = f["fill"] = 0.0
            my, mx = np.nonzero(m)
            if len(my):
                f["r_rel"] = float(np.hypot(my.mean() + y0 - cy, mx.mean() + x0 - cx) / R)
                h, w = m.shape
                patch = img[y0 : y0 + h, x0 : x0 + w].astype(np.float32)
                ring = cv2.dilate(mm, np.ones((15, 15), np.uint8)) & (1 - mm)
                inside = float(patch[m].mean())
                outside = float(patch[ring > 0].mean()) if ring.any() else inside
                f["contrast"] = (outside - inside) / disk_std
                f["inside_rel"] = (inside - disk_mean) / disk_std
                f["inside_std"] = float(patch[m].std()) / disk_std
            else:
                f["r_rel"] = f["contrast"] = f["inside_rel"] = f["inside_std"] = 0.0
            f["max_iou_higher"] = float(self_iou[i, :i].max()) if i else 0.0
            f["n_overlap_higher"] = int((self_iou[i, :i] > 0.1).sum()) if i else 0
            for n, (bi, bs) in ag.items():
                f[f"agree_{n}_iou"], f[f"agree_{n}_score"] = float(bi[i]), float(bs[i])
            if lab is not None:
                f["y_frac"], f["y_iou"], f["n_views"] = float(lab[0][i]), float(lab[1][i]), int(lab[2])
            rows.append(f)
    df = pd.DataFrame(rows)
    df.to_pickle(a.out)
    print(f"{len(df)} candidates over {df['stem'].nunique()} stems, {df.shape[1]} columns -> {a.out}")


def iou_and_score(P: list, d: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    if d is None or not len(d):
        return np.zeros(len(P)), np.zeros(len(P))
    io = iou_mat(P, [to_rle_obj(r) for r in d["rle"]])
    j = io.argmax(1)
    return io.max(1), d["score"].values[j] * (io.max(1) > 0)


def labels(P: list, g: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray, int]:
    if g is None:
        return np.zeros(len(P)), np.zeros(len(P)), 0
    hit, bi = [], []
    for _, v in g.groupby("view"):
        io = iou_mat(P, [to_rle_obj(r) for r in v["segmentation_rle"]])
        hit.append(io.max(1) > 0.5)
        bi.append(io.max(1))
    return np.mean(hit, 0), np.mean(bi, 0), len(hit)


if __name__ == "__main__":
    main()
