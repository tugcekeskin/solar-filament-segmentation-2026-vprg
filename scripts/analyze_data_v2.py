#!/usr/bin/env python
"""Data + error analysis v2 (CPU only). Writes docs/data_analysis_v2.md and docs/figs_v2/*.png.

Answers six questions: disk geometry, orientation, photometry, instance statistics /
annotator disagreement, errors of the current best val predictions, copy-paste suitability.

The report = hand-written `docs/data_analysis_v2_recs.md` (findings + ranked recommendations,
if present) followed by the auto-generated tables of this script.
Read-only on data/ and artifacts/. Val GT is only used for *reporting* (never to fit anything).

  .venv/bin/python scripts/analyze_data_v2.py [--workers 16] [--no-cache]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
cv2.setNumThreads(1)

from filseg.gt import build_gt_df, view_stem  # noqa: E402
from filseg.metrics import fast_evaluate  # noqa: E402
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows  # noqa: E402
from filseg.rle import decode_rle, to_rle_obj  # noqa: E402

RAW = ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026"
TRAIN_IMG = RAW / "train/train_images"
TEST_IMG = RAW / "test/test_images"
COCO = RAW / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
SPLIT = ROOT / "data/splits/split_v1.json"
PRED_DIR = ROOT / "artifacts/y11m_1536_flip_ref"
PP = dict(conf=0.25, min_area=100, max_det=20, nms=0.3, erode=0, mask_thr=0.35)
OUT_MD = ROOT / "docs/data_analysis_v2.md"
RECS_MD = ROOT / "docs/data_analysis_v2_recs.md"
FIGS = ROOT / "docs/figs_v2"
SITES = ["Bh", "Ch", "Lh", "Mh", "Th", "Uh"]
PAD = 16
K_OUT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
K_IN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ----------------------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def qs(x, qlist=(1, 5, 25, 50, 75, 95, 99)) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return {f"p{q}": float(np.percentile(x, q)) for q in qlist}


def ks_2samp(a, b) -> float:
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    allv = np.concatenate([a, b])
    cdfa = np.searchsorted(a, allv, side="right") / len(a)
    cdfb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(cdfa - cdfb)))


def cohen_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def circ_axial(theta_deg):
    """Axial (period 180 deg) mean angle in (-90, 90] and resultant length."""
    t = np.deg2rad(2 * np.asarray(theta_deg, float))
    z = np.exp(1j * t).mean()
    return float(np.rad2deg(np.angle(z)) / 2), float(abs(z))


def axial_wrap(a):
    return (np.asarray(a, float) + 90) % 180 - 90


def largest_cc(mask: np.ndarray) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    return lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))


def _circle_fit(x, y):
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    return sol[0], sol[1], math.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 1e-6))


def fit_disk(g2048: np.ndarray) -> tuple[float, float, float, float]:
    """Solar limb (cx, cy, R, ellipse axis ratio) in 2048 px. Some frames have a bright sky halo and a second,
    larger frame circle, so a plain threshold is not reliable: instead take the strongest outward intensity
    drop (limb rim -> sky) along 360 rays of a 512 copy, within R in [780, 1000] px, then robust circle fit,
    then refine in a +-60 px window around the fitted radius."""
    g = cv2.GaussianBlur(cv2.resize(g2048, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 1.5)
    cx, cy, R = 256.0, 256.0, 112.0  # R at 512 scale; coarse prior window [195, 250]
    lo, hi = 195, 250
    for it in range(3):
        pol = cv2.warpPolar(g, (300, 360), (float(cx), float(cy)), 300, cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
        d = np.diff(pol, axis=1)  # outward derivative, d[:, r] = f(r+1)-f(r)
        win = d[:, lo:hi]
        rad = lo + np.argmin(win, axis=1) + 0.5
        th = np.deg2rad(np.arange(360))
        x, y = cx + rad * np.cos(th), cy + rad * np.sin(th)
        keep = np.ones(360, bool)
        for _ in range(3):
            fx, fy, fR = _circle_fit(x[keep], y[keep])
            res = np.hypot(x - fx, y - fy) - fR
            mad = np.median(np.abs(res - np.median(res))) + 1e-3
            keep = np.abs(res - np.median(res)) < 3 * 1.4826 * mad + 0.5
        cx, cy, R = fx, fy, fR
        lo, hi = max(int(R - 15), 150), min(int(R + 15), 290)
    (_, _), (ma, mi), _ = cv2.fitEllipse(np.c_[x[keep], y[keep]].astype(np.float32))
    s = 2048 / 512
    return cx * s, cy * s, R * s, max(ma, mi) / min(ma, mi)


def poly_mask(poly: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(poly - [x0, y0]).astype(np.int32)], 1)
    return m


def mask_feats(m: np.ndarray, y0: int, x0: int, gc: np.ndarray) -> dict:
    """Shape + contrast features of one binary crop mask (offset y0,x0) with grey crop gc."""
    ys, xs = np.nonzero(m)
    area = int(len(ys))
    if area < 3:
        return dict(area=area)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in cnts)
    hull = cv2.contourArea(cv2.convexHull(np.vstack(cnts)))
    pts = np.c_[xs, ys].astype(float)
    cov = np.cov(pts.T) if area > 3 else np.eye(2)
    ev, evec = np.linalg.eigh(cov)
    ang_poly = math.degrees(math.atan2(-evec[1, 1], evec[0, 1]))  # CCW from +x, y up
    out = dict(
        area=area, perim=perim, width=2 * area / max(perim, 1e-6), solidity=area / max(hull, 1.0),
        elong=math.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)), ang_poly=float(axial_wrap(ang_poly)),
        cx=float(xs.mean() + x0), cy=float(ys.mean() + y0),
        bbox_x0=int(xs.min() + x0), bbox_y0=int(ys.min() + y0), bbox_x1=int(xs.max() + x0), bbox_y1=int(ys.max() + y0),
    )
    out["bbox_fill"] = area / ((out["bbox_x1"] - out["bbox_x0"] + 1) * (out["bbox_y1"] - out["bbox_y0"] + 1))
    ring = (cv2.dilate(m, K_OUT) > 0) & ~(cv2.dilate(m, K_IN) > 0)
    inner = float(gc[m > 0].mean())
    out["inner"] = inner
    if ring.sum() >= 10:
        rv = gc[ring].astype(float)
        out["ring_mean"], out["ring_std"] = float(rv.mean()), float(rv.std())
        out["contrast"] = out["ring_mean"] - inner
        out["dprime"] = out["contrast"] / (out["ring_std"] + 1.0)
        out["rel_contrast"] = out["contrast"] / max(out["ring_mean"], 1.0)
    return out


def crop_from_rle(rle: dict, g: np.ndarray):
    full = decode_rle(rle) if isinstance(rle, str) else mask_util.decode(rle)
    ys, xs = np.nonzero(full)
    if len(ys) == 0:
        return None
    y0, y1 = max(ys.min() - PAD, 0), min(ys.max() + PAD + 1, full.shape[0])
    x0, x1 = max(xs.min() - PAD, 0), min(xs.max() + PAD + 1, full.shape[1])
    return full[y0:y1, x0:x1].astype(np.uint8), y0, x0, g[y0:y1, x0:x1]


# ----------------------------------------------------------------------------- per-stem worker
def stem_worker(job: tuple) -> dict:
    stem, split, path, views = job  # views: list of (view_id, annotator, [ann dicts])
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cx, cy, R, ell = fit_disk(g)
    g1 = cv2.resize(g, (1024, 1024), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:1024, 0:1024]
    rr = np.hypot(xx - cx / 2, yy - cy / 2) / (R / 2)
    gf = g1.astype(np.float32)
    on = rr < 0.97
    center = gf[rr < 0.3].mean()
    row = dict(stem=stem, split=split, cx=cx, cy=cy, R=R, ell=ell, disk_mean=float(gf[on].mean()),
               disk_std=float(gf[on].std()), disk_med=float(np.median(gf[on])),
               p1=float(np.percentile(gf[on], 1)), p99=float(np.percentile(gf[on], 99)),
               center_mean=float(center), bg_mean=float(gf[(rr > 1.04) & (rr < 1.15)].mean()),
               corner_mean=float(np.mean([g[:100, :100].mean(), g[:100, -100:].mean(), g[-100:, :100].mean(), g[-100:, -100:].mean()])),
               sat_frac=float((gf[on] >= 250).mean()))
    for i, (a, b) in enumerate([(0, .25), (.25, .5), (.5, .75), (.75, .9), (.9, .97)]):
        row[f"ld{i}"] = float(gf[(rr >= a) & (rr < b)].mean() / center)
    inner = rr < 0.9
    hp = gf - cv2.GaussianBlur(gf, (0, 0), 6)
    row["hp_std"] = float(hp[inner].std())
    lap = cv2.Laplacian(gf, cv2.CV_32F)
    row["sharp"] = float(lap[inner].var() / max(gf[inner].mean(), 1) ** 2)
    anns, mirror = [], []
    disk512 = np.hypot(np.mgrid[0:512, 0:512][1] - cx / 4, np.mgrid[0:512, 0:512][0] - cy / 4) / (R / 4)
    for view_id, annot, alist in views:
        boxes = []
        m512 = np.zeros((512, 512), np.uint8)
        for k, a in enumerate(alist, start=1):
            poly = np.array(a["segmentation"][0], float).reshape(-1, 2)
            x0, y0 = max(int(poly[:, 0].min()) - PAD, 0), max(int(poly[:, 1].min()) - PAD, 0)
            x1, y1 = min(int(poly[:, 0].max()) + PAD + 2, 2048), min(int(poly[:, 1].max()) + PAD + 2, 2048)
            m = poly_mask(poly, y0, x0, y1 - y0, x1 - x0)
            f = mask_feats(m, y0, x0, g[y0:y1, x0:x1])
            f.update(view_id=view_id, stem=stem, annotator=annot, k=k, cat=a["category_id"])
            sp = np.array(a["spine"], float).reshape(-1, 2) if a.get("spine") else None
            if sp is not None and len(sp) >= 2:
                seg = np.hypot(*np.diff(sp, axis=0).T)
                f["spine_len"] = float(seg.sum())
                f["chord"] = float(np.hypot(*(sp[-1] - sp[0])))
                v = sp - sp.mean(0)
                _, evec = np.linalg.eigh(np.cov(v.T) + 1e-9 * np.eye(2))
                f["ang"] = float(axial_wrap(math.degrees(math.atan2(-evec[1, 1], evec[0, 1]))))
            boxes.append((f.get("bbox_x0", 0), f.get("bbox_y0", 0), f.get("bbox_x1", 0), f.get("bbox_y1", 0)))
            f["_bi"] = len(boxes) - 1
            anns.append(f)
            cv2.fillPoly(m512, [np.round(poly / 4).astype(np.int32)], 1)
        # crowding: other boxes intersecting this box padded by 30 px; nearest-neighbour centroid distance
        vf = [f for f in anns if f["view_id"] == view_id]
        for f in vf:
            bx = boxes[f["_bi"]]
            f["crowd_n"] = sum(1 for j, o in enumerate(boxes) if j != f["_bi"] and o[0] <= bx[2] + 30 and o[2] >= bx[0] - 30
                               and o[1] <= bx[3] + 30 and o[3] >= bx[1] - 30)
            d = [math.hypot(f.get("cx", 0) - o.get("cx", 0), f.get("cy", 0) - o.get("cy", 0)) for o in vf if o is not f]
            f["nn_dist"] = min(d) if d else np.nan
            del f["_bi"]
        fl = m512[:, ::-1]  # ultralytics CopyPaste(mode='flip') = horizontal flip about the image centre
        if fl.sum() > 0:
            mirror.append(dict(view_id=view_id, stem=stem, overlap=float((fl & m512).sum() / fl.sum()),
                               off_disk=float(((fl > 0) & (disk512 > 0.98)).sum() / fl.sum()),
                               cover=float(m512.mean())))
    return dict(img=row, anns=anns, mirror=mirror)


def pred_worker(job: tuple) -> list[dict]:
    stem, path, rles, cx, cy, R = job
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    out = []
    for j, r in enumerate(rles):
        c = crop_from_rle(r, g)
        if c is None:
            out.append(dict(stem=stem, j=j))
            continue
        m, y0, x0, gc = c
        f = mask_feats(m, y0, x0, gc)
        f.update(stem=stem, j=j)
        out.append(f)
    return out


# ----------------------------------------------------------------------------- generic breakdown
def add_geom(df: pd.DataFrame, img: pd.DataFrame) -> pd.DataFrame:
    d = df.merge(img[["stem", "cx", "cy", "R"]].rename(columns={"cx": "dcx", "cy": "dcy", "R": "dR"}), on="stem", how="left")
    d["rR"] = np.hypot(d["cx"] - d["dcx"], d["cy"] - d["dcy"]) / d["dR"]
    d["lat"] = (d["dcy"] - d["cy"]) / d["dR"]  # + = north (image up)
    d["lon"] = (d["cx"] - d["dcx"]) / d["dR"]
    return d


def cut(s: pd.Series, edges, labels=None):
    return pd.cut(s, edges, labels=labels, include_lowest=True)


def quint(s: pd.Series, ref: pd.Series | None = None, q: int = 5):
    ref = s if ref is None else ref
    e = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, q + 1)))
    e[0], e[-1] = -np.inf, np.inf
    labs = [f"Q{i + 1}" for i in range(len(e) - 1)]
    rng = np.quantile(ref.dropna(), np.linspace(0, 1, q + 1))
    labs = [f"Q{i + 1} ({rng[i]:.3g}-{rng[i + 1]:.3g})" for i in range(len(e) - 1)]
    return pd.cut(s, e, labels=labs)


def two_side(gt: pd.DataFrame, pr: pd.DataFrame, gcol: str, pcol: str, name: str) -> pd.DataFrame:
    g = gt.groupby(gcol, observed=True).agg(n_gt=("hit", "size"), TP=("hit", "sum"))
    g["FN"] = g.n_gt - g.TP
    g["recall"] = g.TP / g.n_gt
    g["FN_share%"] = 100 * g.FN / g.FN.sum()
    p = pr.groupby(pcol, observed=True).agg(n_pred=("hit", "size"), hitp=("hit", "sum"))
    p["FP"] = p.n_pred - p.hitp
    p["precision"] = p.hitp / p.n_pred
    p["FP_share%"] = 100 * p.FP / p.FP.sum()
    p = p.drop(columns="hitp")
    t = g.join(p, how="outer")
    t.index.name = name
    return t.reset_index()


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "filseg_analysis_v2_cache"))
    args = ap.parse_args()
    nw = min(args.workers, 16)
    t0 = time.time()
    FIGS.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    rep: list[str] = []
    R_ = rep.append

    split = json.load(open(SPLIT))
    val_stems = sorted(split["val"])
    val_set = set(val_stems)
    coco = json.load(open(COCO))
    by_view: dict[str, list] = defaultdict(list)
    for a in coco["annotations"]:
        by_view[a["image_id"]].append(a)
    by_stem: dict[str, list] = defaultdict(list)
    for vid in sorted(by_view):
        by_stem[view_stem(vid)].append((vid, vid.split("-", 1)[0], by_view[vid]))
    manifest = pd.read_csv(ROOT / "data/processed/manifest.csv", dtype={"annotator": str})
    stem_meta = manifest.drop_duplicates("stem").set_index("stem")[["year", "instrument"]]

    # ---- pass 1: all images (train annotated + test)
    pk = cache / "pass1.pkl"
    if pk.exists() and not args.no_cache:
        res = pd.read_pickle(pk)
    else:
        jobs = [(s, "train", str(TRAIN_IMG / f"{s}.jpeg"), v) for s, v in sorted(by_stem.items())]
        test_stems = sorted(p.stem for p in TEST_IMG.glob("*.jpeg"))
        jobs += [(s, "test", str(TEST_IMG / f"{s}.jpeg"), []) for s in test_stems]
        with Pool(nw) as pool:
            res = pool.map(stem_worker, jobs, chunksize=4)
        pd.to_pickle(res, pk)
    print(f"pass1 done {time.time() - t0:.0f}s", flush=True)
    img = pd.DataFrame([r["img"] for r in res])
    img["instrument"] = img.stem.str[-2:]
    img["year"] = img.stem.str[:4].astype(int)
    img["date"] = img.stem.str[:8]
    img["val"] = img.stem.isin(val_set)
    ann = add_geom(pd.DataFrame([a for r in res for a in r["anns"]]), img)
    ann["val"] = ann.stem.isin(val_set)
    ann["year"] = ann.stem.str[:4].astype(int)
    ann["annot_prefix"] = ann.annotator.str[:2]
    mirror = pd.DataFrame([m for r in res for m in r["mirror"]])
    tr, te = img[img.split == "train"], img[img.split == "test"]
    R_("# Auto-generated analysis tables (scripts/analyze_data_v2.py)\n")
    R_(f"{len(tr)} train stems, {len(te)} test stems, {len(ann)} annotations in {ann.view_id.nunique()} views. "
       f"Val = split_v1 val ({len(val_stems)} stems). Disk fit = strongest radial intensity drop (limb) + robust circle fit.\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ============================================================ Q1 geometry
    R_("## Q1. Disk geometry\n")
    rows = []
    for nm, d in [("train", tr), ("test", te), ("val(subset of train)", img[img.val])]:
        rows.append(dict(set=nm, n=len(d), R_mean_px=d.R.mean(), R_std_px=d.R.std(), R_cv_pct=100 * d.R.std() / d.R.mean(),
                         R_min=d.R.min(), R_p1=d.R.quantile(.01), R_p99=d.R.quantile(.99), R_max=d.R.max(),
                         R_over_img_pct=100 * d.R.mean() / 2048))
    R_(md_table(pd.DataFrame(rows), "{:.2f}") + "\n")
    allr = pd.concat([tr, te])
    q1 = []
    for nm, d in [("train", tr), ("test", te)]:
        dx, dy = d.cx - 1024, d.cy - 1024
        off = np.hypot(dx, dy)
        q1.append(dict(set=nm, dx_mean=dx.mean(), dx_std=dx.std(), dy_mean=dy.mean(), dy_std=dy.std(), off_mean=off.mean(),
                       off_p95=off.quantile(.95), off_max=off.max(), off_max_pct_R=100 * off.max() / d.R.mean(),
                       ell_ratio_max=d.ell.max(), margin_min=(1024 - np.maximum(np.abs(dx), np.abs(dy)) - d.R).min()))
    R_("Disk-centre offset from image centre (px at 2048) and QC (ellipse axis ratio, minimum margin disk-edge to frame):\n")
    R_(md_table(pd.DataFrame(q1), "{:.2f}") + "\n")
    rs = []
    for nm, d in [("train", tr), ("test", te)]:
        rs.append(dict(set=nm, R_ratio_p1=d.R.quantile(.01) / d.R.mean(), R_ratio_p99=d.R.quantile(.99) / d.R.mean(),
                       R_ratio_min=d.R.min() / d.R.mean(), R_ratio_max=d.R.max() / d.R.mean()))
    R_("Radius relative to the dataset mean (scale-aug factor needed to cover it):\n")
    R_(md_table(pd.DataFrame(rs), "{:.3f}") + "\n")
    byy = allr.groupby(["split", "year"]).R.agg(["count", "mean", "std"]).reset_index()
    R_("Disk radius by year :\n")
    byp = byy.pivot(index="year", columns="split", values=["count", "mean", "std"]).reset_index().round(2).set_axis(
        ["year", "n_test", "n_train", "R_mean_test", "R_mean_train", "R_std_test", "R_std_train"], axis=1)
    byp["year"] = byp.year.astype(int).astype(str)
    byp["n_test"], byp["n_train"] = byp.n_test.astype(int).astype(str), byp.n_train.astype(int).astype(str)
    R_(md_table(byp, "{:.2f}") + "\n")
    # filament sizes vs scale aug / translate
    lim = ann[(ann.rR > 0.9)]
    R_(f"Filament centroids with r/R>0.9: {len(lim)} ({100 * len(lim) / len(ann):.1f}%); r/R>0.95: {(ann.rR > 0.95).sum()}. "
       f"Max annotation-centroid distance from the image centre: {np.hypot(ann.cx - 1024, ann.cy - 1024).max():.0f} px; "
       f"all annotated pixels lie within the frame (margin to border: min bbox margin "
       f"{min(ann.bbox_x0.min(), ann.bbox_y0.min(), 2047 - ann.bbox_x1.max(), 2047 - ann.bbox_y1.max())} px).\n")
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8), dpi=80)
    ax[0].hist([tr.R, te.R], bins=30, label=["train", "test"], density=True)
    ax[0].set_xlabel("disk R (px @2048)"); ax[0].legend()
    ax[1].scatter(tr.cx - 1024, tr.cy - 1024, s=4, label="train"); ax[1].scatter(te.cx - 1024, te.cy - 1024, s=4, label="test")
    ax[1].set_xlabel("dx (px)"); ax[1].set_ylabel("dy (px)"); ax[1].legend()
    ax[2].scatter(allr.year + (allr.stem.str[4:6].astype(int) - 1) / 12, allr.R, s=3, c=(allr.split == "test"), cmap="coolwarm")
    ax[2].set_xlabel("year"); ax[2].set_ylabel("R (px)")
    fig.tight_layout(); fig.savefig(FIGS / "q1_geometry.png"); plt.close(fig)

    # ============================================================ Q2 orientation
    R_("\n## Q2. Orientation (spine PCA, axial angle in (-90,90], 0 = horizontal/E-W, + = counter-clockwise, image up = north)\n")
    a2 = ann.dropna(subset=["ang"]).copy()
    a2["ang_c"] = axial_wrap(a2.ang)
    h = np.histogram(a2.ang_c, bins=np.arange(-90, 91, 15))[0]
    R_(md_table(pd.DataFrame({"tilt_bin_deg": [f"[{b},{b + 15})" for b in range(-90, 90, 15)], "n": h, "frac%": 100 * h / h.sum()}), "{:.1f}") + "\n")
    dang = np.minimum(np.abs(a2.ang_c), 90)
    mu, Rl = circ_axial(a2.ang_c)
    R_(f"Axial mean tilt {mu:.1f} deg, resultant length (axial) {Rl:.3f} (0 = isotropic). Fractions: |tilt|<22.5 (horizontal) "
       f"{100 * (dang < 22.5).mean():.1f}%, 22.5-67.5 (diagonal) {100 * ((dang >= 22.5) & (dang < 67.5)).mean():.1f}%, >=67.5 (vertical) "
       f"{100 * (dang >= 67.5).mean():.1f}% (uniform expectation 25/50/25). Spine/PCA(polygon) agreement: median abs diff "
       f"{np.median(np.abs(axial_wrap(a2.ang - a2.ang_poly))):.1f} deg.\n")
    a2["latb"] = cut(a2.lat, [-1.01, -0.6, -0.3, 0, 0.3, 0.6, 1.01], ["S>0.6", "S0.3-0.6", "S0-0.3", "N0-0.3", "N0.3-0.6", "N>0.6"])
    rows = []
    for b, g in a2.groupby("latb", observed=True):
        m_, r_ = circ_axial(g.ang_c)
        rows.append(dict(lat_band=b, n=len(g), mean_tilt_deg=m_, resultant=r_, mean_signed_tilt_deg=g.ang_c.mean(),
                         frac_horiz_pct=100 * (np.abs(g.ang_c) < 22.5).mean(), frac_vert_pct=100 * (np.abs(g.ang_c) >= 67.5).mean(),
                         mean_len=g.spine_len.mean()))
    R_("Orientation vs solar latitude proxy (y offset from disk centre / R):\n")
    R_(md_table(pd.DataFrame(rows), "{:.2f}") + "\n")
    rows = []
    for nm, g in [("North (lat>0.05)", a2[a2.lat > 0.05]), ("South (lat<-0.05)", a2[a2.lat < -0.05]), ("|lat|<=0.05", a2[a2.lat.abs() <= 0.05])]:
        m_, r_ = circ_axial(g.ang_c)
        rows.append(dict(hemisphere=nm, n=len(g), mean_tilt_deg=m_, resultant=r_, share_pos_tilt_pct=100 * (g.ang_c > 0).mean(),
                         share_pos_tilt_of_nonhoriz_pct=100 * (g[np.abs(g.ang_c) > 22.5].ang_c > 0).mean()))
    R_("Hemisphere asymmetry (a rule 'tilt = f(latitude)' that is odd in latitude survives flipud but not fliplr):\n")
    R_(md_table(pd.DataFrame(rows), "{:.2f}") + "\n")
    # angle to local radial direction
    phi = np.degrees(np.arctan2(-(a2.cy - a2.dcy), a2.cx - a2.dcx))  # radial direction angle (y up)
    rel = np.abs(axial_wrap(a2.ang_c - phi))  # 0 = radial, 90 = tangential
    a2["rel_rad"] = rel
    rows = []
    for b, g in a2.groupby(cut(a2.rR, [0, .4, .7, .9, 1.1], ["<0.4", "0.4-0.7", "0.7-0.9", ">0.9"]), observed=True):
        rows.append(dict(rR=b, n=len(g), mean_angle_to_radial_deg=g.rel_rad.mean(), frac_tangential_pct=100 * (g.rel_rad > 60).mean()))
    R_("Angle between filament axis and the radial direction (uniform: mean 45, 33% tangential>60):\n")
    R_(md_table(pd.DataFrame(rows), "{:.1f}") + "\n")
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8), dpi=80)
    ax[0].bar(np.arange(-90, 90, 15) + 7.5, h, width=14); ax[0].set_xlabel("axial tilt (deg)")
    ax[1].scatter(a2.lat, a2.ang_c, s=2, alpha=.3); ax[1].set_xlabel("latitude proxy (N+)"); ax[1].set_ylabel("tilt (deg)")
    ax[2].hist(a2.rel_rad, bins=18); ax[2].set_xlabel("angle to radial (deg)")
    fig.tight_layout(); fig.savefig(FIGS / "q2_orientation.png"); plt.close(fig)

    # ============================================================ Q3 photometry
    R_("\n## Q3. Photometry (disk r<0.97R, 1024-px copies; grey levels 0-255)\n")
    feats = ["disk_mean", "disk_std", "ld3", "ld4", "bg_mean", "corner_mean", "hp_std", "sharp", "p99", "sat_frac"]
    desc = {"disk_mean": "mean disk intensity", "disk_std": "std on disk", "ld3": "limb dark. r.75-.9 / centre", "ld4": "r.9-.97 / centre",
            "bg_mean": "sky ring 1.04-1.15R", "corner_mean": "frame corners", "hp_std": "high-pass std (granulation+filaments)",
            "sharp": "sharpness (lapl var/mean^2)", "p99": "p99 disk", "sat_frac": "saturated frac"}
    rows = []
    for f in feats:
        rows.append(dict(metric=desc[f], train_mean=tr[f].mean(), train_std=tr[f].std(), test_mean=te[f].mean(), test_std=te[f].std(),
                         cohen_d=cohen_d(tr[f], te[f]), KS=ks_2samp(tr[f], te[f]),
                         train_cv_pct=100 * tr[f].std() / abs(tr[f].mean())))
    R_("Train vs test (KS 5% critical value for n=707/180 is 0.115):\n")
    R_(md_table(pd.DataFrame(rows), "{:.4f}") + "\n")
    rows = []
    for (sp, s), g in img.groupby(["split", "instrument"]):
        rows.append(dict(split=sp, site=s, n=len(g), disk_mean=g.disk_mean.mean(), disk_std=g.disk_std.mean(), ld3=g.ld3.mean(), ld4=g.ld4.mean(),
                         bg=g.bg_mean.mean(), hp_std=g.hp_std.mean(), sharp=g.sharp.mean(), R=g.R.mean(), mean_disk_cv_pct=100 * g.disk_mean.std() / g.disk_mean.mean()))
    R_("By site:\n")
    R_(md_table(pd.DataFrame(rows).sort_values(["site", "split"]), "{:.3f}") + "\n")
    rows = []
    for (sp, yr), g in img.groupby(["split", "year"]):
        rows.append(dict(split=sp, year=yr, n=len(g), disk_mean=g.disk_mean.mean(), disk_std=g.disk_std.mean(), ld3=g.ld3.mean(), hp_std=g.hp_std.mean(), sharp=g.sharp.mean()))
    R_("By year:\n")
    yt = pd.DataFrame(rows).sort_values(["year", "split"])
    yt["year"] = yt.year.astype(str)
    R_(md_table(yt, "{:.3f}") + "\n")
    ct = pd.concat([tr.instrument.value_counts(normalize=True).rename("train_site_frac"), te.instrument.value_counts(normalize=True).rename("test_site_frac")], axis=1)
    cy_ = pd.concat([tr.year.value_counts(normalize=True).rename("train_year_frac"), te.year.value_counts(normalize=True).rename("test_year_frac")], axis=1).sort_index()
    R_("Composition:\n")
    R_(md_table(ct.reset_index().rename(columns={"index": "site", "instrument": "site"}), "{:.3f}") + "\n")
    cy2 = cy_.reset_index().rename(columns={"index": "year", "year": "year"}).fillna(0)
    cy2["year"] = cy2.year.astype(int).astype(str)
    R_(md_table(cy2, "{:.3f}") + "\n")
    img["halo"] = img.bg_mean > 40  # bright sky halo around the limb (some frames have a ~17-grey halo, others ~60-90)
    ht = img.groupby("split").halo.agg(["size", "mean"]).reset_index().rename(columns={"mean": "halo_bright_frac"})
    R_("\nBright sky halo (mean grey 1.04-1.15R > 40; typical dark halo is ~15-25) share by split / site / year:\n")
    R_(md_table(ht, "{:.3f}") + "\n")
    R_(md_table(img.groupby(["instrument", "split"]).halo.mean().unstack().reset_index(), "{:.3f}") + "\n")
    hy = img.groupby(["year", "split"]).halo.mean().unstack().reset_index()
    hy["year"] = hy.year.astype(str)
    R_(md_table(hy, "{:.3f}") + "\n")
    bright = tr.disk_mean
    R_(f"Disk brightness spread across train images: mean {bright.mean():.1f}, std {bright.std():.1f} ({100 * bright.std() / bright.mean():.1f}%), "
       f"p1-p99 {bright.quantile(.01):.1f}-{bright.quantile(.99):.1f} -> multiplicative range {bright.quantile(.01) / bright.mean():.2f}-{bright.quantile(.99) / bright.mean():.2f}. "
       f"(ultralytics hsv_v g: V *= 1 + U(-g,g)*...; g=0.3 -> x0.70-1.30.) Test: p1-p99 {te.disk_mean.quantile(.01):.1f}-{te.disk_mean.quantile(.99):.1f}.\n")
    # filament contrast
    cc = ann.dropna(subset=["contrast"])
    R_(f"\nFilament contrast (ring mean - inside mean, grey levels): median {cc.contrast.median():.1f}, p5/p95 {cc.contrast.quantile(.05):.1f}/{cc.contrast.quantile(.95):.1f}; "
       f"relative contrast median {100 * cc.rel_contrast.median():.1f}% ; d' (contrast/ring std) median {cc.dprime.median():.2f}, "
       f"share with d'<0.5: {100 * (cc.dprime < .5).mean():.1f}%.\n")
    cj = cc.merge(img[["stem", "disk_mean", "hp_std", "sharp"]], on="stem")
    rows = []
    for s, g in cj.groupby(cj.stem.str[-2:]):
        rows.append(dict(site=s, n_ann=len(g), contrast_med=g.contrast.median(), rel_contrast_med_pct=100 * g.rel_contrast.median(), dprime_med=g.dprime.median()))
    R_(md_table(pd.DataFrame(rows), "{:.2f}") + "\n")
    R_(f"Correlation of per-annotation contrast with image brightness: {np.corrcoef(cj.contrast, cj.disk_mean)[0, 1]:.3f}; "
       f"relative contrast vs brightness: {np.corrcoef(cj.rel_contrast, cj.disk_mean)[0, 1]:.3f}.\n")
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8), dpi=80)
    for s in SITES:
        ax[0].scatter(tr[tr.instrument == s].disk_mean, tr[tr.instrument == s].ld3, s=4, label=s)
    ax[0].set_xlabel("disk mean"); ax[0].set_ylabel("limb darkening ld3"); ax[0].legend(fontsize=6, ncol=2)
    ax[1].hist([tr.disk_mean, te.disk_mean], bins=25, density=True, label=["train", "test"]); ax[1].set_xlabel("disk mean"); ax[1].legend()
    ax[2].boxplot([cc[cc.stem.str[-2:] == s].dprime for s in SITES], tick_labels=SITES, showfliers=False); ax[2].set_ylabel("filament d'")
    fig.tight_layout(); fig.savefig(FIGS / "q3_photometry.png"); plt.close(fig)

    # ============================================================ Q4 instances / annotators
    R_("\n## Q4. Instances and annotators\n")
    vcount = ann.groupby("view_id").size()
    R_(f"Annotations per view: mean {vcount.mean():.2f}, median {vcount.median():.0f}, p5/p95 {vcount.quantile(.05):.0f}/{vcount.quantile(.95):.0f}, max {vcount.max()}; "
       f"views with >20: {(vcount > 20).sum()}.\n")
    rows = []
    for nm, s in [("area px", ann.area), ("width px (2A/P)", ann.width), ("spine length px", ann.spine_len), ("elongation", ann.elong),
                  ("solidity", ann.solidity), ("bbox fill", ann.bbox_fill), ("sinuosity chord/path", ann.chord / ann.spine_len),
                  ("area %img (px/2048^2 *1e4)", ann.area / 2048 ** 2 * 1e4)]:
        rows.append(dict(stat=nm, mean=s.mean(), **qs(s)))
    R_(md_table(pd.DataFrame(rows), "{:.2f}") + "\n")
    sc = 1536 / 2048
    R_(f"At imgsz 1536 (x0.75): share of GT with width<4 px {100 * (ann.width * sc < 4).mean():.1f}%, <3 px {100 * (ann.width * sc < 3).mean():.1f}%; "
       f"area<100 px (@2048) {100 * (ann.area < 100).mean():.1f}%, <300 px {100 * (ann.area < 300).mean():.1f}%; at the proto stride (4) width<1 proto px: "
       f"{100 * (ann.width * sc / 4 < 1).mean():.1f}%. Total area per view as % of disk: median {100 * (ann.groupby('view_id').area.sum() / (math.pi * img.R.mean() ** 2)).median():.2f}%.\n")
    R_("Crowding / overlap (per annotation):\n")
    R_(md_table(pd.DataFrame([dict(stat="neighbours with bbox within 30px (mean)", v=ann.crowd_n.mean()),
                              dict(stat="share with >=1 such neighbour %", v=100 * (ann.crowd_n >= 1).mean()),
                              dict(stat="share with >=3 neighbours %", v=100 * (ann.crowd_n >= 3).mean()),
                              dict(stat="nearest-neighbour centroid dist median px", v=ann.nn_dist.median()),
                              dict(stat="nearest-neighbour dist p10 px", v=ann.nn_dist.quantile(.1))]), "{:.2f}") + "\n")
    # annotator style
    first = ann.groupby("view_id").size().rename("n_view").reset_index().merge(manifest[["view_id", "annotator", "n_views_of_stem"]], on="view_id")
    ann = ann.merge(manifest[["view_id", "n_views_of_stem"]], on="view_id", how="left")
    st = ann.groupby("annotator").agg(n_ann=("area", "size"), n_views=("view_id", "nunique"), area_med=("area", "median"), width_mean=("width", "mean"),
                                      len_med=("spine_len", "median"), contrast_med=("contrast", "median"), frac_small_pct=("area", lambda x: 100 * (x < 500).mean()))
    st["ann_per_view"] = st.n_ann / st.n_views
    st = st.reset_index()
    big = st[st.n_views >= 20].sort_values("n_views", ascending=False)
    R_(f"\nPer annotator ID ({len(st)} IDs; showing those with >=20 views; all-annotator mean ann/view {ann.groupby('view_id').size().mean():.2f}):\n")
    R_(md_table(big[["annotator", "n_views", "ann_per_view", "area_med", "width_mean", "len_med", "contrast_med", "frac_small_pct"]], "{:.1f}") + "\n")
    # paired comparison on shared stems
    multi = {s: v for s, v in by_stem.items() if len(v) >= 2}
    pairs, per_ann_rel = [], []
    gt_all = build_gt_df(str(COCO))
    gt_all["view"] = gt_all.filament_id.str.split("_", n=1).str[0]
    rle_by_view = {v: [to_rle_obj(x) for x in g.segmentation_rle] for v, g in gt_all.groupby("view", sort=False)}
    ann_idx = ann.set_index(["view_id", "k"])
    for s, views in multi.items():
        cnt = np.array([len(a) for _, _, a in views], float)
        for (vid, an, al), c in zip(views, cnt):
            per_ann_rel.append(dict(stem=s, annotator=an, n=c, rel_n=c / cnt.mean(), n_views=len(views)))
        for i, (va, aa, _) in enumerate(views):
            for j, (vb, ab, _) in enumerate(views):
                if i == j:
                    continue
                A, B = rle_by_view[va], rle_by_view[vb]
                iou = np.asarray(mask_util.iou(B, A, [0] * len(A))).T.reshape(len(A), len(B))
                hit = iou > 0.5
                for k in range(len(A)):
                    pairs.append(dict(stem=s, va=va, vb=vb, aa=aa, ab=ab, k=k + 1, matched=bool(hit[k].any()), best_iou=float(iou[k].max()) if len(B) else 0.0))
    pr_ = pd.DataFrame(pairs)
    pr_ = pr_.merge(ann[["view_id", "k", "area", "width", "contrast", "dprime", "spine_len", "rR", "val"]], left_on=["va", "k"], right_on=["view_id", "k"])
    rel = pd.DataFrame(per_ann_rel)
    arow = rel.groupby("annotator").agg(stems=("stem", "size"), rel_count=("rel_n", "mean")).reset_index()
    am = pr_.groupby("aa").agg(n_inst=("matched", "size"), matched_by_other_pct=("matched", lambda x: 100 * x.mean())).reset_index().rename(columns={"aa": "annotator"})
    arow = arow.merge(am, on="annotator")
    R_(f"\nSame-stem comparison (stems with >=2 annotators: {len(multi)}). rel_count = this annotator's count / mean count of all annotators of that stem "
       f"(1.0 = typical). matched_by_other = share of this annotator's instances that another annotator also marked (IoU>0.5):\n")
    R_(md_table(arow[arow.stems >= 15].sort_values("rel_count"), "{:.2f}") + "\n")
    R_(f"Overall: instance marked by another annotator (IoU>0.5) in {100 * pr_.matched.mean():.1f}% of ordered pairs; "
       f"spread of rel_count across annotators with >=15 stems: min {arow[arow.stems >= 15].rel_count.min():.2f}, max {arow[arow.stems >= 15].rel_count.max():.2f}, "
       f"std {arow[arow.stems >= 15].rel_count.std():.2f}.\n")
    sup = pr_.groupby(["va", "k", "aa"]).matched.max().reset_index()
    sup["unsup"] = ~sup.matched
    us = sup.groupby("aa").agg(n=("unsup", "size"), unsupported_pct=("unsup", lambda x: 100 * x.mean())).reset_index().rename(columns={"aa": "annotator"})
    R_(f"Instances in multi-annotator stems marked by NO other annotator: {sup.unsup.sum()} / {len(sup)} = {100 * sup.unsup.mean():.1f}% "
       f"(per annotator ID with >=100 such instances below).\n")
    R_(md_table(us[us.n >= 100].sort_values("unsupported_pct"), "{:.1f}") + "\n")
    nv = sup.merge(pd.DataFrame(per_ann_rel)[["annotator", "stem", "n_views"]].rename(columns={"annotator": "aa"}).assign(va=lambda d: d.aa + "-" + d.stem)[["va", "n_views"]], on="va")
    R_("Unsupported share by number of annotators of the stem: " + ", ".join(f"{n}: {100 * g.unsup.mean():.1f}% (n={len(g)})" for n, g in nv.groupby("n_views")) + "\n")
    pr_["aq"] = quint(pr_.area)
    pr_["wq"] = quint(pr_.width)
    pr_["dq"] = cut(pr_.dprime, [-np.inf, 0.5, 1, 2, 3, np.inf], ["<0.5", "0.5-1", "1-2", "2-3", ">3"])
    for col, nm in [("aq", "area quintile"), ("wq", "width quintile"), ("dq", "contrast d'")]:
        t = pr_.groupby(col, observed=True).agg(n=("matched", "size"), agree_pct=("matched", lambda x: 100 * x.mean()), mean_best_iou=("best_iou", "mean")).reset_index()
        R_(f"\nInter-annotator agreement (instance matched by another annotator) vs {nm}:\n")
        R_(md_table(t, "{:.2f}") + "\n")
    # pair PQ (as G1) per annotator summarised: symmetric
    rows = []
    for s, views in multi.items():
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                A, B = rle_by_view[views[i][0]], rle_by_view[views[j][0]]
                iou = np.asarray(mask_util.iou(B, A, [0] * len(A))).T.reshape(len(A), len(B))
                hit = iou > 0.5
                tp = hit.any(1).sum(); fn = len(A) - tp; fp = len(B) - hit.any(0).sum()
                siou = sum(iou[k].max() for k in range(len(A)) if hit[k].any())
                rows.append(dict(a=views[i][1], b=views[j][1], tp=tp, fp=fp, fn=fn, siou=siou))
    pq = pd.DataFrame(rows)
    pooled_pq = pq.siou.sum() / (pq.tp.sum() + .5 * pq.fp.sum() + .5 * pq.fn.sum())
    R_(f"\nPairwise annotator-vs-annotator pooled PQ (one view as GT, the other as pred; hit semantics as the metric): {pooled_pq:.3f} "
       f"(TP {pq.tp.sum()}, FP {pq.fp.sum()}, FN {pq.fn.sum()}); the model's val PQ 0.4675 evaluates against 1-3 annotators pooled.\n")
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8), dpi=80)
    ax[0].hist(np.log10(ann.area), bins=40); ax[0].set_xlabel("log10 area")
    ax[1].hist(ann.width.clip(upper=60), bins=40); ax[1].set_xlabel("width (px)")
    ax[2].bar(range(len(arow[arow.stems >= 15])), arow[arow.stems >= 15].sort_values("rel_count").rel_count); ax[2].axhline(1, c="k", lw=.5)
    ax[2].set_xlabel("annotator (>=15 stems)"); ax[2].set_ylabel("relative count")
    fig.tight_layout(); fig.savefig(FIGS / "q4_instances.png"); plt.close(fig)

    # ============================================================ Q5 errors
    R_("\n## Q5. Errors of y11m_1536_flip_ref on val\n")
    raw = pd.read_pickle(PRED_DIR / "val_raw.pkl")
    params = PPParams(**PP)
    rows = postprocess_to_rows(group_by_stem(raw), params, val_stems, group_soft_by_stem(raw))
    pred_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    gt_val = build_gt_df(str(COCO), stems=val_set)
    ev = fast_evaluate(gt_val, pred_df)
    assert (ev.tp, ev.fp, ev.fn) == (994, 393, 551), (ev.tp, ev.fp, ev.fn)
    R_(f"Reconciled with the cached evaluation: PQ {ev.pq:.4f}, SQ {ev.sq:.3f}, RQ {ev.rq:.3f}, TP/FP/FN = {ev.tp}/{ev.fp}/{ev.fn} "
       f"({ev.n_views} GT views, {ev.n_gt} GT, {ev.n_pred} (view,pred) pairs; params {PP}). FP/TP/FN are counted per (view, instance) as in the organizer metric.\n")
    pred_df["stem"] = pred_df.filament_id.str.rsplit("_", n=1).str[0]
    pred_df["j"] = pred_df.filament_id.str.rsplit("_", n=1).str[1].astype(int) - 1
    pred_rles = {s: [to_rle_obj(x) for x in g.sort_values("j").segmentation_rle] for s, g in pred_df.groupby("stem")}
    raw_g = {s: (g["score"].to_numpy(), [to_rle_obj(x) for x in g["rle"]]) for s, g in raw.groupby("stem")}
    # pred features
    pjobs = [(s, str(TRAIN_IMG / f"{s}.jpeg"), [pred_df[(pred_df.stem == s)].sort_values("j").segmentation_rle.tolist()][0],
              *img.set_index("stem").loc[s, ["cx", "cy", "R"]].tolist()) for s in sorted(pred_rles)]
    with Pool(nw) as pool:
        pf = pd.DataFrame([r for rs_ in pool.map(pred_worker, pjobs) for r in rs_])
    pf = add_geom(pf, img)
    pf_idx = pf.set_index(["stem", "j"])

    gt_rec, pr_rec = [], []
    views_of = {s: [v[0] for v in by_stem[s]] for s in val_stems}
    sup_pair_stats = []
    for s in val_stems:
        vws = views_of[s]
        P = pred_rles.get(s, [])
        Gv = {v: rle_by_view[v] for v in vws}
        pa = np.asarray(mask_util.area(P), float) if P else np.zeros(0)
        rs_scores, RR = raw_g.get(s, (np.zeros(0), []))
        # GT-GT cross-view agreement
        gg = {}
        for v in vws:
            for w in vws:
                if v != w:
                    A, B = Gv[v], Gv[w]
                    gg[(v, w)] = (np.asarray(mask_util.iou(B, A, [0] * len(A))).T.reshape(len(A), len(B)) > 0.5).any(1)
        pred_hitviews = np.zeros(len(P), int)
        for v in vws:
            A = Gv[v]
            if P:
                iou = np.asarray(mask_util.iou(P, A, [0] * len(A))).T.reshape(len(A), len(P))
            else:
                iou = np.zeros((len(A), 0))
            hit = iou.astype(np.float32) > np.float32(0.5)
            ga = np.asarray(mask_util.area(A), float)
            inter = np.rint(iou * (ga[:, None] + pa[None, :]) / (1 + iou)) if len(P) else iou
            pred_hitviews += hit.any(0)
            # raw proposals
            if len(RR):
                riou = np.asarray(mask_util.iou(RR, A, [0] * len(A))).T.reshape(len(A), len(RR))
            else:
                riou = np.zeros((len(A), 0))
            for k in range(len(A)):
                sup = sum(int(gg[(v, w)][k]) for w in vws if w != v)
                row = dict(view=v, stem=s, k=k + 1, hit=bool(hit[k].any()), best_iou=float(iou[k].max()) if len(P) else 0.0, support=sup, n_other=len(vws) - 1)
                # fragmentation: preds that lie mostly (>=50%) inside this GT
                if len(P):
                    fr = np.where((inter[k] / np.maximum(pa, 1) >= 0.5) & (inter[k] > 0))[0]
                    row["n_frag"] = len(fr)
                    if len(fr) >= 2:
                        row["frag_union_iou"] = float(inter[k, fr].sum() / (ga[k] + pa[fr].sum() - inter[k, fr].sum()))
                else:
                    row["n_frag"] = 0
                if len(RR):
                    bi = riou[k]
                    ok = bi > 0.5
                    row["raw_best_iou"] = float(bi.max())
                    row["raw_score_best_match"] = float(rs_scores[ok].max()) if ok.any() else np.nan
                else:
                    row["raw_best_iou"] = 0.0
                    row["raw_score_best_match"] = np.nan
                row["cov_by_pred"] = float(inter[k].sum() / ga[k]) if len(P) else 0.0
                gt_rec.append(row)
            for j in range(len(P)):
                col = inter[:, j] if len(A) else np.zeros(0)
                pr_rec.append(dict(view=v, stem=s, j=j, hit=bool(hit[:, j].any()) if len(A) else False,
                                   best_iou=float(iou[:, j].max()) if len(A) else 0.0,
                                   inside=float(col.max() / pa[j]) if len(A) else 0.0,
                                   area_ratio=float(pa[j] / ga[int(np.argmax(iou[:, j]))]) if len(A) else np.nan,
                                   n_gt_covered=int(((col / np.maximum(ga, 1)) >= 0.5).sum()) if len(A) else 0))
        for j in range(len(P)):
            sup_pair_stats.append((s, j, int(pred_hitviews[j]), len(vws)))
    G = pd.DataFrame(gt_rec)
    Pp = pd.DataFrame(pr_rec)
    ph = pd.DataFrame(sup_pair_stats, columns=["stem", "j", "views_hit", "n_views"])
    Pp = Pp.merge(ph, on=["stem", "j"])
    assert G.hit.sum() == ev.tp and (~G.hit).sum() == ev.fn and (~Pp.hit).sum() == ev.fp, "reconcile"
    G = G.merge(ann[["view_id", "k", "area", "width", "contrast", "dprime", "rR", "lat", "spine_len", "ang", "annotator", "crowd_n", "elong"]],
                left_on=["view", "k"], right_on=["view_id", "k"], how="left")
    G["site"] = G.stem.str[-2:]
    G = G.merge(img[["stem", "halo", "disk_mean", "hp_std"]], on="stem", how="left")
    G["year"] = G.stem.str[:4].astype(int)
    Pp = Pp.merge(pf, on=["stem", "j"], how="left")
    Pp["site"] = Pp.stem.str[-2:]
    Pp = Pp.merge(img[["stem", "halo"]], on="stem", how="left")

    # --- 5.0 where the errors are: IoU/localisation
    nfn, nfp = (~G.hit).sum(), (~Pp.hit).sum()
    bins = [-1, 1e-9, 0.3, 0.5]
    gb = cut(G[~G.hit].best_iou, bins, ["0 (no overlap)", "(0,0.3]", "(0.3,0.5]"])
    pb = cut(Pp[~Pp.hit].best_iou, bins, ["0 (no overlap)", "(0,0.3]", "(0.3,0.5]"])
    t = pd.DataFrame({"FN": gb.value_counts().sort_index(), "FP": pb.value_counts().sort_index()}).reset_index().rename(columns={"index": "best IoU of error"})
    t["FN%"] = 100 * t.FN / nfn
    t["FP%"] = 100 * t.FP / nfp
    R_("### 5.1 Error by best IoU against the other side\n")
    R_(md_table(t, "{:.1f}") + "\n")

    # --- ranking vs detection
    fn = G[~G.hit].copy()
    conf = PP["conf"]

    def cat(r):
        if r.raw_best_iou > 0.5:
            return "A: raw IoU>0.5 and score>=conf, lost in post-proc (NMS/overlap/area/mask_thr)" if r.raw_score_best_match >= conf else "B: raw IoU>0.5 but score<conf (ranking/threshold)"
        if r.raw_best_iou > 0.3:
            return "C: raw best IoU 0.3-0.5 (localisation / mask quality)"
        return "D: no raw proposal with IoU>0.3 at score>=0.10 (detection miss)"
    fn["cause"] = fn.apply(cat, axis=1)
    t = fn.cause.value_counts().sort_index().rename("FN").reset_index()
    t["FN%"] = 100 * t.FN / len(fn)
    R_("### 5.2 FN: ranking issue vs detection issue (raw cache has score>=0.10; IoU vs binary raw masks)\n")
    R_(md_table(t, "{:.1f}") + "\n")
    b = fn[fn.raw_score_best_match < conf]
    R_(f"Category B score of best matching raw proposal (quantiles 10/25/50/75/90): " + " / ".join(f"{np.nanpercentile(b.raw_score_best_match, q):.3f}" for q in (10, 25, 50, 75, 90)) + ".\n")
    # marginal precision of low-score raw proposals
    pr_hits = []
    for s in val_stems:
        sc, RR = raw_g.get(s, (np.zeros(0), []))
        if not len(RR):
            continue
        areas = np.asarray(mask_util.area(RR), float)
        for v in views_of[s]:
            A = rle_by_view[v]
            iou = np.asarray(mask_util.iou(RR, A, [0] * len(A))).reshape(len(RR), len(A))
            for i in range(len(RR)):
                pr_hits.append((sc[i], areas[i], bool((iou[i] > .5).any())))
    ph_ = pd.DataFrame(pr_hits, columns=["score", "area", "hit"])
    ph_ = ph_[ph_.area >= PP["min_area"]]
    ph_["sb"] = cut(ph_.score, [0, .15, .2, .25, .35, .5, 1], ["0.10-0.15", "0.15-0.20", "0.20-0.25", "0.25-0.35", "0.35-0.50", ">0.5"])
    t = ph_.groupby("sb", observed=True).agg(n_pairs=("hit", "size"), hit_rate=("hit", "mean")).reset_index()
    R_("Hit rate (IoU>0.5 vs one annotator view; raw masks, before NMS) of raw proposals by score band -- marginal precision of what lowering conf would add. "
       "Break-even hit rate for adding a proposal is ~PQ/(2*SQ) = 0.34 (a hit adds IoU to the numerator and 0.5 to the denominator, a miss adds 0.5 to the denominator); rates here are optimistic because duplicates/overlapping raw masks are not de-duplicated:\n")
    R_(md_table(t, "{:.3f}") + "\n")
    fp = Pp[~Pp.hit]
    R_(f"Non-hit preds (FP pairs) by overlap: {100 * (fp.best_iou < 1e-9).mean():.1f}% touch no GT, {100 * ((fp.best_iou >= 1e-9) & (fp.best_iou <= 0.3)).mean():.1f}% IoU<=0.3, "
       f"{100 * (fp.best_iou > 0.3).mean():.1f}% IoU 0.3-0.5 (mask-quality FP).\n")

    nh = Pp[(~Pp.hit) & (Pp.best_iou > 0.1)].copy()
    nh["ratio_bin"] = cut(nh.area_ratio, [0, .5, .8, 1.25, 2, 1e9], ["<0.5 (pred much smaller)", "0.5-0.8", "0.8-1.25 (same size, misaligned)", "1.25-2", ">2 (pred much larger)"])
    th_ = Pp[Pp.hit]
    R_("\nNon-hit preds that overlap some GT (IoU>0.1) by pred/GT area ratio (shape-quality FP):\n")
    R_(md_table(nh.ratio_bin.value_counts().sort_index().rename("FP pairs").reset_index().rename(columns={"index": "area ratio"}), "{:.0f}") + "\n")
    R_(f"For reference TP pairs: median pred/GT area ratio {th_.area_ratio.median():.2f}, p10/p90 {th_.area_ratio.quantile(.1):.2f}/{th_.area_ratio.quantile(.9):.2f}.\n")

    # --- breakdowns
    R_("\n### 5.3 Breakdowns (GT side: recall/FN; pred side: precision/FP; quintile edges from val GT)\n")
    G["aq"] = quint(G.area)
    G["wq"] = quint(G.width)
    G["dq"] = cut(G.dprime, [-np.inf, 0.5, 1, 2, 3, np.inf], ["<0.5", "0.5-1", "1-2", "2-3", ">3"])
    G["rb"] = cut(G.rR, [0, .35, .6, .8, .92, 2], ["<0.35", "0.35-0.6", "0.6-0.8", "0.8-0.92", ">0.92"])
    Pp["aq"] = pd.cut(Pp.area, [-np.inf] + list(np.quantile(G.area, [.2, .4, .6, .8])) + [np.inf], labels=G.aq.cat.categories)
    Pp["wq"] = pd.cut(Pp.width, [-np.inf] + list(np.quantile(G.width.dropna(), [.2, .4, .6, .8])) + [np.inf], labels=G.wq.cat.categories)
    Pp["dq"] = cut(Pp.dprime, [-np.inf, 0.5, 1, 2, 3, np.inf], ["<0.5", "0.5-1", "1-2", "2-3", ">3"])
    Pp["rb"] = cut(Pp.rR, [0, .35, .6, .8, .92, 2], ["<0.35", "0.35-0.6", "0.6-0.8", "0.8-0.92", ">0.92"])
    for gc, pc, nm in [("aq", "aq", "area quintile (px@2048)"), ("wq", "wq", "width quintile (px@2048, 2A/P)"), ("dq", "dq", "contrast d' (ring-inside)/ring std"),
                       ("rb", "rb", "disk position r/R (mask centroid)"), ("site", "site", "site"), ("halo", "halo", "bright sky halo")]:
        R_(f"\n**{nm}**\n")
        R_(md_table(two_side(G, Pp, gc, pc, nm), "{:.3f}") + "\n")
    G["lb"] = cut(G.lat, [-2, -.5, -.15, .15, .5, 2], ["S>0.5", "S.15-.5", "|lat|<.15", "N.15-.5", "N>0.5"])
    G["ob"] = cut(np.abs(G.ang), [0, 22.5, 67.5, 90.1], ["horizontal(<22.5)", "diagonal", "vertical(>67.5)"])
    G["lenq"] = quint(G.spine_len)
    for col, nm in [("lb", "latitude proxy"), ("ob", "orientation of spine"), ("lenq", "spine length quintile (px)")]:
        t = G.groupby(col, observed=True).agg(n_gt=("hit", "size"), TP=("hit", "sum")).reset_index()
        t["recall"] = t.TP / t.n_gt
        R_(f"\n**GT recall by {nm}**\n")
        R_(md_table(t, "{:.3f}") + "\n")
    t = G.groupby("year").agg(n_gt=("hit", "size"), TP=("hit", "sum")).reset_index()
    t["recall"] = t.TP / t.n_gt
    pt = Pp.assign(year=Pp.stem.str[:4].astype(int)).groupby("year").agg(n_pred=("hit", "size"), hp=("hit", "sum")).reset_index()
    pt["precision"] = pt.hp / pt.n_pred
    R_("\n**by year**\n")
    R_(md_table(t.merge(pt, on="year").drop(columns="hp"), "{:.3f}") + "\n")
    G["cq"] = cut(G.crowd_n, [-1, 0, 2, 5, 100], ["0", "1-2", "3-5", ">5"])
    t = G.groupby("cq", observed=True).agg(n_gt=("hit", "size"), TP=("hit", "sum")).reset_index()
    t["recall"] = t.TP / t.n_gt
    R_("\n**GT recall by number of neighbouring filaments (bbox within 30 px)**\n")
    R_(md_table(t, "{:.3f}") + "\n")

    # --- annotator support
    R_("\n### 5.4 Annotator disagreement share of the errors (multi-annotator val stems)\n")
    mv = G[G.n_other >= 1]
    t = mv.groupby(["n_other", "support"]).agg(n_gt=("hit", "size"), TP=("hit", "sum")).reset_index()
    t["FN"] = t.n_gt - t.TP
    t["recall"] = t.TP / t.n_gt
    R_("GT recall by the number of OTHER annotator views that marked the same instance (IoU>0.5):\n")
    R_(md_table(t, "{:.3f}") + "\n")
    sup0 = mv[(mv.support == 0)]
    sup_any = mv[mv.support >= 1]
    n_fn_mv = (~mv.hit).sum()
    R_(f"Multi-annotator val GT: {len(mv)} instances, FN {n_fn_mv}. Of those FN: {(~sup0.hit).sum()} ({100 * (~sup0.hit).sum() / n_fn_mv:.1f}%) were marked by NO other annotator "
       f"(idiosyncratic), {(~sup_any.hit).sum()} ({100 * (~sup_any.hit).sum() / n_fn_mv:.1f}%) had support from >=1 other annotator. "
       f"Recall: support 0 = {sup0.hit.mean():.3f}, support>=1 = {sup_any.hit.mean():.3f}. Single-annotator stems: {(G.n_other == 0).sum()} GT, {(~G[G.n_other == 0].hit).sum()} FN.\n")
    pm = Pp[Pp.n_views >= 2]
    fpm = pm[~pm.hit]
    R_(f"Multi-annotator val preds: {len(pm)} (view,pred) pairs, FP pairs {len(fpm)}. Of those FP pairs, {(fpm.views_hit >= 1).sum()} ({100 * (fpm.views_hit >= 1).mean():.1f}%) are preds that "
       f"another annotator did mark (disagreement-only FP); {(fpm.views_hit == 0).sum()} ({100 * (fpm.views_hit == 0).mean():.1f}%) are unsupported by every annotator.\n")
    # counterfactual ceiling
    tp_, fp_, fn_ = ev.tp, ev.fp, ev.fn
    siou = ev.sq * ev.tp
    fp_adj = fp_ - (fpm.views_hit >= 1).sum()
    fn_adj = fn_ - (~sup0.hit).sum()
    R_(f"Counterfactual (report only): if every FN that no other annotator marked and every FP that another annotator marked were NOT counted, PQ would be "
       f"{siou / (tp_ + .5 * fp_adj + .5 * fn_adj):.4f} (from {ev.pq:.4f}); removing only the unmarked-by-others FN gives {siou / (tp_ + .5 * fp_ + .5 * fn_adj):.4f}, only the other-annotator-FP gives "
       f"{siou / (tp_ + .5 * fp_adj + .5 * fn_):.4f}. Remaining errors = true model misses/hallucinations plus mask-quality (IoU<0.5) losses.\n")
    # recall by annotator
    t = G.groupby("annotator").agg(n_gt=("hit", "size"), recall=("hit", "mean")).reset_index()
    R_("\nVal recall per annotator ID (n_gt>=40):\n")
    R_(md_table(t[t.n_gt >= 40].sort_values("recall"), "{:.3f}") + "\n")

    # --- fragmentation / merge
    R_("\n### 5.5 Fragmentation and merges (final preds are mutually disjoint)\n")
    fr = G[G.n_frag >= 2]
    fr_fn = fr[~fr.hit]
    R_(f"GT instances covered by >=2 preds that lie >=50% inside them (fragmentation): {len(fr)} ({100 * len(fr) / len(G):.1f}% of GT); of these {len(fr_fn)} are FN "
       f"({100 * len(fr_fn) / (~G.hit).sum():.1f}% of all FN) and {len(fr) - len(fr_fn)} still TP (one fragment > IoU 0.5). "
       f"Union of the fragments of the FN ones would have IoU>0.5 in {int((fr_fn.frag_union_iou > 0.5).sum())} cases (median union IoU {fr_fn.frag_union_iou.median():.2f}).\n")
    mg = Pp[Pp.n_gt_covered >= 2]
    R_(f"Preds covering >=50% of >=2 GT in the same view (merges): {len(mg)} pairs ({100 * len(mg) / len(Pp):.1f}% of preds); {int((~mg.hit).sum())} of them are FP. "
       f"Metric diagnostics: splits {ev.splits}, merges {ev.merges} (pairs with IoU>0 against >=2).\n")
    inside = Pp[(~Pp.hit) & (Pp.inside >= 0.5)]
    R_(f"FP pairs lying >=50% inside a GT without hitting it (fragments/partial masks): {len(inside)} ({100 * len(inside) / len(fp):.1f}% of FP).\n")
    under = G[(~G.hit) & (G.cov_by_pred >= 0.5)]
    R_(f"FN GT that are >=50% covered by preds but with IoU<=0.5 (fragmented or over-grown masks): {len(under)} ({100 * len(under) / (~G.hit).sum():.1f}% of FN).\n")

    # ============================================================ Q6 copy-paste
    R_("\n## Q6. Copy-paste suitability\n")
    cc = ann.dropna(subset=["dprime"])
    R_(md_table(pd.DataFrame([
        dict(stat="filament contrast d' median", v=cc.dprime.median()),
        dict(stat="share with d'>=1.5 (clearly visible dark object) %", v=100 * (cc.dprime >= 1.5).mean()),
        dict(stat="share with d'<0.5 (barely distinguishable) %", v=100 * (cc.dprime < .5).mean()),
        dict(stat="median ring std (local background texture, grey levels)", v=cc.ring_std.median()),
        dict(stat="median filament contrast (grey levels)", v=cc.contrast.median()),
        dict(stat="share isolated (no neighbour bbox within 30px) %", v=100 * (ann.crowd_n == 0).mean()),
        dict(stat="median solidity (area/convex hull)", v=ann.solidity.median()),
        dict(stat="median bbox fill (mask area/bbox area)", v=ann.bbox_fill.median()),
        dict(stat="share with width<6px @2048 (pasting needs soft edges) %", v=100 * (ann.width < 6).mean()),
        dict(stat="share on limb (r/R>0.9) %", v=100 * (ann.rR > .9).mean())]), "{:.2f}") + "\n")
    R_("`copy_paste_mode='flip'` (ultralytics default) pastes the horizontally mirrored (about the image centre) instances of the same image onto it. "
       "Statistics of the mirrored-mask layer over all train views:\n")
    R_(md_table(pd.DataFrame([
        dict(stat="mean fraction of mirrored mask pixels landing on existing GT pixels (overlap)", v=100 * mirror.overlap.mean()),
        dict(stat="share of views with >10% such overlap", v=100 * (mirror.overlap > .1).mean()),
        dict(stat="mean fraction of mirrored pixels outside the disk (r>0.98R) (unphysical)", v=100 * mirror.off_disk.mean()),
        dict(stat="share of views with >5% mirrored pixels off-disk", v=100 * (mirror.off_disk > .05).mean()),
        dict(stat="mean mask coverage per view %", v=100 * mirror.cover.mean())]), "{:.2f}") + "\n")
    R_(f"Annotations per view are {vcount.mean():.1f} on average: the flipped layer would add roughly as many filaments again at positions that mirror the original ones "
       f"(a view's disk centre is offset by only {np.hypot(tr.cx - 1024, tr.cy - 1024).mean():.0f} px from the frame centre, so the mirror stays on the disk).\n")

    # ---- seed of figure for errors
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8), dpi=80)
    t = G.groupby("aq", observed=True).hit.mean()
    ax[0].bar(range(len(t)), t.values); ax[0].set_xticks(range(len(t)), [f"Q{i + 1}" for i in range(len(t))]); ax[0].set_title("recall by area quintile")
    t = G.groupby("dq", observed=True).hit.mean()
    ax[1].bar(range(len(t)), t.values); ax[1].set_xticks(range(len(t)), t.index.astype(str)); ax[1].set_title("recall by contrast d'")
    t = mv.groupby("support").hit.mean()
    ax[2].bar(range(len(t)), t.values); ax[2].set_xticks(range(len(t)), [f"{i}" for i in t.index]); ax[2].set_title("recall by annotator support")
    fig.tight_layout(); fig.savefig(FIGS / "q5_errors.png"); plt.close(fig)

    # save per-instance tables as CSV next to the figs (small)
    img.round(3).to_csv(FIGS / "image_stats.csv", index=False)

    dt = time.time() - t0
    head = f"<!-- generated by scripts/analyze_data_v2.py in {dt:.0f}s -->\n"
    recs = RECS_MD.read_text() + "\n\n---\n\n" if RECS_MD.exists() else "# Data and error analysis v2\n\n"
    OUT_MD.write_text(head + recs + "\n".join(rep) + f"\n\nFigures: docs/figs_v2/q1_geometry.png, q2_orientation.png, q3_photometry.png, q4_instances.png, q5_errors.png. Runtime {dt:.0f}s.\n")
    print(f"wrote {OUT_MD} in {dt:.0f}s")


if __name__ == "__main__":
    main()
