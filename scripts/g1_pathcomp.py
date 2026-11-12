#!/usr/bin/env python
"""G1: image-guided, orientation-aware path completion of final instances (after NMS + overlap resolution).

For every selected instance, each of its two ends is extended along the local filament direction by a greedy
cone search on a darkness-contrast map: c = (local background - image) / local background, with background a
large median. A step continues while c at the next point stays >= alpha * (median c inside the instance); short
sub-threshold stretches (gaps) up to `gap` px are bridged, trailing gaps are trimmed. The path is painted with
the instance's mean half-width, only on unclaimed pixels. Merge safety: a path stops before it comes within
`clear` px of another instance, and outside the solar disc (eroded). Original instances are never shrunk,
removed or re-scored, and only the image is used (no GT) at inference.

Usage: g1_pathcomp.py [--cache artifacts/rescore/val_blend01.pkl] [--grid] [--split val|test --out CSV]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from filseg.infer import load_cache  # noqa: E402
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows  # noqa: E402
from filseg.refine import IMG_DIR, IMG_DIR_TEST  # noqa: E402
from filseg.rle import decode_rle, encode_mask  # noqa: E402


def contrast_map(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, 31), img.shape[::-1], interpolation=cv2.INTER_LINEAR).astype(np.float32)
    sm = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
    c = (bg - sm) / np.maximum(bg, 10.0)
    disc = cv2.erode((cv2.GaussianBlur(img, (0, 0), 4) > 25).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    return c, disc


def ends_and_dirs(m: np.ndarray, near: int = 25):
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mu = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
    ax = vt[0]
    proj = (pts - mu) @ ax
    length = float(proj.max() - proj.min()) + 1
    out = []
    for sgn, idx in ((-1, proj.argmin()), (1, proj.argmax())):
        e = pts[idx]
        loc = pts[np.linalg.norm(pts - e, axis=1) < near]
        if len(loc) >= 5:
            _, _, v2 = np.linalg.svd(loc - loc.mean(0), full_matrices=False)
            d = v2[0]
        else:
            d = ax
        if np.dot(d, ax * sgn) < 0:
            d = -d
        out.append((e, d / np.linalg.norm(d)))
    return out, length


def trace(e, d, c, ok, thr, lmax, gap, step=3.0, theta=25.0):
    angs = np.deg2rad(np.linspace(-theta, theta, 7))
    H, W = c.shape
    pos, path, last_on, run_gap, tot = e.copy(), [], -1, 0.0, 0.0
    while tot < lmax:
        best, bp, bd = -9.0, None, None
        for a in angs:
            ca, sa = np.cos(a), np.sin(a)
            nd = np.array([d[0] * ca - d[1] * sa, d[0] * sa + d[1] * ca])
            p = pos + step * nd
            x, y = int(round(p[0])), int(round(p[1]))
            if not (0 <= x < W and 0 <= y < H) or not ok[y, x]:
                continue
            if c[y, x] > best:
                best, bp, bd = c[y, x], p, nd
        if bp is None:
            break
        pos, tot = bp, tot + step
        d = 0.7 * d + 0.3 * bd
        d /= np.linalg.norm(d)
        path.append(pos.copy())
        if best >= thr:
            last_on, run_gap = len(path) - 1, 0.0
        else:
            run_gap += step
            if run_gap > gap:
                break
    return path[: last_on + 1]


def complete_stem(masks: list, c, disc, alpha, lmax_frac, lmax_abs, gap, clear=4):
    lab = np.zeros(c.shape, np.int32)
    for k, m in enumerate(masks, 1):
        lab[m] = k
    out = [m.copy() for m in masks]
    ker = np.ones((2 * clear + 1, 2 * clear + 1), np.uint8)
    for k, m in enumerate(masks, 1):
        others = cv2.dilate(((lab > 0) & (lab != k)).astype(np.uint8), ker) > 0
        ok = disc & ~others
        mc = float(np.median(c[m]))
        if mc <= 0.01:
            continue
        ends, length = ends_and_dirs(m)
        r = max(1, int(round(m.sum() / length / 2)))
        lmax = min(lmax_abs, lmax_frac * length)
        canvas = np.zeros(c.shape, np.uint8)
        for e, d in ends:
            path = trace(e, d, c, ok, alpha * mc, lmax, gap)
            if len(path) < 2:
                continue
            pts = np.round(np.vstack([e[None], np.array(path)])).astype(np.int32)
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, 1, thickness=2 * r + 1)
        add = (canvas > 0) & (lab == 0) & ok
        out[k - 1] = m | add
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="artifacts/rescore/val_blend01.pkl")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--mask-thr", type=float, default=0.35)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--lmax-frac", type=float, default=0.5)
    ap.add_argument("--lmax-abs", type=float, default=80)
    ap.add_argument("--gap", type=float, default=9)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--stems", default="", help="comma list or json file with a list (default: split stems)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    raw = load_cache(a.cache)
    g, s = group_by_stem(raw), group_soft_by_stem(raw)
    if a.stems:
        stems = json.load(open(a.stems)) if a.stems.endswith(".json") else a.stems.split(",")
    elif a.split == "val":
        stems = sorted(json.load(open("data/splits/split_v1.json"))["val"])
    else:
        stems = sorted(g)
    rows = postprocess_to_rows(g, PPParams(a.conf, 100, 20, 0.3, 0, "square", a.mask_thr), stems, s)
    by = {}
    for k, r in rows:
        by.setdefault(k.rsplit("_", 1)[0], []).append(r)
    cfgs = (list(itertools.product([0.5, 0.7, 0.9], [0.3, 0.6], [40, 120], [0, 9])) if a.grid
            else [(a.alpha, a.lmax_frac, a.lmax_abs, a.gap)])
    new = {cf: [] for cf in cfgs}
    idir = IMG_DIR if a.split == "val" else IMG_DIR_TEST
    for i, st in enumerate(sorted(by)):
        img = cv2.imread(f"{idir}/{st}.jpeg", cv2.IMREAD_GRAYSCALE)
        c, disc = contrast_map(img)
        masks = [decode_rle(r).astype(bool) for r in by[st]]
        for cf in cfgs:
            out = complete_stem(masks, c, disc, *cf)
            new[cf] += [(f"{st}_{j}", encode_mask(m.astype(np.uint8))) for j, m in enumerate(out, 1)]
        if i % 20 == 0:
            print(f"  {i}/{len(by)}", flush=True)
    if a.split == "val":
        from decode_diag import GT_JSON, evaluate
        from filseg.gt import build_gt_df
        gt = build_gt_df(GT_JSON, set(stems))
        b = evaluate(gt, rows)
        print(f"baseline PQ {b.pq:.4f} TP {b.tp} FP {b.fp} FN {b.fn}")
        res = []
        for cf in cfgs:
            r = evaluate(gt, new[cf])
            res.append(dict(alpha=cf[0], lmax_frac=cf[1], lmax_abs=cf[2], gap=cf[3], pq=r.pq, tp=r.tp, fp=r.fp, fn=r.fn))
            print(f"alpha {cf[0]} lmax {cf[1]}x/{cf[2]}px gap {cf[3]}: PQ {r.pq:.4f} ({r.pq - b.pq:+.4f}) "
                  f"TP {r.tp} FP {r.fp} FN {r.fn}", flush=True)
        Path("artifacts/g1").mkdir(parents=True, exist_ok=True)
        json.dump(dict(baseline=b.pq, results=res), open(f"artifacts/g1/grid_{Path(a.cache).stem}.json", "w"), indent=1)
    if a.out:
        import pandas as pd
        pd.DataFrame(new[cfgs[0]], columns=["filament_id", "segmentation_rle"]).to_csv(a.out, index=False)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
