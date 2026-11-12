#!/usr/bin/env python
"""G1 (spec version): skeleton-endpoint path completion with a Dijkstra ridge search, after selection + overlap
resolution. Image-only at inference.

Per selected instance: morphological skeleton -> endpoints (skeleton pixels with one neighbour; the two farthest
apart are kept, so small branches are ignored) -> outward tangent (endpoint minus the mean of skeleton pixels
within 15 px). In a ROI of radius L ahead of the endpoint (half-plane along the tangent) a Dijkstra search runs on
cost = 1 + w_dark * relu(1 - c / c_inst) + w_ang * (1 - cos(angle to tangent)), where c is the background-
normalised darkness (g1_pathcomp.contrast_map) and c_inst the instance's median darkness; pixels of other
instances (dilated 4 px), off-disc pixels and very dark blobs (c > 3 c_inst, sunspot-like) are forbidden. The
cheapest path to the ROI rim is trimmed from its far end until its last pixel has c >= alpha * c_inst; it is kept
if >= 8 px and its mean darkness >= alpha * c_inst. Growth is painted with the instance's mean width, on unclaimed
pixels only; the original masks are never shrunk and the instance count/order is asserted unchanged.
Usage: g1_dijkstra.py [--cache ...] [--grid] [--L 64 --alpha 0.8] [--overlays N]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_pathcomp import contrast_map  # noqa: E402
from filseg.infer import load_cache  # noqa: E402
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows  # noqa: E402
from filseg.refine import IMG_DIR, mask_bbox  # noqa: E402
from filseg.rle import decode_rle, encode_mask  # noqa: E402

W_DARK, W_ANG = 4.0, 2.0


def skeleton_crop(m: np.ndarray):
    y0, y1, x0, x1 = mask_bbox(m)
    t = torch.from_numpy(np.pad(m[y0:y1, x0:x1], 2).astype(np.float32))[None, None]
    er = lambda x: -torch.nn.functional.max_pool2d(-x, 3, 1, 1)  # noqa: E731
    di = lambda x: torch.nn.functional.max_pool2d(x, 3, 1, 1)  # noqa: E731
    img, sk = t, torch.relu(t - di(er(t)))
    for _ in range(40):
        img = er(img)
        if img.sum() == 0:
            break
        d = torch.relu(img - di(er(img)))
        sk = sk + torch.relu(d - sk * d)
    sk = (sk[0, 0] > 0).numpy()
    ys, xs = np.nonzero(sk)
    return np.stack([xs + x0 - 2, ys + y0 - 2], 1), sk, (x0 - 2, y0 - 2)


def endpoints(m: np.ndarray):
    pts, sk, (ox, oy) = skeleton_crop(m)
    if len(pts) < 5:
        return [], len(pts)
    nb = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - 1
    ys, xs = np.nonzero(sk & (nb == 1))
    cand = np.stack([xs + ox, ys + oy], 1).astype(np.float32)
    if len(cand) < 2:  # loop or blob: fall back to the extreme skeleton points along the principal axis
        c = pts - pts.mean(0)
        ax = np.linalg.svd(c, full_matrices=False)[2][0]
        pr = c @ ax
        cand = pts[[pr.argmin(), pr.argmax()]].astype(np.float32)
    dd = np.linalg.norm(cand[:, None] - cand[None], axis=2)
    i, j = np.unravel_index(dd.argmax(), dd.shape)
    out = []
    for e in (cand[i], cand[j]):
        near = pts[np.linalg.norm(pts - e, axis=1) < 15]
        t = e - near.mean(0)
        if np.linalg.norm(t) < 1e-3:
            continue
        out.append((e, t / np.linalg.norm(t)))
    return out, len(pts)


def grid_graph(cost: np.ndarray, ok: np.ndarray):
    h, w = cost.shape
    idx = np.arange(h * w).reshape(h, w)
    rs, cs, ws = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = idx[max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)]
        b = idx[max(0, dy) :, max(0, dx) :][: a.shape[0], : a.shape[1]] if dx >= 0 else idx[dy:, : w + dx][: a.shape[0], : a.shape[1]]
        okk = ok.ravel()[a.ravel()] & ok.ravel()[b.ravel()]
        wt = 0.5 * (cost.ravel()[a.ravel()] + cost.ravel()[b.ravel()]) * (np.sqrt(2) if dx and dy else 1.0)
        rs += [a.ravel()[okk], b.ravel()[okk]]
        cs += [b.ravel()[okk], a.ravel()[okk]]
        ws += [wt[okk], wt[okk]]
    return coo_matrix((np.concatenate(ws), (np.concatenate(rs), np.concatenate(cs))), shape=(h * w, h * w)).tocsr()


def extend(e, t, c, ok_full, c_inst, L, alpha):
    H, W = c.shape
    x0, y0 = int(max(0, e[0] - L)), int(max(0, e[1] - L))
    x1, y1 = int(min(W, e[0] + L + 1)), int(min(H, e[1] + L + 1))
    cc = c[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    vx, vy = xx - e[0], yy - e[1]
    r = np.hypot(vx, vy)
    cosang = (vx * t[0] + vy * t[1]) / np.maximum(r, 1e-6)
    ex, ey = int(round(e[0])) - x0, int(round(e[1])) - y0
    ok = ok_full[y0:y1, x0:x1] & (r <= L) & ((cosang > 0.3) | (r < 2)) & (cc < 3 * c_inst)
    if not (0 <= ey < ok.shape[0] and 0 <= ex < ok.shape[1]):
        return None
    ok[ey, ex] = True
    cost = 1 + W_DARK * np.maximum(0, 1 - cc / c_inst) + W_ANG * (1 - np.clip(cosang, -1, 1))
    G = grid_graph(cost, ok)
    src = ey * ok.shape[1] + ex
    dist, pred = dijkstra(G, indices=src, return_predecessors=True)
    rim = ok & (r >= L - 2) & np.isfinite(dist.reshape(ok.shape))
    if not rim.any():
        reach = ok & np.isfinite(dist.reshape(ok.shape)) & (r > 4)
        if not reach.any():
            return None
        tgt = int(np.argmax(np.where(reach, r, -1)))
    else:
        tgt = int(np.argmin(np.where(rim, dist.reshape(ok.shape), np.inf)))
    path = []
    k = tgt
    while k != src and k >= 0:
        path.append(k)
        k = pred[k]
    path = path[::-1]
    if not path:
        return None
    py, px = np.divmod(np.array(path), ok.shape[1])
    v = cc[py, px]
    good = np.flatnonzero(v >= alpha * c_inst)
    if len(good) == 0:
        return None
    end = good[-1] + 1
    py, px, v = py[:end], px[:end], v[:end]
    if end < 8 or v.mean() < alpha * c_inst:
        return None
    return np.stack([px + x0, py + y0], 1)


def complete_stem(masks, c, disc, L, alpha, clear=4):
    lab = np.zeros(c.shape, np.int32)
    for k, m in enumerate(masks, 1):
        lab[m] = k
    out, changed = [m.copy() for m in masks], 0
    ker = np.ones((2 * clear + 1, 2 * clear + 1), np.uint8)
    for k, m in enumerate(masks, 1):
        c_inst = float(np.median(c[m]))
        if c_inst <= 0.01:
            continue
        ends, slen = endpoints(m)
        if not ends:
            continue
        others = cv2.dilate(((lab > 0) & (lab != k)).astype(np.uint8), ker) > 0
        ok = disc & ~others
        width = max(1, int(round(m.sum() / max(slen, 1))))
        canvas = np.zeros(c.shape, np.uint8)
        for e, t in ends:
            p = extend(e, t, c, ok, c_inst, L, alpha)
            if p is not None:
                cv2.polylines(canvas, [p.reshape(-1, 1, 2).astype(np.int32)], False, 1, thickness=width)
        add = (canvas > 0) & (lab == 0) & ok
        if add.any():
            out[k - 1] = m | add
            changed += 1
    assert len(out) == len(masks)
    return out, changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="artifacts/rescore/val_blend01.pkl")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--mask-thr", type=float, default=0.35)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--grid", action="store_true", help="6 settings: L 32/64/96 x alpha 0.8/1.0")
    ap.add_argument("--overlays", type=int, default=0, help="save N before/after overlays for the first setting")
    ap.add_argument("--out-dir", default="artifacts/g1_dijkstra")
    a = ap.parse_args()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    stems = sorted(json.load(open("data/splits/split_v1.json"))["val"])
    raw = load_cache(a.cache)
    g, s = group_by_stem(raw), group_soft_by_stem(raw)
    rows = postprocess_to_rows(g, PPParams(a.conf, 100, 20, 0.3, 0, "square", a.mask_thr), stems, s)
    by = {}
    for k, r in rows:
        by.setdefault(k.rsplit("_", 1)[0], []).append(r)
    cfgs = list(itertools.product([32, 64, 96], [0.8, 1.0])) if a.grid else [(a.L, a.alpha)]
    new = {cf: [] for cf in cfgs}
    nchg = {cf: 0 for cf in cfgs}
    tcfg = {cf: 0.0 for cf in cfgs}
    ov = 0
    for i, st in enumerate(sorted(by)):
        img = cv2.imread(f"{IMG_DIR}/{st}.jpeg", cv2.IMREAD_GRAYSCALE)
        c, disc = contrast_map(img)
        masks = [decode_rle(r).astype(bool) for r in by[st]]
        for cf in cfgs:
            t0 = time.time()
            out, ch = complete_stem(masks, c, disc, *cf)
            tcfg[cf] += time.time() - t0
            nchg[cf] += ch
            new[cf] += [(f"{st}_{j}", encode_mask(m.astype(np.uint8))) for j, m in enumerate(out, 1)]
            if cf == cfgs[0] and ov < a.overlays:
                for m0, m1 in zip(masks, out):
                    if ov >= a.overlays or (m1 & ~m0).sum() == 0:
                        continue
                    y0, y1, x0, x1 = mask_bbox(m1)
                    y0, x0 = max(0, y0 - 40), max(0, x0 - 40)
                    y1, x1 = y1 + 40, x1 + 40
                    rgb = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
                    rgb[m0[y0:y1, x0:x1]] = (0.5 * rgb[m0[y0:y1, x0:x1]] + [0, 120, 0]).astype(np.uint8)
                    rgb[(m1 & ~m0)[y0:y1, x0:x1]] = (0, 0, 255)
                    cv2.imwrite(str(od / f"overlay_{ov:02d}_{st}.png"), rgb)
                    ov += 1
        if i % 20 == 0:
            print(f"  {i}/{len(by)}", flush=True)
    from decode_diag import GT_JSON, evaluate
    from filseg.gt import build_gt_df
    gt = build_gt_df(GT_JSON, set(stems))
    b = evaluate(gt, rows)
    print(f"baseline PQ {b.pq:.4f} SQ {b.sq:.4f} RQ {b.rq:.4f} TP {b.tp} FP {b.fp} FN {b.fn}")
    res = []
    for cf in cfgs:
        r = evaluate(gt, new[cf])
        res.append(dict(L=cf[0], alpha=cf[1], pq=r.pq, sq=r.sq, rq=r.rq, tp=r.tp, fp=r.fp, fn=r.fn, changed=nchg[cf],
                        n_inst=len(rows), sec_per_image=tcfg[cf] / len(by)))
        print(f"L {cf[0]} alpha {cf[1]}: PQ {r.pq:.4f} ({r.pq - b.pq:+.4f}) SQ {r.sq:.4f} RQ {r.rq:.4f} TP {r.tp} "
              f"FP {r.fp} FN {r.fn} | changed {nchg[cf]}/{len(rows)} | {tcfg[cf] / len(by):.2f} s/img", flush=True)
    json.dump(dict(baseline=dict(pq=b.pq, sq=b.sq, rq=b.rq, tp=b.tp, fp=b.fp, fn=b.fn), results=res),
              open(od / "grid.json", "w"), indent=1)


if __name__ == "__main__":
    main()
