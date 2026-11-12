#!/usr/bin/env python
"""G0 (spec completion): per-view table, near-miss CSV, match multiplicity, and offline opportunity ceilings.

Analysis only; GT is used to build the oracle edits, never at inference. Every ceiling is an exact
re-evaluation of pooled per-view PQ on frozen copies of the baseline output.
  per-view: n_gt, n_pred, TP, FP, FN, N = sum IoU_TP, D = TP + FP/2 + FN/2, PQ
  near_misses.csv: every FN with best IoU in [0.30, 0.50], category, coverage/precision, skeleton length and width
  multiplicity: predictions with IoU > .5 to more than one GT of a view, GT with more than one such prediction
  (i)  bounded continuation: grow a fragment's prediction geodesically inside its GT by L = 32/64/96 px
  (ii) candidate union: union of the selected prediction with pre-NMS candidates (score >= .01) that lie mostly
       inside the GT (precision >= .75)
  (iv) skeleton length and mean width of prediction vs GT for fragment near-misses
Usage: g0_opportunity.py [--cache artifacts/rescore/val_blend01.pkl --out-dir reports]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_diag import GT_JSON, evaluate, iou  # noqa: E402
from g0_audit import classify  # noqa: E402
from filseg.gt import build_gt_df  # noqa: E402
from filseg.infer import load_cache  # noqa: E402
from filseg.postprocess import (PPParams, _resolve_rles, group_by_stem, group_soft_by_stem,  # noqa: E402
                                postprocess_to_rows, transform_stem)
from filseg.refine import mask_bbox  # noqa: E402
from filseg.rle import decode_rle, encode_mask, to_rle_obj  # noqa: E402


def skel_len(m: np.ndarray) -> int:
    """Pixel count of the morphological skeleton (erosion-based, same as the G2 loss target, untubed)."""
    b = mask_bbox(m)
    if b is None:
        return 0
    y0, y1, x0, x1 = b
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
    return int((sk > 0).sum())


def geodesic_grow(p: np.ndarray, g: np.ndarray, steps: int) -> np.ndarray:
    y0, y1, x0, x1 = mask_bbox(p | g)
    pc, gc = p[y0:y1, x0:x1].astype(np.uint8), (p | g)[y0:y1, x0:x1].astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    for _ in range(steps):
        nxt = cv2.dilate(pc, k) & gc
        if (nxt == pc).all():
            break
        pc = nxt
    out = p.copy()
    out[y0:y1, x0:x1] |= pc.astype(bool)
    return out


def apply_mods(pred: dict, mods: dict, drop: dict | None = None) -> list:
    rows = []
    for st, lst in pred.items():
        if st not in mods:
            rows += lst
            continue
        rl, sc = [], []
        for j, (k, r) in enumerate(lst):
            if drop and j in drop.get(st, set()):
                continue
            if j in mods[st]:
                rl.append(encode_mask(mods[st][j].astype(np.uint8)))
                sc.append(2.0)  # edited masks win collisions (oracle ceiling)
            else:
                rl.append(r)
                sc.append(1.0 - j * 1e-3)
        masks, _ = _resolve_rles(rl, np.asarray(sc))
        rows += [(f"{st}_{i}", encode_mask(m)) for i, m in enumerate(masks, 1) if m is not None and m.sum() >= 100]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="artifacts/rescore/val_blend01.pkl")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--mask-thr", type=float, default=0.35)
    ap.add_argument("--out-dir", default="reports")
    a = ap.parse_args()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    stems = sorted(json.load(open("data/splits/split_v1.json"))["val"])
    gt = build_gt_df(GT_JSON, set(stems))
    fid = gt["filament_id"].str.rsplit("_", n=1).str[0]
    views = defaultdict(lambda: defaultdict(list))
    for v, r in zip(fid, gt["segmentation_rle"]):
        views[v.split("-", 1)[1]][v].append(to_rle_obj(r))
    raw = load_cache(a.cache)
    g, s = group_by_stem(raw), group_soft_by_stem(raw)
    rows = postprocess_to_rows(g, PPParams(a.conf, 100, 20, 0.3, 0, "square", a.mask_thr), stems, s)
    base = evaluate(gt, rows)
    pred = defaultdict(list)
    for k, r in rows:
        pred[k.rsplit("_", 1)[0]].append((k, r))

    # per-view table + multiplicity + near misses
    pv, nm = [], []
    multi_pred = multi_gt = 0
    for st, vs in views.items():
        P = [to_rle_obj(r) for _, r in pred.get(st, [])]
        for v, Gs in vs.items():
            M = iou(Gs, P) if P else np.zeros((len(Gs), 0))
            hit = M > 0.5
            multi_pred += int((hit.sum(0) > 1).sum())
            multi_gt += int((hit.sum(1) > 1).sum())
            tp_iou = M[hit]
            tp, fn = int(hit.any(1).sum()), int((~hit.any(1)).sum())
            fp = int((~hit.any(0)).sum()) if P else 0
            # scorer counts every hit pair as TP; with disjoint masks a GT has at most one hit
            n = float(tp_iou.sum())
            d = int(hit.sum()) + 0.5 * fp + 0.5 * fn
            pv.append(dict(view=v, stem=st, instrument=st[-2:], date=st[:8], n_gt=len(Gs), n_pred=len(P),
                           tp=int(hit.sum()), fp=fp, fn=fn, N=n, D=d, pq=n / d if d else float("nan")))
            for gi, G in enumerate(Gs):
                if hit[gi].any() or not P:
                    continue
                j = int(M[gi].argmax())
                b = float(M[gi, j])
                if not (0.30 <= b <= 0.50):
                    continue
                cat, _ = classify(G, P, j, Gs)
                Gm = mu.decode(G).astype(bool)
                Pm = decode_rle(pred[st][j][1]).astype(bool)
                inter = int((Gm & Pm).sum())
                sg, sp = skel_len(Gm), skel_len(Pm)
                nm.append(dict(view=v, stem=st, gt_idx=gi, pred_idx=j, best_iou=b, category=cat,
                               coverage=inter / Gm.sum(), precision=inter / Pm.sum(), gt_area=int(Gm.sum()),
                               pred_area=int(Pm.sum()), gt_skel_len=sg, pred_skel_len=sp,
                               gt_width=Gm.sum() / max(sg, 1), pred_width=Pm.sum() / max(sp, 1)))
    pvd, nmd = pd.DataFrame(pv), pd.DataFrame(nm)
    pvd.to_csv(od / "per_view.csv", index=False)
    nmd.to_csv(od / "near_misses.csv", index=False)
    N, D = pvd.N.sum(), pvd.D.sum()
    gt_all, fn_all, M_, F_ = int(pvd.n_gt.sum()), int(pvd.fn.sum()), len(nmd), int((nmd.category == "fragment").sum())
    res["baseline"] = dict(pq_scorer=base.pq, tp=base.tp, fp=base.fp, fn=base.fn, pq_from_per_view=N / D, N=N, D=D)
    res["counts"] = dict(images=len(views), views=len(pvd), dates=int(pvd.date.nunique()),
                         instruments=pvd.groupby("instrument").stem.nunique().to_dict(),
                         GT_all=gt_all, FN_all=fn_all, M=M_, F=F_, M_over_FN=M_ / fn_all, M_over_GT=M_ / gt_all,
                         F_over_M=F_ / M_, F_over_GT=F_ / gt_all,
                         categories=nmd.category.value_counts().to_dict(),
                         preds_hitting_2plus_gt=multi_pred, gt_hit_by_2plus_preds=multi_gt)
    fr = nmd[nmd.category == "fragment"]
    res["iv_fragment_shape"] = dict(
        median_len_ratio_pred_over_gt=float((fr.pred_skel_len / fr.gt_skel_len).median()),
        median_width_ratio_pred_over_gt=float((fr.pred_width / fr.gt_width).median()),
        median_gt_skel_len=float(fr.gt_skel_len.median()), median_missing_len=float((fr.gt_skel_len - fr.pred_skel_len).median()))
    print(json.dumps(res, indent=1, default=float))

    # (i) bounded geodesic continuation of fragments inside their GT
    res["i_continuation"] = {}
    for L in (32, 64, 96):
        mods = defaultdict(dict)
        for x in fr.itertuples():
            G = mu.decode(views[x.stem][x.view][x.gt_idx]).astype(bool)
            cur = mods[x.stem].get(x.pred_idx, decode_rle(pred[x.stem][x.pred_idx][1]).astype(bool))
            mods[x.stem][x.pred_idx] = geodesic_grow(cur, G, L)
        r = evaluate(gt, apply_mods(pred, mods))
        # how many of the edited fragments now clear IoU .5 in their own view
        res["i_continuation"][L] = dict(pq=r.pq, gain=r.pq - base.pq, tp=r.tp, fp=r.fp, fn=r.fn)
        print(f"(i) continuation {L} px: PQ {r.pq:.4f} ({r.pq - base.pq:+.4f}) TP {r.tp} FP {r.fp} FN {r.fn}", flush=True)

    # (ii) union with pre-NMS candidates lying inside the GT
    mods, used = defaultdict(dict), 0
    for x in fr.itertuples():
        sc, rl, ar = transform_stem(*g[x.stem], s.get(x.stem), 0, "square", a.mask_thr, 0.01)
        G = views[x.stem][x.view][x.gt_idx]
        Gm = mu.decode(G).astype(bool)
        cur = mods[x.stem].get(x.pred_idx, decode_rle(pred[x.stem][x.pred_idx][1]).astype(bool))
        for r_ in rl:
            o = to_rle_obj(r_)
            inter = mu.area(mu.merge([o, G], intersect=True))
            if inter / max(mu.area(o), 1) >= 0.75:
                cur = cur | mu.decode(o).astype(bool)
                used += 1
        mods[x.stem][x.pred_idx] = cur
    r = evaluate(gt, apply_mods(pred, mods))
    res["ii_candidate_union"] = dict(pq=r.pq, gain=r.pq - base.pq, tp=r.tp, fp=r.fp, fn=r.fn, candidates_used=used)
    print(f"(ii) pre-NMS candidate union: PQ {r.pq:.4f} ({r.pq - base.pq:+.4f}) TP {r.tp} FP {r.fp} FN {r.fn}, "
          f"{used} candidate masks used", flush=True)
    json.dump(res, open(od / "geometry_audit_opportunity.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
