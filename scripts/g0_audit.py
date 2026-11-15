#!/usr/bin/env python
"""G0 geometry / metric audit on the frozen baseline val output (fold 0).

Sections (all exact re-evaluations with fast_evaluate unless stated):
  A. baseline reproduction (PQ, TP/FP/FN) for the frozen cache;
  B. FN denominator: every GT of every view by best IoU with any prediction;
  C. near-miss geometry (best IoU in (lo, 0.5]) with counts, for two ranges, and where the missing part of
     a fragment lies (lateral = within 5 px of the prediction, i.e. width; axial = farther, i.e. length);
  D. oracle repairs, exact metric: replace the offending prediction(s) by a GT-informed repair and
     re-score all views of the image (a repair can break a TP in another view, so this is honest):
       fragment -> pred U GT;  split -> union of touching preds (no GT pixels);  too big -> pred n dilate(GT, 3);
       all near-miss -> GT itself (ceiling of perfect geometry on near-misses);
  E. GT overlap inside a view (pairs with intersection > 0, and pixel share).
Usage: g0_audit.py [--cache artifacts/rescore/val_blend01.pkl --conf 0.2 --mask-thr 0.35]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_diag import GT_JSON, evaluate, iou  # noqa: E402
from filseg.gt import build_gt_df  # noqa: E402
from filseg.infer import load_cache  # noqa: E402
from filseg.postprocess import PPParams, _resolve_rles, group_by_stem, group_soft_by_stem, postprocess_to_rows  # noqa: E402
from filseg.rle import decode_rle, encode_mask, to_rle_obj  # noqa: E402


def classify(G, P, j, same_view):
    inter = mu.area(mu.merge([G, P[j]], intersect=True))
    cov, prec = inter / mu.area(G), inter / mu.area(P[j])
    touch_pred = [k for k, x in enumerate(iou([G], P)[0]) if x > 0.05]
    n_gt_touch = int((iou([P[j]], same_view)[0] > 0.05).sum())
    if n_gt_touch >= 2:
        return "merge", touch_pred
    if len(touch_pred) >= 2:
        return "split", touch_pred
    if cov >= 0.75 and prec < 0.6:
        return "too_big", touch_pred
    if prec >= 0.75 and cov < 0.6:
        return "fragment", touch_pred
    return "shifted", touch_pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="artifacts/rescore/val_blend01.pkl")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--mask-thr", type=float, default=0.35)
    ap.add_argument("--out", default="artifacts/g0/audit.json")
    a = ap.parse_args()
    out: dict = {"cache": a.cache, "conf": a.conf, "mask_thr": a.mask_thr}
    stems = sorted(json.load(open("data/splits/split_v1.json"))["val"])
    gt = build_gt_df(GT_JSON, set(stems))
    fid = gt["filament_id"].str.rsplit("_", n=1).str[0]
    views_of = defaultdict(lambda: defaultdict(list))  # stem -> view -> [rle obj]
    for v, r in zip(fid, gt["segmentation_rle"]):
        views_of[v.split("-", 1)[1]][v].append(to_rle_obj(r))
    raw = load_cache(a.cache)
    g, s = group_by_stem(raw), group_soft_by_stem(raw)
    rows = postprocess_to_rows(g, PPParams(a.conf, 100, 20, 0.3, 0, "square", a.mask_thr), stems, s)
    base = evaluate(gt, rows)
    out["A_baseline"] = dict(pq=base.pq, tp=base.tp, fp=base.fp, fn=base.fn, n_rows=len(rows), n_views=len(set(fid)),
                             n_gt=len(gt))
    print(f"A. baseline PQ {base.pq:.4f} TP {base.tp} FP {base.fp} FN {base.fn} | rows {len(rows)} views "
          f"{len(set(fid))} GT {len(gt)}")

    pred = defaultdict(list)  # stem -> [(fid, counts)]
    for k, r in rows:
        pred[k.rsplit("_", 1)[0]].append((k, r))

    # B + C
    bins = [(-1, 0.0, "no overlap (0)"), (0.0, 0.05, "(0, .05]"), (0.05, 0.3, "(.05, .3]"), (0.3, 0.5, "(.3, .5]"),
            (0.5, 1.01, "> .5 (TP-covered)")]
    bcount = defaultdict(int)
    near = []  # (stem, view, gt_idx, j, cat, touch, best)
    for st, vs in views_of.items():
        P = [to_rle_obj(r) for _, r in pred.get(st, [])]
        for v, Gs in vs.items():
            for gi, G in enumerate(Gs):
                io = iou([G], P)[0] if P else np.zeros(0)
                b = float(io.max()) if len(io) else 0.0
                for lo, hi, name in bins:
                    if (b > lo and b <= hi) or (name.startswith("no") and b == 0.0):
                        bcount[name] += 1
                        break
                if 0.05 < b <= 0.5:
                    j = int(io.argmax())
                    cat, touch = classify(G, P, j, Gs)
                    near.append((st, v, gi, j, cat, touch, b))
    fn_total = sum(n for k, n in bcount.items() if not k.startswith(">"))
    out["B_fn_denominator"] = dict(bcount, fn_total=fn_total)
    print("\nB. GT by best IoU with any prediction (FN = all but the last bin):")
    for _, _, name in bins:
        print(f"   {name:20s} {bcount[name]:5d}")
    print(f"   FN total {fn_total} (metric FN {base.fn})")

    print("\nC. near-miss geometry")
    out["C_near_miss"] = {}
    for lo in (0.3, 0.05):
        sub = [x for x in near if x[6] > lo]
        c = defaultdict(int)
        for x in sub:
            c[x[4]] += 1
        out["C_near_miss"][f"({lo},0.5]"] = dict(n=len(sub), **c)
        print(f"   best IoU in ({lo}, 0.5]: n = {len(sub)} of FN {fn_total}: "
              + ", ".join(f"{k} {n} ({n / len(sub):.0%})" for k, n in sorted(c.items(), key=lambda t: -t[1])))
    # missing-part location for fragments
    lat, axial, nfr = 0, 0, 0
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for st, v, gi, j, cat, touch, b in near:
        if cat != "fragment" or b <= 0.3:
            continue
        Gm = mu.decode(views_of[st][v][gi]).astype(bool)
        Pm = decode_rle(pred[st][j][1]).astype(bool)
        miss = Gm & ~Pm
        near5 = cv2.dilate(Pm.astype(np.uint8), k5).astype(bool)
        lat += int((miss & near5).sum())
        axial += int((miss & ~near5).sum())
        nfr += 1
    out["C_fragment_missing_part"] = dict(n=nfr, lateral_px=lat, axial_px=axial,
                                          axial_share=axial / max(lat + axial, 1))
    print(f"   fragments (IoU .3-.5, n={nfr}): missing GT pixels within 5 px of pred (width) {lat}, "
          f"farther (length / gaps) {axial} -> axial share {axial / max(lat + axial, 1):.0%}")

    # D. exact oracle repairs
    def repaired(kind: str, lo: float) -> object:
        mods = defaultdict(dict)  # stem -> j -> new bool mask (cumulative)
        drop = defaultdict(set)
        for st, v, gi, j, cat, touch, b in near:
            if b <= lo or (kind != "all" and cat != kind):
                continue
            G = mu.decode(views_of[st][v][gi]).astype(bool)
            cur = mods[st].get(j)
            if cur is None:
                cur = decode_rle(pred[st][j][1]).astype(bool)
            if kind == "fragment":
                cur = cur | G
            elif kind == "too_big":
                cur = cur & cv2.dilate(G.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            elif kind == "split":
                for k in touch:
                    if k != j:
                        cur = cur | decode_rle(pred[st][k][1]).astype(bool)
                        drop[st].add(k)
            else:  # all -> perfect geometry for this GT
                cur = G
            mods[st][j] = cur
        new_rows = []
        for st, lst in pred.items():
            if st not in mods:
                new_rows += lst
                continue
            rl, sc = [], []
            for j, (k, r) in enumerate(lst):
                if j in drop[st] and j not in mods[st]:
                    continue
                if j in mods[st]:
                    rl.append(encode_mask(mods[st][j]))
                    sc.append(2.0)
                else:
                    rl.append(r)
                    sc.append(1.0 - j * 1e-3)
            masks, _ = _resolve_rles(rl, np.asarray(sc))
            new_rows += [(f"{st}_{i}", encode_mask(m)) for i, m in enumerate(masks, 1) if m is not None and m.sum() >= 100]
        return evaluate(gt, new_rows)

    print("\nD. oracle geometry repairs (exact metric, all views re-scored)")
    out["D_oracle"] = {}
    for kind in ("fragment", "split", "too_big", "all"):
        for lo in (0.3, 0.05):
            r = repaired(kind, lo)
            out["D_oracle"][f"{kind}@({lo},0.5]"] = dict(pq=r.pq, tp=r.tp, fp=r.fp, fn=r.fn, gain=r.pq - base.pq)
            print(f"   {kind:9s} near-miss ({lo}, .5]: PQ {r.pq:.4f} ({r.pq - base.pq:+.4f}) TP {r.tp} FP {r.fp} FN {r.fn}")

    # E. GT overlaps inside a view
    pairs, ov_views, nviews, ov_px, tot_px = 0, 0, 0, 0, 0
    for st, vs in views_of.items():
        for v, Gs in vs.items():
            nviews += 1
            any_ov = False
            for i in range(len(Gs)):
                tot_px += int(mu.area(Gs[i]))
                for k in range(i + 1, len(Gs)):
                    x = int(mu.area(mu.merge([Gs[i], Gs[k]], intersect=True)))
                    if x > 0:
                        pairs += 1
                        ov_px += x
                        any_ov = True
            ov_views += any_ov
    out["E_gt_overlap"] = dict(pairs=pairs, views_with_overlap=ov_views, views=nviews, overlap_px=ov_px, gt_px=tot_px)
    print(f"\nE. GT overlap within a view: {pairs} overlapping pairs in {ov_views}/{nviews} views; "
          f"overlap pixels {ov_px} of {tot_px} GT pixels ({ov_px / max(tot_px, 1):.2%})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
