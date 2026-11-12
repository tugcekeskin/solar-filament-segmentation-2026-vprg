#!/usr/bin/env python
"""Headroom of the decision step (which candidates to emit), on a refined val cache.

Under the organizer PQ (each view scored separately, TP = IoU > 0.5, no one-to-one matching), adding a
candidate c changes the pooled sums by  N += sum_v IoU_v [IoU_v > .5]  and  D += 0.5 per view (a TP adds
1 and removes 1/2 FN, an FP adds 1/2). So with lambda = current PQ, c is worth emitting iff
    q(c) = mean_v IoU_v [IoU_v > .5]  >  lambda / 2.
Sections
  1. oracle: q from the true GT, greedy non-overlapping selection -> PQ ceiling of perfect selection
     (split into "only drop FPs from the current output" and "free choice from all candidates");
  2. near-miss geometry: FN whose best prediction has IoU in (0.3, 0.5] — too big, fragment, or shifted;
  3. per-instrument conf thresholds, stem-grouped 5-fold CV vs one global threshold.
Usage: decode_diag.py CACHE [--conf 0.2 --mask-thr 0.35]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.gt import build_gt_df  # noqa: E402
from filseg.infer import load_cache  # noqa: E402
from filseg.metrics import fast_evaluate  # noqa: E402
from filseg.postprocess import (PPParams, _resolve_rles, group_by_stem, group_soft_by_stem,  # noqa: E402
                                postprocess_to_rows, transform_stem)
from filseg.rle import encode_mask, to_rle_obj  # noqa: E402

GT_JSON = "data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def iou(a: list, b: list) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou(a, b, [0] * len(b)))


def evaluate(gt: pd.DataFrame, rows: list) -> object:
    return fast_evaluate(gt, pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]))


def select(stem: str, rles: list, gain: np.ndarray, allowed: np.ndarray) -> list:
    """Greedy by gain, skip candidates overlapping an already chosen one by > 20% of their area, then
    resolve residual overlaps exactly like the real pipeline."""
    objs = [to_rle_obj(r) for r in rles]
    areas = np.array([mu.area(o) for o in objs], float)
    chosen = []
    for i in np.argsort(-gain):
        if gain[i] <= 0 or not allowed[i]:
            continue
        if chosen:
            inter = np.array([mu.area(mu.merge([objs[i], objs[j]], intersect=True)) for j in chosen])
            if (inter / areas[i]).max() > 0.2:
                continue
        chosen.append(i)
    if not chosen:
        return []
    masks, _ = _resolve_rles([rles[i] for i in chosen], gain[chosen])
    return [(f"{stem}_{k}", encode_mask(m)) for k, m in enumerate(masks, 1) if m.sum() >= 100]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--mask-thr", type=float, default=0.35)
    a = ap.parse_args()
    stems = sorted(json.load(open("data/splits/split_v1.json"))["val"])
    gt = build_gt_df(GT_JSON, set(stems))
    fid = gt["filament_id"].str.rsplit("_", n=1).str[0]
    gt_views = defaultdict(list)
    for (v, r) in zip(fid, gt["segmentation_rle"]):
        gt_views[v.split("-", 1)[1]].append((v, r))
    raw = load_cache(a.cache)
    g, s = group_by_stem(raw), group_soft_by_stem(raw)
    base_p = PPParams(a.conf, 100, 20, 0.3, 0, "square", a.mask_thr)
    base_rows = postprocess_to_rows(g, base_p, stems, s)
    base = evaluate(gt, base_rows)
    print(f"baseline conf {a.conf} thr {a.mask_thr}: PQ {base.pq:.4f} TP {base.tp} FP {base.fp} FN {base.fn}")

    # candidates after the same soft threshold, all scores >= 0.01, area >= 100
    cand = {}
    for st in stems:
        if st not in g:
            continue
        sc, rl, ar = transform_stem(*g[st], s.get(st), 0, "square", a.mask_thr, 0.0)
        keep = [i for i in range(len(rl)) if ar[i] >= 100]
        P = [to_rle_obj(rl[i]) for i in keep]
        views = defaultdict(list)
        for v, r in gt_views[st]:
            views[v].append(to_rle_obj(r))
        q = np.zeros(len(P))
        for V in views.values():
            io = iou(P, V).max(1) if V else np.zeros(len(P))
            q += io * (io > 0.5)
        cand[st] = (np.asarray(sc)[keep], [rl[i] for i in keep], q / max(len(views), 1), len(views))
    base_keys = {(st, r) for st in cand for r in cand[st][1]}

    # 1. oracle (two lambda iterations)
    print("\n1. oracle selection (q from true GT)")
    for name, mode in [("drop-FP only (from candidates >= conf)", "drop"), ("free choice (all candidates >= 0.01)", "free")]:
        lam = base.pq
        for _ in range(2):
            rows = []
            for st, (sc, rl, q, nv) in cand.items():
                allowed = sc >= a.conf if mode == "drop" else np.ones(len(sc), bool)
                rows += select(st, rl, q - lam / 2, allowed)
            r = evaluate(gt, rows)
            lam = r.pq
        print(f"   {name}: PQ {r.pq:.4f} TP {r.tp} FP {r.fp} FN {r.fn}")
    # realistic: the same rule with YOLO score as q-proxy is just the threshold; show q vs score agreement
    allq = np.concatenate([c[2] for c in cand.values()])
    alls = np.concatenate([c[0] for c in cand.values()])
    worth = allq > base.pq / 2
    for lo, hi in [(0.01, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01)]:
        m = (alls >= lo) & (alls < hi)
        print(f"   score [{lo:.2f},{hi:.2f}): {m.sum():5d} cands, worth emitting {worth[m].mean():.1%}")

    # 2. near-miss geometry of baseline FN
    print("\n2. near-miss FN (best IoU in (0.3, 0.5]) geometry")
    pred = defaultdict(list)
    for k, r in base_rows:
        pred[k.rsplit("_", 1)[0]].append(to_rle_obj(r))
    cats = defaultdict(int)
    for st, lst in gt_views.items():
        P = pred.get(st, [])
        for v, r in lst:
            G = to_rle_obj(r)
            if not P:
                continue
            io = iou([G], P)[0]
            j = int(io.argmax())
            if not (0.3 < io[j] <= 0.5):
                continue
            inter = mu.area(mu.merge([G, P[j]], intersect=True))
            cov, prec = inter / mu.area(G), inter / mu.area(P[j])
            n_pred_touch = int((iou([G], P)[0] > 0.05).sum())
            same_view = [to_rle_obj(x) for vv, x in lst if vv == v]
            n_gt_touch = int((iou([P[j]], same_view)[0] > 0.05).sum())
            if n_gt_touch >= 2:
                c = "pred merges >=2 GT of the view"
            elif n_pred_touch >= 2:
                c = "GT split over >=2 preds"
            elif cov >= 0.75 and prec < 0.6:
                c = "pred too big (covers GT)"
            elif prec >= 0.75 and cov < 0.6:
                c = "pred too small (fragment of GT)"
            else:
                c = "shifted / shape mismatch"
            cats[c] += 1
    tot = sum(cats.values())
    for c, n in sorted(cats.items(), key=lambda t: -t[1]):
        print(f"   {c}: {n} ({n / tot:.0%})")

    # 3. per-instrument threshold CV
    print("\n3. per-instrument conf, stem-grouped 5-fold CV")
    confs = [0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    rows_by = {c: defaultdict(list) for c in confs}
    for c in confs:
        for k, r in postprocess_to_rows(g, PPParams(c, 100, 20, 0.3, 0, "square", a.mask_thr), stems, s):
            rows_by[c][k.rsplit("_", 1)[0]].append((k, r))
    inst = {st: st[-2:] for st in stems}
    print("   instruments:", pd.Series(inst).value_counts().to_dict())
    rng = np.random.default_rng(0)
    order = rng.permutation(stems)
    folds = {st: i % 5 for i, st in enumerate(order)}
    gt_stem = fid.str.split("-", n=1).str[1]

    def pq_of(sts: list, choice: dict) -> float:
        rows = [x for st in sts for x in rows_by[choice[st]][st]]
        return evaluate(gt[gt_stem.isin(sts)], rows).pq

    out_rows = []
    for f in range(5):
        tr = [st for st in stems if folds[st] != f]
        te = [st for st in stems if folds[st] == f]
        glob = max(confs, key=lambda c: pq_of(tr, {st: c for st in tr}))
        choice = {}
        for ins in set(inst.values()):
            sub = [st for st in tr if inst[st] == ins]
            best = max(confs, key=lambda c: pq_of(sub, {st: c for st in sub})) if len(sub) >= 5 else glob
            for st in te:
                if inst[st] == ins:
                    choice[st] = best
        for st in te:
            choice.setdefault(st, glob)
        out_rows += [x for st in te for x in rows_by[choice[st]][st]]
    r = evaluate(gt, out_rows)
    rg = evaluate(gt, [x for st in stems for x in rows_by[a.conf if a.conf in confs else 0.2][st]])
    print(f"   per-instrument CV: PQ {r.pq:.4f} (TP {r.tp} FP {r.fp} FN {r.fn}) vs global conf {a.conf}: {rg.pq:.4f}")


if __name__ == "__main__":
    main()
