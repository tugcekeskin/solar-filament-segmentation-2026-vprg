#!/usr/bin/env python
"""Quantitative error analysis of a run's val predictions under the organizer PQ semantics.

Writes docs/error_analysis_<run>.md and overlays under artifacts/<run>/analysis/.
Totals are asserted to equal ``fast_evaluate`` on val_pred.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from filseg.gt import build_gt_df, view_stem  # noqa: E402
from filseg.metrics import fast_evaluate  # noqa: E402
from filseg.rle import decode_rle, to_rle_obj  # noqa: E402

DATA = "data/raw/MAGFiLO_1.0_Kaggle_2026/train"
GT_JSON = f"{DATA}/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
F32_HALF = np.float32(0.5)
CAT_NAMES = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}
RAD_EDGES = [0.5, 0.8, 0.95]
RAD_LABELS = ["<0.5", "0.5-0.8", "0.8-0.95", ">0.95 (limb)"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="data/splits/split_v1.json")
    p.add_argument("--n-overlays", type=int, default=8)
    return p.parse_args()


# ----------------------------------------------------------------------------- helpers
def iou_mat(a: list[dict], b: list[dict]) -> np.ndarray:
    """IoU (len(a), len(b)); mask_util.iou(a, b) already is that orientation."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mask_util.iou(a, b, [0] * len(b)), dtype=np.float64).reshape(len(a), len(b))


def hit_of(iou: np.ndarray) -> np.ndarray:
    return iou.astype(np.float32) > F32_HALF


def pq_calc(tp: float, fp: float, fn: float, sum_iou: float) -> float:
    den = tp + 0.5 * fp + 0.5 * fn
    return sum_iou / den if den > 0 else 0.0


def centroid(m: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean())


def disk_geometry(stem: str) -> tuple[float, float, float]:
    """(cx, cy, R) of the solar disk: threshold, open, largest CC, fill holes; R from area."""
    g = cv2.imread(f"{DATA}/train_images/{stem}.jpeg", 0)
    b = cv2.morphologyEx((g > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    comp = (lab == 1 + st[1:, 4].argmax()).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    cx, cy = centroid(filled)
    return cx, cy, float(np.sqrt(filled.sum() / np.pi))


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.astype(object).iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def best_bin(x: float) -> str:
    if x <= 0:
        return "0 (no overlap)"
    if x <= 0.3:
        return "(0, 0.3]"
    return "(0.3, 0.5]"


BIN_ORDER = ["0 (no overlap)", "(0, 0.3]", "(0.3, 0.5]"]


# ----------------------------------------------------------------------------- main data pass
def main() -> None:
    args = parse_args()
    art = ROOT / "artifacts" / args.run
    out_dir = art / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_path = ROOT / "docs" / f"error_analysis_{args.run}.md"
    import os
    os.chdir(ROOT)

    val_stems = set(json.load(open(args.split))["val"])
    gt = build_gt_df(GT_JSON, val_stems)
    pred = pd.read_csv(art / "val_pred.csv", dtype=str, keep_default_na=False)
    raw = pd.read_pickle(art / "val_raw.pkl")
    man = pd.read_csv("data/processed/manifest.csv").set_index("view_id")
    res = fast_evaluate(gt, pred)
    pv = res.per_view.copy()

    # GT metadata (category, json area) keyed by organizer filament_id
    coco = json.load(open(GT_JSON))
    by_view = defaultdict(list)
    for a in coco["annotations"]:
        by_view[a["image_id"]].append(a)
    meta = {}
    for v, anns in by_view.items():
        for k, a in enumerate(anns, 1):
            meta[f"{v}_{k}"] = a["category_id"]

    gt["view"] = gt["filament_id"].str.split("_", n=1).str[0]
    gt["stem"] = gt["view"].map(view_stem)
    pred["stem"] = pred["filament_id"].str.split("_", n=1).str[0]
    gt_by_view = {v: g for v, g in gt.groupby("view", sort=False)}
    views_of_stem = defaultdict(list)
    for v in pd.unique(gt["view"]):
        views_of_stem[view_stem(v)].append(v)
    pred_by_stem = {s: g for s, g in pred.groupby("stem")}
    raw_by_stem = {s: g for s, g in raw.groupby("stem")}

    disk = {s: disk_geometry(s) for s in sorted(views_of_stem)}

    gt_rows, fp_rows = [], []
    frag_rows, merge_rows = [], []
    tp_ious_all = []
    hits_by_pred = defaultdict(lambda: defaultdict(set))  # stem -> pred idx -> views hit
    iou_cache = {}
    for view in pd.unique(gt["view"]):
        stem = view_stem(view)
        g = gt_by_view[view]
        grle = [to_rle_obj(c) for c in g["segmentation_rle"]]
        pdf = pred_by_stem.get(stem)
        prle = [to_rle_obj(c) for c in pdf["segmentation_rle"]] if pdf is not None else []
        rdf = raw_by_stem.get(stem)
        rrle = [to_rle_obj(c) for c in rdf["rle"]] if rdf is not None else []
        rscore = rdf["score"].to_numpy() if rdf is not None else np.zeros(0)
        iou = iou_mat(grle, prle)
        hit = hit_of(iou)
        riou = iou_mat(grle, rrle)
        iou_cache[view] = (iou, hit)
        cx, cy, R = disk[stem]
        garea = np.asarray(mask_util.area(grle), dtype=np.float64)
        parea = np.asarray(mask_util.area(prle), dtype=np.float64) if prle else np.zeros(0)
        gcen = [centroid(decode_rle(c)) for c in g["segmentation_rle"]]
        pcen = [centroid(decode_rle(c)) for c in pdf["segmentation_rle"]] if pdf is not None else []
        pr_r = [np.hypot(x - cx, y - cy) / R for x, y in pcen]
        for i, fid in enumerate(g["filament_id"]):
            best = float(iou[i].max()) if prle else 0.0
            rbest = float(riou[i].max()) if rrle else 0.0
            gt_rows.append(dict(
                fid=fid, view=view, stem=stem, area=garea[i], r=np.hypot(gcen[i][0] - cx, gcen[i][1] - cy) / R,
                cat=CAT_NAMES[meta[fid]], hit=bool(hit[i].any()), best=best, raw_best=rbest,
                raw_best_score=float(rscore[riou[i].argmax()]) if rrle else 0.0,
                n_hit=int(hit[i].sum()), tp_iou=float(iou[i][hit[i]].sum()) if prle else 0.0,
                n_pos=int((iou[i] > 0).sum()) if prle else 0,
            ))
        for j in range(len(prle)):
            h = bool(hit[:, j].any())
            if h:
                hits_by_pred[stem][j].add(view)
            fp_rows.append(dict(
                view=view, stem=stem, j=j, area=parea[j], r=pr_r[j], hit=h,
                best=float(iou[:, j].max()), best_gt=int(iou[:, j].argmax()),
                score=float(pdf["score"].iloc[j]) if "score" in pdf else np.nan,
            ))
        # fragmentation (GT covered by >=2 preds, none hits) / merging (pred over >=2 GT, none hit)
        pos = iou > 0
        for i in range(len(grle)):
            idx = np.nonzero(pos[i])[0] if prle else []
            if len(idx) >= 2 and not hit[i].any():
                u = mask_util.merge([prle[j] for j in idx])
                ui = float(iou_mat([grle[i]], [u])[0, 0])
                frag_rows.append(dict(view=view, i=i, preds=[int(j) for j in idx],
                                      n_fp=int(sum(not hit[:, j].any() for j in idx)), union_iou=ui))
        for j in range(len(prle)):
            idx = np.nonzero(pos[:, j])[0]
            if len(idx) >= 2 and not hit[:, j].any():
                u = mask_util.merge([grle[i] for i in idx])
                ui = float(iou_mat([u], [prle[j]])[0, 0])
                merge_rows.append(dict(view=view, j=j, gts=[int(i) for i in idx],
                                       n_fn=int(sum(not hit[i].any() for i in idx)), union_iou=ui))
    G = pd.DataFrame(gt_rows)
    P = pd.DataFrame(fp_rows)
    P["fp"] = ~P["hit"]

    # ---- reconcile with fast_evaluate
    tp_pairs = sum(int(h.sum()) for _, h in iou_cache.values())
    fp_n = int((~P["hit"]).sum())
    fn_n = int((~G["hit"]).sum())
    assert (tp_pairs, fp_n, fn_n) == (res.tp, res.fp, res.fn), ((tp_pairs, fp_n, fn_n), (res.tp, res.fp, res.fn))
    assert (res.tp, res.fp, res.fn) == (834, 560, 711) or args.run != "y11s_1024"
    sum_iou = float(pv["sum_iou"].sum())
    assert abs(pq_calc(res.tp, res.fp, res.fn, sum_iou) - res.pq) < 1e-9
    TP, FP, FN = res.tp, res.fp, res.fn

    md: list[str] = []
    add = md.append
    add(f"# Error analysis: {args.run} (val split)\n")
    add("Generated by `scripts/analyze_errors.py --run %s`; organizer PQ semantics (hit = IoU > 0.5, no 1:1 matching, "
        "pooled over GT annotator-views). All numbers are on val_pred.csv (final calibrated predictions) unless noted.\n" % args.run)
    add(f"Reconciliation (asserted): PQ {res.pq:.4f}, SQ {res.sq:.3f}, RQ {res.rq:.3f}, TP/FP/FN = {TP}/{FP}/{FN}, "
        f"{res.n_views} views, {len(views_of_stem)} stems, {res.n_gt} GT, {res.n_pred} preds "
        f"(FP/TP/FN are counted per (view, instance): one pred is scored once for every annotator view of its stem).\n")

    # ---- 1. FN / FP breakdown
    add("## 1. FN and FP breakdown by best IoU\n")
    fnG = G[~G["hit"]].copy()
    fnG["bin"] = fnG["best"].map(best_bin)
    fpP = P[P["fp"]].copy()
    fpP["bin"] = fpP["best"].map(best_bin)
    t1 = pd.DataFrame({
        "best IoU bin": BIN_ORDER,
        "FN (GT, best IoU vs any pred)": [int((fnG["bin"] == b).sum()) for b in BIN_ORDER],
        "FN %": [100 * (fnG["bin"] == b).mean() for b in BIN_ORDER],
        "FP (pred, best IoU vs GT in view)": [int((fpP["bin"] == b).sum()) for b in BIN_ORDER],
        "FP %": [100 * (fpP["bin"] == b).mean() for b in BIN_ORDER],
    })
    add(md_table(t1, "{:.1f}") + "\n")
    near_fn = int((fnG["bin"] == "(0.3, 0.5]").sum())
    zero_fn = int((fnG["bin"] == "0 (no overlap)").sum())
    near_fp = int((fpP["bin"] == "(0.3, 0.5]").sum())
    zero_fp = int((fpP["bin"] == "0 (no overlap)").sum())
    add(f"Zero-overlap = detection problem ({zero_fn} FN, {zero_fp} FP); 0.3-0.5 = mask/shape-quality problem "
        f"({near_fn} FN, {near_fp} FP).\n")

    # ---- 2. recoverability
    add("## 2. FN recoverability from raw cached instances (conf >= 0.01)\n")
    fnG["rec"] = fnG["raw_best"].astype(np.float32) > F32_HALF
    nrec = int(fnG["rec"].sum())
    add(f"Raw instance with IoU > 0.5 exists for **{nrec} / {FN} FN ({100 * nrec / FN:.1f}%)** "
        f"= recoverable by threshold/post-processing; **{FN - nrec} ({100 * (FN - nrec) / FN:.1f}%) never proposed**.\n")
    rows = []
    for b in BIN_ORDER:
        s = fnG[fnG["bin"] == b]
        rows.append(dict(bin=b, n=len(s), recoverable=int(s["rec"].sum()), never_proposed=int((~s["rec"]).sum())))
    add(md_table(pd.DataFrame(rows)) + "\n")
    rec = fnG[fnG["rec"]]
    if len(rec):
        q = rec["raw_best_score"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        add("Confidence of the best matching raw instance for recoverable FNs (quantiles 10/25/50/75/90): "
            + " / ".join(f"{v:.3f}" for v in q) + ". Mean raw best-IoU of recoverable FNs: "
            f"{rec['raw_best'].mean():.3f}.\n")
    nonrec = fnG[~fnG["rec"]]
    add(f"Never-proposed FNs: median area {nonrec['area'].median():.0f} px (vs {G['area'].median():.0f} px for all GT), "
        f"mean best raw IoU {nonrec['raw_best'].mean():.3f}.\n")

    # ---- 3. breakdowns
    add("## 3. Breakdowns\n")
    qe = np.quantile(G["area"], [0.2, 0.4, 0.6, 0.8])
    qlab = [f"Q{i + 1}" for i in range(5)]
    qlab[0] += f" (<{qe[0]:.0f})"
    qlab[1] += f" ({qe[0]:.0f}-{qe[1]:.0f})"
    qlab[2] += f" ({qe[1]:.0f}-{qe[2]:.0f})"
    qlab[3] += f" ({qe[2]:.0f}-{qe[3]:.0f})"
    qlab[4] += f" (>{qe[3]:.0f})"
    G["area_bin"] = pd.Categorical([qlab[i] for i in np.searchsorted(qe, G["area"], side="right")], qlab, ordered=True)
    P["area_bin"] = pd.Categorical([qlab[i] for i in np.searchsorted(qe, P["area"], side="right")], qlab, ordered=True)
    for d in (G, P):
        d["rad_bin"] = pd.Categorical([RAD_LABELS[i] for i in np.searchsorted(RAD_EDGES, d["r"], side="right")],
                                      RAD_LABELS, ordered=True)

    def inst_table(col: str, title: str) -> str:
        g = G.groupby(col, observed=True).agg(n_gt=("hit", "size"), TP=("hit", "sum"),
                                              mean_tp_iou=("tp_iou", lambda s: s[s > 0].mean()))
        g["FN"] = g["n_gt"] - g["TP"]
        g["recall"] = g["TP"] / g["n_gt"]
        g["FN share %"] = 100 * g["FN"] / FN
        if col in P.columns:
            p = P.groupby(col, observed=True).agg(n_pred_views=("hit", "size"), FP=("fp", "sum"))
            p["precision"] = 1 - p["FP"] / p["n_pred_views"]
            p["FP share %"] = 100 * p["FP"] / FP
            g = g.join(p)
        g = g.reset_index().rename(columns={col: title})
        for c in ("n_gt", "TP", "FN", "n_pred_views", "FP"):
            if c in g:
                g[c] = g[c].astype(int)
        return md_table(g)

    add("Instance-level (GT attribute for recall/FN; pred attribute for FP/precision, using the same bins). "
        "`mean_tp_iou` = mean IoU of hit pairs.\n")
    add("### 3a. GT area quintile (pixels at 2048, val GT quintile edges)\n")
    add(inst_table("area_bin", "area bin") + "\n")
    add("### 3b. Radial position r/R_disk (mask centroid; disk center and R estimated by threshold)\n")
    add(inst_table("rad_bin", "r/R") + "\n")
    add(f"({int((G['r'] > 0.95).sum())} GT and {int((P['r'] > 0.95).sum())} preds have centroid beyond 0.95 R: limb is "
        f"effectively unannotated/undetected, so no limb conclusions.)\n")
    add("### 3c. Category (GT only; predictions are class-agnostic)\n")
    add(inst_table("cat", "category") + "\n")

    # stem-level attributes from per-view table
    pv = pv.merge(man[["n_views_of_stem", "instrument", "year"]], left_on="view", right_index=True, how="left")
    pv["year_grp"] = pv["year"].map(lambda y: str(y) if y <= 2017 else "2018-2022")
    pv["n_gt_bin"] = pd.cut(pv["n_gt"], [0, 3, 6, 10, 1000], labels=["1-3", "4-6", "7-10", ">10"])

    def view_table(col: str, title: str) -> str:
        rows = []
        for k, s in pv.groupby(col, observed=True):
            tp, fp, fn, si = s["tp"].sum(), s["fp"].sum(), s["fn"].sum(), s["sum_iou"].sum()
            rows.append({title: k, "views": len(s), "GT": int(s["n_gt"].sum()), "TP": int(tp), "FP": int(fp),
                         "FN": int(fn), "pooled PQ": pq_calc(tp, fp, fn, si), "SQ": si / tp if tp else 0.0,
                         "RQ": tp / (tp + .5 * fp + .5 * fn), "mean per-view PQ": s["pq"].mean()})
        return md_table(pd.DataFrame(rows))

    add("View-level (pooled PQ over the group's views; mean per-view PQ for comparison).\n")
    add("### 3d. n_views_of_stem\n")
    add(view_table("n_views_of_stem", "n_views") + "\n")
    add("### 3e. Instrument\n")
    add(view_table("instrument", "instrument") + "\n")
    add("### 3f. Year (2018-2022 pooled; few views each)\n")
    add(view_table("year_grp", "year") + "\n")
    add("### 3g. GT instances per view\n")
    add(view_table("n_gt_bin", "n_gt in view") + "\n")

    # ---- 4. annotator consensus
    add("## 4. Annotator consensus\n")
    support = {}
    for stem, vs in views_of_stem.items():
        if len(vs) < 2:
            continue
        rl = {v: [to_rle_obj(c) for c in gt_by_view[v]["segmentation_rle"]] for v in vs}
        for v in vs:
            for i in range(len(rl[v])):
                sup = 0
                for w in vs:
                    if w != v and hit_of(iou_mat([rl[v][i]], rl[w]))[0].any():
                        sup += 1
                support[(v, i)] = sup
    G["i"] = G.groupby("view").cumcount()
    G["support"] = [support.get((v, i), -1) for v, i in zip(G["view"], G["i"])]
    G["n_others"] = G["stem"].map(lambda s: len(views_of_stem[s]) - 1)
    rows = []
    multi = G[G["n_others"] >= 1]
    for nv in (2, 3):
        for sup in range(nv):
            s = multi[(multi["n_others"] == nv - 1) & (multi["support"] == sup)]
            if len(s):
                rows.append({"stem views": nv, "other views agreeing": sup, "GT": len(s),
                             "recall": s["hit"].mean(), "TP": int(s["hit"].sum()), "FN": int((~s["hit"]).sum())})
    s1 = G[G["n_others"] == 0]
    rows.append({"stem views": 1, "other views agreeing": "n/a", "GT": len(s1), "recall": s1["hit"].mean(),
                 "TP": int(s1["hit"].sum()), "FN": int((~s1["hit"]).sum())})
    add("Recall vs support (support = number of OTHER annotator views of the same stem with a GT instance at IoU > 0.5):\n")
    add(md_table(pd.DataFrame(rows)) + "\n")
    allsup = multi.groupby("support").agg(GT=("hit", "size"), recall=("hit", "mean"))
    add("Pooled over multi-view stems: " + "; ".join(
        f"support {k}: n={int(r.GT)}, recall {r.recall:.3f}" for k, r in allsup.iterrows()) + ".\n")
    # FPs on multi-view stems
    P["n_views_stem"] = P["stem"].map(lambda s: len(views_of_stem[s]))
    P["views_hit"] = [len(hits_by_pred[s][j]) for s, j in zip(P["stem"], P["j"])]
    pm = P[P["n_views_stem"] >= 2]
    rows = []
    for nv in (2, 3):
        s = pm[pm["n_views_stem"] == nv]
        for h in range(nv + 1):
            fpn = int(((s["views_hit"] == h) & s["fp"]).sum())
            rows.append({"stem views": nv, "views the pred hits": h,
                         "preds (unique)": int(((s["views_hit"] == h)).sum() / nv),
                         "FP (view, pred) pairs": fpn})
    add("For preds on multi-view stems: number of annotator views each pred hits (0 = unsupported by every annotator). "
        "Each pred is a FP in every view it does not hit:\n")
    add(md_table(pd.DataFrame(rows)) + "\n")
    fp_m = pm[pm["fp"]]
    n_fpm = len(fp_m)
    sup_fp = int((fp_m["views_hit"] >= 1).sum())
    add(f"On multi-view stems {n_fpm} of {FP} FP pairs; **{sup_fp} ({100 * sup_fp / max(n_fpm, 1):.1f}%) are preds that another annotator "
        f"did mark** (they are FP only w.r.t. this annotator: pure inter-annotator disagreement), "
        f"{n_fpm - sup_fp} are unsupported by every annotator.\n")
    add(f"Single-annotator stems: {int(P[P['n_views_stem'] == 1]['fp'].sum())} FP pairs.\n")

    # ---- 5. split/merge
    add("## 5. Split / merge analysis\n")
    F = pd.DataFrame(frag_rows)
    M = pd.DataFrame(merge_rows)
    nfrag, nmerge = len(F), len(M)
    frag_fp = {(r.view, j) for r in F.itertuples() for j in r.preds if not hit_of_view(iou_cache, r.view, j)} if nfrag else set()
    merge_fn = {(r.view, i) for r in M.itertuples() for i in r.gts if not iou_cache[r.view][1][i].any()} if nmerge else set()
    frag_fp_n = len(frag_fp)
    merge_fn_n = len(merge_fn)
    rows = [
        {"case": "Fragmentation: GT overlapped by >=2 preds, none hits", "count": nfrag,
         "preds involved": int(sum(len(x) for x in F["preds"])) if nfrag else 0,
         "union of fragments hits (IoU>0.5)": int((F["union_iou"].astype(np.float32) > F32_HALF).sum()) if nfrag else 0,
         "FN caused": nfrag, "FP caused": frag_fp_n},
        {"case": "Merging: pred overlapping >=2 GT, none hits", "count": nmerge,
         "preds involved": nmerge,
         "union of fragments hits (IoU>0.5)": int((M["union_iou"].astype(np.float32) > F32_HALF).sum()) if nmerge else 0,
         "FN caused": merge_fn_n, "FP caused": nmerge},
    ]
    add(md_table(pd.DataFrame(rows)) + "\n")
    add(f"(Fast-eval raw counts including cases that also have a hit: splits {res.splits}, merges {res.merges}.) "
        f"Union column: IoU of union of the fragments (resp. GTs) with the single counterpart. FN/FP caused are the "
        f"deduplicated FN GT and FP (view, pred) items involved, i.e. an upper bound on what fixing splits/merges removes "
        f"({nfrag + merge_fn_n} FN, {frag_fp_n + nmerge} FP of {FN}/{FP}).\n")

    # ---- 6. what-ifs
    add("## 6. Upper-bound what-ifs (offline, not achievable bounds)\n")
    base = pq_calc(TP, FP, FN, sum_iou)
    a_pq = pq_calc(TP, FP, FN, TP)
    # near-miss FN -> TP at 0.55; also drop the FP that is the best-matching pred if it has no hit
    nm = fnG[fnG["bin"] == "(0.3, 0.5]"]
    nm_best_fp = set()
    for r in nm.itertuples():
        iou, hit = iou_cache[r.view]
        i = G.index[G["fid"] == r.fid][0]
        j = int(iou[int(G.at[i, "i"])].argmax())
        if not hit[:, j].any():
            nm_best_fp.add((r.view, j))
    n_nm = len(nm)
    b1 = pq_calc(TP + n_nm, FP, FN - n_nm, sum_iou + 0.55 * n_nm)
    b2 = pq_calc(TP + n_nm, FP - len(nm_best_fp), FN - n_nm, sum_iou + 0.55 * n_nm)
    # recoverable FN (raw instance exists) -> TP at its raw IoU, FP unchanged
    n_rec = int(fnG["rec"].sum())
    c_rec = pq_calc(TP + n_rec, FP, FN - n_rec, sum_iou + fnG.loc[fnG["rec"], "raw_best"].sum())
    # zero-support FP removal (multi-view stems, hit by no view)
    unsup = int(((P["fp"]) & (P["n_views_stem"] >= 2) & (P["views_hit"] == 0)).sum())
    d_pq = pq_calc(TP, FP - unsup, FN, sum_iou)
    # fragmentation/merge fixes: frag GT -> TP (at union iou or 0.55), its FP fragments dropped
    fr_ok = int((F["union_iou"].astype(np.float32) > F32_HALF).sum()) if nfrag else 0
    e_pq = pq_calc(TP + nfrag, FP - frag_fp_n, FN - nfrag, sum_iou + 0.55 * nfrag)
    # never proposed FN -> TP at mean TP IoU
    n_np = FN - n_rec
    f_pq = pq_calc(TP + n_np, FP, FN - n_np, sum_iou + res.sq * n_np)
    all_pq = pq_calc(TP + n_nm, FP - len(nm_best_fp), FN - n_nm, TP + n_nm)
    rows = [
        {"scenario": "Baseline", "TP": TP, "FP": FP, "FN": FN, "PQ": base, "dPQ": 0.0},
        {"scenario": "(a) oracle masks: TP IoU -> 1", "TP": TP, "FP": FP, "FN": FN, "PQ": a_pq, "dPQ": a_pq - base},
        {"scenario": f"(b1) {n_nm} near-miss FN -> TP @0.55 (FP unchanged)", "TP": TP + n_nm, "FP": FP,
         "FN": FN - n_nm, "PQ": b1, "dPQ": b1 - base},
        {"scenario": f"(b2) same, also drop {len(nm_best_fp)} FP that were the best-match pred", "TP": TP + n_nm,
         "FP": FP - len(nm_best_fp), "FN": FN - n_nm, "PQ": b2, "dPQ": b2 - base},
        {"scenario": f"(g) {n_rec} threshold-recoverable FN -> TP at raw IoU (FP unchanged)", "TP": TP + n_rec,
         "FP": FP, "FN": FN - n_rec, "PQ": c_rec, "dPQ": c_rec - base},
        {"scenario": f"(h) {n_np} never-proposed FN -> TP at SQ={res.sq:.2f}", "TP": TP + n_np, "FP": FP,
         "FN": FN - n_np, "PQ": f_pq, "dPQ": f_pq - base},
        {"scenario": f"(i) fix {nfrag} fragmentations (-{frag_fp_n} FP) @0.55", "TP": TP + nfrag,
         "FP": FP - frag_fp_n, "FN": FN - nfrag, "PQ": e_pq, "dPQ": e_pq - base},
        {"scenario": f"(j) drop {unsup} FP supported by no annotator (multi-view stems)", "TP": TP, "FP": FP - unsup,
         "FN": FN, "PQ": d_pq, "dPQ": d_pq - base},
        {"scenario": "(a)+(b2) combined", "TP": TP + n_nm, "FP": FP - len(nm_best_fp), "FN": FN - n_nm,
         "PQ": all_pq, "dPQ": all_pq - base},
    ]
    add(md_table(pd.DataFrame(rows)) + "\n")
    add("(c) PQ by annotator multiplicity of the stem (the inter-annotator PQ ceiling is ~0.337, SQ 0.631):\n")
    pv["multi"] = np.where(pv["n_views_of_stem"] >= 2, "multi-annotator stems", "single-annotator stems")
    add(view_table("multi", "stems") + "\n")
    # gt hit counts restricted
    add(f"Single-annotator views are scored against one opinion only, so extra preds that other annotators would have "
        f"marked count as FP there; multi-annotator views average 2-3 opinions of the same preds.\n")

    # ---- 7. overlays
    add("## 7. Worst-view overlays\n")
    cand = pv[pv["n_gt"] >= 3].sort_values(["pq", "n_gt"], ascending=[True, False])
    cand = cand.drop_duplicates("stem").head(args.n_overlays)
    rows = []
    for rank, r in enumerate(cand.itertuples(), 1):
        stem = r.stem
        img = cv2.resize(cv2.imread(f"{DATA}/train_images/{stem}.jpeg", 0), (1024, 1024), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(cv2.convertScaleAbs(img, alpha=1.3, beta=0), cv2.COLOR_GRAY2BGR)
        iou, hit = iou_cache[r.view]

        def contours(c: str) -> list:
            m = cv2.resize(decode_rle(c), (1024, 1024), interpolation=cv2.INTER_NEAREST)
            return cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        pdf = pred_by_stem.get(stem)
        if pdf is not None:
            for j, c in enumerate(pdf["segmentation_rle"]):
                col = (0, 255, 255) if hit[:, j].any() else (0, 0, 255)
                cv2.drawContours(img, contours(c), -1, col, 1)
        for c in gt_by_view[r.view]["segmentation_rle"]:
            cv2.drawContours(img, contours(c), -1, (0, 255, 0), 1)
        cv2.putText(img, f"{r.view} PQ={r.pq:.2f} gt={r.n_gt} tp={r.tp} fp={r.fp} fn={r.fn}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        fn = out_dir / f"overlay_{rank:02d}_{r.view}.png"
        cv2.imwrite(str(fn), img)
        rows.append({"rank": rank, "view": r.view, "PQ": r.pq, "GT": r.n_gt, "TP": r.tp, "FP": r.fp, "FN": r.fn,
                     "file": f"artifacts/{args.run}/analysis/{fn.name}"})
    add("Worst per-view PQ among views with >= 3 GT (one view per stem; ties by more GT first). GT contours green, "
        "TP-matched preds yellow, other preds red.\n")
    add(md_table(pd.DataFrame(rows)) + "\n")

    # ---- implications (ranked by computed upper-bound gain)
    fp_unsup_share = unsup / FP
    cands = [
        (a_pq - base, f"Mask quality (oracle masks on already-detected instances): +{a_pq - base:.3f} PQ upper bound; SQ is only "
         f"{res.sq:.3f} and inter-annotator SQ is also 0.631, so only part of this is reachable (annotator noise). "
         f"{near_fn} near-miss FN and {near_fp} near-miss FP (IoU 0.3-0.5) point to shape/boundary errors: higher "
         f"resolution/thin-structure handling, boundary-aware loss, mask refinement."),
        (b2 - base, f"Near-miss conversion: turning the {n_nm} near-miss FNs (and their {len(nm_best_fp)} FPs) into TPs at 0.55 gives "
         f"+{b2 - base:.3f} PQ; same remedies as mask quality (they overlap with the item above, not additive)."),
        (c_rec - base, f"Threshold/ranking recovery: {n_rec} of {FN} FNs have a raw instance with IoU>0.5 "
         f"(median raw score {rec['raw_best_score'].median():.3f}); upper bound +{c_rec - base:.3f} PQ ignoring the FPs "
         f"admitted at lower thresholds. Worth trying: per-image adaptive threshold, rescoring/learned confidence, "
         f"keeping low-score instances when the image has few detections."),
        (f_pq - base, f"Detection recall: {n_np} FNs ({100 * n_np / FN:.0f}%) are never proposed even at conf 0.01 "
         f"(median area {nonrec['area'].median():.0f} px; see 3a/3b for where). Recovering them needs a better "
         f"detector (larger imgsz, more data/augmentation, longer training); gain shown assumes SQ-level IoU for all."),
        (e_pq - base, f"Fragmentation: {nfrag} GT instances covered by >=2 non-hitting preds ({frag_fp_n} FP); union of fragments "
         f"would hit {fr_ok} of them; fixing gives up to +{e_pq - base:.3f} PQ. Merging {nmerge} preds spans >=2 GT "
         f"({merge_fn_n} FN involved). Try mask-union / NMS-by-containment post-processing."),
        (d_pq - base, f"Unsupported FPs: {unsup} FP pairs on multi-view stems are hit by no annotator at all; removing them would give "
         f"+{d_pq - base:.3f} PQ. The other {sup_fp} FP pairs on multi-view stems are filaments some annotator marked: they "
         f"are annotator disagreement, not model error, and cannot be fixed beyond choosing which annotator style to follow "
         f"(e.g. favour high-consensus structures: precision vs. support, see section 4)."),
    ]
    cands.sort(key=lambda t: -t[0])
    add("## Implications (ranked by estimated PQ upper-bound gain; not additive)\n")
    for k, (_, txt) in enumerate(cands, 1):
        add(f"{k}. {txt}")
    add(f"{len(cands) + 1}. Ceiling: the inter-annotator PQ is 0.337 (SQ 0.631) while the model scores {res.pq:.3f}; "
        f"multi-annotator stems score {pq_calc(*[pv[pv['multi'] == 'multi-annotator stems'][c].sum() for c in ('tp', 'fp', 'fn', 'sum_iou')]):.3f} "
        f"vs single-annotator {pq_calc(*[pv[pv['multi'] == 'single-annotator stems'][c].sum() for c in ('tp', 'fp', 'fn', 'sum_iou')]):.3f}. "
        f"Gains from further tuning are bounded by label noise; prioritise changes that are cheap and robust.")
    doc_path.write_text("\n".join(md) + "\n")
    print(f"wrote {doc_path} and {len(rows)} overlays in {out_dir}")
    print(f"PQ {res.pq:.4f} TP/FP/FN {TP}/{FP}/{FN} OK")


def hit_of_view(cache: dict, view: str, j: int) -> bool:
    return bool(cache[view][1][:, j].any())


if __name__ == "__main__":
    main()
