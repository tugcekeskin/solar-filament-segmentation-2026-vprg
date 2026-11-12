#!/usr/bin/env python
"""Evaluate a prediction CSV (or --gt-as-pred) against COCO GT with organizer PQ."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from filseg.gt import build_gt_df, first_view_as_pred, view_stem  # noqa: E402
from filseg.metrics import fast_evaluate, reference_overlap_df, reference_pq_counts  # noqa: E402

DEFAULT_GT = "data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", default=DEFAULT_GT)
    p.add_argument("--pred-csv")
    p.add_argument("--gt-as-pred", action="store_true", help="first annotator view per stem as predictions")
    p.add_argument("--split", help="split json {'train':[stems],'val':[stems]}")
    p.add_argument("--subset", default="val")
    p.add_argument("--stems-file", help="text file, one stem per line")
    p.add_argument("--max-views", type=int, help="restrict GT to N evenly spaced views (sorted by id)")
    p.add_argument("--out", help="write results json")
    p.add_argument("--reference", action="store_true", help="also run slow reference and assert equality")
    return p.parse_args()


def load_stems(args: argparse.Namespace) -> set[str] | None:
    """Stem filter from --stems-file or --split/--subset (None = all)."""
    if args.stems_file:
        return {l.strip() for l in open(args.stems_file) if l.strip()}
    if args.split:
        return set(json.load(open(args.split))[args.subset])
    return None


def main() -> None:
    args = parse_args()
    if not args.gt_as_pred and not args.pred_csv:
        sys.exit("need --pred-csv or --gt-as-pred")
    gt = build_gt_df(args.gt_json, load_stems(args))
    pred = first_view_as_pred(gt) if args.gt_as_pred else pd.read_csv(args.pred_csv, dtype=str, keep_default_na=False)
    if args.gt_as_pred:
        # Drop the chosen view from GT (no self-matching) and keep only multi-view stems, so the
        # score is a genuine inter-annotator agreement (~0.34 PQ on train), not an inflated self-score.
        gt_view = gt["filament_id"].str.split("_", n=1).str[0]
        chosen = {view_stem(v): v for v in sorted(gt_view.unique(), reverse=True)}
        gt = gt[~gt_view.isin(set(chosen.values()))]
    if args.max_views:
        views = sorted(gt["filament_id"].str.split("_", n=1).str[0].unique())
        keep = [views[i] for i in np.linspace(0, len(views) - 1, min(args.max_views, len(views))).astype(int)]
        gt = gt[gt["filament_id"].str.split("_", n=1).str[0].isin(keep)]
        stems = {view_stem(v) for v in keep}
        pred = pred[pred["filament_id"].str.split("_", n=1).str[0].isin(stems)]
    t = time.time()
    res = fast_evaluate(gt, pred)
    fast_s = time.time() - t
    print(f"{'PQ':>7}{'SQ':>7}{'RQ':>7}{'TP':>7}{'FP':>7}{'FN':>7}{'Dice':>7}{'splits':>7}{'merges':>7}{'dups':>6}")
    print(f"{res.pq:7.4f}{res.sq:7.4f}{res.rq:7.4f}{res.tp:7d}{res.fp:7d}{res.fn:7d}{res.dice:7.4f}"
          f"{res.splits:7d}{res.merges:7d}{res.dup_hits:6d}")
    print(f"views={res.n_views} gt={res.n_gt} pred={res.n_pred} fast_time={fast_s:.1f}s")
    out = res.to_dict(per_view=True)
    if args.reference:
        t = time.time()
        rpq, rtp, rfp, rfn = reference_pq_counts(reference_overlap_df(gt, pred))
        ref_s = time.time() - t
        assert (rtp, rfp, rfn) == (res.tp, res.fp, res.fn), f"count mismatch ref={(rtp, rfp, rfn)} fast={(res.tp, res.fp, res.fn)}"
        assert abs(rpq - res.pq) < 1e-6, f"PQ mismatch ref={rpq} fast={res.pq}"
        print(f"reference PQ={rpq:.6f} TP/FP/FN={rtp}/{rfp}/{rfn} |diff|={abs(rpq - res.pq):.2e} ref_time={ref_s:.1f}s  OK")
        out["reference_pq"] = rpq
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
