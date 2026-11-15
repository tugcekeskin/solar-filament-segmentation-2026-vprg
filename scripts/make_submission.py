#!/usr/bin/env python
"""Apply calibrated params to the test cache, write + validate submission.csv (and val_pred.csv/eval json)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from filseg.gt import build_gt_df  # noqa: E402
from filseg.infer import load_cache  # noqa: E402
from filseg.metrics import fast_evaluate  # noqa: E402
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows  # noqa: E402
from filseg.submission import COLUMNS, validate_submission, write_submission  # noqa: E402

ROOT = "data/raw/MAGFiLO_1.0_Kaggle_2026"
DEFAULT_GT = f"{ROOT}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="artifacts/<run> (holds calib.json, *_raw.pkl)")
    p.add_argument("--test-cache")
    p.add_argument("--val-cache")
    p.add_argument("--gt-json", default=DEFAULT_GT)
    p.add_argument("--split", default="data/splits/split_v1.json")
    a = p.parse_args()
    d = Path(a.run_dir)
    bp = json.load(open(d / "calib.json"))["best_params"]
    params = PPParams(bp["conf"], bp["min_area"], bp["max_det"], bp["nms"], bp.get("erode", 0),
                      bp.get("erode_shape", "square"), bp.get("mask_thr"))
    print("params", params)

    # val predictions + eval
    val_stems = sorted(json.load(open(a.split))["val"])
    vraw = load_cache(a.val_cache or d / "val_raw.pkl")
    vg, vs = group_by_stem(vraw), (group_soft_by_stem(vraw) if "soft" in vraw.columns else None)
    vdf = write_submission(postprocess_to_rows(vg, params, val_stems, vs), str(d / "val_pred.csv"))
    res = fast_evaluate(build_gt_df(a.gt_json, set(val_stems)), vdf)
    (d / "val_eval.json").write_text(json.dumps(res.to_dict(per_view=False), indent=1))
    print(f"val PQ={res.pq:.4f} SQ={res.sq:.4f} RQ={res.rq:.4f} TP={res.tp} FP={res.fp} FN={res.fn}")

    # test submission
    test_stems = sorted(p_.stem for p_ in Path(f"{ROOT}/test/test_images").glob("*.jpeg"))
    traw = load_cache(a.test_cache or d / "test_raw.pkl")
    tg, ts = group_by_stem(traw), (group_soft_by_stem(traw) if "soft" in traw.columns else None)
    sub = write_submission(postprocess_to_rows(tg, params, test_stems, ts), str(d / "submission.csv"))
    rep = validate_submission(str(d / "submission.csv"), set(test_stems))
    print(f"validator ok={rep.ok} rows={rep.n_rows} stems={rep.n_stems}/{len(test_stems)} "
          f"overlap_pairs={rep.n_overlap_pairs} errors={rep.errors[:5]} warnings={rep.warnings}")
    print(f"stems with zero predictions: {len(rep.stems_without_predictions)}")
    print(f"mean instances/stem (test): {len(sub) / len(test_stems):.2f}  (train GT mean ~7.1 per view)")
    if not rep.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
