from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from filseg.gt import build_gt_df, first_view_as_pred
from filseg.metrics import fast_evaluate, reference_overlap_df, reference_pq_counts
from filseg.rle import decode_rle, encode_mask

JSON = Path(__file__).resolve().parents[1] / "data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
needs_data = pytest.mark.skipif(not JSON.exists(), reason="train json missing")


def rect(r0, r1, c0, c1):
    m = np.zeros((2048, 2048), bool)
    m[r0:r1, c0:c1] = True
    return m


def toy_dfs():
    gt = pd.DataFrame({
        "filament_id": ["a-s1_1", "a-s1_2"],
        "segmentation_rle": [encode_mask(rect(0, 10, 0, 10)), encode_mask(rect(100, 110, 0, 10))],
    })
    pred = pd.DataFrame({
        "filament_id": ["s1_1", "s1_2", "s1_3", "other_1"],
        "segmentation_rle": [encode_mask(rect(0, 10, 0, 8)),      # IoU 0.8 with GT0 -> TP
                             encode_mask(rect(0, 10, 8, 20)),     # IoU 20/200=0.1 with GT0 -> FP, split
                             encode_mask(rect(100, 110, 0, 5)),   # IoU 0.5 exactly -> not a hit
                             encode_mask(rect(0, 10, 0, 10))],    # stem without GT -> ignored
    })
    return gt, pred


def test_toy_hand_computed():
    gt, pred = toy_dfs()
    r = fast_evaluate(gt, pred)
    assert (r.tp, r.fp, r.fn) == (1, 2, 1)
    assert r.pq == pytest.approx(0.8 / 2.5)
    assert r.sq == pytest.approx(0.8) and r.rq == pytest.approx(1 / 2.5)
    assert r.dice == pytest.approx(2 * 80 / 180)
    assert r.splits == 1 and r.merges == 0 and r.dup_hits == 0
    assert r.n_views == 1 and r.n_gt == 2 and r.n_pred == 3
    assert list(r.per_view.columns) == ["view", "stem", "n_gt", "n_pred", "tp", "fp", "fn", "sum_iou", "pq"]
    rpq, tp, fp, fn = reference_pq_counts(reference_overlap_df(gt, pred))
    assert (tp, fp, fn) == (1, 2, 1) and rpq == pytest.approx(r.pq, abs=1e-6)


def test_merge_counted():
    gt = pd.DataFrame({"filament_id": ["a-s_1", "a-s_2"],
                       "segmentation_rle": [encode_mask(rect(0, 10, 0, 10)), encode_mask(rect(0, 10, 10, 20))]})
    pred = pd.DataFrame({"filament_id": ["s_1"], "segmentation_rle": [encode_mask(rect(0, 10, 0, 20))]})
    r = fast_evaluate(gt, pred)
    assert r.merges == 1 and r.tp == 0 and r.fp == 1 and r.fn == 2  # IoU 0.5 each: no hit


def test_empty_preds():
    gt, _ = toy_dfs()
    empty = pd.DataFrame({"filament_id": pd.Series([], dtype=object), "segmentation_rle": pd.Series([], dtype=object)})
    r = fast_evaluate(gt, empty)
    assert r.pq == 0.0 and r.fn == len(gt) and r.tp == 0 and r.fp == 0
    assert reference_pq_counts(reference_overlap_df(gt, empty))[3] == len(gt)


@pytest.fixture(scope="module")
def real_gt():
    if not JSON.exists():
        pytest.skip("train json missing")
    import json
    stems = sorted({im["id"].split("-", 1)[1] for im in json.load(open(JSON))["images"]})
    return build_gt_df(str(JSON), set(stems[:25]))


@needs_data
def test_gt_vs_gt(real_gt):
    view = real_gt["filament_id"].str.split("_", n=1).str[0]
    stem = view.str.split("-", n=1).str[1]
    single = stem.map(stem.groupby(stem).apply(lambda x: view[x.index].nunique())) == 1
    g = real_gt[single]
    assert g["filament_id"].str.split("_", n=1).str[0].nunique() >= 10
    p = pd.DataFrame({"filament_id": g["filament_id"].str.split("-", n=1).str[1].values,
                      "segmentation_rle": g["segmentation_rle"].values})
    r = fast_evaluate(g, p)
    assert r.pq == pytest.approx(1.0) and r.fp == 0 and r.fn == 0 and r.tp == len(g)
    assert reference_pq_counts(reference_overlap_df(g, p))[0] == pytest.approx(1.0)


def perturb(mask, kind, rng):
    m = mask.copy()
    if kind == "dilate":
        for _ in range(3):
            m = m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
    elif kind == "erode":
        for _ in range(2):
            m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    elif kind == "shift":
        m = np.roll(m, (int(rng.integers(-6, 7)), int(rng.integers(-6, 7))), (0, 1))
    return m


@needs_data
def test_fast_equals_reference_perturbed(real_gt):
    rng = np.random.default_rng(1)
    base = first_view_as_pred(real_gt)
    stems = base["filament_id"].str.split("_", n=1).str[0]
    rows = []
    kinds = ["none", "dilate", "erode", "shift", "drop", "shift"]
    for i, (fid, rle) in enumerate(zip(base["filament_id"], base["segmentation_rle"])):
        kind = kinds[rng.integers(len(kinds))]
        if kind == "drop":
            continue
        m = perturb(decode_rle(rle).astype(bool), kind, rng)
        if m.any():
            rows.append((fid, encode_mask(m)))
    for stem in stems.unique()[::3]:  # extra random blobs
        r0, c0 = rng.integers(0, 1900, 2)
        rows.append((f"{stem}_999", encode_mask(rect(r0, r0 + 80, c0, c0 + 40))))
    pred = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    n_views = real_gt["filament_id"].str.split("_", n=1).str[0].nunique()
    assert n_views >= 30
    fast = fast_evaluate(real_gt, pred)
    rpq, tp, fp, fn = reference_pq_counts(reference_overlap_df(real_gt, pred))
    assert (fast.tp, fast.fp, fast.fn) == (tp, fp, fn)
    assert abs(fast.pq - rpq) < 1e-6
    assert 0.2 < fast.pq < 1.0 and fast.fp > 0 and fast.fn > 0


def test_duplicate_flagged():
    gt, pred = toy_dfs()
    dup = pd.concat([pred, pd.DataFrame({"filament_id": ["s1_4"], "segmentation_rle": [pred["segmentation_rle"][0]]})])
    r = fast_evaluate(gt, dup)
    assert r.dup_hits == 1 and r.tp == 2
