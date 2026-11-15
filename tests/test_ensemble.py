from __future__ import annotations

import numpy as np

from filseg.ensemble import ensemble_caches, ensemble_stem
from filseg.infer import pack_soft, unpack_soft
from filseg.rle import decode_rle
import pandas as pd


def _inst(model, score, y0, x0, h, w, val=255):
    return model, score, pack_soft(y0, x0, np.full((h, w), val, np.uint8))


def test_single_model_identity():
    insts = [_inst(0, 0.9, 10, 10, 20, 20), _inst(0, 0.5, 100, 100, 5, 8, 200)]
    out = ensemble_stem(insts, 1)
    assert [round(s, 6) for s, _ in out] == [0.9, 0.5]
    for (_, _, src), (_, got) in zip(insts, out):
        a, b = unpack_soft(src), unpack_soft(got)
        assert a[:2] == b[:2] and np.array_equal(a[2], b[2])


def test_overlapping_pair_merges():
    insts = [_inst(0, 0.8, 0, 0, 10, 10, 200), _inst(1, 0.6, 0, 0, 10, 10, 140)]
    out = ensemble_stem(insts, 2)
    assert len(out) == 1
    score, soft = out[0]
    assert abs(score - 0.7) < 1e-9
    _, _, crop = unpack_soft(soft)
    assert (crop == 170).all()


def test_non_overlapping_split_and_min_votes():
    insts = [_inst(0, 0.8, 0, 0, 10, 10), _inst(1, 0.6, 500, 500, 10, 10)]
    out = ensemble_stem(insts, 2)
    assert len(out) == 2
    for _, soft in out:
        assert unpack_soft(soft)[2].max() == 128  # 255/2 rounded, still >= 128
    assert [round(s, 6) for s, _ in out] == [0.4, 0.3]
    assert ensemble_stem(insts, 2, min_votes=2) == []


def test_one_member_per_model():
    insts = [_inst(0, 0.9, 0, 0, 10, 10), _inst(0, 0.8, 0, 0, 10, 10), _inst(1, 0.7, 0, 0, 10, 10)]
    out = ensemble_stem(insts, 2)
    assert len(out) == 2
    assert abs(out[0][0] - 0.8) < 1e-9 and abs(out[1][0] - 0.4) < 1e-9


def test_ensemble_caches_schema():
    def df(model_scores):
        rows = [{"stem": "a", "score": s, "rle": "", "area": 0, "soft": pack_soft(4, 4, np.full((6, 6), 255, np.uint8))} for s in model_scores]
        return pd.DataFrame(rows)

    out = ensemble_caches([df([0.9]), df([0.7])], 0.5, 1)
    assert list(out.columns) == ["stem", "score", "rle", "area", "soft"]
    assert len(out) == 1 and out["area"].iloc[0] == 36
    m = decode_rle(out["rle"].iloc[0])
    assert m.sum() == 36 and m[4:10, 4:10].all()
