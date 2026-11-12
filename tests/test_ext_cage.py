import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg import ext_cage  # noqa: E402
from filseg.infer import pack_soft, unpack_soft  # noqa: E402
from filseg.rle import decode_rle  # noqa: E402


def _probs():
    p = np.zeros((3, 1024, 1024), np.uint8)
    p[0, 100:140, 100:300] = 230  # mask inside hull
    p[1, 90:150, 90:310] = 230
    p[0, 600:620, 600:700] = 230  # fallback: mask w/o hull
    p[0, 800:805, 800:810] = 230  # tiny -> dropped
    return p


def test_mode_b_instances_and_rows():
    insts = ext_cage.mode_b_instances(_probs())
    assert len(insts) == 2
    rows = ext_cage.instances_to_rows("s", insts)
    df = ext_cage.rows_to_df(rows)
    assert list(df.columns) == ["stem", "score", "rle", "area", "soft"]
    r = df.iloc[0]
    assert 0.85 < r["score"] <= 1.0
    assert decode_rle(r["rle"]).sum() == r["area"] > 500
    y0, x0, crop = unpack_soft(r["soft"])
    assert y0 <= 180 and crop.max() > 200
    # soft is zero outside the hull component
    assert crop[0, 0] == 0


def test_refine_alpha_and_score_unchanged():
    import pandas as pd

    df = ext_cage.rows_to_df(ext_cage.instances_to_rows("s", ext_cage.mode_b_instances(_probs())))
    cage = {"s": np.full((2048, 2048), 100, np.uint8)}
    out = ext_cage.refine_cache(df, cage, 0.5)
    assert (out["score"] == df["score"]).all()
    y0, x0, c0 = unpack_soft(df.iloc[0]["soft"])
    _, _, c1 = unpack_soft(out.iloc[0]["soft"])
    assert abs(int(c1.max()) - (0.5 * int(c0.max()) + 50)) <= 1
    # unknown stem untouched
    d2 = df.assign(stem="zz")
    assert ext_cage.refine_cache(d2, cage, 0.5)["rle"].tolist() == d2["rle"].tolist()
    # alpha=1 keeps soft
    o1 = ext_cage.refine_cache(df, cage, 1.0)
    assert unpack_soft(o1.iloc[0]["soft"])[2].tolist() == c0.tolist()


def test_preprocess_range():
    x = ext_cage.preprocess(np.full((2048, 2048), 128, np.uint8))
    assert x.shape == (1024, 1024) and abs(x[0, 0] - 0.0078 * 1) < 0.02
