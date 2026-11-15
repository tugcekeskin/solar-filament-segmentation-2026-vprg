import numpy as np
import pandas as pd

from filseg.postprocess import PPParams, group_by_stem, postprocess_stem, postprocess_to_rows
from filseg.rle import decode_rle, encode_mask


def _rect(y0, y1, x0, x1):
    m = np.zeros((2048, 2048), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def _raw():
    ms = [_rect(0, 100, 0, 100), _rect(0, 100, 50, 150), _rect(500, 520, 500, 520), _rect(900, 905, 900, 905)]
    sc = [0.9, 0.6, 0.3, 0.8]
    return pd.DataFrame({"stem": "a", "score": sc, "rle": [encode_mask(m) for m in ms], "area": [int(m.sum()) for m in ms]})


def test_overlap_resolution_disjoint():
    g = group_by_stem(_raw())
    masks, scores = postprocess_stem(*g["a"], PPParams(conf=0.05))
    assert len(masks) >= 2
    tot = np.sum([m.astype(int) for m in masks], axis=0)
    assert tot.max() <= 1
    assert scores == sorted(scores, reverse=True)


def test_conf_and_min_area_filters():
    g = group_by_stem(_raw())
    assert len(postprocess_stem(*g["a"], PPParams(conf=0.7))[0]) == 2  # 0.9 and 0.8
    assert len(postprocess_stem(*g["a"], PPParams(conf=0.05, min_area=100))[0]) == 3  # tiny dropped
    assert len(postprocess_stem(*g["a"], PPParams(conf=0.05, max_det=1))[0]) == 1


def test_nms_and_rows_unique_ids():
    g = group_by_stem(_raw())
    n_nms = len(postprocess_stem(*g["a"], PPParams(conf=0.05, nms=0.3))[0])
    n_none = len(postprocess_stem(*g["a"], PPParams(conf=0.05))[0])
    assert n_nms <= n_none
    rows = postprocess_to_rows(g, PPParams(conf=0.05))
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)) and ids[0] == "a_1"
    assert decode_rle(rows[0][1]).shape == (2048, 2048)


def test_erode_matches_cv2_and_drops_vanished():
    import cv2

    from filseg.postprocess import erode_window, transform_stem

    m = _rect(100, 110, 200, 260)  # 10 x 60 bar
    thin = _rect(500, 502, 500, 560)  # 2 px thin: vanishes with k=1
    edge = _rect(0, 10, 0, 60)  # touches image edge: default cv2 border keeps edge
    for shape, k in [("square", 1), ("cross", 1), ("square", 2), ("ellipse", 2)]:
        from filseg.postprocess import _kernel

        for mm in (m, edge):
            ys, xs = np.nonzero(mm)
            y0, x0 = ys.min(), xs.min()
            ey0, ex0, w = erode_window(mm[y0 : ys.max() + 1, x0 : xs.max() + 1], y0, x0, k, shape)
            full = np.zeros_like(mm)
            full[ey0 : ey0 + w.shape[0], ex0 : ex0 + w.shape[1]] = w
            assert np.array_equal(full, cv2.erode(mm, _kernel(k, shape)))
    ms = [m, thin]
    raw = pd.DataFrame({"stem": "a", "score": [0.9, 0.8], "rle": [encode_mask(x) for x in ms], "area": [int(x.sum()) for x in ms]})
    sc, rl, ar = transform_stem(*group_by_stem(raw)["a"], None, erode=1)
    assert len(rl) == 1 and ar[0] == 8 * 58
    rows = postprocess_to_rows(group_by_stem(raw), PPParams(conf=0.05, erode=1))
    assert len(rows) == 1


def test_soft_threshold():
    from filseg.infer import pack_soft
    from filseg.postprocess import transform_stem

    soft = np.zeros((20, 30), np.uint8)
    soft[:, :10] = 200
    soft[:, 10:20] = 140
    m = _rect(50, 70, 100, 120)
    raw = pd.DataFrame({"stem": "a", "score": [0.9], "rle": [encode_mask(m)], "area": [400], "soft": [pack_soft(50, 100, soft)]})
    from filseg.postprocess import group_soft_by_stem

    g, sg = group_by_stem(raw), group_soft_by_stem(raw)
    for thr, expect in [(0.5, 400), (0.6, 200), (0.9, 0)]:
        sc, rl, ar = transform_stem(*g["a"], sg["a"], mask_thr=thr)
        assert (ar[0] if len(ar) else 0) == expect
    rows = postprocess_to_rows(g, PPParams(conf=0.05, mask_thr=0.6, erode=1), soft_grouped=sg)
    m2 = decode_rle(rows[0][1])
    assert m2.sum() == 18 * 8


def test_unflip_soft_roundtrip():
    import numpy as np
    from filseg.infer import flip_array, pack_soft, unflip_soft, unpack_soft

    crop = np.arange(12, dtype=np.uint8).reshape(3, 4)
    full = np.zeros((2048, 2048), np.uint8)
    full[10:13, 20:24] = crop
    for f in ("h", "v", "hv"):
        flipped = flip_array(full, f)
        ys, xs = np.nonzero(flipped)
        y0, x0 = ys.min(), xs.min()
        t = pack_soft(y0, x0, flipped[y0 : y0 + 3, x0 : x0 + 4])
        y1, x1, c = unpack_soft(unflip_soft(t, f))
        assert (y1, x1) == (10, 20) and (c == crop).all()
