import numpy as np
import pytest
from pycocotools import mask as mask_util

from filseg.rle import decode_many, decode_rle, encode_mask, rle_run_total


def test_roundtrip_random_and_thin():
    rng = np.random.default_rng(0)
    rand = rng.random((2048, 2048)) > 0.7
    thin = np.zeros((2048, 2048), bool)
    thin[500, 10:2000] = True
    thin[::7, 3] = True
    diag = np.eye(2048, dtype=bool)
    for m in (rand, thin, diag, np.zeros((2048, 2048), bool), np.ones((2048, 2048), bool)):
        s = encode_mask(m)
        assert isinstance(s, str)
        assert np.array_equal(decode_rle(s), m.astype(np.uint8))
        assert rle_run_total(s) == 2048 * 2048
    many = decode_many([encode_mask(thin), encode_mask(diag)])
    assert many.shape == (2, 2048, 2048) and np.array_equal(many[1], diag)
    assert decode_many([]).shape == (0, 2048, 2048)


def test_orientation_asymmetric():
    m = np.zeros((2048, 2048), np.uint8)
    m[10:20, 100:400] = 1  # wide block near top, left-ish
    m[1500, 5] = 1
    d = decode_rle(encode_mask(m))
    assert np.array_equal(d, m)
    assert d[15, 200] == 1 and d[200, 15] == 0 and d[1500, 5] == 1
    # matches pycocotools Fortran-order convention directly
    ref = mask_util.encode(np.asfortranarray(m))["counts"].decode()
    assert encode_mask(m) == ref


def test_shape_check():
    with pytest.raises(ValueError):
        encode_mask(np.zeros((100, 100), bool))
    s = encode_mask(np.eye(100, dtype=bool), size=(100, 100))
    assert np.array_equal(decode_rle(s, 100, 100), np.eye(100, dtype=np.uint8))
