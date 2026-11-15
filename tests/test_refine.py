import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.infer import pack_soft, unpack_soft  # noqa: E402
from filseg.refine import SZ, crop_window, cut, mask_bbox, paste_back, place, soft_bbox_window, to_sz  # noqa: E402
from filseg.rle import H, W  # noqa: E402


def test_window_square_padded_inside():
    wy0, wx0, s = crop_window(100, 140, 500, 900)  # 40x400: pad 100 -> side 600
    assert s == 600
    assert 0 <= wy0 <= H - s and 0 <= wx0 <= W - s
    assert wx0 <= 500 - 100 and wx0 + s >= 900 + 100 - 1
    assert wy0 <= 100 and wy0 + s >= 140  # contains bbox vertically


def test_window_min_pad_and_clip():
    wy0, wx0, s = crop_window(0, 10, 0, 10)
    assert s == 42 and (wy0, wx0) == (0, 0)
    wy0, wx0, s = crop_window(2000, 2048, 2000, 2048)
    assert wy0 + s == H and wx0 + s == W
    assert crop_window(0, 2048, 0, 2048)[2] == 2048


def test_place_and_cut():
    win = (10, 20, 50)
    a = np.zeros((H, W), np.uint8)
    a[30:40, 30:40] = 7
    c = place(25, 25, a[25:45, 25:45], win)
    assert c.shape == (50, 50) and (c == cut(a, win)).sum() >= (c > 0).sum()
    assert c[30 - 10, 30 - 20] == 7


def _roundtrip(mask):
    ys, xs = np.nonzero(mask)
    win = crop_window(ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    soft = (mask * 255).astype(np.uint8)
    prior = to_sz(cut(soft, win))
    assert prior.shape == (SZ, SZ)
    packed = paste_back(prior.astype(np.float32) / 255.0, win)  # identity "refiner"
    y0, x0, crop = unpack_soft(packed)
    out = np.zeros((H, W), bool)
    out[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop >= 128
    return out


def test_roundtrip_thin_mask_iou():
    m = np.zeros((H, W), np.uint8)
    m[800:1000, 600:603] = 1  # 200x3 line
    m[1000:1003, 600:700] = 1
    out = _roundtrip(m)
    inter = (out & (m > 0)).sum()
    iou = inter / ((out | (m > 0)).sum())
    assert iou > 0.97, iou


def test_roundtrip_diagonal_thin_mask():
    import cv2

    m = np.zeros((H, W), np.uint8)
    cv2.line(m, (300, 300), (700, 520), 1, 3)
    out = _roundtrip(m)
    iou = (out & (m > 0)).sum() / (out | (m > 0)).sum()
    assert iou > 0.97, iou


def test_soft_bbox_window():
    crop = np.zeros((30, 40), np.uint8)
    crop[5:10, 8:30] = 200
    crop[0, 0] = 100  # below 128: ignored
    t = pack_soft(100, 200, crop)
    win = soft_bbox_window(t)
    assert win == crop_window(105, 110, 208, 230)
    assert soft_bbox_window(pack_soft(0, 0, np.zeros((4, 4), np.uint8))) is None
