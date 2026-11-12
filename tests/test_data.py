import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg import data as D  # noqa: E402


def test_normalize_roundtrip():
    seg = [100, 200, 300, 200, 300, 400, 100, 400, 100, 200]  # closed
    n = D.normalize_polygon(seg)
    assert len(n) == 8 and all(0 <= v <= 1 for v in n)
    back = D.denormalize_polygon(n)
    assert np.allclose(back, seg[:-2])


def test_clip_and_degenerate():
    n = D.normalize_polygon([-5, 0, 3000, 0, 3000, 100])
    assert min(n) == 0.0 and max(n) == 1.0
    assert D.normalize_polygon([1, 1, 2, 2, 1, 1]) is None


def test_split_no_leakage():
    stems = [f"{y}{m:02d}{d:02d}1200000{i}" for y in range(2011, 2015)
             for m in range(1, 13) for d in (3, 17) for i in range(3)]
    sp = D.make_split(stems)
    assert set(sp["folds"]) == set(stems)
    assert not ({s[:8] for s in sp["train"]} & {s[:8] for s in sp["val"]})
    assert len(sp["train"]) + len(sp["val"]) == len(stems)
    assert sp == D.make_split(stems)


def test_label_line_format():
    n = D.normalize_polygon([0, 0, 1024, 0, 1024, 1024, 0, 1024])
    line = D.yolo_line(n)
    assert line.split()[0] == "0"
    assert D.parse_yolo_line(line) == [float(f"{v:.6f}") for v in n]


def test_rasterize_square_area():
    m = D.rasterize([10, 10, 30, 10, 30, 40, 10, 40])
    assert m.shape == (2048, 2048) and m.sum() == 20 * 30
    assert D.rasterize_at([10, 10, 30, 10, 30, 40, 10, 40], 1024).sum() == 4 * 10 * 15
