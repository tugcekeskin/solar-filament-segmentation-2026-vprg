import numpy as np
import pandas as pd

from filseg.rle import decode_rle, encode_mask
from filseg.submission import masks_to_rows, resolve_overlaps, validate_submission, write_submission


def rect(r0, r1, c0, c1):
    m = np.zeros((2048, 2048), bool)
    m[r0:r1, c0:c1] = True
    return m


def make_csv(tmp_path, rows):
    p = tmp_path / "sub.csv"
    write_submission(rows, str(p))
    return str(p)


def test_roundtrip_exact_strings(tmp_path):
    rng = np.random.default_rng(0)
    masks = [rng.random((2048, 2048)) > 0.5 for _ in range(3)]
    masks.append(rect(5, 9, 5, 1500))
    rows = masks_to_rows("stemA", masks)
    assert [r[0] for r in rows] == ["stemA_1", "stemA_2", "stemA_3", "stemA_4"]
    assert any("\\" in r[1] for r in rows), "want backslash coverage"
    p = make_csv(tmp_path, rows)
    assert open(p).readline().strip() == "filament_id,segmentation_rle"
    back = pd.read_csv(p)
    assert back["segmentation_rle"].tolist() == [r[1] for r in rows]
    assert '"' not in open(p).read()
    assert np.array_equal(decode_rle(back["segmentation_rle"][3]), masks[3].astype(np.uint8))


def test_validate_ok_and_warn(tmp_path):
    rows = masks_to_rows("a", [rect(0, 10, 0, 10), rect(50, 60, 0, 10)])
    rep = validate_submission(make_csv(tmp_path, rows), {"a", "b"})
    assert rep.ok and rep.n_overlap_pairs == 0
    assert rep.stems_without_predictions == ["b"] and rep.warnings


def test_validate_flags_duplicate_and_overlap(tmp_path):
    m = rect(0, 10, 0, 10)
    rows = masks_to_rows("a", [m, m])
    rep = validate_submission(make_csv(tmp_path, rows), {"a"})
    assert not rep.ok and rep.overlap_pairs == {"a": 1}


def test_validate_errors(tmp_path):
    good = masks_to_rows("a", [rect(0, 10, 0, 10)])
    # unknown stem
    rep = validate_submission(make_csv(tmp_path, good), {"z"})
    assert not rep.ok
    # duplicate ids
    rep = validate_submission(make_csv(tmp_path, good + good), {"a"})
    assert any("duplicated" in e for e in rep.errors)
    # empty mask, wrong size
    empty = [("a_1", encode_mask(np.zeros((2048, 2048), bool)))]
    assert not validate_submission(make_csv(tmp_path, empty), {"a"}).ok
    from pycocotools import mask as mu
    small = [("a_1", mu.encode(np.asfortranarray(np.ones((100, 100), np.uint8)))["counts"].decode())]
    assert not validate_submission(make_csv(tmp_path, small), {"a"}).ok
    # bad columns
    p = tmp_path / "bad.csv"
    pd.DataFrame({"id": ["a_1"], "rle": ["x"]}).to_csv(p, index=False)
    assert not validate_submission(str(p), {"a"}).ok


def test_resolve_overlaps():
    a = rect(0, 10, 0, 10)          # 100 px
    b = rect(0, 10, 5, 15)          # overlaps a on 50 px
    c = rect(0, 10, 0, 10)          # identical to a, lower score
    d = rect(0, 10, 8, 10)          # fully inside a/b region
    masks, idx = resolve_overlaps([a, b, c, d], [0.9, 0.8, 0.1, 0.05])
    assert idx == [0, 1]  # c, d are empty -> dropped
    assert not (masks[0] & masks[1]).any()
    assert masks[0].sum() == 100 and masks[1].sum() == 50
    # fully swallowed -> dropped; loses >50% -> dropped; loses exactly 50% -> kept
    assert resolve_overlaps([rect(0, 10, 0, 10), rect(0, 10, 3, 10)], [0.9, 0.5])[1] == [0]
    assert resolve_overlaps([rect(0, 10, 0, 10), rect(0, 10, 2, 14)], [0.9, 0.5])[1] == [0]  # keeps 40/120
    m3, i3 = resolve_overlaps([rect(0, 10, 0, 10), rect(0, 10, 5, 15)], [0.9, 0.5])
    assert i3 == [0, 1] and m3[1].sum() == 50
