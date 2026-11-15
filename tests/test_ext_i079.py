from pathlib import Path

import numpy as np
import pytest
import torch

from filseg import ext_i079 as E

ROOT = Path(__file__).resolve().parents[1]
have = (ROOT / E.DETECTOR_PATH).exists() and (ROOT / E.REFINER_PATH).exists()


def test_square_bounds():
    m = np.zeros((2048, 2048), bool)
    m[10:20, 5:15] = True
    x0, y0, x1, y1 = E.square_bounds(m)
    assert (x1 - x0) == (y1 - y0) == 96 and x0 >= 0 and y0 >= 0


def test_topology_guard_rejects_empty():
    s = np.ones((256, 256), np.float32)
    _, ok, why = E.topology_safe_select(s, np.zeros_like(s))
    assert not ok and why == "empty_tta_mask"


@pytest.mark.skipif(not have, reason="i079 weights absent")
def test_refiner_loads_weights_only_and_shapes():
    r = E.load_refiner(ROOT / E.REFINER_PATH, "cpu")
    crops = np.random.default_rng(0).integers(0, 255, (2, 256, 256), dtype=np.uint8)
    p = E.refine_probabilities(r, crops, torch.device("cpu"), tta=True)
    assert p.shape == (2, 256, 256) and p.min() >= 0 and p.max() <= 1


@pytest.mark.skipif(not have, reason="i079 weights absent")
def test_detector_loads_weights_only_and_runs_dummy():
    d = E.load_detector(ROOT / E.DETECTOR_PATH, "cpu")
    img = np.zeros((2048, 2048), np.uint8)
    masks, scores = E.detect(d, img, torch.device("cpu"), 0.3)
    assert masks.ndim == 3 and len(masks) == len(scores)
