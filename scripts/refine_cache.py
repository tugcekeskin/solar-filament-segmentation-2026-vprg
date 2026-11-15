#!/usr/bin/env python
"""Refine the soft masks of a raw cache with the crop refiner; output has the same schema."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from filseg.infer import SOFT_COLUMN, load_cache, save_cache, unpack_soft  # noqa: E402
from filseg.refine import IMG_DIR, IMG_DIR_TEST, load_model, refine_stem  # noqa: E402
from filseg.rle import H, W, encode_mask  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-score", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--crop-frac", type=float, default=0.25, help="window padding fraction (match the refiner's training)")
    a = p.parse_args()
    raw = load_cache(a.cache)
    raw = raw[raw["score"] >= a.min_score]
    model = load_model(a.weights)
    t = time.time()
    rows = []
    for stem, g in raw.groupby("stem", sort=True):
        path = Path(IMG_DIR) / f"{stem}.jpeg"
        if not path.exists():
            path = Path(IMG_DIR_TEST) / f"{stem}.jpeg"
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        new = refine_stem(model, img, g[SOFT_COLUMN].tolist(), a.batch, frac=a.crop_frac)
        for sc, nt in zip(g["score"], new):
            if nt is None:
                continue
            y0, x0, crop = unpack_soft(nt)
            full = np.zeros((H, W), np.uint8)
            full[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop >= 128
            area = int(full.sum())
            if area == 0:
                continue
            rows.append((stem, float(sc), encode_mask(full), area, nt))
    df = pd.DataFrame(rows, columns=["stem", "score", "rle", "area", SOFT_COLUMN])
    save_cache(df, a.out)
    print(f"refined {len(raw)} -> {len(df)} instances over {df['stem'].nunique()} stems in {time.time() - t:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
