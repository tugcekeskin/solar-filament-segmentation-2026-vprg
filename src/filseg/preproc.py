"""Contrast normalisation for H-alpha full-disk images (applied identically to train, val and test).

``flat3`` returns a 3-channel uint8 image:
  ch0 = original grey image (unchanged, so nothing is lost),
  ch1 = flat-fielded disk: image / local disk background (normalised Gaussian convolution,
        sigma ~60 px at 2048, restricted to the disk so the limb does not bleed), stretched so that
        a 10 % relative darkening (median filament contrast) moves ~40 grey levels,
  ch2 = CLAHE (clip 2, 8x8 tiles) of ch1.
Limb darkening is removed by the flat-field, so the relative contrast of a filament no longer depends
on its distance from the disk centre.
"""
from __future__ import annotations

import cv2
import numpy as np

SIGMA = 60.0
GAIN = 400.0  # grey levels per unit relative contrast (0.1 -> 40)


def disk_mask(g: np.ndarray, thr: int = 60) -> np.ndarray:
    m = (g > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == k).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    return m


def flat3(img: np.ndarray) -> np.ndarray:
    g = img if img.ndim == 2 else img[..., 0]
    g32 = g.astype(np.float32)
    m = disk_mask(g)
    mf = m.astype(np.float32)
    s = SIGMA * g.shape[0] / 2048.0
    num = cv2.GaussianBlur(g32 * mf, (0, 0), s)
    den = cv2.GaussianBlur(mf, (0, 0), s)
    bg = num / np.maximum(den, 1e-3)
    ratio = g32 / np.maximum(bg, 1.0)
    flat = np.clip(128.0 + (ratio - 1.0) * GAIN, 0, 255)
    flat[m == 0] = 0
    flat = flat.astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flat)
    cl[m == 0] = 0
    return np.dstack([g, flat, cl])
