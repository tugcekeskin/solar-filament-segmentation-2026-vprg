"""COCO compressed-RLE helpers matching the Kaggle organizer conventions.

Masks are 2048x2048; RLE is a pycocotools compressed ``counts`` string of the
Fortran-ordered uint8 mask (``size`` is implicit and fixed).
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util

H = 2048
W = 2048


def encode_mask(mask: np.ndarray, size: tuple[int, int] = (H, W)) -> str:
    """Encode a binary HxW mask to a compressed COCO counts string.

    Raises ValueError if ``mask.shape != size`` (default 2048x2048).
    """
    if mask.ndim != 2 or tuple(mask.shape) != tuple(size):
        raise ValueError(f"mask shape {mask.shape} != expected {tuple(size)}")
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    return counts.decode("utf-8") if isinstance(counts, bytes) else counts


def to_rle_obj(counts: str, h: int = H, w: int = W) -> dict:
    """Wrap a counts string as a pycocotools RLE dict."""
    return {"size": [h, w], "counts": counts}


def decode_rle(counts: str, h: int = H, w: int = W) -> np.ndarray:
    """Decode one counts string to a uint8 (h, w) mask."""
    return mask_util.decode(to_rle_obj(counts, h, w))


def decode_many(counts_list: list[str], h: int = H, w: int = W) -> np.ndarray:
    """Decode counts strings to a uint8 (n, h, w) stack (empty -> (0, h, w))."""
    if not counts_list:
        return np.zeros((0, h, w), dtype=np.uint8)
    masks = mask_util.decode([to_rle_obj(c, h, w) for c in counts_list])
    return np.ascontiguousarray(masks.transpose(2, 0, 1))


def rle_run_total(counts: str) -> int:
    """Sum of run lengths encoded in a compressed counts string.

    Pure-python port of pycocotools ``rleFrString``; lets callers verify that a
    string describes exactly h*w pixels before handing it to the C decoder.
    Raises ValueError on malformed strings or negative runs.
    """
    cnts: list[int] = []
    p = 0
    n = len(counts)
    while p < n:
        x = 0
        k = 0
        more = True
        while more:
            if p >= n:
                raise ValueError("truncated RLE string")
            c = ord(counts[p]) - 48
            if c < 0 or c > 63:
                raise ValueError(f"invalid RLE character {counts[p]!r}")
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[len(cnts) - 2]
        if x < 0:
            raise ValueError("negative run length")
        cnts.append(x)
    return sum(cnts)
