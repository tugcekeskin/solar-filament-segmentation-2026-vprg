"""Merge fragmented instance predictions (post-processing on final, non-overlapping rows).

Two predicted instances of the same stem are linked when one mask, dilated by ``radius`` px,
touches the other. Linked groups (connected components, at most ``max_group`` members) become one
instance: the union of the masks, optionally with the gap bridged by a morphological closing.
Only pairs whose principal axes are roughly aligned (|angle difference| <= ``max_angle`` deg)
are linked, so that distinct neighbouring filaments crossing at an angle stay apart.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from filseg.rle import decode_rle, encode_mask


def _axis_angle(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return 0.0
    c = np.cov(np.stack([xs, ys]).astype(np.float64))
    w, v = np.linalg.eigh(c)
    vx, vy = v[:, 1]
    return math.degrees(math.atan2(vy, vx)) % 180.0


def _bbox(m: np.ndarray) -> tuple[int, int, int, int]:
    ys = np.nonzero(m.any(1))[0]
    xs = np.nonzero(m.any(0))[0]
    return ys[0], ys[-1] + 1, xs[0], xs[-1] + 1


def merge_stem(rles: list[str], radius: int, max_angle: float = 180.0, max_group: int = 3,
               close: bool = False) -> list[str]:
    """Return RLEs after merging fragments. Input masks are assumed non-overlapping."""
    n = len(rles)
    if n < 2 or radius <= 0:
        return list(rles)
    masks = [decode_rle(r).astype(np.uint8) for r in rles]
    boxes = [_bbox(m) for m in masks]
    angles = [_axis_angle(m) for m in masks]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    parent = list(range(n))
    size = [1] * n

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        y0, y1, x0, x1 = boxes[i]
        Y0, Y1, X0, X1 = max(y0 - radius, 0), y1 + radius, max(x0 - radius, 0), x1 + radius
        dil = None
        for j in range(i + 1, n):
            b = boxes[j]
            if b[0] >= Y1 or b[1] <= Y0 or b[2] >= X1 or b[3] <= X0:
                continue
            d = abs(angles[i] - angles[j]) % 180.0
            if min(d, 180.0 - d) > max_angle:
                continue
            if dil is None:
                dil = cv2.dilate(masks[i][Y0:Y1, X0:X1], k)
            if not (dil & masks[j][Y0:Y1, X0:X1]).any():
                continue
            ri, rj = find(i), find(j)
            if ri != rj and size[ri] + size[rj] <= max_group:
                parent[rj] = ri
                size[ri] += size[rj]
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out: list[str] = []
    for root in sorted(groups):
        idx = groups[root]
        if len(idx) == 1:
            out.append(rles[idx[0]])
            continue
        u = np.zeros_like(masks[0])
        for i in idx:
            u |= masks[i]
        if close:
            c = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k)
            others = np.zeros_like(u)
            for j in range(n):
                if j not in idx:
                    others |= masks[j]
            u = c & (1 - others)  # never create overlaps with other instances
        out.append(encode_mask(u))
    return out
