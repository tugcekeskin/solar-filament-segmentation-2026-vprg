"""Offline post-processing of cached raw instances.

Pipeline per stem (all deterministic):
  1. drop instances with score < conf or area < min_area
  2. optional mask-IoU NMS (greedy by score; drop if IoU > nms with a kept one)
  3. keep top ``max_det`` by score
  4. resolve overlaps: contested pixels go to the higher score; instances that
     lose more than 50% of their area (or become < min_area) are dropped
Optional mask-thickness control, applied per instance BEFORE the steps above
(see ``transform_stem``): re-binarize cached soft probabilities at ``mask_thr``
and/or morphologically erode with a (2*erode+1) kernel of ``erode_shape``;
instances that vanish are dropped.
Output masks are pairwise disjoint binary 2048x2048 arrays.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from filseg.infer import SOFT_COLUMN, unpack_soft
from filseg.rle import H, W, decode_rle, encode_mask, to_rle_obj
from filseg.submission import resolve_overlaps


@dataclass(frozen=True)
class PPParams:
    conf: float = 0.25
    min_area: int = 0
    max_det: int = 100
    nms: float | None = None  # mask-IoU threshold; None = off
    erode: int = 0  # erosion radius k: kernel (2k+1)x(2k+1); 0 = off
    erode_shape: str = "square"  # square | cross | ellipse
    mask_thr: float | None = None  # soft mask probability threshold; None = cached binary (0.5)

    def to_dict(self) -> dict:
        return asdict(self)


def group_by_stem(raw: pd.DataFrame) -> dict[str, tuple[np.ndarray, list[str], np.ndarray]]:
    """raw cache -> {stem: (scores, rles, areas)} sorted by descending score."""
    out = {}
    for stem, g in raw.groupby("stem", sort=True):
        g = g.sort_values("score", ascending=False, kind="stable")
        out[stem] = (g["score"].to_numpy(), g["rle"].tolist(), g["area"].to_numpy())
    return out


def group_soft_by_stem(raw: pd.DataFrame) -> dict[str, list]:
    """raw cache with a ``soft`` column -> {stem: packed soft crops} in ``group_by_stem`` order."""
    out = {}
    for stem, g in raw.groupby("stem", sort=True):
        g = g.sort_values("score", ascending=False, kind="stable")
        out[stem] = g[SOFT_COLUMN].tolist()
    return out


def _kernel(k: int, shape: str) -> np.ndarray:
    size = 2 * k + 1
    if shape == "square":
        return np.ones((size, size), np.uint8)
    if shape == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    if shape == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    raise ValueError(f"unknown erode_shape {shape!r}")


def erode_window(crop: np.ndarray, y0: int, x0: int, k: int, shape: str) -> tuple[int, int, np.ndarray]:
    """Erode a mask given as a window (crop at offset y0,x0 in the 2048x2048 frame).

    Equivalent to ``cv2.erode`` on the full frame (default border, i.e. the image
    edge does not erode) provided pixels outside the crop are 0: the window is
    extended by k with zeros, clipped at the image edge.
    Returns (y0', x0', eroded uint8 window).
    """
    h, w = crop.shape
    ey0, ex0 = max(y0 - k, 0), max(x0 - k, 0)
    ey1, ex1 = min(y0 + h + k, H), min(x0 + w + k, W)
    canvas = np.zeros((ey1 - ey0, ex1 - ex0), np.uint8)
    canvas[y0 - ey0 : y0 - ey0 + h, x0 - ex0 : x0 - ex0 + w] = crop
    return ey0, ex0, cv2.erode(canvas, _kernel(k, shape))


def _window_of(full: np.ndarray) -> tuple[int, int, np.ndarray] | None:
    ys, xs = np.nonzero(full.any(axis=1))[0], np.nonzero(full.any(axis=0))[0]
    if len(ys) == 0:
        return None
    return int(ys[0]), int(xs[0]), full[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]


def transform_stem(
    scores: np.ndarray, rles: list[str], areas: np.ndarray, softs: list | None,
    erode: int = 0, erode_shape: str = "square", mask_thr: float | None = None, min_score: float = 0.0,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Re-binarize (``mask_thr``, needs ``softs``) and/or erode each instance of one stem.

    Instances scoring below ``min_score`` or that become empty are dropped. With
    erode == 0 and mask_thr None this is just the score filter.
    """
    ks, out_r, out_a = [], [], []
    thr_q = None if mask_thr is None else int(np.ceil(mask_thr * 255 - 1e-9))
    for i in range(len(rles)):
        if scores[i] < min_score:
            continue
        if erode == 0 and thr_q is None:
            ks.append(i), out_r.append(rles[i]), out_a.append(int(areas[i]))
            continue
        if thr_q is not None:
            if softs is None or softs[i] is None:
                raise ValueError("mask_thr requires a cache produced with --soft")
            y0, x0, soft = unpack_soft(softs[i])
            win = (soft >= thr_q).astype(np.uint8)
        else:
            w_ = _window_of(decode_rle(rles[i]))
            if w_ is None:
                continue
            y0, x0, win = w_
        if erode > 0:
            y0, x0, win = erode_window(win, y0, x0, erode, erode_shape)
        a = int(win.sum())
        if a == 0:
            continue
        full = np.zeros((H, W), np.uint8)
        full[y0 : y0 + win.shape[0], x0 : x0 + win.shape[1]] = win
        ks.append(i), out_r.append(encode_mask(full)), out_a.append(a)
    return scores[ks] if len(ks) else np.zeros(0), out_r, np.asarray(out_a, dtype=np.int64)


def transform_grouped(
    grouped: dict, soft_grouped: dict | None, erode: int = 0, erode_shape: str = "square",
    mask_thr: float | None = None, min_score: float = 0.0,
) -> dict:
    """``transform_stem`` over all stems -> new grouped dict (same layout as ``group_by_stem``)."""
    return {
        stem: transform_stem(sc, rl, ar, None if soft_grouped is None else soft_grouped.get(stem),
                             erode, erode_shape, mask_thr, min_score)
        for stem, (sc, rl, ar) in grouped.items()
    }


def _nms_indices(rles: list[str], thr: float) -> list[int]:
    """Greedy mask-IoU NMS over RLEs already sorted by descending score."""
    if len(rles) < 2:
        return list(range(len(rles)))
    objs = [to_rle_obj(c) for c in rles]
    iou = np.asarray(mask_util.iou(objs, objs, [0] * len(objs)))
    kept: list[int] = []
    for i in range(len(rles)):
        if all(iou[i, j] <= thr for j in kept):
            kept.append(i)
    return kept


def _resolve_rles(rles: list[str], scores: np.ndarray) -> tuple[list[np.ndarray | None], list[int]]:
    """Overlap resolution done per connected overlap group on cropped windows.

    Equivalent to ``resolve_overlaps`` on full masks (ownership is local to a
    group) but avoids full-frame work for non-overlapping instances.
    Returns (full-size bool masks for kept instances, kept indices in order).
    """
    n = len(rles)
    if n == 0:
        return [], []
    objs = [to_rle_obj(c) for c in rles]
    inter = np.asarray(mask_util.iou(objs, objs, [0] * n)) > 0
    np.fill_diagonal(inter, False)
    # connected components
    comp = list(range(n))

    def find(x):
        while comp[x] != x:
            comp[x] = comp[comp[x]]
            x = comp[x]
        return x

    for i, j in zip(*np.nonzero(np.triu(inter))):
        comp[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    result: dict[int, np.ndarray] = {}
    for members in groups.values():
        if len(members) == 1:
            result[members[0]] = decode_rle(rles[members[0]]).astype(bool)
            continue
        full = [decode_rle(rles[i]) for i in members]
        ys = np.nonzero(np.any([m.any(axis=1) for m in full], axis=0))[0]
        xs = np.nonzero(np.any([m.any(axis=0) for m in full], axis=0))[0]
        y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
        crops = [m[y0:y1, x0:x1] for m in full]
        kept_masks, kept_idx = resolve_overlaps(crops, scores[members])
        for km, ki in zip(kept_masks, kept_idx):
            out = np.zeros(full[0].shape, dtype=bool)
            out[y0:y1, x0:x1] = km
            result[members[ki]] = out
    kept = sorted(result)
    return [result[i] for i in kept], kept


def postprocess_stem(
    scores: np.ndarray, rles: list[str], areas: np.ndarray, params: PPParams
) -> tuple[list[np.ndarray], list[float]]:
    """Apply the pipeline to one stem (inputs sorted by descending score).

    Returns (disjoint bool masks, their scores) sorted by descending score.
    """
    idx = [i for i in range(len(rles)) if scores[i] >= params.conf and areas[i] >= params.min_area]
    if params.nms is not None and idx:
        keep = _nms_indices([rles[i] for i in idx], params.nms)
        idx = [idx[k] for k in keep]
    idx = idx[: params.max_det]
    masks, kept = _resolve_rles([rles[i] for i in idx], scores[idx] if idx else np.zeros(0))
    out_m, out_s = [], []
    for m, k in zip(masks, kept):
        if int(m.sum()) < max(params.min_area, 1):
            continue
        out_m.append(m)
        out_s.append(float(scores[idx[k]]))
    return out_m, out_s


def postprocess_to_rows(
    grouped: dict, params: PPParams, stems: list[str] | None = None, soft_grouped: dict | None = None
) -> list[tuple[str, str]]:
    """Whole subset -> [(f'{stem}_{k}', rle)] with k = 1..n per stem (score order).

    ``params.erode`` / ``params.mask_thr`` are applied here (``soft_grouped`` is
    required for mask_thr); ``postprocess_stem`` itself assumes inputs already transformed.
    """
    rows: list[tuple[str, str]] = []
    for stem in stems if stems is not None else sorted(grouped):
        if stem not in grouped:
            continue
        scores, rles, areas = grouped[stem]
        if params.erode > 0 or params.mask_thr is not None:
            scores, rles, areas = transform_stem(
                scores, rles, areas, None if soft_grouped is None else soft_grouped.get(stem),
                params.erode, params.erode_shape, params.mask_thr, params.conf)
        masks, _ = postprocess_stem(scores, rles, areas, params)
        rows.extend((f"{stem}_{k}", encode_mask(m)) for k, m in enumerate(masks, start=1))
    return rows
