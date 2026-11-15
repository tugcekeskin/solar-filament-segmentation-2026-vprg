"""CPU-only ensembling of per-instance soft predictions from several models.

Per stem: instances of all models are clustered greedily by mask IoU (binary
mask = soft >= 128), at most one member per model per cluster. The fused soft
crop is the sum of member soft probabilities divided by ``n_models`` (absent
models count as 0) over the union window of the member crops; the fused score
is the mean member score over ``n_models``.
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd

from filseg.infer import SOFT_COLUMN, pack_soft, unpack_soft
from filseg.rle import H, W, encode_mask

BIN_THR = 128


def _pair_iou(a: tuple, b: tuple) -> float:
    """IoU of two binary windows (y0, x0, mask, area), computed in the window intersection."""
    ay0, ax0, am, aa = a
    by0, bx0, bm, ba = b
    y0, x0 = max(ay0, by0), max(ax0, bx0)
    y1, x1 = min(ay0 + am.shape[0], by0 + bm.shape[0]), min(ax0 + am.shape[1], bx0 + bm.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0.0
    inter = int(np.count_nonzero(am[y0 - ay0 : y1 - ay0, x0 - ax0 : x1 - ax0] & bm[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0]))
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def _fuse(members: list[tuple[int, int, np.ndarray]], n_models: int) -> tuple[int, int, np.ndarray]:
    y0 = min(m[0] for m in members)
    x0 = min(m[1] for m in members)
    y1 = max(m[0] + m[2].shape[0] for m in members)
    x1 = max(m[1] + m[2].shape[1] for m in members)
    acc = np.zeros((y1 - y0, x1 - x0), np.float32)
    for my, mx, crop in members:
        acc[my - y0 : my - y0 + crop.shape[0], mx - x0 : mx - x0 + crop.shape[1]] += crop
    return y0, x0, np.clip(np.rint(acc / n_models), 0, 255).astype(np.uint8)


def ensemble_stem(
    insts: list[tuple], n_models: int, iou_thr: float = 0.5, min_votes: int = 1
) -> list[tuple]:
    """Fuse one stem. ``insts`` = [(model_idx, score, packed soft)] -> [(fused score, packed soft)]."""
    order = sorted(range(len(insts)), key=lambda i: -insts[i][1])  # stable
    crops = []
    for i in order:
        y0, x0, crop = unpack_soft(insts[i][2])
        m = crop >= BIN_THR
        crops.append((int(y0), int(x0), crop, m, int(np.count_nonzero(m))))
    models = [insts[i][0] for i in order]
    scores = [float(insts[i][1]) for i in order]
    n = len(order)
    by_model: dict[int, list[int]] = {}
    for k in range(n):
        by_model.setdefault(models[k], []).append(k)
    assigned = [False] * n
    out: list[tuple[float, tuple]] = []
    for s in range(n):
        if assigned[s]:
            continue
        assigned[s] = True
        members = [s]
        sb = (crops[s][0], crops[s][1], crops[s][3], crops[s][4])
        for mdl in sorted(by_model):
            if mdl == models[s]:
                continue
            best, best_iou = -1, -1.0
            for c in by_model[mdl]:
                if assigned[c]:
                    continue
                iou = _pair_iou(sb, (crops[c][0], crops[c][1], crops[c][3], crops[c][4]))
                if iou > best_iou:
                    best, best_iou = c, iou
            if best >= 0 and best_iou >= iou_thr:
                assigned[best] = True
                members.append(best)
        if len(members) < min_votes:
            continue
        y0, x0, fused = _fuse([(crops[k][0], crops[k][1], crops[k][2]) for k in members], n_models)
        if not (fused >= BIN_THR).any():
            continue
        rec = (sum(scores[k] for k in members) / n_models, pack_soft(y0, x0, fused))
        if len(insts[order[s]]) > 3:  # optional class index of the top-scored member
            rec += (insts[order[s]][3],)
        out.append(rec)
    out.sort(key=lambda t: -t[0])
    return out


def _stem_job(args: tuple) -> tuple[str, list[tuple[float, str, int, tuple]]]:
    stem, insts, n_models, iou_thr, min_votes = args
    rows = []
    for score, soft, *cls in ensemble_stem(insts, n_models, iou_thr, min_votes):
        y0, x0, crop = unpack_soft(soft)
        win = crop >= BIN_THR
        full = np.zeros((H, W), np.uint8)
        full[y0 : y0 + win.shape[0], x0 : x0 + win.shape[1]] = win
        rows.append((score, encode_mask(full), int(win.sum()), soft, *cls))
    return stem, rows


def ensemble_caches(
    dfs: list[pd.DataFrame], iou_thr: float = 0.5, min_votes: int = 1, workers: int = 1
) -> pd.DataFrame:
    """Fuse raw caches (each with a ``soft`` column) into one raw cache with the same schema."""
    n_models = len(dfs)
    per_stem: dict[str, list] = {}
    for mi, df in enumerate(dfs):
        if SOFT_COLUMN not in df.columns:
            raise ValueError(f"cache {mi} has no soft column")
        cls = df["cls"] if "cls" in df.columns else [None] * len(df)
        for stem, score, soft, c in zip(df["stem"], df["score"], df[SOFT_COLUMN], cls):
            per_stem.setdefault(stem, []).append((mi, float(score), soft) + (() if c is None else (int(c),)))
    jobs = [(stem, per_stem[stem], n_models, iou_thr, min_votes) for stem in sorted(per_stem)]
    if workers > 1 and len(jobs) > 1:
        with Pool(workers) as pool:
            results = pool.map(_stem_job, jobs, chunksize=4)
    else:
        results = [_stem_job(j) for j in jobs]
    with_cls = all("cls" in df.columns for df in dfs)
    recs = [
        {"stem": stem, "score": r[0], "rle": r[1], "area": r[2], SOFT_COLUMN: r[3], **({"cls": r[4]} if with_cls else {})}
        for stem, rows in results
        for r in rows
    ]
    return pd.DataFrame(recs, columns=["stem", "score", "rle", "area", SOFT_COLUMN] + (["cls"] if with_cls else []))
