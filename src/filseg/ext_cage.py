"""Inference adapter for the external CAGE 3-head U-Net (heads: mask, hull, interior).

Weights are CC BY-NC 4.0 and an untrusted pickle: they are loaded ONLY with
``torch.load(..., weights_only=True)``. Recipe (model card): grayscale 2048 image ->
1024 (INTER_AREA), ``(x/255 - 0.5)/0.25``, sigmoid, h-flip TTA average. Instances
("mode B"): each connected hull component is one filament (mask pixels inside it);
mask pixels outside any hull fall back to components of the 51-px elliptically closed
mask; instances < 500 px are dropped.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from filseg.infer import RAW_COLUMNS, SOFT_COLUMN, SOFT_PAD, pack_soft, unpack_soft
from filseg.rle import H, W, encode_mask

IN_SIZE = 1024
MIN_INST_AREA = 500
CLOSE_K = 51


def load_model(weights: str, device: str = "cuda"):
    import segmentation_models_pytorch as smp
    import torch

    model = smp.Unet("resnet34", encoder_weights=None, in_channels=1, classes=3)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    return model.to(device).eval()


def preprocess(gray2048: np.ndarray) -> np.ndarray:
    """uint8 grayscale (2048x2048) -> float32 [1024,1024] normalised."""
    x = cv2.resize(gray2048, (IN_SIZE, IN_SIZE), interpolation=cv2.INTER_AREA)
    return ((x.astype(np.float32) / 255.0 - 0.5) / 0.25).astype(np.float32)


def predict_probs(model, grays: list[np.ndarray], device: str = "cuda") -> np.ndarray:
    """List of uint8 gray images -> uint8 probs [N,3,1024,1024] (mask, hull, interior), h-flip TTA."""
    import torch

    x = torch.from_numpy(np.stack([preprocess(g) for g in grays]))[:, None].to(device)
    with torch.no_grad():
        p = torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1)
        p = (p / 2).clamp_(0, 1)
    return torch.round(p * 255).to(torch.uint8).cpu().numpy()


def upsample_prob(p_u8: np.ndarray) -> np.ndarray:
    """uint8 [h,w] prob -> uint8 [2048,2048] (bilinear)."""
    return np.clip(np.rint(cv2.resize(p_u8.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)), 0, 255).astype(np.uint8)


def mode_b_instances(probs: np.ndarray, min_area: int = MIN_INST_AREA) -> list[dict]:
    """probs uint8 [3,h,w] -> [{score, mask(bool 2048), region(bool 2048), p_mask(uint8 2048)}].

    ``region`` is the hull component (or the closed-mask component for fallbacks).
    Deterministic ordering: hull components in label order, then fallbacks.
    """
    pm = upsample_prob(probs[0])
    ph = upsample_prob(probs[1])
    mask = pm > 127  # p > 0.5 on the 0-255 scale
    hull = ph > 127
    out: list[dict] = []
    n, lab = cv2.connectedComponents(hull.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        region = lab == k
        inst = mask & region
        if int(inst.sum()) >= min_area:
            out.append(dict(mask=inst, region=region))
    rest = mask & ~hull
    if rest.any():
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
        closed = cv2.morphologyEx(rest.astype(np.uint8), cv2.MORPH_CLOSE, ker)
        n2, lab2 = cv2.connectedComponents(closed, connectivity=8)
        for k in range(1, n2):
            region = lab2 == k
            inst = rest & region
            if int(inst.sum()) >= min_area:
                out.append(dict(mask=inst, region=region))
    for d in out:
        d["score"] = float(pm[d["mask"]].mean() / 255.0)
        d["p_mask"] = pm
    return out


def instances_to_rows(stem: str, insts: list[dict], pad: int = SOFT_PAD) -> list[tuple]:
    rows = []
    for d in insts:
        reg = d["region"]
        ys, xs = np.nonzero(reg.any(axis=1))[0], np.nonzero(reg.any(axis=0))[0]
        y0, y1 = max(int(ys[0]) - pad, 0), min(int(ys[-1]) + 1 + pad, H)
        x0, x1 = max(int(xs[0]) - pad, 0), min(int(xs[-1]) + 1 + pad, W)
        crop = np.where(reg[y0:y1, x0:x1], d["p_mask"][y0:y1, x0:x1], 0).astype(np.uint8)
        rows.append((stem, d["score"], encode_mask(d["mask"].astype(np.uint8)), int(d["mask"].sum()), pack_soft(y0, x0, crop)))
    return rows


def rows_to_df(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RAW_COLUMNS + [SOFT_COLUMN])


def fuse_soft(packed: tuple, cage_p: np.ndarray, alpha: float) -> tuple:
    """alpha*yolo_soft + (1-alpha)*cage p_mask (uint8 2048) within the YOLO soft window."""
    y0, x0, crop = unpack_soft(packed)
    h, w = crop.shape
    f = alpha * crop.astype(np.float32) + (1 - alpha) * cage_p[y0 : y0 + h, x0 : x0 + w].astype(np.float32)
    return pack_soft(y0, x0, np.clip(np.rint(f), 0, 255).astype(np.uint8))


def refine_cache(raw: pd.DataFrame, cage_pmask: dict[str, np.ndarray], alpha: float) -> pd.DataFrame:
    """Fuse each YOLO instance's soft crop with CAGE p_mask; rle/area re-derived at soft >= 0.5.

    ``cage_pmask``: stem -> uint8 p_mask (any size; upsampled to 2048). Score unchanged.
    Stems missing from ``cage_pmask`` are kept as is. If the fused mask is empty the
    original rle/area are kept.
    """
    up: dict[str, np.ndarray] = {}
    out = raw.copy()
    rles, areas, softs = [], [], []
    for _, r in raw.iterrows():
        stem = r["stem"]
        if stem not in cage_pmask:
            rles.append(r["rle"]), areas.append(r["area"]), softs.append(r[SOFT_COLUMN])
            continue
        if stem not in up:
            p = cage_pmask[stem]
            up[stem] = p if p.shape == (H, W) else upsample_prob(p)
        s = fuse_soft(r[SOFT_COLUMN], up[stem], alpha)
        y0, x0, crop = unpack_soft(s)
        m = crop >= 128
        if m.any():
            full = np.zeros((H, W), np.uint8)
            full[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = m
            rles.append(encode_mask(full)), areas.append(int(m.sum()))
        else:
            rles.append(r["rle"]), areas.append(r["area"])
        softs.append(s)
    out["rle"], out["area"], out[SOFT_COLUMN] = rles, areas, softs
    return out
