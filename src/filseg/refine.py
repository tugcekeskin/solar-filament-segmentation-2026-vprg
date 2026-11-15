"""Full-resolution crop mask refiner: sharpen YOLO instance masks with a small U-Net.

Each instance is cropped from the 2048x2048 image in a square window around its
(padded) mask bbox, resized to ``SZ`` and fed to a 2-channel U-Net (per-crop
normalised image, prior probability). The output probability is resized back to
the window and stored like any other soft mask (see ``filseg.infer``).
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from filseg.infer import SOFT_COLUMN, pack_soft, unpack_soft
from filseg.rle import H, W, decode_rle, encode_mask, to_rle_obj

SZ = 256
IMG_DIR = "data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images"
GT_JSON = "data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMG_DIR_TEST = "data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images"


# ----------------------------------------------------------------- windows
def crop_window(y0: int, y1: int, x0: int, x1: int, min_pad: int = 16, frac: float = 0.25) -> tuple[int, int, int]:
    """Square window (wy0, wx0, side) around the half-open bbox [y0,y1)x[x0,x1).

    Padded by max(min_pad, frac * longer side) on each side, made square about the
    centre, then shifted/clipped to lie inside the image (side <= min(H, W)).
    """
    h, w = y1 - y0, x1 - x0
    pad = max(min_pad, int(round(frac * max(h, w))))
    side = min(max(h, w) + 2 * pad, H, W)
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    wy0 = int(round(cy - side / 2.0))
    wx0 = int(round(cx - side / 2.0))
    wy0 = min(max(wy0, 0), H - side)
    wx0 = min(max(wx0, 0), W - side)
    return wy0, wx0, int(side)


def mask_bbox(m: np.ndarray) -> tuple[int, int, int, int] | None:
    ys = np.flatnonzero(m.any(axis=1))
    if len(ys) == 0:
        return None
    xs = np.flatnonzero(m.any(axis=0))
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1


def cut(a: np.ndarray, win: tuple[int, int, int]) -> np.ndarray:
    wy0, wx0, s = win
    return a[wy0 : wy0 + s, wx0 : wx0 + s]


def place(y0: int, x0: int, crop: np.ndarray, win: tuple[int, int, int]) -> np.ndarray:
    """Paste a crop given at (y0,x0) in the image frame into an s x s canvas for ``win`` (clipped, zero fill)."""
    wy0, wx0, s = win
    out = np.zeros((s, s), crop.dtype)
    ya, xa = max(y0, wy0), max(x0, wx0)
    yb, xb = min(y0 + crop.shape[0], wy0 + s), min(x0 + crop.shape[1], wx0 + s)
    if yb > ya and xb > xa:
        out[ya - wy0 : yb - wy0, xa - wx0 : xb - wx0] = crop[ya - y0 : yb - y0, xa - x0 : xb - x0]
    return out


def to_sz(a: np.ndarray, nearest: bool = False) -> np.ndarray:
    if nearest:
        return cv2.resize(a, (SZ, SZ), interpolation=cv2.INTER_NEAREST)
    # bilinear when enlarging; area averaging when shrinking (plain bilinear aliases thin structures)
    return cv2.resize(a, (SZ, SZ), interpolation=cv2.INTER_AREA if a.shape[0] > SZ else cv2.INTER_LINEAR)


def paste_back(prob: np.ndarray, win: tuple[int, int, int], floor: int = 26) -> tuple | None:
    """SZ x SZ float prob in [0,1] -> packed uint8 soft crop in the image frame (trimmed to values >= floor/255)."""
    wy0, wx0, s = win
    big = cv2.resize(prob.astype(np.float32), (s, s), interpolation=cv2.INTER_LINEAR)
    q = np.clip(np.rint(big * 255), 0, 255).astype(np.uint8)
    q[q < floor] = 0
    bb = mask_bbox(q)
    if bb is None:
        return None
    a, b, c, d = bb
    return pack_soft(wy0 + a, wx0 + c, q[a:b, c:d])


def soft_bbox_window(packed: tuple, thr: int = 128, frac: float = 0.25) -> tuple[int, int, int] | None:
    """Window from the >= thr bbox of a packed soft crop."""
    y0, x0, crop = unpack_soft(packed)
    bb = mask_bbox(crop >= thr)
    if bb is None:
        return None
    a, b, c, d = bb
    return crop_window(y0 + a, y0 + b, x0 + c, x0 + d, frac=frac)


# ----------------------------------------------------------------- model
def make_model(pretrained: bool = True):
    import segmentation_models_pytorch as smp

    return smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=2, classes=1)


def norm_img(img_u8):
    """(B,SZ,SZ) uint8/float tensor -> per-crop standardised float."""
    x = img_u8.float()
    m = x.mean(dim=(-2, -1), keepdim=True)
    s = x.std(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    return (x - m) / s


def make_input(img_u8, prior_u8):
    import torch

    return torch.stack([norm_img(img_u8), prior_u8.float() / 255.0], dim=1)


def load_model(weights: str, device: str = "cuda"):
    import torch

    if "sam" in Path(weights).name:
        from filseg.sam_refine import load_sam_refiner

        return load_sam_refiner(weights, device)

    m = make_model(pretrained=False)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    return m.to(device).eval()


def predict_crops(model, imgs: np.ndarray, priors: np.ndarray, batch: int = 64, tta: bool = True, device: str = "cuda") -> np.ndarray:
    """(N,SZ,SZ) uint8 image crops + uint8 priors -> (N,SZ,SZ) float32 probs (h-flip TTA average)."""
    import torch

    out = np.zeros(imgs.shape, np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        for i in range(0, len(imgs), batch):
            x = make_input(torch.from_numpy(imgs[i : i + batch]).to(device), torch.from_numpy(priors[i : i + batch]).to(device))
            p = torch.sigmoid(model(x).float())
            if tta:
                p = 0.5 * (p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1))
            out[i : i + batch] = p[:, 0].cpu().numpy()
    return out


def refine_stem(model, img: np.ndarray, softs: list, batch: int = 64, device: str = "cuda", frac: float = 0.25) -> list:
    """Refine every packed soft of one stem against its full image -> list of packed soft (or None if vanished)."""
    wins, imgs, priors, idx = [], [], [], []
    for i, t in enumerate(softs):
        win = soft_bbox_window(t, frac=frac)
        if win is None:
            continue
        y0, x0, crop = unpack_soft(t)
        wins.append(win), idx.append(i)
        imgs.append(to_sz(cut(img, win))), priors.append(to_sz(place(y0, x0, crop, win)))
    out: list = [None] * len(softs)
    if not idx:
        return out
    probs = predict_crops(model, np.stack(imgs), np.stack(priors), batch, True, device)
    for k, i in enumerate(idx):
        out[i] = paste_back(probs[k], wins[k])
    return out


# ----------------------------------------------------------------- dataset
def _degrade(gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """GT mask (full uint8) -> degraded soft prior (uint8 0-255) mimicking a blurry YOLO mask."""
    bb = mask_bbox(gt)
    y0, y1, x0, x1 = bb
    m = max(24, int(rng.integers(2, 8)) * 4)
    ya, yb, xa, xb = max(y0 - m, 0), min(y1 + m, H), max(x0 - m, 0), min(x1 + m, W)
    sub = gt[ya:yb, xa:xb].astype(np.uint8)
    k = int(rng.integers(1, 5))
    sub = cv2.dilate(sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    dy, dx = (int(v) for v in rng.integers(-3, 4, size=2))
    sub = np.roll(sub, (dy, dx), axis=(0, 1))
    sig = float(rng.uniform(1.0, 4.0))
    f = cv2.GaussianBlur(sub.astype(np.float32), (0, 0), sig)
    f = np.clip((f - 0.5) * float(rng.uniform(1.0, 2.5)) + 0.5, 0, 1)  # random sharpness around the edge
    out = np.zeros((H, W), np.uint8)
    out[ya:yb, xa:xb] = np.rint(f * 255).astype(np.uint8)
    return out


def _fragment(prior: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep a random contiguous 35-80% slice of the prior along its longer bbox axis (mimics a fragment)."""
    bb = mask_bbox(prior >= 128)
    if bb is None:
        return prior
    y0, y1, x0, x1 = bb
    ax, lo, hi = (0, y0, y1) if (y1 - y0) >= (x1 - x0) else (1, x0, x1)
    n = hi - lo
    keep = max(4, int(round(n * rng.uniform(0.35, 0.8))))
    a = lo + int(rng.integers(0, max(1, n - keep + 1)))
    out = np.zeros_like(prior)
    if ax == 0:
        out[a : a + keep] = prior[a : a + keep]
    else:
        out[:, a : a + keep] = prior[:, a : a + keep]
    return out


def _stem_samples(args: tuple) -> dict:
    """Samples (real proposal/GT pairs + synthetic degraded GT priors) of one stem."""
    # props: [(score, packed soft)]; gts: {view: [rle str]}
    stem, props, gts, syn_frac, seed, win_mode, frac_lo, frac_hi, frag_p = args
    rng = np.random.default_rng(seed)
    img = cv2.imread(f"{IMG_DIR}/{stem}.jpeg", cv2.IMREAD_GRAYSCALE)
    gt_objs = {v: [to_rle_obj(r) for r in rl] for v, rl in gts.items()}
    # proposal full masks (binary) as RLE objs
    full_p, canv = [], []
    for _, t in props:
        y0, x0, crop = unpack_soft(t)
        f = np.zeros((H, W), np.uint8)
        f[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
        canv.append(f)
        full_p.append(mask_util.encode(np.asfortranarray((f >= 128).astype(np.uint8))))
    I, P, T, SYN = [], [], [], []

    def add(prior_full: np.ndarray, gt_full: np.ndarray, syn: int):
        pm = prior_full >= 128
        un = pm | (gt_full > 0)
        bb = mask_bbox(un)
        if bb is None or not gt_full.any():
            return
        if win_mode == "union":  # original recipe: the whole GT is always inside the window
            win = crop_window(*bb)
        else:  # "prior": window from the prior alone, as at inference, with random context
            bp = mask_bbox(pm)
            if bp is None:
                return
            win = crop_window(*bp, frac=float(rng.uniform(frac_lo, frac_hi)))
        I.append(to_sz(cut(img, win))), P.append(to_sz(cut(prior_full, win))), T.append(to_sz(cut(gt_full, win), True)), SYN.append(syn)

    n_real = 0
    for v, objs in gt_objs.items():
        if not objs or not full_p:
            continue
        iou = np.asarray(mask_util.iou(full_p, objs, [0] * len(objs)))
        for pi in range(len(full_p)):
            gj = int(iou[pi].argmax())
            if iou[pi, gj] >= 0.2:
                gt_full = decode_rle(gts[v][gj])
                add(canv[pi], gt_full, 0)
                n_real += 1
    n_syn = int(round(n_real * syn_frac / (1 - syn_frac))) if n_real else 0
    allg = [(v, j) for v, rl in gts.items() for j in range(len(rl))]
    if allg and n_syn:
        for k in rng.choice(len(allg), size=min(n_syn, len(allg)), replace=False):
            v, j = allg[int(k)]
            g = decode_rle(gts[v][j])
            if g.any():
                pr = _degrade(g, rng)
                if frag_p and rng.random() < frag_p:
                    pr = _fragment(pr, rng)
                add(pr, g, 1)
    n = len(I)
    return dict(stem=stem, img=np.stack(I) if n else np.zeros((0, SZ, SZ), np.uint8),
                prior=np.stack(P) if n else np.zeros((0, SZ, SZ), np.uint8),
                tgt=np.stack(T) if n else np.zeros((0, SZ, SZ), np.uint8), syn=np.array(SYN, np.uint8))


def build_dataset(raw: pd.DataFrame, stems: list[str], out_path: str, gt_json: str = GT_JSON,
                  min_score: float = 0.1, syn_frac: float = 0.3, workers: int = 16, seed: int = 0,
                  win_mode: str = "union", frac_lo: float = 0.25, frac_hi: float = 0.25, frag_p: float = 0.0) -> dict:
    from filseg.gt import build_gt_df, view_stem

    gt = build_gt_df(gt_json, set(stems))
    view = gt["filament_id"].str.split("_", n=1).str[0]
    gt_by: dict[str, dict[str, list[str]]] = {}
    for v, fid, r in zip(view, gt["filament_id"], gt["segmentation_rle"]):
        gt_by.setdefault(view_stem(v), {}).setdefault(v, []).append(r)
    raw = raw[raw["score"] >= min_score]
    props_by = {s: list(zip(g["score"], g[SOFT_COLUMN])) for s, g in raw.groupby("stem")}
    jobs = [(s, props_by.get(s, []), gt_by.get(s, {}), syn_frac, seed * 100003 + i, win_mode, frac_lo, frac_hi, frag_p) for i, s in enumerate(sorted(stems)) if s in gt_by]
    with Pool(workers) as pool:
        res = pool.map(_stem_samples, jobs, chunksize=4)
    res = [r for r in res if len(r["syn"])]
    d = dict(
        img=np.concatenate([r["img"] for r in res]), prior=np.concatenate([r["prior"] for r in res]),
        tgt=np.concatenate([r["tgt"] for r in res]), syn=np.concatenate([r["syn"] for r in res]),
        stem=np.concatenate([[r["stem"]] * len(r["syn"]) for r in res]).astype("U16"),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **d)
    return d
