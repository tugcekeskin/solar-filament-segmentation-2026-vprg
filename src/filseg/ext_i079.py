"""Inference adapter for the external "i079" Mask R-CNN detector + crop U-Net refiner.

Ported (not executed) from the public i079 inference notebook: detector at 1024,
square native-res crops (context 1.8, min 96) resized to 256 for a ResNet18 U-Net
refiner (grayscale replicated to 3 channels; the refiner is image-only), 4-view
flip TTA with topology guards, threshold 0.70, positive-overlap merging and
detector-seeded splitting with a crowding abstention.

Weights are untrusted pickles: they are loaded ONLY with ``weights_only=True``.
Weights were trained on all 707 train images (incl. our val stems), so val
numbers are in-sample.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch import nn
from torchvision.models import resnet18
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from filseg.infer import SOFT_PAD, pack_soft, unpack_soft
from filseg.rle import H, W, encode_mask

IMAGE_SIZE = 1024
DETECTOR_MASK_THRESHOLD = 0.50
DEFAULT_SCORE_THRESHOLD = 0.70
REFINER_THRESHOLD = 0.70
MAXIMUM_SEED_IOU = 0.10
MINIMUM_UNIQUE_SEED_PIXELS = 16
MINIMUM_SEED_DISTANCE = 8.0
MAXIMUM_SPLIT_TASKS = 1
TTA_TRANSFORMS = ("identity", "hflip", "vflip", "hvflip")
TTA_AREA_RATIO_MIN = 0.60
TTA_AREA_RATIO_MAX = 1.70
TTA_DICE_MIN = 0.62
TTA_MIN_CONTROL_AREA_FOR_DICE = 32

DETECTOR_PATH = "ext/i079/final_detector.pt"
REFINER_PATH = "ext/i079/final_refiner.pt"


# ----------------------------------------------------------------------------- models
def build_detector() -> nn.Module:
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, hidden, 2)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=28, sampling_ratio=2
    )
    model.transform.min_size = (IMAGE_SIZE,)
    model.transform.max_size = IMAGE_SIZE
    return model


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class CropUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        enc = resnet18(weights=None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), ConvBlock(96, 32)
        self.up0, self.head = nn.ConvTranspose2d(32, 16, 2, 2), nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.mean) / self.std
        x0 = self.stem(image)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        v = self.dec4(torch.cat([self.up4(x4), x3], 1))
        v = self.dec3(torch.cat([self.up3(v), x2], 1))
        v = self.dec2(torch.cat([self.up2(v), x1], 1))
        v = self.dec1(torch.cat([self.up1(v), x0], 1))
        return self.head(self.up0(v))


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Safe load: weights_only=True, no fallback (raises if the pickle needs arbitrary code)."""
    value = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or "model" not in value:
        raise RuntimeError(f"unexpected checkpoint structure: {path}")
    return value


def load_detector(path: str | Path = DETECTOR_PATH, device: str | torch.device = "cpu") -> nn.Module:
    m = build_detector()
    m.load_state_dict(load_checkpoint(path)["model"], strict=True)
    return m.to(device).eval()


def load_refiner(path: str | Path = REFINER_PATH, device: str | torch.device = "cpu") -> nn.Module:
    m = CropUNet()
    m.load_state_dict(load_checkpoint(path)["model"], strict=True)
    return m.to(device).eval()


# ----------------------------------------------------------------------------- geometry
def square_bounds(mask: np.ndarray, context: float = 1.8, minimum: int = 96) -> tuple[int, int, int, int]:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or not value.any():
        raise ValueError("square_bounds requires one active 2D mask")
    height, width = value.shape
    ys, xs = np.where(value)
    cx = (xs.min() + xs.max() + 1) / 2
    cy = (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return int(x0), int(y0), int(x1), int(y1)


def _groups_from_adjacency(adj: np.ndarray) -> list[list[int]]:
    groups: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(adj)):
        if start in seen:
            continue
        stack, group = [start], []
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            group.append(i)
            stack.extend(np.flatnonzero(adj[i]).tolist())
        groups.append(group)
    return groups


def positive_overlap_groups_rle(rles: list[dict]) -> list[list[int]]:
    """Connected components of the 'masks share >=1 pixel' graph."""
    if len(rles) < 2:
        return [[i] for i in range(len(rles))]
    adj = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=np.float64) > 0
    return _groups_from_adjacency(adj)


def _enc(mask: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int(np.sum(a | b))
    return int(np.sum(a & b)) / u if u else 0.0


def split_mask_with_detector_seeds(refined: np.ndarray, members: list[np.ndarray], scores: np.ndarray) -> list[np.ndarray]:
    refined = np.asarray(refined, dtype=bool)
    masks = [np.asarray(m, dtype=bool) for m in members]
    scores = np.asarray(scores, dtype=np.float32)
    accepted: list[int] = []
    for i in sorted(range(len(masks)), key=lambda k: (-scores[k], k)):
        if all(_mask_iou(masks[i], masks[o]) <= MAXIMUM_SEED_IOU for o in accepted):
            accepted.append(i)
    if len(accepted) < 2:
        return [refined]
    seeds: list[tuple[int, int]] = []
    for i in accepted:
        others = np.zeros_like(refined)
        for o in accepted:
            if o != i:
                others |= masks[o]
        ys, xs = np.nonzero(masks[i] & refined & ~others)
        if len(ys) < MINIMUM_UNIQUE_SEED_PIXELS:
            continue
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        n = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        seeds.append((int(ys[n]), int(xs[n])))
    if len(seeds) < 2:
        return [refined]
    for a in range(len(seeds)):
        for b in range(a + 1, len(seeds)):
            if np.hypot(seeds[a][0] - seeds[b][0], seeds[a][1] - seeds[b][1]) < MINIMUM_SEED_DISTANCE:
                return [refined]
    fy, fx = np.nonzero(refined)
    dist = np.stack([(fy - sy) ** 2 + (fx - sx) ** 2 for sy, sx in seeds], axis=1)
    assign = np.argmin(dist, axis=1)
    out = []
    for k in range(len(seeds)):
        sel = assign == k
        if not sel.any():
            return [refined]
        part = np.zeros_like(refined)
        part[fy[sel], fx[sel]] = True
        out.append(part)
    return out


# ----------------------------------------------------------------------------- refiner
def _tf(a: np.ndarray, t: str) -> np.ndarray:
    # flips are involutions, so the same op transforms and inverts
    if t == "identity":
        return np.ascontiguousarray(a)
    if t == "hflip":
        return np.ascontiguousarray(a[:, :, ::-1])
    if t == "vflip":
        return np.ascontiguousarray(a[:, ::-1, :])
    if t == "hvflip":
        return np.ascontiguousarray(a[:, ::-1, ::-1])
    raise ValueError(t)


def run_refiner(refiner: nn.Module, crops: np.ndarray, device, batch_size: int = 32) -> np.ndarray:
    if crops.ndim != 3 or crops.shape[1:] != (256, 256):
        raise ValueError(f"refiner crops must have shape [N,256,256], got {crops.shape}")
    outs = []
    with torch.inference_mode():
        for s in range(0, len(crops), batch_size):
            b = torch.from_numpy(np.ascontiguousarray(crops[s : s + batch_size])).float().unsqueeze(1).repeat(1, 3, 1, 1)
            outs.append(refiner(b.to(device) / 255).sigmoid().cpu().numpy()[:, 0].astype(np.float32))
    return np.concatenate(outs, 0) if outs else np.zeros((0, 256, 256), np.float32)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    d = int(a.sum() + b.sum())
    return 1.0 if d == 0 else 2.0 * int(np.logical_and(a, b).sum()) / d


def _ncomp(mask: np.ndarray) -> int:
    return int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)


def topology_safe_select(single: np.ndarray, tta: np.ndarray) -> tuple[np.ndarray, bool, str]:
    sm = single >= np.float32(REFINER_THRESHOLD)
    tm = tta >= np.float32(REFINER_THRESHOLD)
    sa, ta = int(sm.sum()), int(tm.sum())
    if ta == 0:
        return single, False, "empty_tta_mask"
    if sa == 0:
        return single, False, "empty_single_view_mask"
    r = ta / max(sa, 1)
    if r < TTA_AREA_RATIO_MIN:
        return single, False, "area_shrink_guard"
    if r > TTA_AREA_RATIO_MAX:
        return single, False, "area_growth_guard"
    if _ncomp(tm) > max(_ncomp(sm) + 1, 2):
        return single, False, "fragmentation_guard"
    if sa >= TTA_MIN_CONTROL_AREA_FOR_DICE and _dice(sm, tm) < TTA_DICE_MIN:
        return single, False, "overlap_guard"
    return tta, True, "accepted"


def refine_probabilities(refiner: nn.Module, crops: np.ndarray, device, tta: bool = True) -> np.ndarray:
    """Per-crop refiner probabilities [N,256,256]; with ``tta`` the 4-view mean under topology guards."""
    single = run_refiner(refiner, crops, device)
    if not tta or len(crops) == 0:
        return single
    acc = single.copy()
    for t in TTA_TRANSFORMS[1:]:
        acc += _tf(run_refiner(refiner, _tf(crops, t), device), t)
    mean = acc / np.float32(len(TTA_TRANSFORMS))
    return np.stack([topology_safe_select(s, m)[0] for s, m in zip(single, mean)]).astype(np.float32)


def make_crop(image: np.ndarray, mask: np.ndarray):
    """i079 crop construction: square bounds around ``mask`` (native res), image crop resized to 256."""
    b = square_bounds(mask)
    x0, y0, x1, y1 = b
    return b, cv2.resize(image[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_LINEAR)


def crop_prob_native(prob: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return cv2.resize(prob.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)


def _soft_window(mask: np.ndarray, tasks: list[dict], pad: int = SOFT_PAD) -> tuple | None:
    """Pack per-pixel refined probability around ``mask``'s bbox (max over contributing tasks)."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    by0, by1, bx0, bx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    y0, y1, x0, x1 = max(by0 - pad, 0), min(by1 + pad, H), max(bx0 - pad, 0), min(bx1 + pad, W)
    soft = np.zeros((y1 - y0, x1 - x0), np.float32)
    for t in tasks:
        tx0, ty0, tx1, ty1 = t["bounds"]
        ix0, iy0, ix1, iy1 = max(tx0, x0), max(ty0, y0), min(tx1, x1), min(ty1, y1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        p = t["prob_native"][iy0 - ty0 : iy1 - ty0, ix0 - tx0 : ix1 - tx0]
        sub = soft[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0]
        np.maximum(sub, p, out=sub)
    keep = np.zeros_like(soft, dtype=bool)
    keep[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0] = True
    soft[~keep] = 0
    q = np.round(soft * 255).clip(0, 255).astype(np.uint8)
    # guarantee soft >= 0.5 reproduces the final binary mask
    m = mask[y0:y1, x0:x1]
    q[m & (q < 128)] = 128
    return pack_soft(y0, x0, q)


# ----------------------------------------------------------------------------- detector
def prep_detector_input(image: np.ndarray, device) -> torch.Tensor:
    r = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    return torch.from_numpy(r).unsqueeze(0).repeat(3, 1, 1).to(device) / 255


def detect(detector: nn.Module, image: np.ndarray, device, min_score: float):
    """Detector masks (bool [N,1024,1024]) and scores above ``min_score`` with non-empty mask."""
    with torch.inference_mode():
        out = detector([prep_detector_input(image, device)])[0]
    keep = out["scores"] >= min_score
    scores = out["scores"][keep].cpu().numpy().astype(np.float32)
    masks = out["masks"][keep, 0].cpu().numpy() >= DETECTOR_MASK_THRESHOLD
    active = np.flatnonzero(masks.any(axis=(1, 2)))
    return np.ascontiguousarray(masks[active]), np.ascontiguousarray(scores[active])


def _native(mask1024: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    return cv2.resize(mask1024.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def infer_full(refiner: nn.Module, image: np.ndarray, masks: np.ndarray, scores: np.ndarray, device,
               min_score: float = DEFAULT_SCORE_THRESHOLD, tta: bool = True):
    """i079 candidate pipeline on already-detected masks (filtered at ``min_score``).

    Returns list of dicts {mask(bool 2048^2), score, soft(packed)} sorted like i079 (y, x of bbox).
    """
    sel = np.flatnonzero(scores >= min_score)
    masks, scores = masks[sel], scores[sel]
    if not len(masks):
        return []
    h, w = image.shape
    rles1024 = [_enc(m) for m in masks]
    tasks = []
    for group in positive_overlap_groups_rle(rles1024):
        fb = _native(np.logical_or.reduce([masks[i] for i in group]), (h, w))
        bounds, crop = make_crop(image, fb)
        tasks.append({"bounds": bounds, "crop": crop, "fallback": fb,
                      "members": [_native(masks[i], (h, w)) for i in group], "mscores": scores[group]})
    probs = refine_probabilities(refiner, np.stack([t["crop"] for t in tasks]), device, tta=tta)
    n_split = 0
    task_parts: list[list[np.ndarray]] = []
    control: list[np.ndarray] = []
    for t, p in zip(tasks, probs):
        t["prob_native"] = crop_prob_native(p, t["bounds"])
        x0, y0, x1, y1 = t["bounds"]
        cm = t["prob_native"] >= np.float32(REFINER_THRESHOLD)
        if cm.any():
            full = np.zeros((h, w), bool)
            full[y0:y1, x0:x1] = cm
            t["empty"] = False
        else:
            full = t["fallback"]  # detector fallback; soft = mask itself
            t["empty"] = True
            t["prob_native"] = np.where(full[y0:y1, x0:x1], np.float32(1.0), np.float32(0.0))
        control.append(full)
        if len(t["members"]) >= 2:
            parts = split_mask_with_detector_seeds(full, t["members"], t["mscores"])
        else:
            parts = [full]
        n_split += int(len(parts) > 1)
        task_parts.append(parts)
    abstain = n_split > MAXIMUM_SPLIT_TASKS
    # (mask, score, contributing tasks, is_part)
    items: list[tuple[np.ndarray, float, list[dict], bool]] = []
    for t, parts, ctrl in zip(tasks, task_parts, control):
        use = [ctrl] if abstain else parts
        sc = float(t["mscores"].max())
        for part in use:
            items.append((part, sc, [t], (not abstain) and len(parts) > 1))
    # merge positive-overlap outputs
    rl = [_enc(it[0]) for it in items]
    out = []
    for g in positive_overlap_groups_rle(rl):
        mask = np.logical_or.reduce([items[i][0] for i in g])
        tl = [t for i in g for t in items[i][2]]
        soft = _soft_window(mask, tl)
        if soft is None:
            continue
        sc = max(items[i][1] for i in g)
        if len(g) == 1 and items[g[0]][3]:  # soft of one split part must not leak into its sibling
            y0, x0, q = unpack_soft(soft)
            q = np.where(mask[y0 : y0 + q.shape[0], x0 : x0 + q.shape[1]], q, 0).astype(np.uint8)
            soft = pack_soft(y0, x0, q)
        out.append({"mask": mask, "score": sc, "soft": soft})
    out.sort(key=lambda d: (np.nonzero(d["mask"].any(1))[0][0], np.nonzero(d["mask"].any(0))[0][0]))
    return out


def rows_from_instances(stem: str, insts: list[dict]) -> list[tuple]:
    return [(stem, d["score"], encode_mask(d["mask"].astype(np.uint8)), int(d["mask"].sum()), d["soft"]) for d in insts]


# ----------------------------------------------------------------------------- refine-only
def refine_proposals(refiner: nn.Module, image: np.ndarray, masks: list[np.ndarray], softs: list[tuple],
                     device, tta: bool = True):
    """For each proposal mask build the i079 crop, refine, paste back.

    Returns list of (mask bool, packed soft) aligned with ``masks``; proposals whose refined
    mask is empty keep their original mask and soft.
    """
    h, w = image.shape
    bounds, crops = [], []
    for m in masks:
        b, c = make_crop(image, m)
        bounds.append(b)
        crops.append(c)
    probs = refine_probabilities(refiner, np.stack(crops), device, tta=tta) if crops else []
    res = []
    for m, sf, b, p in zip(masks, softs, bounds, probs):
        x0, y0, x1, y1 = b
        pn = crop_prob_native(p, b)
        cm = pn >= np.float32(REFINER_THRESHOLD)
        if not cm.any():
            res.append((m, sf))
            continue
        full = np.zeros((h, w), bool)
        full[y0:y1, x0:x1] = cm
        res.append((full, pack_soft(y0, x0, np.round(pn * 255).clip(0, 255).astype(np.uint8))))
    return res
