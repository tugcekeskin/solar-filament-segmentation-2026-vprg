"""Organizer-matched pooled Panoptic Quality.

Two implementations:

* ``reference_*``: vendored from the organizers' self-evaluation notebook
  (ref/organizer_eval/self-evaluation-notebook.ipynb), torch float32 dense
  masks. Slow; ground truth for testing. The only change is that
  ``reference_pq_counts`` additionally returns TP/FP/FN counts; PQ logic is
  unchanged.
* ``fast_evaluate``: pycocotools RLE IoU. Same semantics (hit = IoU > 0.5, no
  one-to-one matching, pooled over GT annotator-views). The IoU is cast to
  float32 before thresholding to mirror the reference's float32 arithmetic.
  Dice is derived from IoU and areas.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

from filseg.gt import view_stem
from filseg.rle import H, W, to_rle_obj

IOU_THRESHOLD = 0.5


# --------------------------------------------------------------------------
# Reference implementation (vendored from the organizer notebook)
# --------------------------------------------------------------------------
def get_overlap_matrices(gt_layers: torch.Tensor, pred_layers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU and Dice, shapes (n_gt, n_pred). Verbatim from organizers."""
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]
    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)
    intersection = torch.matmul(gt_flat, pred_flat.t())
    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)
    union = gt_areas + pred_areas - intersection
    iou_matrix = torch.where(union == 0, torch.tensor(0., device=gt_layers.device), intersection / union)
    dice_matrix = torch.where(
        union == 0, torch.tensor(0., device=gt_layers.device), 2 * intersection / (gt_areas + pred_areas)
    )
    return iou_matrix, dice_matrix


def fp_count_hit(hit_matrix: torch.Tensor) -> int:
    """Predictions (columns) with no hit. Verbatim from organizers."""
    return (hit_matrix.sum(dim=0) == 0).sum().item()


def fn_count_hit(hit_matrix: torch.Tensor) -> int:
    """GT rows with no hit. Verbatim from organizers."""
    return (hit_matrix.sum(dim=1) == 0).sum().item()


def rles_to_layers(rles: list[str], height: int = H, width: int = W) -> np.ndarray:
    """Decode RLE strings to float32 (n, H, W). Verbatim from organizers."""
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    masks = mask_util.decode(rle_dicts)
    return masks.transpose(2, 0, 1).astype(np.float32)


def process_entry(annotator_image, gt_annotator_image_ids, pred_image_ids, gt_df, pred_df):
    """Overlap matrices for one annotator-view. Verbatim from organizers."""
    image_id = annotator_image.split("-", maxsplit=1)[1]
    gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_image, "segmentation_rle"].tolist()
    gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
    pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist()
    pred_layers = torch.from_numpy(rles_to_layers(pred_rles))
    n_gt = len(gt_rles)
    n_pred = len(pred_rles)
    iou_matrix, dice_matrix = get_overlap_matrices(gt_layers, pred_layers)
    return iou_matrix, dice_matrix, n_gt, n_pred


def reference_overlap_df(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Per-annotator-view overlap table (organizer ``get_overlap_df``)."""
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.split("_", n=1).str[0]
    annotator_images = gt_annotator_image_ids.unique()
    overlap_df = pd.DataFrame({"annotator_image": annotator_images})
    results = overlap_df["annotator_image"].apply(
        process_entry,
        gt_annotator_image_ids=gt_annotator_image_ids,
        pred_image_ids=pred_image_ids,
        gt_df=gt_df,
        pred_df=pred_df,
    )
    overlap_df["iou_matrix"], overlap_df["dice_matrix"], overlap_df["n_gt"], overlap_df["n_pred"] = zip(*results)
    return overlap_df


def reference_pq_counts(overlap_df: pd.DataFrame) -> tuple[float, int, int, int]:
    """Organizer ``get_pq_score`` logic; returns (pq, tp, fp, fn)."""
    iou_threshold: float = IOU_THRESHOLD
    tp_iou_scores: list[float] = []
    fp_count = 0
    fn_count = 0
    for row in overlap_df.itertuples(index=False):
        iou_matrix = row.iou_matrix
        n_gt = row.n_gt
        n_pred = row.n_pred
        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue
        hit_matrix = iou_matrix > iou_threshold
        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += fp_count_hit(hit_matrix)
        fn_count += fn_count_hit(hit_matrix)
    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count
    pq_score = sum(tp_iou_scores) / denominator if denominator > 0 else 0.0
    return pq_score, tp_count, fp_count, fn_count


def reference_pq(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """Convenience: organizer PQ from the two dataframes."""
    return reference_pq_counts(reference_overlap_df(gt_df, pred_df))[0]


# --------------------------------------------------------------------------
# Fast implementation
# --------------------------------------------------------------------------
@dataclass
class EvalResult:
    """Pooled metrics plus diagnostics. ``per_view`` has one row per GT view."""

    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    dice: float
    n_views: int
    n_gt: int
    n_pred: int
    splits: int
    merges: int
    dup_hits: int
    per_view: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def to_dict(self, per_view: bool = False) -> dict:
        """Plain dict (JSON-able); per-view table optional."""
        d = {k: v for k, v in asdict(self).items() if k != "per_view"}
        if per_view:
            d["per_view"] = self.per_view.to_dict(orient="records")
        return d


def _group_rles(df: pd.DataFrame, key: pd.Series) -> dict[str, list[dict]]:
    """key value -> list of RLE dicts (row order preserved)."""
    out: dict[str, list[dict]] = {}
    for k, c in zip(key.tolist(), df["segmentation_rle"].tolist()):
        out.setdefault(k, []).append(to_rle_obj(c))
    return out


def _view_matrices(gt_rles: list[dict], pred_rles: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(iou, dice) float arrays shape (n_gt, n_pred) via pycocotools."""
    iou = mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles)).T  # (n_gt, n_pred)
    iou = np.asarray(iou, dtype=np.float64).reshape(len(gt_rles), len(pred_rles))
    ga = np.asarray(mask_util.area(gt_rles), dtype=np.float64)[:, None]
    pa = np.asarray(mask_util.area(pred_rles), dtype=np.float64)[None, :]
    inter = np.rint(iou * (ga + pa) / (1.0 + iou))  # exact: IoU = I/(a+b-I)
    denom = ga + pa
    dice = np.divide(2 * inter, denom, out=np.zeros_like(iou), where=denom > 0)
    return iou, dice


def _empty_per_view() -> pd.DataFrame:
    return pd.DataFrame(columns=["view", "stem", "n_gt", "n_pred", "tp", "fp", "fn", "sum_iou", "pq"])


def fast_evaluate(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> EvalResult:
    """Pooled organizer PQ (+ diagnostics) using RLE IoU; see module docstring."""
    gt_view = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_stem = pred_df["filament_id"].astype(str).str.split("_", n=1).str[0]
    gt_groups = _group_rles(gt_df, gt_view)
    pred_groups = _group_rles(pred_df, pred_stem)

    rows: list[dict] = []
    tp_ious: list[float] = []
    tp_dice: list[float] = []
    splits = merges = dups = 0
    for view in pd.unique(gt_view):
        stem = view_stem(view)
        g = gt_groups[view]
        p = pred_groups.get(stem, [])
        n_gt, n_pred = len(g), len(p)
        if n_pred == 0:
            rows.append(dict(view=view, stem=stem, n_gt=n_gt, n_pred=0, tp=0, fp=0, fn=n_gt, sum_iou=0.0))
            continue
        iou, dice = _view_matrices(g, p)
        hit = iou.astype(np.float32) > np.float32(IOU_THRESHOLD)
        tp_ious.extend(iou[hit].tolist())
        tp_dice.extend(dice[hit].tolist())
        pos = iou > 0
        splits += int((pos.sum(axis=1) >= 2).sum())
        merges += int((pos.sum(axis=0) >= 2).sum())
        dups += int((hit.sum(axis=1) >= 2).sum())
        rows.append(dict(
            view=view, stem=stem, n_gt=n_gt, n_pred=n_pred, tp=int(hit.sum()),
            fp=int((hit.sum(axis=0) == 0).sum()), fn=int((hit.sum(axis=1) == 0).sum()),
            sum_iou=float(iou[hit].sum()),
        ))
    pv = pd.DataFrame(rows) if rows else _empty_per_view()
    if rows:
        den = pv["tp"] + 0.5 * pv["fp"] + 0.5 * pv["fn"]
        pv["pq"] = np.where(den > 0, pv["sum_iou"] / den.where(den > 0, 1), 0.0)
    tp, fp, fn = int(pv["tp"].sum()), int(pv["fp"].sum()), int(pv["fn"].sum())
    den = tp + 0.5 * fp + 0.5 * fn
    s = sum(tp_ious)
    return EvalResult(
        pq=s / den if den > 0 else 0.0,
        sq=s / tp if tp else 0.0,
        rq=tp / den if den > 0 else 0.0,
        tp=tp, fp=fp, fn=fn,
        dice=float(np.mean(tp_dice)) if tp_dice else 0.0,
        n_views=len(pv), n_gt=int(pv["n_gt"].sum()), n_pred=int(pv["n_pred"].sum()),
        splits=splits, merges=merges, dup_hits=dups, per_view=pv,
    )
