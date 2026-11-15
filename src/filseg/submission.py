"""Submission building and validation.

CSV format: header ``filament_id,segmentation_rle``; ids '<stem>_<k>'. RLE
counts strings contain characters in '0'..'o' (including backslash) but never
commas, quotes or newlines, so pandas' default QUOTE_MINIMAL writes them
verbatim (unquoted) and ``pd.read_csv`` (the official reader) returns identical
strings. Tests verify this round trip.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from filseg.rle import H, W, decode_rle, encode_mask, rle_run_total, to_rle_obj

COLUMNS = ["filament_id", "segmentation_rle"]


def masks_to_rows(stem: str, masks: list[np.ndarray]) -> list[tuple[str, str]]:
    """Instance masks of one test stem -> [(f'{stem}_{k}', rle)], k from 1."""
    return [(f"{stem}_{k}", encode_mask(m)) for k, m in enumerate(masks, start=1)]


def write_submission(rows: list[tuple[str, str]], path: str) -> pd.DataFrame:
    """Write rows to CSV (header + unquoted RLE); returns the dataframe."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(path, index=False)
    return df


def resolve_overlaps(
    masks: list[np.ndarray], scores: list[float] | np.ndarray
) -> tuple[list[np.ndarray], list[int]]:
    """Give each contested pixel to the highest-score instance.

    Instances left empty or that lose more than 50% of their original area are
    dropped (their pixels stay unassigned). Returns (kept masks as bool,
    kept original indices) in original order.
    """
    if not masks:
        return [], []
    scores = np.asarray(scores, dtype=np.float64)
    owner = np.full(masks[0].shape, -1, dtype=np.int32)
    for i in np.argsort(-scores, kind="stable"):
        owner[(owner < 0) & (masks[i] > 0)] = i
    kept_masks: list[np.ndarray] = []
    kept_idx: list[int] = []
    for i, m in enumerate(masks):
        orig = int((m > 0).sum())
        new = owner == i
        area = int(new.sum())
        if area == 0 or area < 0.5 * orig:
            continue
        kept_masks.append(new)
        kept_idx.append(i)
    return kept_masks, kept_idx


@dataclass
class ValidationReport:
    """Result of ``validate_submission``. ``ok`` is True iff no errors."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    overlap_pairs: dict[str, int] = field(default_factory=dict)  # stem -> #overlapping pairs
    n_rows: int = 0
    n_stems: int = 0
    stems_without_predictions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def n_overlap_pairs(self) -> int:
        return sum(self.overlap_pairs.values())


def _check_rle(counts: str) -> str | None:
    """Return an error string if the RLE is invalid, else None."""
    try:
        if rle_run_total(counts) != H * W:
            return "run lengths do not sum to 2048*2048"
        if int(mask_util.area(to_rle_obj(counts))) == 0:
            return "empty mask"
        if decode_rle(counts).shape != (H, W):
            return "bad decoded shape"
    except Exception as e:  # noqa: BLE001 - report any decode problem
        return f"undecodable: {e}"
    return None


def _overlapping_pairs(rles: list[str]) -> int:
    """Number of instance pairs with intersection > 0."""
    if len(rles) < 2:
        return 0
    objs = [to_rle_obj(c) for c in rles]
    iou = np.asarray(mask_util.iou(objs, objs, [0] * len(objs)))
    return int((np.triu(iou, k=1) > 0).sum())


def validate_submission(csv_path: str, test_stems: set[str]) -> ValidationReport:
    """Validate a submission CSV against the competition format."""
    rep = ValidationReport()
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    rep.n_rows = len(df)
    if list(df.columns) != COLUMNS:
        rep.errors.append(f"columns {list(df.columns)} != {COLUMNS}")
        return rep
    dup = df["filament_id"][df["filament_id"].duplicated()]
    if len(dup):
        rep.errors.append(f"{dup.nunique()} duplicated ids, e.g. {dup.iloc[0]}")
    stems = df["filament_id"].str.split("_", n=1).str[0]
    bad = sorted(set(stems) - set(test_stems))
    if bad:
        rep.errors.append(f"{len(bad)} ids with stem not in test set, e.g. {bad[:3]}")
    for fid, counts in zip(df["filament_id"], df["segmentation_rle"]):
        err = _check_rle(counts)
        if err:
            rep.errors.append(f"{fid}: {err}")
    rep.n_stems = stems.nunique()
    rep.stems_without_predictions = sorted(set(test_stems) - set(stems))
    if rep.stems_without_predictions:
        rep.warnings.append(f"{len(rep.stems_without_predictions)} test stems have zero predictions")
    if not rep.errors:
        for stem, grp in df.groupby(stems.values):
            n = _overlapping_pairs(grp["segmentation_rle"].tolist())
            if n:
                rep.overlap_pairs[stem] = n
        if rep.overlap_pairs:
            rep.errors.append(
                f"{rep.n_overlap_pairs} overlapping instance pairs in {len(rep.overlap_pairs)} stems"
            )
    return rep
