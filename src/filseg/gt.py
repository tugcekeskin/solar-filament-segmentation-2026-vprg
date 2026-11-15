"""Build organizer-format ground-truth dataframes from the COCO train JSON."""
from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd
from pycocotools import mask as mask_util


def view_stem(view_id: str) -> str:
    """Annotator-view id '<annotator>-<stem>' -> '<stem>'."""
    return view_id.split("-", 1)[1]


def ann_to_rle_str(seg, h: int, w: int) -> str:
    """Polygon(s)/RLE segmentation -> compressed counts string.

    Identical to pycocotools ``COCO.annToRLE`` (frPyObjects -> merge).
    """
    if isinstance(seg, list):
        rle = mask_util.merge(mask_util.frPyObjects(seg, h, w))
    elif isinstance(seg["counts"], list):
        rle = mask_util.frPyObjects(seg, h, w)
    else:
        rle = seg
    counts = rle["counts"]
    return counts.decode("utf-8") if isinstance(counts, bytes) else counts


def build_gt_df(coco_json_path: str, stems: set[str] | None = None) -> pd.DataFrame:
    """COCO json -> DataFrame[filament_id, segmentation_rle].

    filament_id is '<view_id>_<k>' with k = 1..n per view (annotation order in
    the json). If ``stems`` is given only views whose stem is in it are kept.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    by_view: dict[str, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        by_view[ann["image_id"]].append(ann)
    rows: list[tuple[str, str]] = []
    for view_id in sorted(by_view):
        if stems is not None and view_stem(view_id) not in stems:
            continue
        im = images[view_id]
        for k, ann in enumerate(by_view[view_id], start=1):
            rows.append((f"{view_id}_{k}", ann_to_rle_str(ann["segmentation"], im["height"], im["width"])))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def first_view_as_pred(gt_df: pd.DataFrame) -> pd.DataFrame:
    """Use one annotator view per stem (first by view id) as predictions.

    Output ids are '<stem>_<k>'. Inter-annotator sanity / upper-bound helper.
    """
    view = gt_df["filament_id"].str.split("_", n=1).str[0]
    stem = view.map(view_stem)
    chosen: dict[str, str] = {}
    for v in sorted(view.unique()):
        chosen.setdefault(view_stem(v), v)
    keep = view == stem.map(chosen)
    sub = gt_df[keep]
    k = sub["filament_id"].str.rsplit("_", n=1).str[1]
    ids = stem[keep] + "_" + k
    return pd.DataFrame({"filament_id": ids.values, "segmentation_rle": sub["segmentation_rle"].values})
