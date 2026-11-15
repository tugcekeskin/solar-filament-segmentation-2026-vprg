#!/usr/bin/env python
"""Positive-unlabeled variant of data/yolo/v1: in train stems with a single annotator view, detector
predictions that no GT instance overlaps (IoU < --max-iou) and that score >= --min-score are added as
class 1 ("ignore"). ``filseg.pu_loss`` gives anchors assigned to ignore instances no cls/box/mask loss,
so likely-missed filaments stop being taught as hard negatives. Multi-view stems and val are unchanged:
there the disagreement between views already acts as a soft label that matches the per-view metric.

Usage: .venv/bin/python scripts/make_pu_dataset.py --pred artifacts/refiner/train_raw.pkl --out data/yolo/v5_pu
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg import data as D  # noqa: E402
from filseg.infer import load_cache  # noqa: E402
from filseg.rle import H, W, decode_rle, to_rle_obj  # noqa: E402


def polygons(mask: np.ndarray, min_area: float = 30.0) -> list[list[float]]:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(np.float64)
        if len(c) < 3:
            continue
        c[:, 0] /= W
        c[:, 1] /= H
        out.append(np.clip(c, 0, 1).reshape(-1).tolist())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", default="artifacts/refiner/train_raw.pkl")
    p.add_argument("--src", default="data/yolo/v1")
    p.add_argument("--out", default="data/yolo/v5_pu")
    p.add_argument("--split", default="data/splits/split_v1.json")
    p.add_argument("--min-score", type=float, default=0.1)
    p.add_argument("--max-iou", type=float, default=0.1)
    p.add_argument("--min-area", type=int, default=100)
    a = p.parse_args()
    coco = D.load_coco()
    train = set(json.load(open(a.split))["train"])
    views = defaultdict(list)
    for im in coco["images"]:
        views[Path(im["file_name"]).stem].append(im["id"])
    anns = defaultdict(list)
    for x in coco["annotations"]:
        anns[x["image_id"]].append(x)
    src, out = Path(a.src), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for sub in ("train", "val"):
        (out / "images" / sub).mkdir(parents=True)
        (out / "labels" / sub).mkdir(parents=True)
        for f in (src / "images" / sub).iterdir():
            (out / "images" / sub / f.name).symlink_to(f.resolve())
        for f in (src / "labels" / sub).glob("*.txt"):
            shutil.copy(f, out / "labels" / sub / f.name)
    pred = load_cache(a.pred)
    pred = pred[(pred["score"] >= a.min_score) & (pred["area"] >= a.min_area)]
    n_ign, n_stems = 0, 0
    for stem, g in pred.groupby("stem"):
        if stem not in train or len(views[stem]) != 1:
            continue
        vid = views[stem][0]
        gt = [mu.frPyObjects(x["segmentation"], H, W) for x in anns[vid]]
        gt = [mu.merge(r) if isinstance(r, list) else r for r in gt]
        P = [to_rle_obj(r) for r in g["rle"]]
        iou = np.asarray(mu.iou(P, gt, [0] * len(gt))).max(1) if gt else np.zeros(len(P))
        lines = []
        for r, i in zip(g["rle"], iou):
            if i >= a.max_iou:
                continue
            for poly in polygons(decode_rle(r)):
                lines.append(D.yolo_line(poly, 1))
        if lines:
            lab = out / "labels" / "train" / f"{vid}.txt"
            txt = lab.read_text()
            lab.write_text(txt + ("" if not txt or txt.endswith("\n") else "\n") + "\n".join(lines) + "\n")
            n_ign += len(lines)
            n_stems += 1
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n  1: ignore\n")
    print(f"{n_ign} ignore polygons added to {n_stems} single-view train stems -> {out}")


if __name__ == "__main__":
    main()
