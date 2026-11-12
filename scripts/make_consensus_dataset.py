#!/usr/bin/env python
"""YOLO dataset where, in train stems annotated by 3 annotators, filaments that no other annotator marked
(no IoU > 0.5 match in any other view) are dropped. Val/1-2 annotator stems unchanged.

Usage: .venv/bin/python scripts/make_consensus_dataset.py --out data/yolo/v2_consensus
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg import data as D  # noqa: E402

RES = 1024  # IoU computed at 1024 (cheap, close to full-res for >= 0.5 decisions)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/yolo/v2_consensus")
    p.add_argument("--split", default="data/splits/split_v1.json")
    p.add_argument("--min-views", type=int, default=3)
    p.add_argument("--iou", type=float, default=0.5)
    a = p.parse_args()
    coco = D.load_coco()
    split = json.load(open(a.split))
    train = set(split["train"])
    views = defaultdict(list)
    for im in coco["images"]:
        views[Path(im["file_name"]).stem].append(im["id"])
    anns = defaultdict(list)
    for x in coco["annotations"]:
        anns[x["image_id"]].append(x)
    drop, total = set(), 0
    for stem, vids in views.items():
        if stem not in train or len(vids) < a.min_views:
            continue
        masks = {v: [(x["id"], D.rasterize_at(x["segmentation"][0], RES)) for x in anns[v]] for v in vids}
        for v in vids:
            for aid, m in masks[v]:
                total += 1
                ok = any(D.mask_iou(m, m2) > a.iou for v2 in vids if v2 != v for _, m2 in masks[v2])
                if not ok:
                    drop.add(aid)
    c2 = copy.deepcopy(coco)
    c2["annotations"] = [x for x in coco["annotations"] if x["id"] not in drop]
    counts = D.write_yolo_dataset(c2, split, Path(a.out))
    print(f"3-annotator train instances {total}, dropped {len(drop)} ({len(drop) / max(total, 1):.1%}); {counts}")


if __name__ == "__main__":
    main()
