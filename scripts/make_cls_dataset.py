#!/usr/bin/env python
"""YOLO dataset identical to data/yolo/v1 but with 2 classes: 0 = identified (Left/Right),
1 = unidentified (Unidentifiable/Ambiguous). Lets the detector learn a separate confidence for the
class that holds ~half of our false negatives, so it can get its own threshold at inference.

Usage: .venv/bin/python scripts/make_cls_dataset.py --out data/yolo/v4_cls
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg import data as D  # noqa: E402

CLS = {1: 0, 2: 0, 3: 1, 4: 1}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/yolo/v4_cls")
    p.add_argument("--split", default="data/splits/split_v1.json")
    a = p.parse_args()
    coco = D.load_coco()
    split = json.load(open(a.split))
    out = Path(a.out)
    part = {s: "train" for s in split["train"]}
    part.update({s: "val" for s in split["val"]})
    anns = defaultdict(list)
    for x in coco["annotations"]:
        anns[x["image_id"]].append(x)
    for sub in ("train", "val"):
        for kind in ("images", "labels"):
            (out / kind / sub).mkdir(parents=True, exist_ok=True)
    n = defaultdict(int)
    for im in coco["images"]:
        vid, stem = str(im["id"]), Path(im["file_name"]).stem
        sub = part[stem]
        link = out / "images" / sub / f"{vid}.jpeg"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((D.IMG_DIR / im["file_name"]).resolve())
        lines = []
        for x in anns.get(im["id"], []):
            norm = D.normalize_polygon(x["segmentation"][0])
            if norm is None:
                continue
            c = CLS[x["category_id"]]
            lines.append(D.yolo_line(norm, c))
            n[c] += 1
        (out / "labels" / sub / f"{vid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: identified\n  1: unidentified\n")
    print(f"labels per class: {dict(n)} -> {out}")


if __name__ == "__main__":
    main()
