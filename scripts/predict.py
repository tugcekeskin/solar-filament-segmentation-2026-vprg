#!/usr/bin/env python
"""Run a YOLO-seg model on the val or test images and cache raw instances (pickle)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.infer import predict_raw, save_cache  # noqa: E402

ROOT = "data/raw/MAGFiLO_1.0_Kaggle_2026"
SPLIT = "data/splits/split_v1.json"


def subset_images(subset: str, root: str | None = None) -> list[str]:
    """Raw image paths; with ``root`` (e.g. data/proc/flat3) the preprocessed copies with the same stems."""
    paths = _subset_images(subset)
    if root:
        paths = [f"{root}/{'test_images' if subset == 'test' else 'train_images'}/{Path(x).name}" for x in paths]
    return paths


def _subset_images(subset: str) -> list[str]:
    if subset == "val":
        stems = json.load(open(SPLIT))["val"]
        return [f"{ROOT}/train/train_images/{s}.jpeg" for s in stems]
    if subset == "train":
        stems = json.load(open(SPLIT))["train"]
        return [f"{ROOT}/train/train_images/{s}.jpeg" for s in stems]
    return sorted(str(p) for p in Path(f"{ROOT}/test/test_images").glob("*.jpeg"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--subset", choices=["val", "train", "test"], required=True)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.01)
    p.add_argument("--iou", type=float, default=0.7, help="ultralytics box NMS iou")
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--tta", action="store_true", help="ultralytics augment=True")
    p.add_argument("--soft", action="store_true", help="also cache uint8 mask probabilities (bbox crops)")
    p.add_argument("--flip", default="", choices=["", "h", "v", "hv"], help="manual flip TTA pass")
    p.add_argument("--image-root", help="preprocessed image dir with train_images/ and test_images/ (e.g. data/proc/flat3)")
    p.add_argument("--cls", action="store_true", help="also cache the predicted class index (multi-class models)")
    p.add_argument("--out-cache", required=True)
    a = p.parse_args()
    paths = subset_images(a.subset, a.image_root)
    t = time.time()
    df = predict_raw(a.weights, paths, a.imgsz, a.conf, a.iou, a.max_det, a.batch, tta=a.tta, soft=a.soft, flip=a.flip, with_cls=a.cls)
    save_cache(df, a.out_cache)
    print(f"{a.subset}: {len(paths)} images, {len(df)} instances, {df['stem'].nunique()} stems with preds, "
          f"{time.time() - t:.1f}s -> {a.out_cache}")


if __name__ == "__main__":
    main()
