"""Data utilities: COCO(MAGFiLO) loading, annotator de-duplication, train/val split, YOLO-seg export."""
import json, os, random, collections
import numpy as np
import cv2
from pycocotools import mask as cm

ROOT = "/home/baykar/Desktop/FLAMENT/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
ANN = f"{ROOT}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMG_DIR = f"{ROOT}/train/train_images"
S = 2048


def load_coco():
    d = json.load(open(ANN))
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    by_file = collections.defaultdict(lambda: collections.defaultdict(list))  # file -> annotator-entry -> anns
    for a in d["annotations"]:
        by_file[id2file[a["image_id"]]][a["image_id"]].append(a)
    return by_file


def poly_to_mask(seg, s=S):
    m = np.zeros((s, s), np.uint8)
    for p in seg:
        pts = np.round(np.array(p, np.float32).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m


def split_files(files, val_frac=0.15, seed=42):
    """Split by unique image, stratified by observatory code (last 2 chars before extension)."""
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for f in sorted(files):
        groups[f[-7:-5]].append(f)
    tr, va = [], []
    for g in groups.values():
        rng.shuffle(g)
        n = max(1, round(len(g) * val_frac))
        va += g[:n]; tr += g[n:]
    return sorted(tr), sorted(va)


def iou(a, b):
    i = np.logical_and(a, b).sum()
    return i / max(1, np.logical_or(a, b).sum())


def dedupe_annotators(entries, thr=0.3, drop_cats=()):
    """Merge several annotators' instances of the same image: keep an instance only if it does not
    overlap (IoU>thr) an already kept one. Returns list of (polygon, category)."""
    kept, masks = [], []
    for ents in sorted(entries.values(), key=lambda x: -len(x)):
        for a in ents:
            if a["category_id"] in drop_cats:
                continue
            m = poly_to_mask(a["segmentation"])
            if any(iou(m, k) > thr for k in masks):
                continue
            masks.append(m); kept.append(a)
    return kept


def build_yolo(out="data_yolo", val_frac=0.15, seed=42, drop_cats=(), thr=0.3):
    by_file = load_coco()
    tr, va = split_files(list(by_file), val_frac, seed)
    for split, files in (("train", tr), ("val", va)):
        os.makedirs(f"{out}/images/{split}", exist_ok=True)
        os.makedirs(f"{out}/labels/{split}", exist_ok=True)
        for f in files:
            dst = f"{out}/images/{split}/{f}"
            if not os.path.exists(dst):
                os.symlink(f"{IMG_DIR}/{f}", dst)
            lines = []
            for a in dedupe_annotators(by_file[f], thr, drop_cats):
                for p in a["segmentation"]:
                    if len(p) < 6:
                        continue
                    xy = (np.array(p).reshape(-1, 2) / S).clip(0, 1)
                    lines.append("0 " + " ".join(f"{v:.5f}" for v in xy.ravel()))
            open(f"{out}/labels/{split}/{f[:-5]}.txt", "w").write("\n".join(lines))
    os.makedirs(out, exist_ok=True)
    open(f"{out}/data.yaml", "w").write(
        f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    json.dump({"train": tr, "val": va}, open(f"{out}/split.json", "w"))
    return tr, va


def val_gt(files):
    """Per-annotator GT mask lists for each val file: {file: [[mask,...] per annotator entry]}."""
    by_file = load_coco()
    return {f: [[poly_to_mask(a["segmentation"]) for a in ents] for ents in by_file[f].values()] for f in files}


if __name__ == "__main__":
    tr, va = build_yolo()
    print(len(tr), len(va))
