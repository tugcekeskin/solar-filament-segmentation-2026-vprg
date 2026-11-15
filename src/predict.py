"""Test inference -> competition submission CSV (filament_id, segmentation_rle; COCO RLE, Fortran order).
Usage: python src/predict.py --weights W --imgsz 1024 --conf .4 --nms .5 --min_area 20 --thin 3 --out submission.csv"""
import argparse, os, glob, csv, sys
import numpy as np
from pycocotools import mask as cm
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(__file__))
from dataset import ROOT
from postprocess import raw_predict, post

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True); ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--conf", type=float, default=0.3); ap.add_argument("--nms", type=float, default=0.3)
ap.add_argument("--min_area", type=int, default=20); ap.add_argument("--thin", default="5")
ap.add_argument("--out", required=True)
a = ap.parse_args()
model = YOLO(a.weights)
rows = []
for p in sorted(glob.glob(f"{ROOT}/test/test_images/*.jp*g")):
    stem = os.path.splitext(os.path.basename(p))[0]
    masks = post(*raw_predict(model, p, a.imgsz), a.conf, a.nms, a.min_area, a.thin)
    for i, m in enumerate(masks, 1):
        rle = cm.encode(np.asfortranarray(m.astype(np.uint8)))
        rows.append((f"{stem}_{i}", rle["counts"].decode("utf-8")))
with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["filament_id", "segmentation_rle"]); w.writerows(rows)
print(a.out, len(rows), "filaments", f"{os.path.getsize(a.out)/1e6:.2f} MB")
