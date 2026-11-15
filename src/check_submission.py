"""Validate a submission CSV against the competition format and the TEST image set.
Usage: python src/check_submission.py submission.csv   (exit 1 on any failure)"""
import sys, os, glob, csv, collections
import numpy as np
from pycocotools import mask as cm
sys.path.insert(0, os.path.dirname(__file__))
from dataset import ROOT

path = sys.argv[1]
test = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{ROOT}/test/test_images/*.jp*g")}
train = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{ROOT}/train/train_images/*.jp*g")}
errs = []
rows = list(csv.reader(open(path, newline="")))
if rows[0] != ["filament_id", "segmentation_rle"]: errs.append(f"bad header {rows[0]}")
ids = [r[0] for r in rows[1:]]
if len(ids) != len(set(ids)): errs.append("duplicate filament_id")
per = collections.Counter()
for fid, rle in rows[1:]:
    stem, _, n = fid.rpartition("_")
    if stem not in test: errs.append(f"{fid}: image not in TEST set"); continue
    if stem in train: errs.append(f"{fid}: image is a TRAIN image")
    if not n.isdigit(): errs.append(f"{fid}: bad index")
    m = cm.decode({"size": [2048, 2048], "counts": rle.encode()})
    if m.sum() == 0: errs.append(f"{fid}: empty mask")
    per[stem] += 1
missing = test - set(per)
size = os.path.getsize(path) / 1e6
if size > 10: errs.append(f"file {size:.1f} MB > 10 MB")
print(f"{path}: {len(ids)} filaments over {len(per)}/{len(test)} test images; {len(missing)} test images with no prediction; {size:.2f} MB")
if errs:
    print("FAIL:\n" + "\n".join(errs[:20])); sys.exit(1)
print("OK")
