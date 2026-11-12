"""YOLO-seg training. Usage: python src/train.py --name run1 --model yolo11s-seg.pt --imgsz 1024 --epochs 80 --batch 4"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--model", default="yolo11s-seg.pt")
ap.add_argument("--imgsz", type=int, default=1024); ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--batch", type=int, default=4); ap.add_argument("--data", default="data_yolo/data.yaml")
a = ap.parse_args()
YOLO(a.model).train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, name=a.name,
                    project="runs", exist_ok=True, workers=8, seed=0, patience=0, plots=False, val=True)
