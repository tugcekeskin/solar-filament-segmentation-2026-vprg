"""Cache raw predictions on val and grid over postprocess params. Usage: python src/evaluate.py --weights runs/run1/weights/best.pt --imgsz 1024"""
import argparse, json, os, pickle, itertools, time
import numpy as np, torch
from ultralytics import YOLO
import sys; sys.path.insert(0, os.path.dirname(__file__))
from dataset import val_gt, IMG_DIR
from metrics import evaluate
from postprocess import raw_predict, post

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--fast", action="store_true"); ap.add_argument("--split", default="data_yolo/split.json")
    a = ap.parse_args()
    val = json.load(open(a.split))["val"]
    gt = val_gt(val)
    model = YOLO(a.weights)
    torch.cuda.reset_peak_memory_stats(); t = time.time()
    raw = {f: raw_predict(model, f"{IMG_DIR}/{f}", a.imgsz) for f in val}
    per_img = (time.time() - t) / len(val)
    mem = torch.cuda.max_memory_allocated() / 1e9
    res = []
    for conf, nms, ma, thin in itertools.product([0.3, 0.4, 0.5], [0.5], [20], ["3", "5"]) if a.fast else itertools.product([0.2, 0.3, 0.4, 0.5], [0.3, 0.5], [20, 100], ["none", "3", "5", "7"]):
        preds = {f: post(*raw[f], conf, nms, ma, thin) for f in val}
        r = evaluate(preds, gt); r.update(conf=conf, nms=nms, min_area=ma, thin=thin); res.append(r)
    res.sort(key=lambda r: -r["PQ"])
    for r in res[:8]: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    base = [r for r in res if (r["conf"], r["nms"], r["min_area"], r["thin"]) == (0.3, 0.3, 20, "5")]
    if base: print("baseline cfg (conf.3,nms.3,area20,thin5):", base[0])
    print(f"inference {per_img:.2f}s/img (incl. mask transfer) gpu_mem {mem:.2f}GB")
    json.dump(res, open(a.weights.replace(".pt", "_grid.json"), "w"))
