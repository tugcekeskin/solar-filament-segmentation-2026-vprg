"""Train a YOLO-seg direct-instance baseline on the filament dataset.

Example:
    .venv/bin/python scripts/train.py --model weights/yolo11s-seg.pt --imgsz 1024 --name y11s_1024
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="weights/yolo11s-seg.pt")
    p.add_argument("--data", default="data/yolo/v1/data.yaml")
    p.add_argument("--pu", action="store_true", help="PU loss: class 1 = ignore regions (needs --set single_cls=false)")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=float, default=-1, help="-1 = autobatch; 0<x<1 = VRAM fraction")
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--mask-ratio", type=int, default=4)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", default="ram")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override any ultralytics train arg, e.g. --set hsv_v=0.12 copy_paste=0.2 box=10")
    return p.parse_args()


def _val(v: str):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def main() -> None:
    a = parse_args()
    if a.pu:
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from filseg.pu_loss import install

        install()
    batch = int(a.batch) if a.batch >= 1 or a.batch == -1 else a.batch
    model = YOLO(str(ROOT / a.model))
    kw = dict(
        data=str(ROOT / a.data),
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=batch,
        optimizer="AdamW",
        lr0=a.lr0,
        cos_lr=True,
        warmup_epochs=3,
        amp=True,
        single_cls=not a.pu,  # PU needs the ignore class
        mask_ratio=a.mask_ratio,
        overlap_mask=True,
        # Geometry augmentations: the solar disk has no preferred orientation for detection.
        fliplr=0.5,
        flipud=0.5,
        degrees=15.0,
        scale=0.3,
        translate=0.05,
        mosaic=0.5,
        close_mosaic=15,
        # Grayscale H-alpha: no hue/saturation jitter, mild brightness only.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        max_det=100,
        patience=30,
        seed=a.seed,
        deterministic=False,
        workers=a.workers,
        cache=a.cache,
        project=str(ROOT / "runs"),
        name=a.name,
        exist_ok=False,
        plots=True,
    )
    kw.update({k: _val(v) for k, v in (x.split("=", 1) for x in a.set)})
    print("overrides:", a.set)
    model.train(**kw)


if __name__ == "__main__":
    main()
