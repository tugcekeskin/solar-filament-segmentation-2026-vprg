"""Run the external CAGE U-Net: save prob maps + mode-B raw cache, or refine a YOLO cache.

  predict_cage.py --subset val --out-dir artifacts/cage/val
  predict_cage.py --subset val --out-dir artifacts/cage/val --refine-cache C.pkl --alpha 0.5 --out OUT.pkl
(refine reuses the saved npz maps in --out-dir; run the predict step first.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filseg import ext_cage  # noqa: E402
from filseg.infer import load_cache, save_cache  # noqa: E402

IMG = ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images"


def stems_for(subset: str) -> tuple[list[str], Path]:
    if subset == "val":
        return sorted(json.load(open(ROOT / "data/splits/split_v1.json"))["val"]), IMG
    d = ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    return sorted(p.stem for p in d.glob("*.jpeg")), d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["val", "test"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--weights", default=str(ROOT / "ext/cage/unet1024_s5.pt"))
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--refine-cache")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--out")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems, img_dir = stems_for(a.subset)

    if a.refine_cache:
        assert a.alpha is not None and a.out
        pm = {s: np.load(out_dir / f"{s}.npz")["probs"][0] for s in stems if (out_dir / f"{s}.npz").exists()}
        save_cache(ext_cage.refine_cache(load_cache(a.refine_cache), pm, a.alpha), a.out)
        return

    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ext_cage.load_model(a.weights, dev)
    rows: list[tuple] = []
    for i in range(0, len(stems), a.batch):
        chunk = stems[i : i + a.batch]
        grays = [cv2.imread(str(img_dir / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE) for s in chunk]
        probs = ext_cage.predict_probs(model, grays, dev)
        for s, p in zip(chunk, probs):
            np.savez_compressed(out_dir / f"{s}.npz", probs=p)
            rows += ext_cage.instances_to_rows(s, ext_cage.mode_b_instances(p))
    save_cache(ext_cage.rows_to_df(rows), str(out_dir / "raw.pkl"))
    print(f"{len(stems)} images, {len(rows)} instances")


if __name__ == "__main__":
    main()
