#!/usr/bin/env python
"""Figures for the report: the pipeline diagram and example segmentations on validation images.

examples: final pipeline (calibrated parameters) vs one annotator's labels, on three validation images chosen
at the 20th, 50th and 80th percentile of per-image PQ (images with 5-12 filaments), full disk plus a zoom.
Usage: report_figures.py [--out docs/report/figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_diag import GT_JSON  # noqa: E402
from filseg.gt import build_gt_df  # noqa: E402
from filseg.infer import load_cache, unpack_soft  # noqa: E402
from filseg.refine import soft_bbox_window  # noqa: E402
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows  # noqa: E402
from filseg.rle import decode_rle  # noqa: E402

IMG = Path("data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images")


def _full(packed: tuple, shape=(2048, 2048)) -> np.ndarray:
    y0, x0, crop = unpack_soft(packed)
    f = np.zeros(shape, np.float32)
    h, w = crop.shape
    f[y0:y0 + h, x0:x0 + w] = crop[: shape[0] - y0, : shape[1] - x0] / 255.0
    return f


def pipeline(out: Path, stem: str = "20111206102934Ch", inst_score: float = 0.55) -> None:
    """Pipeline diagram with real thumbnails from one validation image.

    The illustrated instance (fused score 0.55) was chosen as one where refinement clearly helps: its IoU with
    the closest annotation rises from 0.49 (prior, a false negative) to 0.71 (refined, a true positive).
    """
    img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    fused = load_cache("artifacts/ctx/val_flip3_orig.pkl")
    fused = fused[fused.stem == stem].sort_values("score", ascending=False)
    blend = load_cache("artifacts/y11m_1536_flip_blend/val_raw.pkl")
    blend = blend[blend.stem == stem].sort_values("score", ascending=False)
    # the largest confident instance, matched between the fused and refined caches by IoU
    big = fused.iloc[int(np.argmin(np.abs(fused.score.values - inst_score)))]
    prior = _full(big.soft)
    ious = [((_full(b.soft) > .5) & (prior > .5)).sum() / max(((_full(b.soft) > .5) | (prior > .5)).sum(), 1)
            for b in blend.itertuples()]
    refined = _full(blend.iloc[int(np.argmax(ious))].soft)
    y, x, side = soft_bbox_window(big.soft)
    crop = img[y:y + side, x:x + side]
    pc, rc = prior[y:y + side, x:x + side], refined[y:y + side, x:x + side]
    gt = build_gt_df(GT_JSON, {stem})
    G = [decode_rle(r).astype(bool) for r in gt.segmentation_rle]

    def _iou(a, b):
        return (a & b).sum() / max((a | b).sum(), 1)
    g = max(G, key=lambda m: _iou(prior > .5, m))
    iou_p, iou_r = _iou(prior > .5, g), _iou(refined > .35, g)
    gc = g[y:y + side, x:x + side].astype(np.uint8)
    calib = json.load(open("artifacts/y11m_1536_flip_blend/calib.json"))["best_params"]
    raw = load_cache("artifacts/y11m_1536_flip_blend/val_raw.pkl")
    rows = postprocess_to_rows(group_by_stem(raw), PPParams(**calib), [stem], group_soft_by_stem(raw))

    def heat(gray, p, col):
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255
        a = (0.9 * p)[..., None]
        rgb = (1 - a) * rgb + a * np.array(col)
        cs, _ = cv2.findContours(gc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # annotation outline
        rgb = np.ascontiguousarray((rgb * 255).astype(np.uint8))
        cv2.drawContours(rgb, cs, -1, (255, 200, 0), max(1, side // 220))
        return rgb

    disk = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    final = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for _, r in rows:
        cs, _ = cv2.findContours(decode_rle(r).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(final, cs, -1, (255, 0, 255), 16)
    final = cv2.resize(final, (512, 512), interpolation=cv2.INTER_AREA)

    W, H = 7.0, 2.35
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * H / W)
    ax.axis("off")
    Y = 100 * H / W
    C = {"det": ("#e8eef8", "#2f5d9e"), "ref": ("#e7f4ea", "#2e7d4f"), "dec": ("#fbefe3", "#b0602a")}

    def group(x0, x1, key, title):
        fc, ec = C[key]
        ax.add_patch(FancyBboxPatch((x0, 1.2), x1 - x0, Y - 2.4, boxstyle="round,pad=0,rounding_size=1.5",
                                    fc=fc, ec="none", alpha=0.75, zorder=0))
        ax.text((x0 + x1) / 2, Y - 3.6, title, ha="center", va="center", fontsize=9.0, weight="bold", color=ec)

    def box(x, y, w, h, t, key, fs=7.4):
        fc, ec = C[key]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                    fc="white", ec=ec, lw=0.9, zorder=2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.15)

    def thumb(x, y, w, im, cap=None):
        a = fig.add_axes([x / 100, y / Y, w / 100, w / Y])  # square: 1 data unit = W/100 inch on both axes
        a.imshow(im)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#666")
            sp.set_linewidth(0.6)
        if cap:
            ax.text(x + w / 2, y - 1.6, cap, ha="center", va="top", fontsize=7.0, color="#333")

    def arrow(a, b, col="#444"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=col, zorder=4))

    group(13.5, 43.5, "det", "1  Detection")
    group(44.5, 86.0, "ref", "2  Refinement (per instance)")
    group(87.0, 100.0, "dec", "3  Decoding")
    # input
    thumb(0.4, 9.5, 11.6, cv2.cvtColor(disk, cv2.COLOR_GRAY2RGB), "H$\\alpha$ image\n2048$^2$")
    # detection: three passes -> fusion
    for yy, t in ((20.5, "identity"), (12.5, "h-flip"), (4.5, "v-flip")):
        box(15.0, yy, 12.5, 6.6, f"YOLO11m-seg\n{t}", "det", 7.2)
        arrow((12.2, 15.3), (14.8, yy + 3.3))
        arrow((27.7, yy + 3.3), (30.3, 15.3))
    ax.text(28.5, 2.3, "input 1536 px; per-instance soft masks kept", ha="center", fontsize=6.6, color="#2f5d9e")
    box(30.5, 10.3, 11.5, 10.0, "instance\nfusion\nIoU $\\geq$ 0.5\nmean score", "det", 7.2)
    # refinement: crop + prior -> two refiners -> blend
    thumb(45.4, 9.4, 13.4, heat(crop, pc, (1.0, 0.1, 0.9)), f"crop + fused prior\nIoU {iou_p:.2f} (miss)")
    arrow((42.2, 15.3), (45.2, 15.3))
    box(60.4, 17.6, 9.2, 6.6, "U-Net\nResNet34", "ref", 7.0)
    box(60.4, 6.4, 9.2, 6.6, "SAM ViT-B\n+ LoRA", "ref", 7.0)
    arrow((59.0, 16.1), (60.2, 20.9))
    arrow((59.0, 16.1), (60.2, 9.7))
    thumb(72.0, 9.4, 13.4, heat(crop, rc, (0.1, 0.75, 0.2)), f"50/50 blend\nIoU {iou_r:.2f} (hit)")
    arrow((69.8, 20.9), (71.8, 18.0))
    arrow((69.8, 9.7), (71.8, 14.0))
    # decoding
    box(87.8, 17.2, 11.4, 10.4, "score $\\geq$ 0.25\n$p \\geq$ 0.35\nNMS 0.3\n$\\leq$ 20 / image\nno overlaps\n$\\rightarrow$ RLE", "dec", 6.4)
    thumb(88.9, 3.4, 9.2, final)
    arrow((85.8, 15.3), (87.6, 21.0))
    arrow((93.5, 17.0), (93.5, 12.8))
    fig.savefig(out / "pipeline.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


def examples(out: Path) -> None:
    calib = json.load(open("artifacts/y11m_1536_flip_blend/calib.json"))["best_params"]
    stems = sorted(json.load(open("data/splits/split_v1.json"))["val"])
    pv = pd.read_csv("reports/per_view.csv").groupby("stem")[["n_gt", "N", "D"]].sum()
    pv = pv[pv.n_gt.between(5, 12)]
    pv["pq"] = pv.N / pv.D
    pv = pv.sort_values("pq")
    pick = [pv.index[int(len(pv) * q)] for q in (0.2, 0.5, 0.8)]
    raw = load_cache("artifacts/y11m_1536_flip_blend/val_raw.pkl")
    rows = postprocess_to_rows(group_by_stem(raw), PPParams(**calib), pick, group_soft_by_stem(raw))
    pred = defaultdict(list)
    for k, r in rows:
        pred[k.rsplit("_", 1)[0]].append(decode_rle(r).astype(np.uint8))
    gt = build_gt_df(GT_JSON, set(pick))
    gt["view"] = gt["filament_id"].str.rsplit("_", n=1).str[0]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.75))
    for c, st in enumerate(pick):
        img = cv2.imread(str(IMG / f"{st}.jpeg"), cv2.IMREAD_GRAYSCALE)
        view = sorted(gt[gt.view.str.endswith(st)].view.unique())[0]
        G = [decode_rle(r).astype(np.uint8) for r in gt[gt.view == view].segmentation_rle]
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)
        g = np.zeros(img.shape, bool)
        for m in G:
            g |= m.astype(bool)
        rgb[g] = 0.45 * rgb[g] + 0.55 * np.array([0, 230, 0])  # annotation: filled green
        rgb = rgb.astype(np.uint8)
        for m in pred[st]:  # prediction: magenta outline
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, cs, -1, (255, 0, 255), 4)
        axes[0, c].imshow(rgb)
        axes[0, c].set_title(f"{st}  (PQ {pv.loc[st, 'pq']:.2f})", fontsize=7)
        # zoom on the densest region of the union of masks
        u = np.zeros_like(img, np.float32)
        for m in G + pred[st]:
            u += m
        u = cv2.blur(u, (401, 401))
        y, x = np.unravel_index(int(u.argmax()), u.shape)
        h = 300
        y0, x0 = int(np.clip(y - h, 0, 2048 - 2 * h)), int(np.clip(x - h, 0, 2048 - 2 * h))
        axes[0, c].add_patch(plt.Rectangle((x0, y0), 2 * h, 2 * h, fill=False, ec="yellow", lw=0.8))
        axes[1, c].imshow(rgb[y0:y0 + 2 * h, x0:x0 + 2 * h])
        for r in (0, 1):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "examples.jpg", dpi=220, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"quality": 88})
    plt.close(fig)
    print("examples:", pick)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/report/figures")
    ap.add_argument("--pipeline-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pipeline(out)
    if not a.pipeline_only:
        examples(out)


if __name__ == "__main__":
    main()
