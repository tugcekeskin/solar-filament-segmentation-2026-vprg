"""G1 data pipeline: manifest, date-grouped split, YOLO-seg conversion, fidelity audit."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

SIZE = 2048
RAW_DIR = Path("data/raw/MAGFiLO_1.0_Kaggle_2026/train")
ANN_JSON = RAW_DIR / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMG_DIR = RAW_DIR / "train_images"


# ----------------------------------------------------------------- manifest
def load_coco(path: Path = ANN_JSON) -> dict:
    """Load the COCO annotation JSON."""
    with open(path) as f:
        return json.load(f)


def build_manifest(coco: dict, img_dir: Path = IMG_DIR) -> pd.DataFrame:
    """One row per annotator-view with stem/date/instrument metadata and annotation counts."""
    n_ann: Dict[str, int] = defaultdict(int)
    for a in coco["annotations"]:
        n_ann[a["image_id"]] += 1
    rows = []
    for im in coco["images"]:
        vid = str(im["id"])
        stem = Path(im["file_name"]).stem
        rows.append(dict(
            view_id=vid, annotator=vid.split("-")[0], stem=stem, file_name=im["file_name"],
            date=stem[:8], year=int(stem[:4]), instrument=stem[-2:],
            n_ann=n_ann.get(im["id"], 0),
            image_path=str(img_dir / im["file_name"]),
        ))
    df = pd.DataFrame(rows).sort_values("view_id").reset_index(drop=True)
    df["n_views_of_stem"] = df.groupby("stem")["view_id"].transform("count")
    cols = ["view_id", "annotator", "stem", "file_name", "date", "year", "instrument",
            "n_ann", "n_views_of_stem", "image_path"]
    return df[cols]


# -------------------------------------------------------------------- split
def assign_folds(stems: Sequence[str], n_folds: int = 5, seed: int = 42) -> Dict[str, int]:
    """Assign each stem to a fold, grouping by calendar date (YYYYMMDD).

    Within each year dates are shuffled (seeded), sorted by size descending and greedily
    put in the fold with the lowest (year count + total count), balancing stems per fold
    and years across folds. Deterministic for a given input and seed.
    """
    by_date: Dict[str, List[str]] = defaultdict(list)
    for s in sorted(set(stems)):
        by_date[s[:8]].append(s)
    rng = random.Random(seed)
    total = [0] * n_folds
    year_cnt: Dict[str, List[int]] = defaultdict(lambda: [0] * n_folds)
    out: Dict[str, int] = {}
    for year in sorted({d[:4] for d in by_date}):
        dates = sorted(d for d in by_date if d[:4] == year)
        rng.shuffle(dates)
        dates.sort(key=lambda d: -len(by_date[d]))  # stable: keeps shuffled tie order
        for d in dates:
            n = len(by_date[d])
            f = min(range(n_folds), key=lambda k: (total[k] + year_cnt[year][k], total[k], k))
            for s in by_date[d]:
                out[s] = f
            total[f] += n
            year_cnt[year][f] += n
    return out


def make_split(stems: Sequence[str], n_folds: int = 5, seed: int = 42, val_fold: int = 0) -> dict:
    """Build the split_v1 dict and assert there is no stem/date leakage between train and val."""
    folds = assign_folds(stems, n_folds, seed)
    train = sorted(s for s, f in folds.items() if f != val_fold)
    val = sorted(s for s, f in folds.items() if f == val_fold)
    assert not (set(train) & set(val)), "stem in both train and val"
    assert not ({s[:8] for s in train} & {s[:8] for s in val}), "date in both train and val"
    return {"seed": seed, "group_key": "date", "folds": dict(sorted(folds.items())),
            "val_fold": val_fold, "train": train, "val": val}


# ---------------------------------------------------------------- geometry
def polygon_to_points(seg: Sequence[float]) -> np.ndarray:
    """Flat COCO polygon -> (N,2) float array, dropping a closing duplicate point."""
    pts = np.asarray(seg, dtype=np.float64).reshape(-1, 2)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def normalize_polygon(seg: Sequence[float], size: int = SIZE):
    """Normalize a COCO polygon to [0,1] (clipped). Returns flat list, or None if <3 distinct points."""
    pts = polygon_to_points(seg)
    if len(np.unique(np.round(pts, 6), axis=0)) < 3:
        return None
    return np.clip(pts / size, 0.0, 1.0).reshape(-1).tolist()


def denormalize_polygon(norm: Sequence[float], size: int = SIZE) -> List[float]:
    """Inverse of normalize_polygon: [0,1] flat coords -> pixel coords."""
    return (np.asarray(norm, dtype=np.float64) * size).tolist()


def yolo_line(norm: Sequence[float], cls: int = 0) -> str:
    """Format one YOLO-seg label line."""
    return f"{cls} " + " ".join(f"{v:.6f}" for v in norm)


def parse_yolo_line(line: str) -> List[float]:
    """Parse a YOLO-seg line into normalized coords; validates format (even count, >=6)."""
    parts = line.split()
    coords = [float(x) for x in parts[1:]]
    if len(coords) < 6 or len(coords) % 2:
        raise ValueError(f"bad label line: {line[:60]}")
    return coords


def rasterize(poly: Sequence[float], h: int = SIZE, w: int = SIZE) -> np.ndarray:
    """Rasterize a flat pixel polygon to a uint8 mask (same as COCO annToMask)."""
    rles = mask_util.frPyObjects([list(map(float, poly))], h, w)
    return mask_util.decode(mask_util.merge(rles))


def rasterize_at(poly: Sequence[float], res: int, size: int = SIZE) -> np.ndarray:
    """Rasterize at training resolution `res`, then nearest-upsample to `size`."""
    m = rasterize([c * res / size for c in poly], res, res)
    if res == size:
        return m
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two binary masks."""
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def polygon_perimeter(seg: Sequence[float]) -> float:
    """Perimeter of the closed polygon in pixels."""
    pts = polygon_to_points(seg)
    return float(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1).sum())


def greedy_match_pq(rles_a: list, rles_b: list, thr: float = 0.5) -> tuple:
    """Greedy one-to-one match (IoU>thr, desc). Returns (pq, tp, fp, fn, sum_iou_tp)
    treating A as ground truth and B as predictions."""
    na, nb = len(rles_a), len(rles_b)
    if na == 0 or nb == 0:
        return 0.0, 0, nb, na, 0.0
    ious = mask_util.iou(rles_b, rles_a, [0] * na)  # (nb, na)
    cand = sorted(((ious[j, i], i, j) for i in range(na) for j in range(nb) if ious[j, i] > thr),
                  reverse=True)
    ua, ub, s, tp = set(), set(), 0.0, 0
    for v, i, j in cand:
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); s += float(v); tp += 1
    fp, fn = nb - tp, na - tp
    return s / (tp + 0.5 * fp + 0.5 * fn), tp, fp, fn, s


# --------------------------------------------------------------- YOLO export
def write_yolo_dataset(coco: dict, split: dict, out_dir: Path, img_dir: Path = IMG_DIR) -> dict:
    """Write YOLO-seg dataset (symlinked images, label txt per view, data.yaml). Returns counts."""
    out_dir = Path(out_dir)
    part = {s: "train" for s in split["train"]}
    part.update({s: "val" for s in split["val"]})
    anns: Dict[str, list] = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    counts = dict(train_views=0, val_views=0, labels=0, skipped=0)
    for sub in ("train", "val"):
        for kind in ("images", "labels"):
            (out_dir / kind / sub).mkdir(parents=True, exist_ok=True)
    for im in coco["images"]:
        vid = str(im["id"])
        stem = Path(im["file_name"]).stem
        sub = part[stem]
        link = out_dir / "images" / sub / f"{vid}.jpeg"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((img_dir / im["file_name"]).resolve())
        lines = []
        for a in anns.get(im["id"], []):
            norm = normalize_polygon(a["segmentation"][0])
            if norm is None:
                counts["skipped"] += 1
                continue
            lines.append(yolo_line(norm))
        (out_dir / "labels" / sub / f"{vid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        counts[f"{sub}_views"] += 1
        counts["labels"] += len(lines)
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    return counts


# --------------------------------------------------------------------- audit
def audit_stem(args: tuple) -> dict:
    """Audit all views of one stem. args=(coco_images_for_stem, anns_by_image_id, labels_dir_lookup, resolutions).

    Returns per-annotation fidelity rows, overlap pair counts and inter-annotator agreement.
    """
    views, label_paths, resolutions = args
    rows, overlaps, view_rles = [], 0, {}
    for vid, anns in views:
        label_lines = []
        lp = label_paths[vid]
        if lp is not None and Path(lp).exists():
            label_lines = [l for l in Path(lp).read_text().splitlines() if l.strip()]
        li = 0
        rles = []
        for a in anns:
            seg = a["segmentation"][0]
            ref = rasterize(seg)
            rle = mask_util.encode(np.asfortranarray(ref))
            rles.append(rle)
            row = dict(view_id=vid, ann_id=a["id"], area_px=int(ref.sum()),
                       perimeter=polygon_perimeter(seg))
            row["width_est"] = 2 * row["area_px"] / row["perimeter"] if row["perimeter"] else 0.0
            if normalize_polygon(seg) is None:
                row["iou_yolo"] = np.nan
            else:
                rt = rasterize(denormalize_polygon(parse_yolo_line(label_lines[li])))
                li += 1
                row["iou_yolo"] = mask_iou(ref, rt)
            for r in resolutions:
                row[f"iou_{r}"] = mask_iou(ref, rasterize_at(seg, r))
            rows.append(row)
        n = len(rles)
        if n > 1:
            ious = mask_util.iou(rles, rles, [0] * n)
            overlaps += int((np.triu(ious, 1) > 0).sum())
        view_rles[vid] = rles
    agree = []
    ids = sorted(view_rles)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pq, tp, fp, fn, s = greedy_match_pq(view_rles[ids[i]], view_rles[ids[j]])
            agree.append(dict(view_a=ids[i], view_b=ids[j], pq=pq, tp=tp, fp=fp, fn=fn, sum_iou=s))
    return dict(rows=rows, overlaps=overlaps, agree=agree)


def run_audit(coco: dict, yolo_dir: Path, split: dict, resolutions=(1024, 1280, 1536),
              workers: int = 16) -> dict:
    """Run the fidelity audit over all annotations (multiprocessing). Returns dict with
    `df` (per-annotation), `overlaps` (int), `agree` (DataFrame)."""
    from multiprocessing import Pool
    part = {s: "train" for s in split["train"]}
    part.update({s: "val" for s in split["val"]})
    anns: Dict[str, list] = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    by_stem: Dict[str, list] = defaultdict(list)
    labels: Dict[str, str] = {}
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        vid = str(im["id"])
        by_stem[stem].append((vid, anns.get(im["id"], [])))
        labels[vid] = str(Path(yolo_dir) / "labels" / part[stem] / f"{vid}.txt")
    jobs = [(v, {vid: labels[vid] for vid, _ in v}, resolutions) for _, v in sorted(by_stem.items())]
    with Pool(workers) as p:
        res = p.map(audit_stem, jobs, chunksize=4)
    df = pd.DataFrame([r for x in res for r in x["rows"]])
    agree = pd.DataFrame([r for x in res for r in x["agree"]])
    return dict(df=df, overlaps=sum(x["overlaps"] for x in res), agree=agree)


def iou_stats(x: pd.Series) -> dict:
    """Mean/median/p5/min and fraction below 0.5 and 0.8."""
    x = x.dropna()
    return dict(mean=x.mean(), median=x.median(), p5=x.quantile(0.05), min=x.min(),
                lt05=(x < 0.5).mean(), lt08=(x < 0.8).mean())
