"""Realistic ceiling: select candidates with q estimated from the OTHER annotators' views (perfect
knowledge of the consensus), score only on the held-out view. Compare with the baseline output on the
same held-out views. Multi-view val stems only."""
import sys, json
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")  # run from repo root
from decode_diag import iou, select, GT_JSON
from pycocotools import mask as mu
from filseg.gt import build_gt_df
from filseg.infer import load_cache
from filseg.metrics import fast_evaluate
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows, transform_stem
from filseg.rle import to_rle_obj
stems = sorted(json.load(open("data/splits/split_v1.json"))["val"])
gt = build_gt_df(GT_JSON, set(stems)); fid = gt["filament_id"].str.rsplit("_", n=1).str[0]
gt["view"] = fid; gt["stem"] = fid.str.split("-", n=1).str[1]
raw = load_cache("artifacts/rescore/val_blend01.pkl"); g, s = group_by_stem(raw), group_soft_by_stem(raw)
base_rows = postprocess_to_rows(g, PPParams(0.2, 100, 20, 0.3, 0, "square", 0.35), stems, s)
base_by = defaultdict(list)
for k, r in base_rows: base_by[k.rsplit("_", 1)[0]].append((k, r))
ho_gt, rows_or, rows_base, rows_mix = [], [], [], []
lam = 0.474
for st in stems:
    views = {v: list(x.segmentation_rle) for v, x in gt[gt.stem == st].groupby("view")}
    if len(views) < 2 or st not in g: continue
    sc, rl, ar = transform_stem(*g[st], s.get(st), 0, "square", 0.35, 0.0)
    keep = [i for i in range(len(rl)) if ar[i] >= 100]; sc = np.asarray(sc)[keep]; rl = [rl[i] for i in keep]
    P = [to_rle_obj(r) for r in rl]
    qv = {}
    for v, V in views.items():
        io = iou(P, [to_rle_obj(r) for r in V]).max(1); qv[v] = io * (io > 0.5)
    for ho in views:
        others = [v for v in views if v != ho]
        q = np.mean([qv[v] for v in others], 0)
        tag = "S" + ho.replace("-", "x")  # unique pseudo-stem per held-out view
        ho_gt.append(gt[gt.view == ho].assign(filament_id=lambda d: "a-" + tag + "_" + d.filament_id.str.rsplit("_", n=1).str[1]))
        rows_or += [(f"{tag}_{k.rsplit('_',1)[1]}", r) for k, r in select(st, rl, q - lam / 2, np.ones(len(q), bool))]
        rows_base += [(f"{tag}_{k.rsplit('_',1)[1]}", r) for k, r in base_by[st]]
        # mix: consensus knowledge only for candidates the model already scores >= 0.1 (realistic re-ranking)
        rows_mix += [(f"{tag}_{k.rsplit('_',1)[1]}", r) for k, r in select(st, rl, q - lam / 2, sc >= 0.1)]
G = pd.concat(ho_gt)
for name, rows in [("baseline", rows_base), ("consensus-oracle, all cands", rows_or), ("consensus-oracle, cands >= 0.1", rows_mix)]:
    r = fast_evaluate(G, pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]))
    print(f"{name}: PQ {r.pq:.4f} TP {r.tp} FP {r.fp} FN {r.fn}")
