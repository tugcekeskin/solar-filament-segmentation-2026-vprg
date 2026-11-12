"""G0 add-on: for fragment near-misses (best IoU .3-.5), share of missing GT pixels outside the refiner window
(window approximated from the final prediction's bbox, pad 0.25, as at inference). Also: does the fragment's
GT have gaps (several connected components) -> path-completion target."""
import sys, json
from collections import defaultdict
import numpy as np, cv2
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from decode_diag import GT_JSON, iou
import g0_audit as A
from pycocotools import mask as mu
from filseg.gt import build_gt_df
from filseg.infer import load_cache
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows
from filseg.rle import decode_rle, to_rle_obj
from filseg.refine import crop_window, mask_bbox
stems = sorted(json.load(open("data/splits/split_v1.json"))["val"])
gt = build_gt_df(GT_JSON, set(stems)); fid = gt["filament_id"].str.rsplit("_", n=1).str[0]
views = defaultdict(lambda: defaultdict(list))
for v, r in zip(fid, gt["segmentation_rle"]): views[v.split("-", 1)[1]][v].append(to_rle_obj(r))
raw = load_cache("artifacts/rescore/val_blend01.pkl"); g, s = group_by_stem(raw), group_soft_by_stem(raw)
pred = defaultdict(list)
for k, r in postprocess_to_rows(g, PPParams(0.2, 100, 20, 0.3, 0, "square", 0.35), stems, s): pred[k.rsplit("_", 1)[0]].append(r)
inside = outside = n = 0; gt_cc = []; gap_frac = []
for st, vs in views.items():
    P = [to_rle_obj(r) for r in pred.get(st, [])]
    if not P: continue
    for v, Gs in vs.items():
        for G in Gs:
            io = iou([G], P)[0]; j = int(io.argmax())
            if not (0.3 < io[j] <= 0.5): continue
            cat, _ = A.classify(G, P, j, Gs)
            if cat != "fragment": continue
            Gm = mu.decode(G).astype(bool); Pm = decode_rle(pred[st][j]).astype(bool)
            wy, wx, side = crop_window(*mask_bbox(Pm), frac=0.25)
            W = np.zeros_like(Gm); W[wy:wy+side, wx:wx+side] = True
            miss = Gm & ~Pm
            inside += int((miss & W).sum()); outside += int((miss & ~W).sum()); n += 1
            gt_cc.append(cv2.connectedComponents(Gm.astype(np.uint8))[0] - 1)
            # pred components inside the GT: fragment made of gaps?
print(f"fragments n={n}: missing GT pixels inside refiner window {inside}, outside {outside} "
      f"-> outside share {outside/max(inside+outside,1):.0%}")
print("GT connected components per fragment-GT:", dict(zip(*np.unique(gt_cc, return_counts=True))))
