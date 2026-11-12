"""Panoptic Quality (PQ/SQ/RQ) as in the challenge: IoU>0.5 matching, PQ = sum IoU / (TP + .5FP + .5FN).
Each annotator's label set for an image is scored separately (multi-annotator GT) and sums are pooled."""
import numpy as np


def match(preds, gts, thr=0.5):
    """Greedy one-to-one matching (IoU>0.5 is already unique for non-overlapping masks)."""
    if not preds or not gts:
        return [], len(preds), len(gts)
    P = np.stack([p.reshape(-1).astype(bool) for p in preds]).astype(np.float32)
    G = np.stack([g.reshape(-1).astype(bool) for g in gts]).astype(np.float32)
    inter = P @ G.T
    union = P.sum(1)[:, None] + G.sum(1)[None] - inter
    I = inter / np.maximum(union, 1)
    ious = []
    used_p, used_g = set(), set()
    for pi, gi in sorted(zip(*np.where(I > thr)), key=lambda x: -I[x]):
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); ious.append(float(I[pi, gi]))
    return ious, len(preds) - len(used_p), len(gts) - len(used_g)


def evaluate(preds_by_file, gt_by_file):
    sum_iou = tp = fp = fn = 0
    for f, annotators in gt_by_file.items():
        preds = preds_by_file.get(f, [])
        for gts in annotators:
            ious, p, g = match(preds, gts)
            sum_iou += sum(ious); tp += len(ious); fp += p; fn += g
    sq = sum_iou / max(tp, 1)
    rq = tp / max(tp + .5 * fp + .5 * fn, 1e-9)
    return dict(PQ=sq * rq, SQ=sq, RQ=rq, TP=tp, FP=fp, FN=fn)
