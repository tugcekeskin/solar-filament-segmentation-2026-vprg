# G0 — Geometry and metric audit (fold 0)

Status: **done**. All numbers come from the commands below, run against the frozen baseline output.
Machine-readable versions: `reports/geometry_audit.json`, `reports/per_view.csv`, `reports/near_misses.csv`.

## Inputs

- **Frozen baseline cache:** `artifacts/rescore/val_blend01.pkl`. This is the YOLO11m-seg 1536 detector with flip3 TTA, refined by the U-Net and SAM-LoRA refiners and blended 50/50.
- **Decoding:** conf 0.2, mask_thr 0.35, min area 100, NMS 0.3, max_det 20, then overlap resolution.
- **Split:** `data/splits/split_v1.json`, fold 0, grouped by date. The validation set has 142 images, 220 annotator views and 133 dates. Instruments (images per instrument): Bh 25, Ch 24, Lh 23, Mh 25, Th 26, Uh 19.
- **Scorer:** `src/filseg/metrics.py:fast_evaluate`, vendored from the organizer notebook `ref/organizer_eval/self-evaluation-notebook.ipynb`. It scores each view separately, counts a hit when IoU > 0.5 (strict), does no one-to-one matching, and pools across views as PQ = ΣIoU_TP / (TP + FP/2 + FN/2). It is covered by the repo's unit tests (`pytest`, 43 passed).

## Commands

```
.venv/bin/python scripts/g0_audit.py         # baseline, FN bins, taxonomy, oracle repairs, GT overlap -> artifacts/g0/audit.{log,json}
.venv/bin/python scripts/g0_window.py        # missing fragment pixels inside/outside the refiner window -> artifacts/g0/window.log
.venv/bin/python scripts/loao_oracle.py      # cross-annotator oracle (existing script) -> artifacts/g0/loao.log
.venv/bin/python scripts/g0_opportunity.py   # per-view table, near-miss CSV, multiplicity, continuation/union ceilings -> reports/
```

## Baseline reproduction

| | PQ | TP | FP | FN | N = ΣIoU_TP | D |
|---|---|---|---|---|---|---|
| scorer | **0.4740** | 1060 | 485 | 485 | 732.27 | 1545 |
| per-view recomputation | 0.4740 | | | | 732.27 | 1545 |

- The recorded reference is 0.4743, from `artifacts/y11m_1536_flip_blend/val_raw.pkl` with the U-Net cache built separately. The 0.0003 difference comes from the two blend caches.
- Match multiplicity:
  - One prediction has IoU > 0.5 with two GT instances of the same view.
  - No GT instance is hit by more than one prediction.

## FN denominator

GT_all = 1545 and FN_all = 485. Each FN falls into one bin by its best IoU with any selected prediction:

| best IoU | count |
|---|---|
| 0 (no overlap) | 240 |
| (0, 0.05] | 4 |
| (0.05, 0.30) | 76 |
| **[0.30, 0.50] (near miss, M)** | **165** |

- M / FN_all = 0.340.
- M / GT_all = 0.107.
- F (fragments) = 65, so F / M = 0.394 and F / GT_all = 0.042.

## Near-miss taxonomy

The categories are exclusive and assigned in priority order. The rules are in `scripts/g0_audit.py:classify`, measured with IoU > 0.05 contacts:

1. **Merge:** the prediction touches two or more GT instances of the view.
2. **Split:** the GT touches two or more predictions.
3. **Too big:** GT coverage ≥ 0.75 and precision < 0.6.
4. **Fragment:** precision ≥ 0.75 and coverage < 0.6.
5. **Shifted:** everything else.

| category | [0.30, 0.50] | (0.05, 0.50] |
|---|---|---|
| fragment | 65 (39%) | 83 (34%) |
| shifted | 31 (19%) | 37 (15%) |
| too big | 26 (16%) | 27 (11%) |
| merge | 24 (15%) | 61 (25%) |
| split | 19 (12%) | 33 (14%) |
| total | 165 | 241 |

The earlier team figures (39/19/16/15/12 %) are therefore reproduced exactly, with denominator n = 165. They sum to 101% only because of rounding; the categories are exclusive.

## Fragment shape (the 65 fragment near misses)

- Missing length dominates; width is close to correct:
  - Median skeleton length, prediction / GT: **0.54**.
  - Median width, prediction / GT: **0.90**.
  - Median GT skeleton length is 289 px; median missing length is 113 px.
- 67% of the missing GT pixels lie more than 5 px from the prediction (along the length), 33% within 5 px (width).
- 76% of the missing GT pixels lie **inside** the refiner's inference window (bbox + 25% pad); 24% lie outside it.
- Every fragment GT is a single connected polygon. The gaps are in the image, not in the labels.

## GT overlap inside one view

Only 2 overlapping GT pairs, in 2 of 220 views. They cover 0.15% of GT pixels.

## Oracle ceilings

These are offline analyses that use GT; none of them is an inference method. Each one edits frozen copies of the predictions and re-scores every view of the edited image with the exact scorer, so a repair can break a TP in another view.

| oracle edit | PQ | ΔPQ | TP / FP / FN |
|---|---|---|---|
| baseline | 0.4740 | | 1060 / 485 / 485 |
| fragment: prediction ∪ GT, near misses [0.3, 0.5] | 0.4996 | +0.0256 | 1097 / 448 / 448 |
| fragment: prediction ∪ GT, (0.05, 0.5] | 0.5064 | +0.0324 | 1108 / 437 / 437 |
| **(i) bounded continuation: geodesic growth inside GT, 32 px** | 0.4978 | **+0.0239** | 1104 / 441 / 441 |
| (i) same, 64 px | 0.4979 | +0.0239 | 1097 / 448 / 448 |
| (i) same, 96 px | 0.4987 | +0.0247 | 1097 / 448 / 448 |
| **(ii) union with pre-NMS candidates lying inside the GT (precision ≥ 0.75)** | 0.4846 | **+0.0106** | 1089 / 453 / 456 |
| split: union of touching predictions | 0.4775 | +0.0036 | 1059 / 455 / 486 |
| too big: prediction ∩ dilated GT | 0.4816 | +0.0076 | 1077 / 468 / 468 |
| perfect geometry on every near miss [0.3, 0.5] | 0.5424 | +0.0685 | 1139 / 376 / 406 |

- **G0 exit:** the path-completion ceiling is +0.024, above the +0.008 screening bar, so **G1 GO**.
- 76% of the missing pixels are inside the refiner crop, so **G2 GO**.

## Cross-annotator oracle

Re-run with the existing `scripts/loao_oracle.py` on the multi-view validation images, using the same cache and decoding. The numbers match the earlier team report.

| | PQ | TP / FP / FN |
|---|---|---|
| our model on held-out views | 0.4793 | 631 / 307 / 242 |
| other-annotator oracle, all candidates | 0.4524 | 598 / 339 / 275 |
| other-annotator oracle, candidates ≥ 0.1 | 0.4709 | 594 / 261 / 279 |

The same-view oracle (perfect selection) gives 0.5420 when it may only drop FPs and 0.6027 when it may choose freely among all candidates (`artifacts/rescore/decode_diag.log`).

## Refiner facts that matter for G2

- **Loss:** BCE-with-logits plus soft Dice (smoothing 1), unweighted. It is shared by the U-Net and SAM (`scripts/train_refiner.py:loss_fn`). There is no boundary term.
- **Sampling:** one sample per (train-split proposal, annotator view) pair with IoU ≥ 0.2. The target is that single view's mask, not a consensus (`src/filseg/refine.py:_stem_samples`). In addition, 30% of the samples are synthetic degraded-GT priors.
- **Data and checkpoint:**
  - crops: `artifacts/refiner/crops.npz`;
  - checkpoint: `artifacts/refiner/refiner.pt`, 25 epochs, batch 32, AdamW lr 3e-4, cosine schedule;
  - holdout: 10% of train images.
- **The 0.551 → 0.657 crop-IoU figure** is logged in `artifacts/refiner/train.log`. The same holdout was used to pick the checkpoint, so the figure is optimistic.
