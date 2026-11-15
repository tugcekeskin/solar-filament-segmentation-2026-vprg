# Experiment Results

**Validation protocol (my approximation of the organizer metric — the official self-evaluation notebook could not be fetched):**
85/15 split by unique image, stratified by observatory (seed 42; 601 train / 106 val images, `data_yolo/split.json`).
Training labels: all annotators merged per image (instance kept only if IoU<=0.3 with an already-kept one), single class.
Val GT: each annotator's label set scored separately and pooled. Matching IoU>0.5, PQ = SQ x RQ,
RQ = TP/(TP+.5FP+.5FN). Test data is never used for training or tuning.
Caveat: postprocess params are tuned on this same val set, so the best-config numbers are slightly optimistic.

## Run Log

| # | Date | Model | Config | Val PQ | Val SQ | Val RQ | TP | FP | FN | Notes |
|---|------|-------|--------|--------|--------|--------|----|----|----|-------|
| 1 | 2026-11-12 | YOLO11s-seg | 1024px, 80ep, plan cfg (conf .3, NMS .3, thin 5x5) | 0.387 | 0.651 | 0.594 | 818 | 599 | 518 | baseline as specified |
| 2 | 2026-11-12 | YOLO11s-seg (Run 1 weights) | conf .4, NMS .5, min_area 20, thin 3x3 | 0.422 | 0.662 | 0.637 | 800 | 375 | 536 | postprocess grid, no retraining |
| 3 | 2026-11-12 | YOLO11m-seg | 1280px, 80ep, conf .5, NMS .5, min_area 20, thin 3x3 | 0.424 | 0.676 | 0.628 | 726 | 252 | 610 | small grid (6 cfg); gain over Run 2 only +0.002 |

---

## Run Details

### Run 1 — YOLO11s-seg baseline
**Date:** 2026-11-12

#### Preprocessing
- Raw grayscale->3ch JPEG, no CLAHE (friend found no gain); Ultralytics resizes 2048->1024.

#### Model
- Architecture: YOLO11s-seg; Pretrained: COCO (Ultralytics); imgsz 1024; Epochs 80; Batch 4; LR/optimizer: Ultralytics defaults (auto)

#### Augmentation
- Ultralytics defaults (mosaic, flips, HSV, scale).

#### Postprocessing
- Conf 0.30, mask NMS IoU 0.3, min area 20, thinning 5x5 cross erosion, no TTA

#### Results
- Val PQ 0.387 | SQ 0.651 | RQ 0.594 | TP 818 FP 599 FN 518
- Train time: ~9 min | Inference: 0.10 s/img (model + mask transfer, excludes postproc) | GPU memory: 0.51 GB (inference)

#### Why I made this change
The baseline from the plan was implemented as-is. Notes: (a) Ultralytics writes to `runs/segment/runs/run1`. (b) The plan's target was ~0.35; we got 0.387.

### Run 2 — Postprocess grid (conf x NMS x min_area x thinning), 64 configs
**Date:** 2026-11-12. Same weights as Run 1.
- Best: conf 0.4, NMS 0.5, thin 3x3 -> PQ 0.422 (SQ .662, RQ .637, TP 800, FP 375, FN 536). min_area 20 vs 100 identical.
- Higher conf cuts FP strongly (599->375) for a small TP loss; 3x3 thinning beats 5x5 on my val (friend's 5x5 gain was on his own setup/resolution).
- Full grid: `runs/segment/runs/run1/weights/best_grid.json` (not committed; reproduce with `python src/evaluate.py`).

#### Why I made this change
Plan: erosion/conf/min_area grid on top of Run 1. Since it was run on Run 1's cached raw predictions, no retraining was needed. Note: 0.422 was selected on val; it will be confirmed with Run 3 before submitting to Kaggle.

### Run 3 — YOLO11m-seg, 1280px
**Date:** 2026-11-12

#### Preprocessing
- Raw grayscale->3ch, no CLAHE; Ultralytics resizes 2048->1280.

#### Model
- YOLO11m-seg, COCO pretrained, imgsz 1280, 80 epochs, batch 4, Ultralytics default LR/optimizer, same train/val split.

#### Postprocessing
- Conf 0.5, mask NMS IoU 0.5, min area 20, thinning 3x3 cross, no TTA.

#### Results
- Val PQ 0.424 | SQ 0.676 | RQ 0.628 | TP 726 FP 252 FN 610
- Train time: ~23.5 min | Inference time was not recorded for this run (evaluate output was cut off by an error) | GPU: ~10 GB for training
- Other grid points: conf .4/5x5 -> 0.402; conf .3/5x5 -> 0.395; conf .5/5x5 -> 0.393
- Test CSV: `docs/submission_run3.csv` (953 filaments, predictions on 175/180 test images, check_submission OK)

#### Why I made this change
Plan: bigger model + higher resolution (friend's finding: 11m > 11s, 1536 > 1280 > 1024). I shrank the grid (conf .3/.4/.5, NMS .5, area 20, thin 3/5) because min_area and NMS made no difference in Run 2 and the full grid takes ~15 min.
**Honest comment:** The gain over Run 2 is +0.002 PQ, i.e. below the <0.005 threshold and within noise on 106 val images. The best conf 0.5 is at the edge of the grid; higher conf was not tried. 11m raised SQ a bit (.662->.676) but also discarded more predictions (RQ similar). The ~0.42 target in the plan was nominally reached, but this cannot be attributed to the model difference.

---

## Submission files (`docs/`)
For each run, a test prediction in the competition format (`filament_id,segmentation_rle`, COCO RLE, Fortran order, 2048x2048): `docs/submission_runX.csv`
(`python src/predict.py ...`; test data is used only for inference, never for training/tuning).
- `submission_run1.csv` — Run 1 plan cfg (conf .3, NMS .3, thin 5x5): 1386 filaments, 0.34 MB
- `submission_run2.csv` — Run 2 best cfg (conf .4, NMS .5, thin 3x3): 1167 filaments, 0.32 MB
- `submission_run3.csv` — Run 3 best cfg (conf .5, NMS .5, thin 3x3): 953 filaments, 0.26 MB
To be added the same way for later runs. Not yet submitted to Kaggle (5 submissions per day; the decision to submit is up to the user).

### Required steps after each run
1. `python src/predict.py ... --out docs/submission_runX.csv` (inference on `test/test_images` only)
2. `python src/check_submission.py docs/submission_runX.csv` — verifies format + test set + RLE decoding (no commit until it reports OK)
3. Update results.md (Run Log row + Run Details + "Why I made this change")
4. Single commit: `git add . && git commit -m "results: run X completed" && git push` — results.md and the CSV together.
