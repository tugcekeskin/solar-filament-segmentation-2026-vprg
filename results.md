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

#### Neden Bu Değişikliği Yaptım
Plan'daki baseline'ı aynen uyguladım. Notlar: (a) Ultralytics yazdığı `runs/segment/runs/run1`. (b) Plan'ın hedefi ~0.35'ti; 0.387 aldık.

### Run 2 — Postprocess grid (conf x NMS x min_area x thinning), 64 configs
**Date:** 2026-11-12. Same weights as Run 1.
- Best: conf 0.4, NMS 0.5, thin 3x3 -> PQ 0.422 (SQ .662, RQ .637, TP 800, FP 375, FN 536). min_area 20 vs 100 identical.
- Higher conf cuts FP strongly (599->375) for a small TP loss; 3x3 thinning beats 5x5 on my val (friend's 5x5 gain was on his own setup/resolution).
- Full grid: `runs/segment/runs/run1/weights/best_grid.json` (not committed; reproduce with `python src/evaluate.py`).

#### Neden Bu Değişikliği Yaptım
Plan: Run 1 üzerinde erosion/conf/min_area grid'i. Run 1'in cache'lenmiş ham tahminleri üzerinde yapıldığı için yeniden eğitim gerekmedi. Not: 0.422 val üzerinde seçildi; Kaggle'a göndermeden önce Run 3 ile teyit edilecek.

---

## Submission dosyaları (`docs/`)
Her run için yarışma formatında (`filament_id,segmentation_rle`, COCO RLE, Fortran order, 2048x2048) test tahmini: `docs/submission_runX.csv`
(`python src/predict.py ...`; test verisi sadece inference için kullanılır, eğitim/tuning'de asla).
- `submission_run1.csv` — Run 1 plan cfg (conf .3, NMS .3, thin 5x5): 1386 filament, 0.34 MB
- `submission_run2.csv` — Run 2 en iyi cfg (conf .4, NMS .5, thin 3x3): 1167 filament, 0.32 MB
Sonraki run'lar için aynı şekilde eklenecek. Kaggle'a henüz gönderilmedi (günlük 5 hak; gönderim kararı kullanıcıda).

### Her run sonrası zorunlu akış
1. `python src/predict.py ... --out docs/submission_runX.csv` (sadece `test/test_images` üzerinde inference)
2. `python src/check_submission.py docs/submission_runX.csv` — format + test seti + RLE decode doğrulaması (OK vermeden commit yok)
3. results.md güncelle (Run Log satırı + Run Details + "Neden")
4. Tek commit: `git add . && git commit -m "results: run X tamamlandı" && git push` — results.md ve CSV birlikte.
