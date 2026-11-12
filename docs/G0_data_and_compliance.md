# G0 — Data access, structure and compliance (2026-11-08)

## Data (verified on this machine)
- Source: `kaggle competitions download -c filament-segmentation-2026` → `data/raw/MAGFiLO_1.0_Kaggle_2026/`
- Train: 707 JPEGs (2048×2048, 8-bit grayscale), 1 COCO JSON with **1154 annotator-views** (`images`) and **8199 annotations**.
  - Views per physical image: 411×1, 145×2, 151×3. 37 annotator batch prefixes (`image.id = "<6-digit annotator>-<stem>"`).
  - Every view has ≥1 annotation (1–26). One polygon per annotation, `iscrowd=0`. Extra fields: `spine`, `bbox`, `area`.
  - Categories (chirality): Left 2535, Right 2590, Unidentifiable 3074, Ambiguous 0. **Scoring ignores class** → train single-class.
- Test: 180 JPEGs, no annotations. 656 train dates / 176 test dates, 31 dates shared; years 2011–2022 in both (random split).
- Counts match the public participant report (707 / 1154 / 180).

## Scoring (organizer self-evaluation notebook, `ref/organizer_eval/`)
- GT rows `filament_id = "<annotator>-<stem>_<k>"`; prediction rows `"<stem>_<k>"`.
- For **each GT annotator-view**, all predictions of that stem are compared; pooled TP/FP/FN over all views.
  Predictions for stems with no GT view are never counted.
- TP = any pair with IoU > 0.5 (no Hungarian matching); FP = prediction column with no hit; FN = GT row with no hit.
- PQ = ΣIoU(TP) / (|TP| + ½|FP| + ½|FN|).
- Known quirk: overlapping duplicate predictions could each score TP. Organizers announced (Aug 20) they detect
  "hacking the system" and run extra verification. **Decision: never emit overlapping instances; resolve overlaps by
  mask-NMS / pixel ownership.** Do not exploit the metric.
- Organizer guidance: PQ > 0.35 is "of great value".

## Submission format
- CSV columns `filament_id,segmentation_rle`; id = `<stem>_<k>` with unique `k`; RLE counts only (string, no quotes),
  pycocotools compressed RLE of a 2048×2048 Fortran-ordered uint8 mask. 5 submissions/day; 2 final selections.
- Public LB ≈ 50% of test.

## Compliance
- External data/models allowed (rules §6; host Evaluation page). Inference only on provided test H-alpha images.
- "Models may not use any other ground-truth metadata for training" → **never use public MAGFiLO annotations for the
  180 test images** (or any MAGFiLO labels beyond the provided training JSON). COCO-pretrained YOLO weights are
  external models, not ground-truth metadata → permitted. Training-mask-derived boxes are fine.
- Licensing risk: Ultralytics is AGPL-3.0 (OSI-approved, does not prohibit commercial use) — acceptable but note in report.
- Final deliverables (Google form): 4-page report (Overleaf template), **public git repo** with `requirements.txt`,
  a notebook running the full pipeline, **model weights stored in repo**.

## Deadlines
- Kaggle API deadline: **2026-11-15 06:00 UTC** (= Nov 15 09:00 Istanbul). Treat this as the hard deadline for
  leaderboard submissions. Important-Dates page says Nov 15 23:59 CDT for reports/solutions — submit report by the
  Kaggle deadline anyway to be safe.

## Hardware
- RTX 4080 SUPER 16 GB, driver 615.71; venv torch 2.15.0+cu132, ultralytics 8.4.207.
