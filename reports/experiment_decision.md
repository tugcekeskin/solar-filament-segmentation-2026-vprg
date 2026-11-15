# Experiment decision — geometry gates G0–G3 (2026-11-11)

- **Baseline:** commit `2632d6e`, then the G0/G1 scripts added on top of it. See `reports/experiment_manifest.md`.
- **Hardware:** one RTX 4080 SUPER (16 GB).
- **Validation:** fold 0 of `data/splits/split_v1.json`, grouped by date. 142 images, 220 views, 1545 GT.
- **Scorer:** `fast_evaluate`, the organizer's per-view metric with strict IoU > 0.5.

**Final decision: keep the baseline pipeline (validation 0.474, public 0.40).** Neither geometry change cleared its gate, so there is nothing to submit beyond the known-good file.

| Gate | Decision | Evidence |
|---|---|---|
| G0 audit | **GO**: denominators measured; continuation ceiling +0.024 | `reports/geometry_audit.md` |
| G1 path completion | **NO GO**: all 6 spec settings and 28 earlier settings are below baseline | `artifacts/g1/` |
| G2 Skeleton Recall | **NO GO (inconclusive)**: best +0.0027 vs reference and +0.0017 vs a plain-retrain control, below the +0.004 bar | `artifacts/g2/` |
| G3 second fold | **not run**: there is no candidate to validate | — |

## G1: path completion on frozen selected instances

**Commands**

```
.venv/bin/python scripts/g1_dijkstra.py --grid --overlays 20   # spec version
.venv/bin/python scripts/g1_pathcomp.py --grid                  # earlier greedy-cone version
```

**Spec version (`scripts/g1_dijkstra.py`)**

- **Where it runs:** after selection and overlap resolution.
- **Endpoints and tangent:** endpoints come from the morphological skeleton (the two farthest single-neighbour pixels). The outward tangent is taken from the last 15 px of the skeleton.
- **Path search:** Dijkstra inside a ROI of radius L ahead of the endpoint. The step cost combines darkness relative to the instance, angle to the tangent and length.
- **Forbidden pixels:** other instances (dilated by 4 px), pixels off the disc, and sunspot-like blobs (darkness more than 3× the instance's).
- **Trimming and painting:** the path is trimmed back to the last pixel with darkness ≥ α × the instance's, then painted at the instance's width on unclaimed pixels only.
- **Invariants:** the instance count and order are asserted unchanged, and only the image is used.

Baseline: PQ 0.4740, SQ 0.6908, RQ 0.6861, TP 1060 / FP 485 / FN 485.

| L (px) | α | PQ | ΔPQ | SQ | RQ | TP / FP / FN | changed instances | s/img |
|---|---|---|---|---|---|---|---|---|
| 32 | 0.8 | 0.4641 | −0.0098 | 0.6855 | 0.6770 | 1046 / 499 / 499 | 283 / 959 | 0.31 |
| 32 | 1.0 | 0.4710 | −0.0029 | 0.6885 | 0.6841 | 1057 / 488 / 488 | 159 / 959 | 0.29 |
| 64 | 0.8 | 0.4490 | −0.0249 | 0.6855 | 0.6550 | 1012 / 533 / 533 | 217 / 959 | 0.31 |
| 64 | 1.0 | 0.4653 | −0.0086 | 0.6886 | 0.6757 | 1044 / 501 / 501 | 127 / 959 | 0.30 |
| 96 | 0.8 | 0.4450 | −0.0290 | 0.6882 | 0.6466 | 999 / 546 / 546 | 184 / 959 | 0.33 |
| 96 | 1.0 | 0.4652 | −0.0088 | 0.6891 | 0.6751 | 1043 / 502 / 502 | 105 / 959 | 0.32 |

**Earlier greedy-cone version (`scripts/g1_pathcomp.py`)**

- It traces in steps of 3 px within a ±25° cone and can bridge gaps.
- All 26 grid settings and 2 conservative settings are below baseline. Results are worse with more extension and worse with gap bridging. The range is −0.0003 (α 1.4) to −0.094.

**Why it fails**

- The GT-guided ceiling is +0.024, and fragments miss a median 113 px of length.
- Even so, image darkness beyond an endpoint does not predict where the annotator ended the filament. Extensions that overshoot cost TPs in some views as often as completions gain them.
- The effect gets smaller as the method gets more conservative, but never turns positive.
- Before/after overlays for the first setting are in `artifacts/g1_dijkstra/overlay_*.png`: green is the original mask, red the added pixels.

## G2: Skeleton Recall loss in the U-Net refiner

**Commands**

```
artifacts/g2/run.sh          # λ 1.0 seeds 0, 1; control λ 0 seed 1
artifacts/g2/run_lambda.sh   # λ 0.1, 0.3 seed 0
```

**Setup**

- **Change:** `scripts/train_refiner.py --skel-w λ` adds 1 − soft recall of the tubed morphological skeleton of each target. Each target is that annotator view's own mask.
- **Unchanged:** data (`artifacts/refiner/crops.npz`), recipe, crop geometry, the YOLO detector, SAM-LoRA, the 50/50 blend and the fused input (`artifacts/ctx/val_flip3_orig.pkl`).

**Results** (all at the same decoding: conf 0.2, mask_thr 0.35; from `artifacts/g2/fixed_setting.log`)

| refiner | U-Net alone PQ | blend with original SAM: PQ | ΔPQ blend vs reference | TP / FP / FN (blend) |
|---|---|---|---|---|
| original U-Net (reference) | 0.4658 | 0.4743 | — | 1061 / 484 / 484 |
| control: plain retrain, seed 1 | 0.4709 | 0.4753 | +0.0010 | 1064 / 481 / 481 |
| λ 0.1, seed 0 | 0.4673 | **0.4770** | +0.0027 | 1068 / 476 / 477 |
| λ 0.3, seed 0 | 0.4705 | 0.4764 | +0.0021 | 1065 / 480 / 480 |
| λ 1.0, seed 0 | 0.4675 | 0.4746 | +0.0003 | 1062 / 485 / 483 |
| λ 1.0, seed 1 | 0.4646 | 0.4744 | +0.0001 | 1059 / 485 / 486 |

**Reading**

- On the U-Net alone, two retrains with the same recipe differ by 0.005 (0.4658 vs 0.4709).
- The Skeleton Recall variants fall inside that spread (0.4646–0.4705).
- The best blend gain, +0.0027, is below the pre-registered +0.004 bar. Its margin over the plain-retrain control is +0.0017.
- No systematic thickening was observed: SQ is unchanged (0.690–0.692).

## G3

Not run. Neither G1 nor G2 produced a candidate to validate on a second fold.

## Submission

- The known-good file is unchanged: `artifacts/y11m_1536_flip_blend/submission.csv`, public 0.40.
- `scripts/run_pipeline.py` reproduces it from the test images (PQ 0.9995 against the file, 1217 rows).
- No new public submission was made.
