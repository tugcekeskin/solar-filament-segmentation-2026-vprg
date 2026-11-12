# Experiment manifest — geometry gates (2026-11-11)

Base commit: `2632d6e`. The G0 scripts, the G1 Dijkstra script and these reports were committed on top of it.

## Unchanged frozen inputs
- Detector `runs/y11m_1536` (YOLO11m-seg, 1536 px, 80 epochs, seed 0).
- U-Net refiner `artifacts/refiner/refiner.pt`.
- SAM-LoRA refiner `artifacts/refiner_sam/sam_refiner.pt`.
- Validation caches: `artifacts/rescore/val_blend01.pkl` (G0/G1), `artifacts/ctx/val_flip3_orig.pkl` + `artifacts/fe/sam_ref.pkl` (G2).

## New code
- `scripts/g0_audit.py`, `scripts/g0_window.py`, `scripts/g0_opportunity.py`: G0 audit (analysis only, uses GT).
- `scripts/g1_pathcomp.py`: greedy-cone path completion (dropped).
- `scripts/g1_dijkstra.py`: skeleton-endpoint Dijkstra path completion, the spec version (dropped).
- `scripts/train_refiner.py`: new `--skel-w` option (Skeleton Recall loss); the default of 0 leaves the old behaviour unchanged.

## New artefacts
- `artifacts/g0/`, `reports/`: G0 outputs.
- `artifacts/g1/`, `artifacts/g1_dijkstra/`: G1 grids and overlays.
- `artifacts/g2/`: 5 U-Net checkpoints, refined caches, logs, and `fixed_setting.log`.

Nothing was deleted or overwritten. Tests: `.venv/bin/python -m pytest -q` (43 passed).
