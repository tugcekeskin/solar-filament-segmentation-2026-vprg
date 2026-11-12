#!/usr/bin/env bash
# Full honest pipeline eval of a trained run: flip3 TTA (id+h+v) -> fused -> crop refiner -> PQ.
# Usage: full_eval.sh <run> <imgsz> [image_root]   (image_root e.g. data/proc/flat3 for preprocessed models)
set -eu
cd "$(dirname "$0")/.."
run=$1; imgsz=$2; root=${3:-}
A=artifacts/fe/$run; mkdir -p $A
for f in "" h v; do
  [ -f $A/val_raw${f:+_$f}.pkl ] || .venv/bin/python scripts/predict.py --weights runs/$run/weights/best.pt --subset val --imgsz $imgsz \
      --batch ${EVAL_BATCH:-2} --soft ${f:+--flip $f} ${root:+--image-root $root} --out-cache $A/val_raw${f:+_$f}.pkl
done
.venv/bin/python scripts/quick_eval.py --cache $A/val_raw.pkl --name ${run}_single | grep RESULT
.venv/bin/python scripts/ensemble_cache.py --caches $A/val_raw.pkl $A/val_raw_h.pkl $A/val_raw_v.pkl --out $A/val_flip3.pkl --workers 8 >/dev/null
.venv/bin/python scripts/quick_eval.py --cache $A/val_flip3.pkl --name ${run}_flip3 | grep RESULT
.venv/bin/python scripts/refine_cache.py --cache $A/val_flip3.pkl --weights artifacts/refiner/refiner.pt --out $A/val_ref.pkl >/dev/null
bash scripts/eval_refined.sh $A/val_ref.pkl ${run}_ref
