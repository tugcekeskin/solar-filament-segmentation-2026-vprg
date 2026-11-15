#!/usr/bin/env bash
# Val inference + PQ calibration for a finished training run. Usage: eval_run.sh <run_name> <imgsz> [--tta]
set -eu
cd "$(dirname "$0")/.."
run=$1; imgsz=$2; tta=${3:-}
tag=$run${tta:+_tta}
mkdir -p artifacts/$tag
.venv/bin/python scripts/predict.py --weights runs/$run/weights/best.pt --subset val --imgsz $imgsz --batch ${EVAL_BATCH:-2} --soft $tta --out-cache artifacts/$tag/val_raw.pkl
.venv/bin/python scripts/calibrate.py --cache artifacts/$tag/val_raw.pkl --out-dir artifacts/$tag --workers 8 | tail -3
python3 -c "import json;d=json.load(open('artifacts/$tag/calib.json'));b=d['best_metrics'];print('RESULT $tag PQ=%.4f SQ=%.4f RQ=%.4f TP=%d FP=%d FN=%d params=%s'%(b['pq'],b['sq'],b['rq'],b['tp'],b['fp'],b['fn'],d['best_params']))"
