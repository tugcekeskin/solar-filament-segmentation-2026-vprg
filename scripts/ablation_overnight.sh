#!/usr/bin/env bash
# Overnight single-factor ablations on the proxy YOLO11s-seg @1280, 80 epochs, batch 6 (= baseline AutoBatch).
# Each run: train -> (background) val predict with soft masks -> quick_eval (fixed small post-processing grid).
# Results: runs/ablation/results.jsonl ; logs runs/logs/abl_<name>.log
set -u
cd "$(dirname "$0")/.."
evalrun() {  # name
  local n=$1
  .venv/bin/python scripts/predict.py --weights runs/abl_$n/weights/best.pt --subset val --imgsz 1280 --batch 2 --soft \
      --out-cache artifacts/abl/$n/val_raw.pkl > runs/logs/abl_${n}_eval.log 2>&1 \
  && .venv/bin/python scripts/quick_eval.py --cache artifacts/abl/$n/val_raw.pkl --name $n >> runs/logs/abl_${n}_eval.log 2>&1
  grep -a RESULT runs/logs/abl_${n}_eval.log || echo "EVALFAIL $n"
}
run() {  # name [extra train args...]
  local n=$1; shift
  echo "=== $(date -u +%FT%TZ) start $n $*"
  .venv/bin/python scripts/train.py --model weights/yolo11s-seg.pt --imgsz 1280 --batch 6 --name abl_$n "$@" \
      > runs/logs/abl_$n.log 2>&1
  echo "=== $(date -u +%FT%TZ) end $n exit=$?"
  evalrun $n &
}
run base_s1   --seed 1
run tidy      --set hsv_v=0.12 scale=0.2 translate=0.02
run box10     --set box=10
run cp        --set copy_paste=0.2 copy_paste_mode=flip
run cons      --data data/yolo/v2_consensus/data.yaml
run nofliplr  --set fliplr=0.0
run deg0      --set degrees=0.0
run comboA    --set hsv_v=0.12 scale=0.2 translate=0.02 copy_paste=0.2 copy_paste_mode=flip
run comboB    --data data/yolo/v2_consensus/data.yaml --set hsv_v=0.12 scale=0.2 translate=0.02 copy_paste=0.2 copy_paste_mode=flip box=10
run e150      --epochs 150
wait
echo "=== $(date -u +%FT%TZ) ALLDONE"
