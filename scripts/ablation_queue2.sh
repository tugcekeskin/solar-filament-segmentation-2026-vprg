#!/usr/bin/env bash
# Remaining ablations; waits for any val/test inference on the GPU to finish first.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "scripts/predict.py" >/dev/null; do sleep 20; done
run() { local name=$1; shift; echo "=== $(date -u +%FT%TZ) start $name"; .venv/bin/python scripts/train.py --name "$name" "$@" > runs/logs/$name.log 2>&1; echo "=== $(date -u +%FT%TZ) end $name exit=$?"; }
run y11s_1024_mr2 --model weights/yolo11s-seg.pt --imgsz 1024 --mask-ratio 2
run y11s_1280 --model weights/yolo11s-seg.pt --imgsz 1280
