#!/usr/bin/env bash
# Sequential ablation queue (matched training settings except the ablated factor).
set -u
cd "$(dirname "$0")/.."
run() { local name=$1; shift; echo "=== $(date -u +%FT%TZ) start $name"; .venv/bin/python scripts/train.py --name "$name" "$@" > runs/logs/$name.log 2>&1; echo "=== $(date -u +%FT%TZ) end $name exit=$?"; }
run y26s_1024 --model weights/yolo26s-seg.pt --imgsz 1024
run y11s_1536 --model weights/yolo11s-seg.pt --imgsz 1536
run y11s_1024_mr2 --model weights/yolo11s-seg.pt --imgsz 1024 --mask-ratio 2
run y11s_1280 --model weights/yolo11s-seg.pt --imgsz 1280
