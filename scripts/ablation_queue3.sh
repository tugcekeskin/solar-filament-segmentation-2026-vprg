#!/usr/bin/env bash
# Waits for the current training (y11m_1536) to finish, then trains mask-ratio-2 at 1280.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "scripts/train.py" >/dev/null; do sleep 30; done
run() { local name=$1; shift; echo "=== $(date -u +%FT%TZ) start $name"; .venv/bin/python scripts/train.py --name "$name" "$@" > runs/logs/$name.log 2>&1; echo "=== $(date -u +%FT%TZ) end $name exit=$?"; }
run y11s_1280_mr2 --model weights/yolo11s-seg.pt --imgsz 1280 --mask-ratio 2
