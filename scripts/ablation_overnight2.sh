#!/usr/bin/env bash
# Extra seeds (noise is ~±0.01 between seeds): runs after ablation_overnight.sh finishes.
set -u
cd "$(dirname "$0")/.."
until grep -q ALLDONE runs/logs/ablation_overnight.log 2>/dev/null; do sleep 60; done
source <(sed -n '/^evalrun()/,/^}/p;/^run()/,/^}/p' scripts/ablation_overnight.sh)
run base_s2   --seed 2
run comboA_s1 --seed 1 --set hsv_v=0.12 scale=0.2 translate=0.02 copy_paste=0.2 copy_paste_mode=flip
run comboB_s1 --seed 1 --data data/yolo/v2_consensus/data.yaml --set hsv_v=0.12 scale=0.2 translate=0.02 copy_paste=0.2 copy_paste_mode=flip box=10
wait
echo "=== $(date -u +%FT%TZ) ALLDONE2"
