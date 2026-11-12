#!/usr/bin/env bash
# PQ grid for a rescored refined cache (conf is now the rescorer threshold). Prints the top 3.
set -eu
cd "$(dirname "$0")/.."
cache=$1; name=$2
args=()
for c in 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do for t in 0.3 0.35 0.4; do args+=("{\"conf\":$c,\"mask_thr\":$t}"); done; done
.venv/bin/python scripts/eval_params.py $cache artifacts/y11m_1536_flip_ref/calib.json "${args[@]}" | sort -t= -k2 -r | head -3 | sed "s/^/RESULT $name /"
