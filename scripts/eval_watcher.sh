#!/usr/bin/env bash
# Evaluate each run as the ablation queue finishes it.
cd "$(dirname "$0")/.."
declare -A sz=([y26s_1024]=1024 [y11s_1536]=1536 [y11s_1024_mr2]=1024 [y11s_1280]=1280)
done_runs=""
while true; do
  for r in "${!sz[@]}"; do
    if grep -q "end $r " runs/logs/queue.log 2>/dev/null && [[ " $done_runs " != *" $r "* ]]; then
      done_runs="$done_runs $r"
      scripts/eval_run.sh $r ${sz[$r]} > runs/logs/eval_$r.log 2>&1; grep RESULT runs/logs/eval_$r.log || echo "EVAL FAILED $r"
    fi
  done
  [ $(echo $done_runs | wc -w) -eq ${#sz[@]} ] && break
  sleep 30
done
