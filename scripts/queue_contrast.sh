#!/usr/bin/env bash
# Morning queue (2026-11-09): 1792 (running) -> 1536 flat3 contrast -> 1536 seed 1; full evals in background.
cd "$(dirname "$0")/.."
L=runs/logs/queue_contrast.log
log(){ echo "=== $(date -u +%FT%TZ) $*" >> $L; }
train(){ log "start $1"; .venv/bin/python scripts/train.py --model weights/yolo11m-seg.pt --batch 2 "${@:2}" --name $1 > runs/logs/$1.log 2>&1; log "end $1 exit=$?"; }
feval(){ ( bash scripts/full_eval.sh "$@" >> $L 2>runs/logs/fe_$1.err || log "EVALFAIL $1" ) ; }
while ps -eo args | grep -qE "[n]ame y11m_1792|[y]11m_1792/weights/last"; do sleep 60; done
log "y11m_1792 finished"
feval y11m_1792 1792 &
train y11m_1536_flat --imgsz 1536 --data data/yolo/v3_flat/data.yaml --seed 0
feval y11m_1536_flat 1536 data/proc/flat3 &
train y11m_1536_s1 --imgsz 1536 --seed 1
feval y11m_1536_s1 1536
wait
A=artifacts/fe
.venv/bin/python scripts/ensemble_cache.py --caches artifacts/y11m_1536/val_raw.pkl artifacts/y11m_1536_flip/val_raw_h.pkl artifacts/y11m_1536_flip/val_raw_v.pkl \
   $A/y11m_1536_s1/val_raw.pkl $A/y11m_1536_s1/val_raw_h.pkl $A/y11m_1536_s1/val_raw_v.pkl --out $A/seeds01_flip3.pkl --workers 8 >/dev/null \
 && .venv/bin/python scripts/refine_cache.py --cache $A/seeds01_flip3.pkl --weights artifacts/refiner/refiner.pt --out $A/seeds01_ref.pkl >/dev/null \
 && bash scripts/eval_refined.sh $A/seeds01_ref.pkl seeds01_ref >> $L || log "EVALFAIL seeds01"
log QDONE
