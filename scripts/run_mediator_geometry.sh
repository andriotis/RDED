#!/usr/bin/env bash
# Teacher-only mediator sweep: distilled-set geometry (cov_gap_to_real, NC1) for every
# (cell, method, seed) on the CACHED distilled sets — no training, runs in minutes/run.
# Feeds tools/analyze_mediator.py, which joins this with the student Mahalanobis-AUROC.
#
# Config via env (defaults shown):
#   METHODS="stock random covmatch momentmatch"
#   CELLS="cifar100:conv3 cifar100:resnet18_modified tinyimagenet:conv4 tinyimagenet:resnet18_modified"
#   IPCS="10"   SEEDS="42 43 44"   OUT=logs/mediator_geometry.jsonl
set -uo pipefail
cd "$(dirname "$0")/.."

METHODS="${METHODS:-stock random covmatch momentmatch}"
CELLS="${CELLS:-cifar100:conv3 cifar100:resnet18_modified tinyimagenet:conv4 tinyimagenet:resnet18_modified}"
IPCS="${IPCS:-10}"
SEEDS="${SEEDS:-42 43 44}"
OUT="${OUT:-logs/mediator_geometry.jsonl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p logs/runs

for method in $METHODS; do
  for cell in $CELLS; do
    dataset="${cell%%:*}"; arch="${cell##*:}"
    for ipc in $IPCS; do
      for seed in $SEEDS; do
        leaf="syn_data_seed${seed}"
        tag=""; [[ "$method" != "stock" ]] && tag="_sel${method}"
        dir="./exp/${dataset}_${arch}_f1_mipc300_ipc${ipc}_cr5${tag}/${leaf}"
        if [[ ! -d "$dir" ]]; then echo "skip (no set): $dir"; continue; fi
        python tools/diagnose_geometry.py --subset "$dataset" --arch-name "$arch" --ipc "$ipc" \
          --seed "$seed" --syn-leaf "$leaf" --select-method "$method" --results-file "$OUT" \
          >> logs/runs/mediator_geom.log 2>&1 \
          && echo "ok:   $method $dataset/$arch ipc$ipc seed$seed" \
          || echo "FAIL: $method $dataset/$arch ipc$ipc seed$seed (see logs/runs/mediator_geom.log)"
      done
    done
  done
done
echo "Done -> $OUT"
