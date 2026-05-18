#!/usr/bin/env bash
# Phase B.0 baseline matrix: stock RDED over a configurable slice of
# {dataset} x {arch} x {ipc} x {seed}. Each cell appends one row to
# logs/results.jsonl via the in-repo results_logger.
#
# Usage: bash scripts/run_baseline_matrix.sh [options]
#   --seeds 42 43 44                            seeds to sweep
#   --ipcs 1 10 50                              IPCs to sweep
#   --datasets cifar100 tinyimagenet            datasets to sweep
#   --archs conv3 conv4 resnet18_modified       archs to sweep
#   --protocol {unpaired,paired}                seed-vs-synth pairing (default unpaired)
#                                                 unpaired: fresh synth per (cell, seed)
#                                                 paired: synth once per cell; later seeds reuse it
#   --dry-run                                   list cells without executing
#   -h, --help                                  show this help
#
# Calls `python ./main.py` directly per cell (does not dispatch through per-config
# scripts) so that --seed is handled in exactly one place. Incompatible
# (dataset, arch) pairs are silently skipped. set -e is intentionally off so a
# single failing run doesn't abort the rest of the matrix.

set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=(42 43 44)
IPCS=(1 10 50)
DATASETS=(cifar100 tinyimagenet)
ARCHS=(conv3 conv4 resnet18_modified)
PROTOCOL=unpaired
DRY_RUN=0

usage() {
  sed -n '2,19p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      shift; SEEDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --ipcs)
      shift; IPCS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do IPCS+=("$1"); shift; done ;;
    --datasets)
      shift; DATASETS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
    --archs)
      shift; ARCHS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do ARCHS+=("$1"); shift; done ;;
    --protocol)
      shift
      PROTOCOL="$1"
      shift
      case "$PROTOCOL" in
        unpaired|paired) ;;
        *) echo "invalid --protocol: $PROTOCOL (expected unpaired|paired)" >&2; exit 1 ;;
      esac ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage; exit 1 ;;
  esac
done

is_compatible() {
  case "$1-$2" in
    cifar100-conv3)             return 0 ;;
    cifar100-resnet18_modified) return 0 ;;
    tinyimagenet-conv4)             return 0 ;;
    tinyimagenet-resnet18_modified) return 0 ;;
    *)                              return 1 ;;
  esac
}

mkdir -p logs/baseline

echo "Matrix: protocol=$PROTOCOL datasets=(${DATASETS[*]}) archs=(${ARCHS[*]}) ipcs=(${IPCS[*]}) seeds=(${SEEDS[*]})"

for dataset in "${DATASETS[@]}"; do
  for arch in "${ARCHS[@]}"; do
    if ! is_compatible "$dataset" "$arch"; then
      continue
    fi
    for ipc in "${IPCS[@]}"; do
      cell_first_seed_done=0
      for seed in "${SEEDS[@]}"; do
        log="logs/baseline/${dataset}_${arch}_ipc${ipc}_seed${seed}.log"
        py_args=(
          --subset "$dataset"
          --arch-name "$arch"
          --stud-name "$arch"
          --factor 1
          --num-crop 5
          --mipc 300
          --ipc "$ipc"
          --re-epochs 300
          --seed "$seed"
        )
        if [[ "$PROTOCOL" == "paired" && $cell_first_seed_done -eq 1 ]]; then
          py_args+=(--skip-synth)
        fi
        if [[ $DRY_RUN -eq 1 ]]; then
          echo "[dry-run] python ./main.py ${py_args[*]} -> $log"
        else
          echo "[$(date +%H:%M:%S)] dataset=$dataset arch=$arch ipc=$ipc seed=$seed -> $log"
          python ./main.py "${py_args[@]}" 2>&1 | tee "$log"
        fi
        cell_first_seed_done=1
      done
    done
  done
done

echo "Baseline matrix complete. Results appended to logs/results.jsonl"
