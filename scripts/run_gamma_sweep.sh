#!/usr/bin/env bash
# Phase B.2 gamma sweep: vary the OCKL weight on a single (dataset, arch, ipc)
# cell to probe sensitivity, per OCCE paper Section VI-C / Fig 11.
#
# Usage: bash scripts/run_gamma_sweep.sh [options]
#   --seeds 42 43 44                 seeds to sweep
#   --gammas 0.1 0.3 1.0 3.0         w_ockl values to sweep
#   --dataset cifar100               single dataset for the sweep
#   --arch resnet18_modified         single arch (used for both teacher and student)
#   --ipc 10                         single IPC value
#   --dry-run                        list cells without executing
#   -h, --help                       show this help
#
# Each cell calls `python ./main.py` with --student-loss kl+ockl and
# --w-kl 1.0 --w-ockl $gamma to override the preset weights. Results land
# in logs/results.jsonl with student_loss="kl+ockl" and w_ockl=$gamma; the
# analyzer pivots on (gamma, seed).
#
# Calls main.py directly per cell so that --seed is handled in exactly one
# place. set -e intentionally off so a single failing run doesn't abort the
# rest of the sweep.

set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=(42 43 44)
GAMMAS=(0.1 0.3 1.0 3.0)
DATASET=cifar100
ARCH=resnet18_modified
IPC=10
DRY_RUN=0

usage() {
  sed -n '2,21p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      shift; SEEDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --gammas)
      shift; GAMMAS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do GAMMAS+=("$1"); shift; done ;;
    --dataset)
      shift; DATASET="$1"; shift ;;
    --arch)
      shift; ARCH="$1"; shift ;;
    --ipc)
      shift; IPC="$1"; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage; exit 1 ;;
  esac
done

mkdir -p logs/gamma_sweep

echo "Gamma sweep: dataset=$DATASET arch=$ARCH ipc=$IPC gammas=(${GAMMAS[*]}) seeds=(${SEEDS[*]})"

for gamma in "${GAMMAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    gamma_tag="${gamma//./p}"
    log="logs/gamma_sweep/${DATASET}_${ARCH}_ipc${IPC}_seed${seed}_g${gamma_tag}.log"
    py_args=(
      --subset "$DATASET"
      --arch-name "$ARCH"
      --stud-name "$ARCH"
      --factor 1
      --num-crop 5
      --mipc 300
      --ipc "$IPC"
      --re-epochs 300
      --seed "$seed"
      --student-loss kl+ockl
      --w-kl 1.0
      --w-ockl "$gamma"
    )
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[dry-run] python ./main.py ${py_args[*]} -> $log"
    else
      echo "[$(date +%H:%M:%S)] gamma=$gamma seed=$seed -> $log"
      python ./main.py "${py_args[@]}" 2>&1 | tee "$log"
    fi
  done
done

echo "Gamma sweep complete. Results appended to logs/results.jsonl"
