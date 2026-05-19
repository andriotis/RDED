#!/usr/bin/env bash
# Phase B.3 cross-architecture matrix: distill with one arch, train student
# with a different arch. Tests OCKL's hypothesized arch-invariance benefit
# (Simplex ETF is architecture-agnostic; cross-arch is RDED Table 4's cell type).
#
# Usage: bash scripts/run_cross_arch_matrix.sh [options]
#   --seeds 42 43 44                            seeds to sweep
#   --ipcs 1 10 50                              IPCs to sweep
#   --pairs cifar100:conv3:resnet18_modified .. asymmetric (dataset:arch:stud) cells
#   --losses kl kl+ockl                         loss settings to sweep
#   --dry-run                                   list cells without executing
#   -h, --help                                  show this help
#
# Each pair is "$dataset:$arch:$stud" — arch is the teacher used for both
# scoring patches during synthesis AND generating soft labels; stud is the
# student trained on the distilled data. arch != stud is enforced.
#
# Calls main.py directly per cell. set -e intentionally off so a single
# failing run doesn't abort the rest of the matrix.

set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=(42 43 44)
IPCS=(1 10 50)
PAIRS=(
  "cifar100:conv3:resnet18_modified"
  "cifar100:resnet18_modified:conv3"
  "tinyimagenet:conv4:resnet18_modified"
  "tinyimagenet:resnet18_modified:conv4"
)
LOSSES=(kl kl+ockl)
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
    --pairs)
      shift; PAIRS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do PAIRS+=("$1"); shift; done ;;
    --losses)
      shift; LOSSES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do LOSSES+=("$1"); shift; done ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage; exit 1 ;;
  esac
done

for loss in "${LOSSES[@]}"; do
  case "$loss" in
    kl|kl+ockl) ;;
    *) echo "invalid loss token: $loss (expected: kl, kl+ockl)" >&2; exit 1 ;;
  esac
done

for pair in "${PAIRS[@]}"; do
  IFS=":" read -r d a s <<< "$pair"
  if [[ -z "$d" || -z "$a" || -z "$s" ]]; then
    echo "malformed pair: '$pair' (expected dataset:arch:stud)" >&2; exit 1
  fi
  if [[ "$a" == "$s" ]]; then
    echo "pair '$pair' is matched-arch (arch == stud); use run_loss_matrix.sh for those." >&2
    exit 1
  fi
done

mkdir -p logs/cross_arch

echo "Cross-arch matrix: pairs=(${PAIRS[*]}) ipcs=(${IPCS[*]}) seeds=(${SEEDS[*]}) losses=(${LOSSES[*]})"

for pair in "${PAIRS[@]}"; do
  IFS=":" read -r dataset arch stud <<< "$pair"
  for ipc in "${IPCS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for loss in "${LOSSES[@]}"; do
        loss_tag="${loss//+/_}"
        log="logs/cross_arch/${dataset}_${arch}_to_${stud}_ipc${ipc}_seed${seed}_${loss_tag}.log"
        py_args=(
          --subset "$dataset"
          --arch-name "$arch"
          --stud-name "$stud"
          --factor 1
          --num-crop 5
          --mipc 300
          --ipc "$ipc"
          --re-epochs 300
          --seed "$seed"
          --student-loss "$loss"
        )
        if [[ $DRY_RUN -eq 1 ]]; then
          echo "[dry-run] python ./main.py ${py_args[*]} -> $log"
        else
          echo "[$(date +%H:%M:%S)] dataset=$dataset arch=$arch stud=$stud ipc=$ipc seed=$seed loss=$loss -> $log"
          python ./main.py "${py_args[@]}" 2>&1 | tee "$log"
        fi
      done
    done
  done
done

echo "Cross-arch matrix complete. Results appended to logs/results.jsonl"
