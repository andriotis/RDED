#!/usr/bin/env bash
# Phase B loss matrix: sweep loss settings on top of the baseline matrix
# (dataset x arch x ipc x seed). Each cell appends one row to
# logs/results.jsonl via the in-repo results_logger.
#
# Usage: bash scripts/run_loss_matrix.sh [options]
#   --seeds 42 43 44                            seeds to sweep
#   --ipcs 1 10 50                              IPCs to sweep
#   --datasets cifar100 tinyimagenet            datasets to sweep
#   --archs conv3 conv4 resnet18_modified       archs to sweep
#   --losses kl kl+ockl                         loss settings to sweep (default both)
#   --resume                                    skip cells already complete in logs/results.jsonl
#                                                 (default: rotate logs/results.jsonl to
#                                                  logs/results-YYYYMMDD[-N].jsonl for a fresh slate)
#   --dry-run                                   list cells without executing
#   -h, --help                                  show this help
#
# Fully unpaired protocol: each (loss, seed) gets a fresh synth (no --skip-synth).
# Matches the published RDED main.py methodology.
#
# Calls `python ./main.py` directly per cell. Incompatible (dataset, arch) pairs
# are silently skipped. set -e intentionally off so a single failing run doesn't
# abort the rest of the matrix.

set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=(42 43 44)
IPCS=(1 10 50)
DATASETS=(cifar100 tinyimagenet)
ARCHS=(conv3 conv4 resnet18_modified)
LOSSES=(kl kl+ockl)
DRY_RUN=0
RESUME=0

usage() {
  sed -n '2,21p' "$0"
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
    --losses)
      shift; LOSSES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do LOSSES+=("$1"); shift; done ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --resume)
      RESUME=1; shift ;;
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

# Validate loss tokens up-front.
for loss in "${LOSSES[@]}"; do
  case "$loss" in
    kl|kl+ockl) ;;
    *) echo "invalid loss token: $loss (expected: kl, kl+ockl)" >&2; exit 1 ;;
  esac
done

mkdir -p logs/loss_matrix

# Rotate-on-default / resume-on-flag for logs/results.jsonl.
# Default (no --resume): move any existing results.jsonl aside so this run
#   starts from a clean slate (the rotated file becomes results-YYYYMMDD.jsonl,
#   or results-YYYYMMDD-N.jsonl if that already exists).
# With --resume: preload the (dataset, arch, stud, ipc, seed, student_loss)
#   keys of completed rows and skip those cells during the loop. Rows with
#   best_top1 == null (incomplete/failed runs) are not treated as complete.
EXISTING_KEYS=""
if [[ $RESUME -eq 1 ]]; then
  if [[ ! -f logs/results.jsonl ]]; then
    echo "--resume given but logs/results.jsonl missing; running fresh." >&2
  else
    EXISTING_KEYS=$(mktemp)
    trap 'rm -f "$EXISTING_KEYS"' EXIT
    python - > "$EXISTING_KEYS" <<'PY'
import json
with open("logs/results.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("best_top1") is None:
            continue
        print(f"{r.get('dataset')}|{r.get('arch')}|{r.get('stud')}|{r.get('ipc')}|{r.get('seed')}|{r.get('student_loss')}")
PY
    n_keys=$(wc -l < "$EXISTING_KEYS")
    echo "Resume mode: $n_keys completed cells in results.jsonl will be skipped"
  fi
else
  if [[ -f logs/results.jsonl ]]; then
    stamp=$(date +%Y%m%d)
    archive="logs/results-${stamp}.jsonl"
    n=1
    while [[ -e "$archive" ]]; do
      archive="logs/results-${stamp}-${n}.jsonl"
      ((n++))
    done
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[dry-run] would rotate logs/results.jsonl -> $archive"
    else
      mv logs/results.jsonl "$archive"
      echo "Rotated logs/results.jsonl -> $archive"
    fi
  fi
fi

echo "Matrix: datasets=(${DATASETS[*]}) archs=(${ARCHS[*]}) ipcs=(${IPCS[*]}) seeds=(${SEEDS[*]}) losses=(${LOSSES[*]})"

for dataset in "${DATASETS[@]}"; do
  for arch in "${ARCHS[@]}"; do
    if ! is_compatible "$dataset" "$arch"; then
      continue
    fi
    for ipc in "${IPCS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for loss in "${LOSSES[@]}"; do
          if [[ -n "$EXISTING_KEYS" ]]; then
            key="${dataset}|${arch}|${arch}|${ipc}|${seed}|${loss}"
            if grep -Fxq "$key" "$EXISTING_KEYS"; then
              echo "[skip] dataset=$dataset arch=$arch ipc=$ipc seed=$seed loss=$loss (already in results.jsonl)"
              continue
            fi
          fi
          loss_tag="${loss//+/_}"
          log="logs/loss_matrix/${dataset}_${arch}_ipc${ipc}_seed${seed}_${loss_tag}.log"
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
            --student-loss "$loss"
          )
          if [[ $DRY_RUN -eq 1 ]]; then
            echo "[dry-run] python ./main.py ${py_args[*]} -> $log"
          else
            echo "[$(date +%H:%M:%S)] dataset=$dataset arch=$arch ipc=$ipc seed=$seed loss=$loss -> $log"
            python ./main.py "${py_args[@]}" 2>&1 | tee "$log"
          fi
        done
      done
    done
  done
done

echo "Loss matrix complete. Results appended to logs/results.jsonl"
