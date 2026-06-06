#!/usr/bin/env bash
# Stock-RDED (KL-only) trustworthiness diagnostic sweep.
#
# Two stages:
#   geom  - teacher-only geometry (H1) + teacher reference. Fast (minutes total).
#           Writes logs/diagnostics.jsonl.
#   train - train the student on each (already-synthesized) distilled set and
#           measure ECE/OSCR/AUROC/FPR95/NC on it (H2). Slow (hours; 12 runs =
#           4 configs x 3 seeds, ipc 10). Writes the 'diag' column of results.jsonl.
#
# Distilled sets are synthesized automatically if missing (synthesis pre-pass),
# then training uses --skip-synth (no re-synthesis).
#
# Usage (activate the env first: conda activate rded):
#   bash scripts/diagnose.sh         # geom then train (default: all)
#   bash scripts/diagnose.sh geom    # geometry pass only (fast)
#   bash scripts/diagnose.sh train   # training pass only
#   RE_EPOCHS=1 bash scripts/diagnose.sh train   # quick smoke of the train path
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

MODE="${1:-all}"
EPOCHS="${RE_EPOCHS:-300}"
FACTOR=1; MIPC=300; CR=5
IPCS=(1 10 50)
SEEDS=(42 43 44)

# (subset arch) pairs — resnet18_modified + the conv teacher for each dataset.
CONFIGS=(
  "cifar100 resnet18_modified"
  "cifar100 conv3"
  "tinyimagenet resnet18_modified"
  "tinyimagenet conv4"
)

ensure_synth () {
  local subset=$1 arch=$2 ipc=$3
  local syn_path="./exp/${subset}_${arch}_f${FACTOR}_mipc${MIPC}_ipc${ipc}_cr${CR}/syn_data"
  if [[ ! -d "$syn_path" ]] || [[ -z "$(ls -A "$syn_path" 2>/dev/null)" ]]; then
    echo "### SYNTHESIZE  $subset $arch ipc$ipc (missing – generating now)"
    python main.py \
      --subset "$subset" --arch-name "$arch" --stud-name "$arch" \
      --factor $FACTOR --num-crop $CR --mipc $MIPC --ipc "$ipc" \
      --seed "${SEEDS[0]}" \
      --synth-only
  fi
}

run_geo () {
  local subset=$1 arch=$2 ipc=$3 seed=$4
  echo "### GEOMETRY  $subset $arch ipc$ipc seed$seed"
  python tools/diagnose_geometry.py \
    --subset "$subset" --arch-name "$arch" --ipc "$ipc" \
    --factor $FACTOR --mipc $MIPC --num-crop $CR --seed "$seed"
}

run_train () {
  local subset=$1 arch=$2 ipc=$3 seed=$4
  echo "### TRAIN+DIAG  $subset $arch ipc$ipc seed$seed  (epochs=$EPOCHS)"
  python main.py \
    --subset "$subset" --arch-name "$arch" --stud-name "$arch" \
    --factor $FACTOR --num-crop $CR --mipc $MIPC --ipc "$ipc" \
    --re-epochs "$EPOCHS" --seed "$seed" \
    --skip-synth --diagnostics --ood-dataset svhn
}

echo "==== synthesis pre-pass (skip if datasets exist) ===="
for ipc in "${IPCS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do ensure_synth $cfg "$ipc"; done
done

if [[ "$MODE" == "all" || "$MODE" == "geom" ]]; then
  echo "==== geometry pass (H1 + teacher reference) ===="
  # Geometry NC of a fixed on-disk distilled set is seed-invariant (loaders use
  # shuffle=False), so one seed suffices here; seeds are swept on the train pass.
  for ipc in "${IPCS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do run_geo $cfg "$ipc" "${SEEDS[0]}"; done
  done
fi

if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
  echo "==== training pass (H2: student-on-RDED) ===="
  for seed in "${SEEDS[@]}"; do
    for ipc in "${IPCS[@]}"; do
      for cfg in "${CONFIGS[@]}"; do run_train $cfg "$ipc" "$seed"; done
    done
  done
fi

echo "Done. Geometry -> logs/diagnostics.jsonl ; student diagnostics -> 'diag' in logs/results.jsonl"
