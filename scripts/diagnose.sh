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
# GPU use: every run is pinned to one GPU (CUDA_VISIBLE_DEVICES) and runs are
# dispatched across all GPUs concurrently — so no run fans a tiny batch across
# every card via DataParallel, and the slow train pass packs the GPUs. Each
# phase (synth / geom / train) drains before the next starts. Knobs (env):
#   GPUS=0,1,2   GPU id list (default: all from nvidia-smi, else CUDA_VISIBLE_DEVICES)
#   PER_GPU=1    concurrent runs per GPU (default 1; 2 is safe for the lighter
#                configs, but ipc50 resnet18 peaks ~3.7 GB so keep 1-2)
#
# Usage (activate the env first: conda activate rded):
#   bash scripts/diagnose.sh         # geom then train (default: all)
#   bash scripts/diagnose.sh geom    # geometry pass only (fast)
#   bash scripts/diagnose.sh train   # training pass only
#   RE_EPOCHS=1 bash scripts/diagnose.sh train   # quick smoke of the train path
#   PER_GPU=2 GPUS=1,2 bash scripts/diagnose.sh  # 2 runs each on GPUs 1 and 2
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

MODE="${1:-all}"
EPOCHS="${RE_EPOCHS:-300}"
FACTOR=1; MIPC=300; CR=5
IPCS=(1 10 50)
SEEDS=(42 43 44)
PER_GPU="${PER_GPU:-1}"

# (subset arch) pairs — resnet18_modified + the conv teacher for each dataset.
CONFIGS=(
  "cifar100 resnet18_modified"
  "cifar100 conv3"
  "tinyimagenet resnet18_modified"
  "tinyimagenet conv4"
)

# --- GPU pool -------------------------------------------------------------
# A FIFO holds one token per slot (PER_GPU tokens per GPU). run_pooled() blocks
# until a token is free, launches its command pinned to that GPU in the
# background, and returns the token when the command exits — a portable bounded
# job pool. A per-phase `wait` drains all background jobs before the next phase.
if [[ -n "${GPUS:-}" ]]; then
  IFS=',' read -ra GPU_LIST <<< "$GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null || true)
fi
[[ ${#GPU_LIST[@]} -eq 0 ]] && GPU_LIST=(0)
SLOTS=$(( ${#GPU_LIST[@]} * PER_GPU ))

FAILLOG="$(mktemp)"
POOL_FIFO="$(mktemp -u)"; mkfifo "$POOL_FIFO"
exec 3<>"$POOL_FIFO"; rm -f "$POOL_FIFO"
for g in "${GPU_LIST[@]}"; do
  for ((i = 0; i < PER_GPU; i++)); do printf '%s\n' "$g" >&3; done
done
cleanup () { exec 3>&- 2>/dev/null || true; rm -f "$FAILLOG" 2>/dev/null || true; }
trap cleanup EXIT

run_pooled () {   # run "$@" pinned to a free GPU, in the background
  local gpu
  read -r -u 3 gpu
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    if "$@"; then :; else printf 'FAIL gpu=%s: %s\n' "$gpu" "$*" >> "$FAILLOG"; fi
    printf '%s\n' "$gpu" >&3   # release the slot
  ) &
}

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

echo "GPU pool: gpus=[${GPU_LIST[*]}] per-gpu=$PER_GPU concurrent-slots=$SLOTS"

echo "==== synthesis pre-pass (skip if datasets exist) ===="
for ipc in "${IPCS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do run_pooled ensure_synth $cfg "$ipc"; done
done
wait   # all distilled sets exist before any read

if [[ "$MODE" == "all" || "$MODE" == "geom" ]]; then
  echo "==== geometry pass (H1 + teacher reference) ===="
  # Geometry NC of a fixed on-disk distilled set is seed-invariant (loaders use
  # shuffle=False), so one seed suffices here; seeds are swept on the train pass.
  for ipc in "${IPCS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do run_pooled run_geo $cfg "$ipc" "${SEEDS[0]}"; done
  done
  wait
fi

if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
  echo "==== training pass (H2: student-on-RDED) ===="
  for seed in "${SEEDS[@]}"; do
    for ipc in "${IPCS[@]}"; do
      for cfg in "${CONFIGS[@]}"; do run_pooled run_train $cfg "$ipc" "$seed"; done
    done
  done
  wait
fi

echo "Done. Geometry -> logs/diagnostics.jsonl ; student diagnostics -> 'diag' in logs/results.jsonl"
nfail=$(wc -l < "$FAILLOG")
if [[ "$nfail" -gt 0 ]]; then
  echo "WARNING: $nfail run(s) failed:" >&2
  cat "$FAILLOG" >&2
  exit 1
fi
