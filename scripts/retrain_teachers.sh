#!/usr/bin/env bash
# Retrain RDED teacher / observer models from scratch with the recipe embedded
# in the released checkpoints. All 15 official checkpoints share the SAME args:
#   SGD lr=0.2  momentum=0.9  weight_decay=1e-4  batch=256  epochs=100
#   cosine schedule, no warmup, no mixup/cutmix/EMA/AMP/autoaug/label-smoothing
# Only the input size varies (derived per dataset/arch below).
#
# Usage (run from RDED/):
#   bash scripts/retrain_teachers.sh                          # all teachers
#   bash scripts/retrain_teachers.sh --dataset cifar10        # both cifar10 teachers
#   bash scripts/retrain_teachers.sh --arch resnet18          # all resnet18(-modified) teachers
#   bash scripts/retrain_teachers.sh --arch resnet18 --dataset cifar10
#
# --arch is matched as a prefix, so "resnet18" also selects "resnet18_modified".
# Checkpoints are written to ./data/pretrain_models/retrain/ (originals untouched).
set -euo pipefail

# canonical (dataset, arch) teacher pairs — one per released checkpoint
TEACHERS=(
  "cifar10 resnet18_modified"
  "cifar10 conv3"
  "cifar100 resnet18_modified"
  "cifar100 conv3"
  "tinyimagenet resnet18_modified"
  "tinyimagenet conv4"
  "imagenet-nette resnet18"
  "imagenet-nette conv5"
  "imagenet-woof resnet18"
  "imagenet-woof conv5"
  "imagenet-10 resnet18"
  "imagenet-10 conv5"
  "imagenet-100 resnet18"
  "imagenet-100 conv6"
  "imagenet-1k conv4"
)

FILTER_ARCH=""
FILTER_DATASET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)    FILTER_ARCH="$2";    shift 2 ;;
    --dataset) FILTER_DATASET="$2"; shift 2 ;;
    -h|--help)
      echo "usage: bash scripts/retrain_teachers.sh [--dataset D] [--arch A]"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# input size from the embedded checkpoint args (train_crop_size)
input_size() {  # $1=dataset $2=arch
  case "$1" in
    cifar10|cifar100) echo 32 ;;
    tinyimagenet|imagenet-1k) echo 64 ;;
    imagenet-nette|imagenet-woof|imagenet-10|imagenet-100)
      [[ "$2" == conv* ]] && echo 128 || echo 224 ;;
    *) echo "unknown dataset: $1" >&2; exit 1 ;;
  esac
}

export CUDA_VISIBLE_DEVICES=1
OUT_DIR=./data/pretrain_models/retrain
LOG_DIR=./logs/retrain_teachers
mkdir -p "$OUT_DIR" "$LOG_DIR"

ran=0
for pair in "${TEACHERS[@]}"; do
  read -r dataset arch <<< "$pair"
  [[ -n "$FILTER_DATASET" && "$dataset" != "$FILTER_DATASET" ]] && continue
  # prefix match: --arch resnet18 also selects resnet18_modified
  [[ -n "$FILTER_ARCH"    && "$arch" != "$FILTER_ARCH"* ]] && continue

  size=$(input_size "$dataset" "$arch")
  out="$OUT_DIR/${dataset}_${arch}.pth"
  log="$LOG_DIR/${dataset}_${arch}.log"
  echo ">>> training $dataset / $arch  (size=$size)  -> $out  (log: $log)"
  python prepare/train_teacher.py \
    --subset "$dataset" --arch "$arch" --size "$size" \
    --epochs 100 --batch-size 256 --lr 0.2 --weight-decay 1e-4 \
    --warmup-epochs 0 --workers 8 \
    --out "$out" 2>&1 | tee "$log"
  ran=$((ran + 1))
done

if [[ $ran -eq 0 ]]; then
  echo "no teachers matched (dataset='$FILTER_DATASET' arch='$FILTER_ARCH')" >&2
  exit 1
fi
echo ">>> done. retrained $ran teacher(s) -> $OUT_DIR"
