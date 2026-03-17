#!/bin/bash
# Train ConvNet observers for all datasets (except ImageNet-1K).
# Saves to ./data/pretrain_models/{dataset}_conv{N}.pth
#
# Run from RDED project root:
#   ./scripts/train_all_convnet.sh
# Or train individual datasets:
#   python scripts/train_convnet_observer.py --dataset cifar10

set -e
cd "$(dirname "$0")/.."

echo "Training ConvNet observers for RDED"
echo "Output: ./data/pretrain_models/"
echo ""

# for ds in cifar10 cifar100 tinyimagenet imagenet-nette imagenet-woof imagenet-100; do
for ds in cifar10; do
    if [[ -d "./data/$ds/train" ]]; then
        python scripts/train_convnet_observer.py --dataset "$ds" --device cuda
    else
        echo "SKIP $ds: ./data/$ds/train not found (run prepare_datasets.py first)"
    fi
done

echo "All ConvNet observer trainings complete."
ls -la ./data/pretrain_models/*conv*.pth 2>/dev/null || true
