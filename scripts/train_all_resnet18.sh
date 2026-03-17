#!/bin/bash
# Train ResNet-18 observers for all datasets (except ImageNet-1K).
# Saves to ./data/pretrain_models/{dataset}_resnet18[_modified].pth
#
# Run from RDED project root:
#   ./scripts/train_all_resnet18.sh
# Or train individual datasets:
#   python scripts/train_resnet18_observer.py --dataset cifar10

set -e
cd "$(dirname "$0")/.."

echo "Training ResNet-18 observers for RDED"
echo "Output: ./data/pretrain_models/"
echo ""

for ds in cifar10 cifar100 tinyimagenet imagenet-nette imagenet-woof imagenet-100; do
    if [[ -d "./data/$ds/train" ]]; then
        python scripts/train_resnet18_observer.py --dataset "$ds" --device cuda
    else
        echo "SKIP $ds: ./data/$ds/train not found (run prepare_datasets.py first)"
    fi
done

echo "All ResNet-18 observer trainings complete."
ls -la ./data/pretrain_models/*resnet18*.pth 2>/dev/null || true
