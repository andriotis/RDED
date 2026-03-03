#!/bin/bash
# Train ResNet-101 observers for all datasets (except ImageNet-1K).
# Saves to ./data/pretrain_models/{dataset}_resnet101[_modified].pth
#
# Run from RDED project root:
#   ./scripts/train_all_resnet101.sh
# Or train individual datasets:
#   python scripts/train_resnet101_observer.py --dataset cifar10

set -e
cd "$(dirname "$0")/.."

echo "Training ResNet-101 observers for RDED Table 2"
echo "Output: ./data/pretrain_models/"
echo ""

# cifar10 cifar100 tinyimagenet imagenet-nette imagenet-woof imagenet-100
for ds in cifar100; do
    if [[ -d "./data/$ds/train" ]]; then
        python scripts/train_resnet101_observer.py --dataset "$ds"
    else
        echo "SKIP $ds: ./data/$ds/train not found (run prepare_datasets.py first)"
    fi
done

echo "All ResNet-101 observer trainings complete."
ls -la ./data/pretrain_models/*resnet101*.pth 2>/dev/null || true
