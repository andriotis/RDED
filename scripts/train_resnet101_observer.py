#!/usr/bin/env python3
"""
Train ResNet-101 observer models from scratch for RDED Table 2.

Produces checkpoints in ./data/pretrain_models/ compatible with RDED's load_model().
Excludes ImageNet-1K (uses torchvision pretrained).

Usage:
    python scripts/train_resnet101_observer.py --dataset cifar10
    python scripts/train_resnet101_observer.py --dataset all
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.models as models

DATA_ROOT = Path("./data")
PRETRAIN_ROOT = Path("./data/pretrain_models")

# Dataset config: (num_classes, input_size, model_name)
DATASET_CONFIG = {
    "cifar10": (10, 32, "resnet101_modified"),
    "cifar100": (100, 32, "resnet101_modified"),
    "tinyimagenet": (200, 64, "resnet101_modified"),
    "imagenet-nette": (10, 224, "resnet101"),
    "imagenet-woof": (10, 224, "resnet101"),
    "imagenet-100": (100, 224, "resnet101"),
}


def build_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """Build ResNet-101 with correct architecture for dataset."""
    if model_name == "resnet101_modified":
        model = models.resnet101()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.resnet101(num_classes=num_classes)
    return model.to(device)


def get_transforms(size: int, train: bool):
    """Image transforms for training/validation."""
    if size <= 64:
        if train:
            return T.Compose([
                T.RandomCrop(size, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    # 224 for ImageNet subsets
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return T.Compose([
        T.Resize(int(size * 256 / 224)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
        if (i + 1) % 50 == 0:
            print(f"  Epoch {epoch} [{i+1}/{len(loader)}] loss={loss.item():.4f} acc={100*correct/total:.2f}%")
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()) + ["all"],
                        help="Dataset to train on (or 'all')")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (default: dataset-specific)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PRETRAIN_ROOT)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_to_run = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]

    # Epochs: fewer for small datasets, more for ImageNet subsets
    default_epochs = {
        "cifar10": 164,
        "cifar100": 164,
        "tinyimagenet": 90,
        "imagenet-nette": 90,
        "imagenet-woof": 90,
        "imagenet-100": 90,
    }

    for dataset in datasets_to_run:
        num_classes, size, model_name = DATASET_CONFIG[dataset]
        train_dir = DATA_ROOT / dataset / "train"
        val_dir = DATA_ROOT / dataset / "val"

        if not train_dir.exists():
            print(f"SKIP {dataset}: {train_dir} not found")
            continue

        epochs = args.epochs if args.epochs is not None else default_epochs[dataset]
        out_path = args.output_dir / f"{dataset}_{model_name}.pth"

        print(f"\n{'='*60}")
        print(f"Training ResNet-101 on {dataset} (classes={num_classes}, size={size}, arch={model_name})")
        print(f"  Train: {train_dir}")
        print(f"  Val:   {val_dir}")
        print(f"  Epochs: {epochs}, batch_size={args.batch_size}, lr={args.lr}")
        print(f"  Output: {out_path}")
        print("="*60)

        model = build_model(model_name, num_classes, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_tf = get_transforms(size, train=True)
        val_tf = get_transforms(size, train=False)

        train_ds = torchvision.datasets.ImageFolder(str(train_dir), transform=train_tf)
        val_ds = torchvision.datasets.ImageFolder(str(val_dir), transform=val_tf)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )

        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.2f}%  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%  ({elapsed:.1f}s)")
            if val_acc > best_acc:
                best_acc = val_acc
                args.output_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict()}, out_path)
                print(f"  -> Saved best (acc={best_acc:.2f}%)")

        print(f"\nDone {dataset}. Best val acc: {best_acc:.2f}%. Saved to {out_path}\n")


if __name__ == "__main__":
    main()
