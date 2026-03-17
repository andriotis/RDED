#!/usr/bin/env python3
"""
Train ConvNet observer models from scratch for RDED.

Produces checkpoints in ./data/pretrain_models/ compatible with RDED's load_model().
Each dataset uses the ConvNet depth and input size that matches argument.py / load_model().

Best-practice training recipe:
  - SGD + momentum 0.9, weight_decay 5e-4
  - Cosine annealing LR with linear warmup
  - Data augmentation: RandomCrop/RandomResizedCrop, HFlip, Cutout (small images)
  - Label smoothing (0.1)

Usage:
    python scripts/train_convnet_observer.py --dataset cifar10
    python scripts/train_convnet_observer.py --dataset all
    python scripts/train_convnet_observer.py --dataset cifar10 --epochs 1   # smoke test
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

# Add project root so we can import synthesize.models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synthesize.models import ConvNet

DATA_ROOT = Path("./data")
PRETRAIN_ROOT = Path("./data/pretrain_models")

# Dataset config: (num_classes, input_size, model_name, conv_depth)
DATASET_CONFIG = {
    "cifar10":        (10,  32,  "conv3", 3),
    "cifar100":       (100, 32,  "conv3", 3),
    "tinyimagenet":   (200, 64,  "conv4", 4),
    "imagenet-nette": (10,  128, "conv5", 5),
    "imagenet-woof":  (10,  128, "conv5", 5),
    "imagenet-100":   (100, 128, "conv6", 6),
}

DEFAULT_EPOCHS = {
    "cifar10": 200,
    "cifar100": 200,
    "tinyimagenet": 200,
    "imagenet-nette": 200,
    "imagenet-woof": 200,
    "imagenet-100": 200,
}

DEFAULT_BATCH_SIZES = {
    "cifar10": 128,
    "cifar100": 128,
    "tinyimagenet": 128,
    "imagenet-nette": 128,
    "imagenet-woof": 128,
    "imagenet-100": 128,
}

DEFAULT_LRS = {
    "cifar10": 0.1,
    "cifar100": 0.1,
    "tinyimagenet": 0.1,
    "imagenet-nette": 0.1,
    "imagenet-woof": 0.1,
    "imagenet-100": 0.1,
}


# ---------------------------------------------------------------------------
# Cutout augmentation (standard for CIFAR / small-image training)
# ---------------------------------------------------------------------------
class Cutout:
    """Randomly mask out a square region from an image tensor."""

    def __init__(self, length: int):
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones(h, w, dtype=img.dtype)
        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        mask[y1:y2, x1:x2] = 0.0
        return img * mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={self.length})"


# ---------------------------------------------------------------------------
# Cosine schedule with linear warmup
# ---------------------------------------------------------------------------
def cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """CosineAnnealingLR with linear warmup for the first `warmup_epochs`."""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(num_classes: int, depth: int, size: int, device: torch.device) -> nn.Module:
    """Build ConvNet with the architecture matching load_model()."""
    model = ConvNet(
        num_classes=num_classes,
        net_norm="batch",
        net_act="relu",
        net_pooling="avgpooling",
        net_depth=depth,
        net_width=128,
        channel=3,
        im_size=(size, size),
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms(size: int, train: bool):
    """Image transforms for training / validation.

    Training: RandomCrop + HFlip + Cutout for small images;
              RandomResizedCrop + HFlip for larger images.
    """
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if size <= 64:
        if train:
            return T.Compose([
                T.Resize(size),
                T.RandomCrop(size, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
                Cutout(length=size // 2),   # e.g. 16 for CIFAR, 32 for TinyImageNet
            ])
        return T.Compose([
            T.Resize(size),
            T.ToTensor(),
            normalize,
        ])
    # 128+ for ImageNet subsets
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.ToTensor(),
            normalize,
            Cutout(length=size // 4),   # 32 for 128px images
        ])
    return T.Compose([
        T.Resize(int(size * 256 / 224)),
        T.CenterCrop(size),
        T.ToTensor(),
        normalize,
    ])


# ---------------------------------------------------------------------------
# Train / Eval loops
# ---------------------------------------------------------------------------
TQDM_EPOCH_POSITION = 0
TQDM_TRAIN_POSITION = 1
TQDM_VAL_POSITION = 2


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(
        loader, desc=f"Epoch {epoch}/{total_epochs} [train]", unit="batch",
        position=TQDM_TRAIN_POSITION, leave=False,
    )
    for images, targets in pbar:
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
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.2f}%")
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in tqdm(
            loader, desc="Val", unit="batch", position=TQDM_VAL_POSITION, leave=False
        ):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ConvNet observer models for RDED")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()) + ["all"],
                        help="Dataset to train on (or 'all')")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (default: dataset-specific, usually 200)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (default: 0.1)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight decay (default: 5e-4)")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear LR warmup epochs (default: 5)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor (default: 0.1)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PRETRAIN_ROOT)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for training. 'auto' = cuda if available else cpu. "
                             "Use 'cuda' to fail immediately if GPU is not available.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    datasets_to_run = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]

    for dataset in datasets_to_run:
        num_classes, size, model_name, depth = DATASET_CONFIG[dataset]
        train_dir = DATA_ROOT / dataset / "train"
        val_dir = DATA_ROOT / dataset / "val"

        if not train_dir.exists():
            print(f"SKIP {dataset}: {train_dir} not found")
            continue

        epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS[dataset]
        batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZES[dataset]
        lr = args.lr if args.lr is not None else DEFAULT_LRS[dataset]
        out_path = args.output_dir / f"{dataset}_{model_name}.pth"

        print(f"\n{'='*60}")
        print(f"Training ConvNet on {dataset}")
        print(f"  arch={model_name}, depth={depth}, size={size}, classes={num_classes}")
        print(f"  Train: {train_dir}")
        print(f"  Val:   {val_dir}")
        print(f"  Epochs: {epochs}, batch_size={batch_size}, lr={lr}")
        print(f"  weight_decay={args.weight_decay}, label_smoothing={args.label_smoothing}")
        print(f"  warmup_epochs={args.warmup_epochs}")
        print(f"  Output: {out_path}")
        print("=" * 60)

        model = build_model(num_classes, depth, size, device)

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay,
        )
        scheduler = cosine_warmup_scheduler(optimizer, args.warmup_epochs, epochs)

        train_tf = get_transforms(size, train=True)
        val_tf = get_transforms(size, train=False)

        train_ds = torchvision.datasets.ImageFolder(str(train_dir), transform=train_tf)
        val_ds = torchvision.datasets.ImageFolder(str(val_dir), transform=val_tf)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=args.workers,
        )

        best_acc = 0.0
        epoch_pbar = tqdm(
            range(1, epochs + 1), desc="Training", unit="epoch",
            position=TQDM_EPOCH_POSITION,
        )
        for epoch in epoch_pbar:
            t0 = time.time()
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, epochs,
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0
            if val_acc > best_acc:
                best_acc = val_acc
                args.output_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict()}, out_path)
                epoch_pbar.write(f"  -> Saved best (acc={best_acc:.2f}%)")
            epoch_pbar.set_postfix(
                train=f"{train_acc:.2f}%", val=f"{val_acc:.2f}%",
                best=f"{best_acc:.2f}%", lr=f"{optimizer.param_groups[0]['lr']:.5f}",
            )

        epoch_pbar.write(f"\nDone {dataset}. Best val acc: {best_acc:.2f}%. Saved to {out_path}\n")


if __name__ == "__main__":
    main()
