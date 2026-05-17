"""
Retrain a teacher / observer model from scratch with the SRe2L-style recipe
the RDED paper says it follows ("we adapt official Torchvision code"). Use
this to try to recover the ~1pp residual drift between the released CIFAR
checkpoints and what the README claims, or to train teachers on datasets
where no checkpoint is provided.

The output .pth has the same shape the rest of the project expects:
    torch.save({"model": model.state_dict()}, f"./data/pretrain_models/{subset}_{arch}.pth")

Run (CIFAR-10 RN18-mod, ~3-4h on a single 10GB GPU):
    CUDA_VISIBLE_DEVICES=1 python prepare/train_teacher.py \\
        --subset cifar10 --arch resnet18_modified --epochs 200

Tiny-ImageNet RN18-mod (matches the zeyuanyin/tiny-imagenet recipe):
    CUDA_VISIBLE_DEVICES=1 python prepare/train_teacher.py \\
        --subset tinyimagenet --arch resnet18_modified \\
        --epochs 100 --lr 0.2 --batch-size 256 --warmup-epochs 5
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
_argv = sys.argv; sys.argv = ["x"]
from synthesize.utils import load_model  # noqa: E402
sys.argv = _argv

INPUT_SIZE = {
    "cifar10": 32, "cifar100": 32,
    "tinyimagenet": 64,
    "imagenet-nette": 224, "imagenet-woof": 224,
    "imagenet-10": 224, "imagenet-100": 224, "imagenet-1k": 224,
}
NCLASS = {
    "cifar10": 10, "cifar100": 100, "tinyimagenet": 200,
    "imagenet-nette": 10, "imagenet-woof": 10, "imagenet-10": 10,
    "imagenet-100": 100, "imagenet-1k": 1000,
}


def build_transforms(size, train):
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        if size <= 64:  # CIFAR / Tiny-ImageNet style
            return T.Compose([
                T.RandomCrop(size, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(), norm,
            ])
        return T.Compose([
            T.RandomResizedCrop(size, antialias=True),
            T.RandomHorizontalFlip(),
            T.ToTensor(), norm,
        ])
    # torchvision references uses resize = round(crop * 256/224); size*8//7
    # reproduces it exactly: 224->256, 128->146 (matches the released ckpts).
    return T.Compose([
        T.Resize(size * 8 // 7, antialias=True),
        T.CenterCrop(size),
        T.ToTensor(), norm,
    ])


def make_scheduler(opt, epochs, warmup_epochs, warmup_decay):
    """Cosine annealing with optional linear warmup (per-epoch step)."""
    def fn(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return warmup_decay + (1 - warmup_decay) * (epoch / max(1, warmup_epochs))
        # cosine over remaining epochs
        e = epoch - warmup_epochs
        T = max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * e / T))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def accuracy_top1(out, y):
    return (out.argmax(1) == y).float().mean().item() * 100


def evaluate(model, loader):
    model.eval()
    n = correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item()
            n += y.size(0)
    return 100.0 * correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, choices=list(NCLASS))
    ap.add_argument("--arch", required=True,
                    help="resnet18_modified / conv3 / conv4 / conv5 / conv6 / resnet18 / ...")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default=None,
                    help="output .pth path; default ./data/pretrain_models/{subset}_{arch}.pth")
    ap.add_argument("--size", type=int, default=None,
                    help="override input size; conv5/conv6 on imagenet-* need 128")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--warmup-decay", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    nclass = NCLASS[args.subset]
    size = args.size or INPUT_SIZE[args.subset]

    # Build model from scratch (pretrained=False)
    model = load_model(model_name=args.arch, dataset=args.subset,
                       pretrained=False, classes=list(range(nclass)))
    model = nn.DataParallel(model).cuda()

    train_ds = ImageFolder(f"{args.data_root}/{args.subset}/train",
                           transform=build_transforms(size, train=True))
    val_ds = ImageFolder(f"{args.data_root}/{args.subset}/val",
                         transform=build_transforms(size, train=False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=False)
    sched = make_scheduler(opt, args.epochs, args.warmup_epochs, args.warmup_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    out = args.out or f"./data/pretrain_models/{args.subset}_{args.arch}.pth"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    best = 0.0
    print(f"subset={args.subset} arch={args.arch} nclass={nclass} size={size}")
    print(f"epochs={args.epochs} bs={args.batch_size} lr={args.lr} "
          f"wd={args.weight_decay} warmup={args.warmup_epochs}")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run_loss = run_acc = n = 0
        for x, y in train_loader:
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            opt.zero_grad()
            out_logits = model(x)
            loss = loss_fn(out_logits, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.size(0)
            run_acc += (out_logits.argmax(1) == y).sum().item()
            n += y.size(0)
        sched.step()
        top1_val = evaluate(model, val_loader)
        print(f"epoch {epoch:3d}  train_loss={run_loss/n:.4f}  "
              f"train_top1={100*run_acc/n:.2f}  val_top1={top1_val:.2f}  "
              f"lr={opt.param_groups[0]['lr']:.5f}  dt={time.time()-t0:.1f}s")
        if top1_val > best:
            best = top1_val
            torch.save({"model": model.module.state_dict(), "top1": top1_val,
                        "epoch": epoch, "args": vars(args)}, out)
    print(f"best top1={best:.2f} saved -> {out}")


if __name__ == "__main__":
    main()
