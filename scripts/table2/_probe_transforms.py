"""Probe: compare transform variants for woof/nette teachers."""
import os, sys
import torch, torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))
_a = sys.argv; sys.argv = ["probe"]
from synthesize.utils import load_model
sys.argv = _a

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def make_variants(size):
    return {
        "verify (//7*8 center)": T.Compose([T.Resize(size // 7 * 8, antialias=True), T.CenterCrop(size), T.ToTensor(), NORM]),
        "resize256 center": T.Compose([T.Resize(256, antialias=True), T.CenterCrop(size), T.ToTensor(), NORM]),
        "resize(size) center": T.Compose([T.Resize(size, antialias=True), T.CenterCrop(size), T.ToTensor(), NORM]),
        "resize(size,size) direct": T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(), NORM]),
        "no antialias //7*8": T.Compose([T.Resize(size // 7 * 8, antialias=False), T.CenterCrop(size), T.ToTensor(), NORM]),
        "no normalize": T.Compose([T.Resize(size // 7 * 8, antialias=True), T.CenterCrop(size), T.ToTensor()]),
    }

def evaluate(model, val_dir, tfm, nclass, limit_per_class=80):
    ds = ImageFolder(root=val_dir, transform=tfm)
    # subsample
    from collections import defaultdict
    cnt = defaultdict(int); keep = []
    for i, (p, c) in enumerate(ds.samples):
        if cnt[c] < limit_per_class:
            keep.append(i); cnt[c] += 1
    sub = torch.utils.data.Subset(ds, keep)
    loader = DataLoader(sub, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    model.eval(); total = correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return 100.0 * correct / total

CASES = [
    ("imagenet-woof", "resnet18", 224, 75.00),
    ("imagenet-woof", "conv5", 128, 67.40),
    ("imagenet-nette", "resnet18", 224, 90.00),
    ("imagenet-nette", "conv5", 128, 89.60),
]

for subset, arch, size, expected in CASES:
    print(f"\n=== {subset}/{arch}  size={size}  expected README top1={expected} ===")
    model = load_model(model_name=arch, dataset=subset, pretrained=True, classes=list(range(10)))
    model = nn.DataParallel(model).cuda()
    for name, tfm in make_variants(size).items():
        try:
            acc = evaluate(model, f"./data/{subset}/val", tfm, 10)
            print(f"  {name:28s} -> {acc:6.2f}")
        except Exception as e:
            print(f"  {name:28s} -> ERR {type(e).__name__}: {e}")
    del model; torch.cuda.empty_cache()
