"""
Evaluate every pretrained teacher .pth in data/pretrain_models/ on its
corresponding real validation set, and compare top-1 to the value the paper's
README reports for that teacher.

Catches "wrong / corrupt / mismatched checkpoint" before it poisons hundreds
of downstream RDED cells. Skips teachers whose val data isn't on disk yet.

Run from RDED/:
    CUDA_VISIBLE_DEVICES=1 python scripts/table2/verify_teachers.py
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))
# load_model imports `argument` (which calls argparse on sys.argv) -- shield it.
_saved_argv = sys.argv
sys.argv = ["verify_teachers"]
from synthesize.utils import load_model  # noqa: E402

sys.argv = _saved_argv

# (subset, arch, input_size, expected_top1) from RDED/README.md
TEACHERS = [
    ("cifar10", "resnet18_modified", 32, 93.86),
    ("cifar10", "conv3", 32, 82.24),
    ("cifar100", "resnet18_modified", 32, 72.27),
    ("cifar100", "conv3", 32, 61.27),
    ("tinyimagenet", "resnet18_modified", 64, 61.98),
    ("tinyimagenet", "conv4", 64, 49.73),
    ("imagenet-nette", "resnet18", 224, 90.00),
    ("imagenet-nette", "conv5", 128, 89.60),
    ("imagenet-woof", "resnet18", 224, 75.00),
    ("imagenet-woof", "conv5", 128, 67.40),
    ("imagenet-10", "resnet18", 224, 87.40),
    ("imagenet-10", "conv5", 128, 85.4),
    ("imagenet-100", "resnet18", 224, 83.40),
    ("imagenet-100", "conv6", 128, 72.82),
    ("imagenet-1k", "conv4", 64, 43.6),
    # imagenet-1k resnet18 uses torchvision pretrained=True, ~69.76 top-1.
    ("imagenet-1k", "resnet18", 224, 69.76),
]

NCLASS = {
    "cifar10": 10,
    "cifar100": 100,
    "tinyimagenet": 200,
    "imagenet-nette": 10,
    "imagenet-woof": 10,
    "imagenet-10": 10,
    "imagenet-100": 100,
    "imagenet-1k": 1000,
}


def evaluate(model, val_dir, input_size, nclass, batch_size, workers):
    tfm = T.Compose(
        [
            T.Resize(input_size // 7 * 8, antialias=True),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # 5-digit folder names (00000, 00001, ...) sort lexicographically into the
    # same order as numeric class IDs, so torchvision's ImageFolder gives the
    # right label mapping.
    ds = ImageFolder(root=val_dir, transform=tfm)
    assert (
        len(ds.classes) == nclass
    ), f"{val_dir}: expected {nclass} classes, found {len(ds.classes)}"
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tol",
        type=float,
        default=1.0,
        help="flag teachers whose top-1 differs from README by > tol points",
    )
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rows = []
    for subset, arch, size, expected in TEACHERS:
        ckpt = f"./data/pretrain_models/{subset}_{arch}.pth"
        val = f"./data/{subset}/val"
        # imagenet-1k resnet18 loads from torchvision; ckpt path won't exist
        needs_ckpt = not (subset == "imagenet-1k" and arch == "resnet18")
        if needs_ckpt and not os.path.exists(ckpt):
            rows.append((subset, arch, "no ckpt", expected, None))
            continue
        if not os.path.isdir(val):
            rows.append((subset, arch, "no val", expected, None))
            continue
        try:
            model = load_model(
                model_name=arch,
                dataset=subset,
                pretrained=True,
                classes=list(range(NCLASS[subset])),
            )
            model = nn.DataParallel(model).cuda()
            top1 = evaluate(model, val, size, NCLASS[subset], args.batch, args.workers)
            rows.append((subset, arch, "ok", expected, top1))
        except Exception as e:
            rows.append((subset, arch, f"err: {type(e).__name__}", expected, None))
        finally:
            try:
                del model
                torch.cuda.empty_cache()
            except Exception:
                pass

    width = max(len(s) + len(a) + 1 for s, a, *_ in rows) + 2
    print(f"{'teacher'.ljust(width)}  expected   measured     diff   status")
    print("-" * (width + 38))
    flagged = 0
    for subset, arch, status, expected, top1 in rows:
        tag = f"{subset}/{arch}".ljust(width)
        if top1 is None:
            print(f"{tag}  {expected:6.2f}    --       --       {status}")
            continue
        diff = top1 - expected
        bad = abs(diff) > args.tol
        flag = "  !! drift" if bad else ""
        if bad:
            flagged += 1
        print(f"{tag}  {expected:6.2f}    {top1:6.2f}   {diff:+6.2f}   ok{flag}")

    print()
    print(
        f"Evaluated {sum(1 for r in rows if r[4] is not None)} teachers, "
        f"flagged {flagged} at |diff|>{args.tol}."
    )
    if flagged:
        sys.exit(1)


if __name__ == "__main__":
    main()
