"""
Evaluate the imagenet-woof resnet18 teacher under several preprocessing
recipes. If a non-standard recipe drops top-1 by ~20 points, then the README's
75% might just be using a different pipeline. Otherwise the gap is real.

Run from RDED/:
    CUDA_VISIBLE_DEVICES=1 python scripts/table2/eval_preprocessing_sweep.py
"""
import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))
_saved_argv = sys.argv
sys.argv = ["eval_preprocessing_sweep"]
from synthesize.utils import load_model  # noqa: E402
sys.argv = _saved_argv

VAL_DIR = "./data/imagenet-woof/val"
SUBSET  = "imagenet-woof"
ARCH    = "resnet18"
NCLASS  = 10
BATCH   = 256
WORKERS = 4

IM_MEAN = [0.485, 0.456, 0.406]
IM_STD  = [0.229, 0.224, 0.225]

# Each recipe builds a torchvision transform pipeline.
RECIPES = [
    ("baseline (resize 256, ccrop 224, imagenet-norm)",
        T.Compose([
            T.Resize(256, antialias=True),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IM_MEAN, IM_STD),
        ])),
    ("tight (resize 224, ccrop 224)",
        T.Compose([
            T.Resize(224, antialias=True),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IM_MEAN, IM_STD),
        ])),
    ("wide (resize 288, ccrop 224)",
        T.Compose([
            T.Resize(288, antialias=True),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IM_MEAN, IM_STD),
        ])),
    ("squash (resize 224x224, no crop)",
        T.Compose([
            T.Resize((224, 224), antialias=True),
            T.ToTensor(),
            T.Normalize(IM_MEAN, IM_STD),
        ])),
    ("bicubic interpolation (resize 256, ccrop 224)",
        T.Compose([
            T.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IM_MEAN, IM_STD),
        ])),
    ("no antialias (resize 256, ccrop 224)",
        T.Compose([
            T.Resize(256, antialias=False),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IM_MEAN, IM_STD),
        ])),
    ("low-res input (resize 160, ccrop 128)",
        T.Compose([
            T.Resize(160, antialias=True),
            T.CenterCrop(128),
            T.ToTensor(),
            T.Normalize(IM_MEAN, IM_STD),
        ])),
    ("high-res input (resize 320, ccrop 288)",
        T.Compose([
            T.Resize(320, antialias=True),
            T.CenterCrop(288),
            T.ToTensor(),
            T.Normalize(IM_MEAN, IM_STD),
        ])),
    ("no normalization (resize 256, ccrop 224)",
        T.Compose([
            T.Resize(256, antialias=True),
            T.CenterCrop(224),
            T.ToTensor(),
        ])),
    ("0.5/0.5 normalization (resize 256, ccrop 224)",
        T.Compose([
            T.Resize(256, antialias=True),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])),
]


def evaluate(model, tfm):
    ds = ImageFolder(root=VAL_DIR, transform=tfm)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=WORKERS, pin_memory=True)
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def main():
    model = load_model(model_name=ARCH, dataset=SUBSET,
                       pretrained=True, classes=list(range(NCLASS)))
    model = nn.DataParallel(model).cuda()

    print(f"teacher: {SUBSET}/{ARCH}   README expected: 75.00")
    print("-" * 78)
    print(f"{'recipe'.ljust(56)} {'top-1':>8} {'diff':>8}")
    print("-" * 78)
    for name, tfm in RECIPES:
        try:
            top1 = evaluate(model, tfm)
            print(f"{name.ljust(56)} {top1:8.2f} {top1 - 75.0:+8.2f}")
        except Exception as e:
            print(f"{name.ljust(56)} ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
