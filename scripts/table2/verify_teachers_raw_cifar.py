"""
Diagnostic: evaluate CIFAR teachers on RAW torchvision CIFAR arrays
(bypassing our JPEG-prepared val folder). Confirms whether the drift seen by
verify_teachers.py is caused by our JPEG-q95 prep step.
"""
import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))
_saved_argv = sys.argv
sys.argv = ["x"]
from synthesize.utils import load_model  # noqa: E402
sys.argv = _saved_argv

CONFIGS = [
    ("cifar10",  "resnet18_modified", 10, 93.86),
    ("cifar10",  "conv3",             10, 82.24),
    ("cifar100", "resnet18_modified", 100, 72.27),
    ("cifar100", "conv3",             100, 61.27),
]

tfm = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def eval_top1(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


print(f"{'teacher':32} {'expected':>9} {'raw-CIFAR':>10} {'JPEG-prep':>10}   diff_raw")
print("-" * 80)
for subset, arch, nc, expected in CONFIGS:
    cls = datasets.CIFAR10 if subset == "cifar10" else datasets.CIFAR100
    ds = cls("./data/_torchvision_cache", train=False, download=False, transform=tfm)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = load_model(model_name=arch, dataset=subset, pretrained=True,
                       classes=list(range(nc)))
    model = nn.DataParallel(model).cuda()
    raw = eval_top1(model, loader)

    # JPEG-prep accuracy is what verify_teachers.py already reported; reprint
    # for side-by-side context (hardcoded from this session's prior run).
    jpeg = {"cifar10/resnet18_modified": 91.25,
            "cifar10/conv3": 79.28,
            "cifar100/resnet18_modified": 67.84,
            "cifar100/conv3": 57.02}[f"{subset}/{arch}"]

    print(f"{subset+'/'+arch:32} {expected:9.2f} {raw:10.2f} {jpeg:10.2f}   "
          f"{raw - expected:+7.2f}")
    del model
    torch.cuda.empty_cache()
