"""
Test whether the residual ~1pp CIFAR teacher drift comes from normalization.
Compare ImageNet-stats norm vs CIFAR-stats norm on raw torchvision CIFAR.
"""
import os, sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))
_argv = sys.argv; sys.argv = ["x"]
from synthesize.utils import load_model  # noqa: E402
sys.argv = _argv

CONFIGS = [
    ("cifar10",  "resnet18_modified", 10, 93.86),
    ("cifar10",  "conv3",             10, 82.24),
    ("cifar100", "resnet18_modified", 100, 72.27),
    ("cifar100", "conv3",             100, 61.27),
]

NORMS = {
    "imagenet": T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    "cifar10":  T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2470, 0.2435, 0.2616]),
    "cifar100": T.Normalize(mean=[0.5071, 0.4865, 0.4409],
                            std=[0.2673, 0.2564, 0.2762]),
    "none":     T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
}


def eval_top1(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


print(f"{'teacher':32} {'expect':>7}   "
      f"{'imagenet':>9} {'cifar*':>9} {'none':>9}")
print("-" * 78)
for subset, arch, nc, expected in CONFIGS:
    cls = datasets.CIFAR10 if subset == "cifar10" else datasets.CIFAR100
    cifar_norm = NORMS["cifar10"] if subset == "cifar10" else NORMS["cifar100"]

    accs = {}
    for norm_name, norm in [("imagenet", NORMS["imagenet"]),
                            ("cifar*", cifar_norm),
                            ("none", NORMS["none"])]:
        tfm = T.Compose([T.ToTensor(), norm])
        ds = cls("./data/_torchvision_cache", train=False, download=False, transform=tfm)
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        model = load_model(model_name=arch, dataset=subset, pretrained=True,
                           classes=list(range(nc)))
        model = nn.DataParallel(model).cuda()
        accs[norm_name] = eval_top1(model, loader)
        del model
        torch.cuda.empty_cache()

    print(f"{subset+'/'+arch:32} {expected:7.2f}   "
          f"{accs['imagenet']:9.2f} {accs['cifar*']:9.2f} {accs['none']:9.2f}")
