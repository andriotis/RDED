import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))
_saved = sys.argv
sys.argv = ["probe"]
from synthesize.utils import load_model
sys.argv = _saved

import torch, torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def make_tfms(input_size):
    tfms = {}
    # (a) verify_teachers.py / RDED-native
    tfms["rded_native (resize %d, crop %d)" % (input_size // 7 * 8, input_size)] = T.Compose([
        T.Resize(input_size // 7 * 8, antialias=True), T.CenterCrop(input_size),
        T.ToTensor(), NORM])
    # (b) torchvision-classic eval: resize 256, crop 224
    tfms["tv_classic (resize 256, crop 224)"] = T.Compose([
        T.Resize(256, antialias=True), T.CenterCrop(224), T.ToTensor(), NORM])
    # (c) resize-then-crop with ratio 256/224 but at input_size (already = a). add: resize shorter to input_size, no crop
    tfms["resize_only (%d)" % input_size] = T.Compose([
        T.Resize((input_size, input_size), antialias=True), T.ToTensor(), NORM])
    # (d) bicubic interpolation variant
    tfms["rded_native_bicubic"] = T.Compose([
        T.Resize(input_size // 7 * 8, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(input_size), T.ToTensor(), NORM])
    return tfms

def evalit(model, val_dir, tfm, n=600, bs=100):
    ds = ImageFolder(root=val_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4)
    model.eval()
    tot = cor = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            cor += (model(x).argmax(1) == y).sum().item()
            tot += y.size(0)
            if tot >= n:
                break
    return 100.0 * cor / tot, tot

for subset, arch, size, exp in [("imagenet-nette", "resnet18", 224, 90.0),
                                ("imagenet-woof", "resnet18", 224, 75.0)]:
    print("\n==== %s / %s  (README expects %.1f) ====" % (subset, arch, exp))
    model = load_model(model_name=arch, dataset=subset, pretrained=True,
                       classes=list(range(10)))
    model = nn.DataParallel(model).cuda()
    for name, tfm in make_tfms(size).items():
        acc, tot = evalit(model, "./data/%s/val" % subset, tfm)
        print("  %-40s top1=%6.2f  (n=%d)" % (name, acc, tot))
    del model
    torch.cuda.empty_cache()
