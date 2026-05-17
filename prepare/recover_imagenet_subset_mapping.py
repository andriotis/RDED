"""
Recover the WNID -> class-index mapping the pretrained teacher was trained
with, by predicting all val images of each WNID and reading off the modal
predicted class. Works for ImageNette, ImageWoof, ImageNet-10, ImageNet-100.

Output: a {wnid: idx} dict printed and (optionally) written to JSON.

Usage:
    CUDA_VISIBLE_DEVICES=1 python prepare/recover_imagenet_subset_mapping.py \\
        --src ./data/_raw/imagewoof2/val \\
        --subset imagenet-woof --arch resnet18 --input-size 224 \\
        --out prepare/imagewoof_mapping.json
"""
import argparse, collections, json, os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
_argv = sys.argv; sys.argv = ["x"]
from synthesize.utils import load_model  # noqa: E402
sys.argv = _argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="val/ root with <wnid>/ subdirs")
    ap.add_argument("--subset", required=True,
                    choices=["imagenet-nette", "imagenet-woof", "imagenet-10",
                             "imagenet-100"])
    ap.add_argument("--arch", required=True)
    ap.add_argument("--input-size", type=int, required=True)
    ap.add_argument("--nclass", type=int, default=10)
    ap.add_argument("--out", help="write JSON mapping to this path")
    args = ap.parse_args()

    tfm = T.Compose([
        T.Resize(args.input_size // 7 * 8, antialias=True),
        T.CenterCrop(args.input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = ImageFolder(args.src, transform=tfm)
    if len(ds.classes) != args.nclass:
        raise SystemExit(f"expected {args.nclass} classes, found {len(ds.classes)}")
    wnids = ds.classes  # alphabetically sorted

    model = load_model(model_name=args.arch, dataset=args.subset,
                       pretrained=True, classes=list(range(args.nclass)))
    model = nn.DataParallel(model).cuda().eval()

    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4,
                        pin_memory=True)
    # for each wnid (= folder-sorted index), tally predicted class indices
    votes = collections.defaultdict(collections.Counter)
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            preds = model(x).argmax(1).cpu().tolist()
            for wnid_idx, pred in zip(y.tolist(), preds):
                votes[wnid_idx][pred] += 1

    mapping = {}
    pp = {}  # purity per wnid (fraction of dominant vote)
    print(f"{'wnid':16} -> {'idx':>4}   purity   votes")
    for wnid_idx in range(args.nclass):
        wnid = wnids[wnid_idx]
        c = votes[wnid_idx]
        top_idx, top_count = c.most_common(1)[0]
        total = sum(c.values())
        purity = top_count / total
        mapping[wnid] = top_idx
        pp[wnid] = purity
        print(f"{wnid:16} -> {top_idx:4d}   {purity:6.2%}   {dict(c.most_common(3))}")

    # sanity: bijection?
    if len(set(mapping.values())) != args.nclass:
        print(f"\nWARN: {args.nclass - len(set(mapping.values()))} class index "
              f"collisions; teacher predictions for some WNIDs are ambiguous "
              f"(probably wrong arch or wrong subset).")
    else:
        print(f"\nbijective mapping recovered.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"wnid_to_idx": mapping, "purity": pp}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
