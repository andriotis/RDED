"""
Convert torchvision's CIFAR-10 / CIFAR-100 into the 5-digit ImageFolder layout
RDED expects:

    ./data/{subset}/train/{00000..nclass-1}/img_XXXXX.jpg
    ./data/{subset}/val/{00000..nclass-1}/img_XXXXX.jpg

Run from the RDED/ directory:
    python prepare/prepare_cifar.py --subset cifar10
    python prepare/prepare_cifar.py --subset cifar100
"""
import argparse
import os
from PIL import Image
from torchvision import datasets


def dump(ds, out_root, n_classes):
    for c in range(n_classes):
        os.makedirs(f"{out_root}/{c:05d}", exist_ok=True)
    counts = [0] * n_classes
    for img, lab in ds:
        # CIFAR is 32x32. JPEG (even q=95) costs 1.6-3.1pp of teacher accuracy
        # — confirmed by scripts/table2/verify_teachers_raw_cifar.py. Use PNG.
        idx = counts[lab]
        img.save(f"{out_root}/{lab:05d}/img_{idx:05d}.png", optimize=False)
        counts[lab] += 1
    print(f"  wrote {sum(counts)} images to {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["cifar10", "cifar100"], required=True)
    ap.add_argument("--data-root", default="./data", help="output root")
    ap.add_argument("--cache", default="./data/_torchvision_cache",
                    help="where torchvision downloads the raw archives")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    cls = datasets.CIFAR10 if args.subset == "cifar10" else datasets.CIFAR100
    n_classes = 10 if args.subset == "cifar10" else 100

    out_train = f"{args.data_root}/{args.subset}/train"
    out_val   = f"{args.data_root}/{args.subset}/val"
    if os.path.isdir(out_train) and os.path.isdir(out_val):
        print(f"[{args.subset}] already prepared at {args.data_root}/{args.subset}; skipping")
        return

    print(f"[{args.subset}] downloading + converting train split...")
    dump(cls(args.cache, train=True,  download=True), out_train, n_classes)
    print(f"[{args.subset}] downloading + converting val split...")
    dump(cls(args.cache, train=False, download=True), out_val,   n_classes)


if __name__ == "__main__":
    main()
