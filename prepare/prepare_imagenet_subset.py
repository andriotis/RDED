"""
Convert an ImageNet subset (ImageNette, ImageWoof, ImageNet-10, ImageNet-100)
to RDED's 5-digit ImageFolder layout, using a recovered WNID->index mapping.

The mapping must come from prepare/recover_imagenet_subset_mapping.py — running
it against the released pretrained teacher reveals the order the teacher was
trained with, which is the only ordering that produces correct accuracies.

Usage:
    python prepare/prepare_imagenet_subset.py \\
        --src ./data/_raw/imagewoof2 \\
        --out ./data/imagenet-woof \\
        --mapping prepare/imagewoof_mapping.json
"""
import argparse
import json
import os
import shutil


def link_or_copy(src, dst, use_link):
    if os.path.exists(dst):
        return
    if use_link:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def reorganize(src_split, out_split, wnid2idx, use_link):
    nclass = len(wnid2idx)
    for idx in range(nclass):
        os.makedirs(f"{out_split}/{idx:05d}", exist_ok=True)
    n = 0
    for wnid in os.listdir(src_split):
        if wnid not in wnid2idx:
            continue
        idx = wnid2idx[wnid]
        dst_dir = f"{out_split}/{idx:05d}"
        for fn in os.listdir(f"{src_split}/{wnid}"):
            link_or_copy(f"{src_split}/{wnid}/{fn}", f"{dst_dir}/{fn}", use_link)
            n += 1
    print(f"  {n} files into {out_split}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="root with train/ and val/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mapping", required=True, help="JSON {wnid_to_idx:{...}}")
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    with open(args.mapping) as f:
        wnid2idx = json.load(f)["wnid_to_idx"]
    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} not empty; remove first")
    use_link = not args.copy
    for split in ("train", "val"):
        src = f"{args.src}/{split}"
        out = f"{args.out}/{split}"
        os.makedirs(out, exist_ok=True)
        if not os.path.isdir(src):
            print(f"  {src} missing, skipping")
            continue
        print(f"[{split}]")
        reorganize(src, out, wnid2idx, use_link)


if __name__ == "__main__":
    main()
