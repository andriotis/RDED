"""
Convert Tiny-ImageNet-200 into the 5-digit ImageFolder layout RDED expects,
using the exact WNID -> class-index mapping documented in
prepare/tinyimagenet.md. That mapping is what the released pretrained teacher
checkpoints were trained with; any other ordering = 0.5% accuracy disaster.

Source layout (Stanford):
    tiny-imagenet-200/
        train/<wnid>/images/<wnid>_<id>.JPEG
        val/images/val_<id>.JPEG
        val/val_annotations.txt   -- maps each val image to its <wnid>
        wnids.txt
        words.txt

Output layout (RDED):
    data/tinyimagenet/{train,val}/00000..00199/*.JPEG

Run from RDED/:
    python prepare/prepare_tinyimagenet.py
"""
import argparse
import os
import re
import shutil

# Parse the mapping out of prepare/tinyimagenet.md (single source of truth).
MAP_FILE = os.path.join(os.path.dirname(__file__), "tinyimagenet.md")


def load_wnid_map():
    """Return {wnid: class_index} from prepare/tinyimagenet.md."""
    mapping = {}
    pat = re.compile(r"^(\d{5}):\s*(n\d+)\s*$")
    with open(MAP_FILE) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                mapping[m.group(2)] = int(m.group(1))
    if len(mapping) != 200:
        raise SystemExit(f"expected 200 wnids in {MAP_FILE}, got {len(mapping)}")
    return mapping


def link_or_copy(src, dst, use_link):
    if os.path.exists(dst):
        return
    if use_link:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="./data/_raw/tiny-imagenet-200")
    ap.add_argument("--out", default="./data/tinyimagenet")
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of symlinking (uses more disk)")
    args = ap.parse_args()

    use_link = not args.copy
    wnid2idx = load_wnid_map()

    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} not empty; remove first")
    os.makedirs(args.out, exist_ok=True)

    # train: src/train/<wnid>/images/*.JPEG  ->  out/train/{idx:05d}/*.JPEG
    train_src = f"{args.src}/train"
    train_out = f"{args.out}/train"
    os.makedirs(train_out, exist_ok=True)
    n_train = 0
    for wnid in os.listdir(train_src):
        if wnid not in wnid2idx:
            raise SystemExit(f"unknown wnid {wnid} in train")
        idx = wnid2idx[wnid]
        dst_dir = f"{train_out}/{idx:05d}"
        os.makedirs(dst_dir, exist_ok=True)
        img_dir = f"{train_src}/{wnid}/images"
        for fn in os.listdir(img_dir):
            link_or_copy(f"{img_dir}/{fn}", f"{dst_dir}/{fn}", use_link)
            n_train += 1
    print(f"train: linked {n_train} images into {train_out}")

    # val: src/val/images/val_X.JPEG + val_annotations.txt
    val_src_imgs = f"{args.src}/val/images"
    val_ann = f"{args.src}/val/val_annotations.txt"
    val_out = f"{args.out}/val"
    os.makedirs(val_out, exist_ok=True)
    for idx in range(200):
        os.makedirs(f"{val_out}/{idx:05d}", exist_ok=True)
    n_val = 0
    with open(val_ann) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            fn, wnid = parts[0], parts[1]
            if wnid not in wnid2idx:
                raise SystemExit(f"unknown wnid {wnid} in val_annotations")
            idx = wnid2idx[wnid]
            link_or_copy(f"{val_src_imgs}/{fn}", f"{val_out}/{idx:05d}/{fn}", use_link)
            n_val += 1
    print(f"val: linked {n_val} images into {val_out}")


if __name__ == "__main__":
    main()
