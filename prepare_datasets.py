"""
Prepare all datasets for RDED Table 2 reproduction.

Downloads and converts each dataset into the required folder format:
    ./data/{subset}/train/00000/, ./data/{subset}/train/00001/, ...
    ./data/{subset}/val/00000/,   ./data/{subset}/val/00001/,   ...

Usage:
    python prepare_datasets.py --all                          # Prepare everything
    python prepare_datasets.py --cifar10 --cifar100           # Specific datasets
    python prepare_datasets.py --tinyimagenet
    python prepare_datasets.py --imagenette --imagewoof
    python prepare_datasets.py --imagenet1k --imagenet1k-src /path/to/ILSVRC/
    python prepare_datasets.py --imagenet100 --imagenet1k-src /path/to/ILSVRC/
    python prepare_datasets.py --pretrained-models            # Download observer models

Requires: torch, torchvision, Pillow, gdown (for pretrained models)
"""

import argparse
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

DATA_ROOT = Path("./data")

# ─────────────────────────────────────────────────────────────────────────────
# ImageNette / ImageWoof class definitions (sorted ImageNet-1K WordNet IDs)
# ─────────────────────────────────────────────────────────────────────────────
IMAGENETTE_WNIDS = [
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
]

# RDED pretrained models expect this exact class order (see prepare docs)
IMAGEWOOF_WNIDS = [
    "n02096294", "n02093754", "n02111889", "n02088364", "n02086240",
    "n02089973", "n02087394", "n02115641", "n02099601", "n02105641",
]

# ImageNet-100: 100-class subset used in dataset distillation literature.
# Sorted by WordNet ID, consistent with IDM / SRe2L / RDED conventions.
IMAGENET100_WNIDS = [
    "n01498041", "n01531178", "n01534433", "n01558993", "n01580077",
    "n01614925", "n01616318", "n01631663", "n01641577", "n01669191",
    "n01677366", "n01687978", "n01694178", "n01695060", "n01704323",
    "n01728572", "n01770081", "n01770393", "n01774750", "n01784675",
    "n01819313", "n01820546", "n01833805", "n01843383", "n01847000",
    "n01855672", "n01882714", "n01910747", "n01914609", "n01924916",
    "n01944390", "n01985128", "n01986214", "n02007558", "n02009912",
    "n02037110", "n02051845", "n02077923", "n02085620", "n02099601",
    "n02106550", "n02106662", "n02110958", "n02119022", "n02123045",
    "n02123394", "n02124075", "n02125311", "n02129165", "n02132136",
    "n02165456", "n02190166", "n02206856", "n02226429", "n02231487",
    "n02233338", "n02236044", "n02268443", "n02279972", "n02281406",
    "n02321529", "n02364673", "n02395406", "n02403003", "n02410509",
    "n02415577", "n02423022", "n02437312", "n02480495", "n02481823",
    "n02486410", "n02504458", "n02509815", "n02666196", "n02669723",
    "n02699494", "n02730930", "n02769748", "n02788148", "n02791270",
    "n02793495", "n02795169", "n02802426", "n02808440", "n02814533",
    "n02814860", "n02815834", "n02823428", "n02837789", "n02841315",
    "n02843684", "n02883205", "n02892201", "n02906734", "n02909870",
    "n02917067", "n02927161", "n02948072", "n02950826", "n02963159",
]

# Tiny-ImageNet: mapping from RDED prepare/tinyimagenet.md
TINYIMAGENET_WNIDS = [
    "n01443537", "n01629819", "n01641577", "n01644900", "n01698640",
    "n01742172", "n01768244", "n01770393", "n01774384", "n01774750",
    "n01784675", "n01855672", "n01882714", "n01910747", "n01917289",
    "n01944390", "n01945685", "n01950731", "n01983481", "n01984695",
    "n02002724", "n02056570", "n02058221", "n02074367", "n02085620",
    "n02094433", "n02099601", "n02099712", "n02106662", "n02113799",
    "n02123045", "n02123394", "n02124075", "n02125311", "n02129165",
    "n02132136", "n02165456", "n02190166", "n02206856", "n02226429",
    "n02231487", "n02233338", "n02236044", "n02268443", "n02279972",
    "n02281406", "n02321529", "n02364673", "n02395406", "n02403003",
    "n02410509", "n02415577", "n02423022", "n02437312", "n02480495",
    "n02481823", "n02486410", "n02504458", "n02509815", "n02666196",
    "n02669723", "n02699494", "n02730930", "n02769748", "n02788148",
    "n02791270", "n02793495", "n02795169", "n02802426", "n02808440",
    "n02814533", "n02814860", "n02815834", "n02823428", "n02837789",
    "n02841315", "n02843684", "n02883205", "n02892201", "n02906734",
    "n02909870", "n02917067", "n02927161", "n02948072", "n02950826",
    "n02963159", "n02977058", "n02988304", "n02999410", "n03014705",
    "n03026506", "n03042490", "n03085013", "n03089624", "n03100240",
    "n03126707", "n03160309", "n03179701", "n03201208", "n03250847",
    "n03255030", "n03355925", "n03388043", "n03393912", "n03400231",
    "n03404251", "n03424325", "n03444034", "n03447447", "n03544143",
    "n03584254", "n03599486", "n03617480", "n03637318", "n03649909",
    "n03662601", "n03670208", "n03706229", "n03733131", "n03763968",
    "n03770439", "n03796401", "n03804744", "n03814639", "n03837869",
    "n03838899", "n03854065", "n03891332", "n03902125", "n03930313",
    "n03937543", "n03970156", "n03976657", "n03977966", "n03980874",
    "n03983396", "n03992509", "n04008634", "n04023962", "n04067472",
    "n04070727", "n04074963", "n04099969", "n04118538", "n04133789",
    "n04146614", "n04149813", "n04179913", "n04251144", "n04254777",
    "n04259630", "n04265275", "n04275548", "n04285008", "n04311004",
    "n04328186", "n04356056", "n04366367", "n04371430", "n04376876",
    "n04398044", "n04399382", "n04417672", "n04456115", "n04465501",
    "n04486054", "n04487081", "n04501370", "n04507155", "n04532106",
    "n04532670", "n04540053", "n04560804", "n04562935", "n04596742",
    "n04597913", "n06596364", "n07579787", "n07583066", "n07614500",
    "n07615774", "n07695742", "n07711569", "n07715103", "n07720875",
    "n07734744", "n07747607", "n07749582", "n07753592", "n07768694",
    "n07871810", "n07873807", "n07875152", "n07920052", "n09193705",
    "n09246464", "n09256479", "n09332890", "n09428293", "n12267677",
]

# Google Drive folder ID for pretrained observer models
GDRIVE_FOLDER_ID = "1HmrheO6MgX453a5UPJdxPHK4UTv-4aVt"

PRETRAINED_FILES = {
    "cifar10_conv3":                  "1_PLACEHOLDER_cifar10_conv3",
    "cifar10_resnet18_modified":      "1_PLACEHOLDER_cifar10_rn18m",
    "cifar100_conv3":                 "1_PLACEHOLDER_cifar100_conv3",
    "cifar100_resnet18_modified":     "1_PLACEHOLDER_cifar100_rn18m",
    "tinyimagenet_conv4":             "1_PLACEHOLDER_tiny_conv4",
    "tinyimagenet_resnet18_modified": "1_PLACEHOLDER_tiny_rn18m",
    "imagenet-nette_conv5":           "1_PLACEHOLDER_nette_conv5",
    "imagenet-nette_resnet18":        "1_PLACEHOLDER_nette_rn18",
    "imagenet-woof_conv5":            "1_PLACEHOLDER_woof_conv5",
    "imagenet-woof_resnet18":         "1_PLACEHOLDER_woof_rn18",
    "imagenet-100_conv6":             "1_PLACEHOLDER_in100_conv6",
    "imagenet-100_resnet18":          "1_PLACEHOLDER_in100_rn18",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_and_extract(url: str, dest: Path, desc: str = "Downloading"):
    """Download a tar/zip archive and extract it."""
    dest.mkdir(parents=True, exist_ok=True)
    archive_name = url.split("/")[-1]
    archive_path = dest / archive_name

    if not archive_path.exists():
        print(f"  Downloading {desc} ...")
        torch.hub.download_url_to_file(url, str(archive_path))
    else:
        print(f"  Archive already exists: {archive_path}")

    print(f"  Extracting {archive_name} ...")
    if archive_name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=str(dest))
    elif archive_name.endswith(".tar"):
        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(path=str(dest))
    elif archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=str(dest))

    return dest


def save_torchvision_as_folders(dataset, output_dir: Path, num_classes: int):
    """Save a torchvision dataset into class-numbered folders of JPEG images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for c in range(num_classes):
        (output_dir / f"{c:05d}").mkdir(exist_ok=True)

    counters = [0] * num_classes
    for img, label in tqdm(dataset, desc=f"  Saving to {output_dir.name}"):
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        fname = f"{output_dir}/{label:05d}/img_{counters[label]:05d}.jpg"
        img.save(fname)
        counters[label] += 1


def copy_class_folder(src_dir: Path, dst_dir: Path, wnid: str, class_idx: int):
    """Copy images from a WordNet ID folder to a numbered class folder."""
    src = src_dir / wnid
    if not src.exists():
        print(f"  WARNING: source folder not found: {src}")
        return 0

    dst = dst_dir / f"{class_idx:05d}"
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_file in sorted(src.iterdir()):
        if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".JPEG"):
            shutil.copy2(img_file, dst / img_file.name)
            count += 1
    return count


def remap_imagenet_folders(src_root: Path, dst_root: Path, wnid_list: list, split: str):
    """Copy ImageNet-style wnid folders into numbered class folders."""
    src_dir = src_root / split
    dst_dir = dst_root / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print(f"  WARNING: source split folder not found: {src_dir}")
        return

    total = 0
    for idx, wnid in enumerate(tqdm(wnid_list, desc=f"  Remapping {split}")):
        n = copy_class_folder(src_dir, dst_dir, wnid, idx)
        total += n

    print(f"  {split}: {total} images across {len(wnid_list)} classes")


def build_imagenet1k_wnid_order(imagenet_train_dir: Path) -> list:
    """Return sorted list of all 1000 WordNet IDs from an ImageNet train dir."""
    wnids = sorted([
        d.name for d in imagenet_train_dir.iterdir()
        if d.is_dir() and d.name.startswith("n")
    ])
    assert len(wnids) == 1000, (
        f"Expected 1000 class folders in {imagenet_train_dir}, found {len(wnids)}"
    )
    return wnids


# ─────────────────────────────────────────────────────────────────────────────
# Dataset preparation functions
# ─────────────────────────────────────────────────────────────────────────────

def prepare_cifar10():
    """Download CIFAR-10 via torchvision and convert to folder structure."""
    print("\n=== Preparing CIFAR-10 ===")
    out = DATA_ROOT / "cifar10"
    if (out / "train" / "00009").exists():
        print("  Already prepared. Skipping.")
        return

    cache = DATA_ROOT / "_cache"
    cache.mkdir(parents=True, exist_ok=True)

    trainset = torchvision.datasets.CIFAR10(root=str(cache), train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=str(cache), train=False, download=True)

    save_torchvision_as_folders(trainset, out / "train", num_classes=10)
    save_torchvision_as_folders(testset, out / "val", num_classes=10)
    print("  Done.")


def prepare_cifar100():
    """Download CIFAR-100 via torchvision and convert to folder structure."""
    print("\n=== Preparing CIFAR-100 ===")
    out = DATA_ROOT / "cifar100"
    if (out / "train" / "00099").exists():
        print("  Already prepared. Skipping.")
        return

    cache = DATA_ROOT / "_cache"
    cache.mkdir(parents=True, exist_ok=True)

    trainset = torchvision.datasets.CIFAR100(root=str(cache), train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root=str(cache), train=False, download=True)

    save_torchvision_as_folders(trainset, out / "train", num_classes=100)
    save_torchvision_as_folders(testset, out / "val", num_classes=100)
    print("  Done.")


def prepare_tinyimagenet():
    """Download Tiny-ImageNet and rename folders per RDED mapping."""
    print("\n=== Preparing Tiny-ImageNet ===")
    out = DATA_ROOT / "tinyimagenet"
    if (out / "train" / "00199").exists():
        print("  Already prepared. Skipping.")
        return

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    cache = DATA_ROOT / "_cache"
    download_and_extract(url, cache, desc="Tiny-ImageNet")

    raw = cache / "tiny-imagenet-200"

    # --- Train split ---
    print("  Processing train split ...")
    train_out = out / "train"
    train_out.mkdir(parents=True, exist_ok=True)
    for idx, wnid in enumerate(tqdm(TINYIMAGENET_WNIDS, desc="  Remapping train")):
        src = raw / "train" / wnid / "images"
        if not src.exists():
            print(f"  WARNING: {src} not found")
            continue
        dst = train_out / f"{idx:05d}"
        dst.mkdir(exist_ok=True)
        for img in sorted(src.iterdir()):
            if img.suffix.lower() in (".jpeg", ".jpg", ".png"):
                shutil.copy2(img, dst / img.name)

    # --- Val split (Tiny-ImageNet val has a different structure) ---
    print("  Processing val split ...")
    val_out = out / "val"
    val_out.mkdir(parents=True, exist_ok=True)
    for idx in range(200):
        (val_out / f"{idx:05d}").mkdir(exist_ok=True)

    # Build wnid-to-index mapping
    wnid_to_idx = {wnid: idx for idx, wnid in enumerate(TINYIMAGENET_WNIDS)}

    val_annotations = raw / "val" / "val_annotations.txt"
    if val_annotations.exists():
        with open(val_annotations, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, wnid = parts[0], parts[1]
                if wnid in wnid_to_idx:
                    idx = wnid_to_idx[wnid]
                    src_img = raw / "val" / "images" / fname
                    if src_img.exists():
                        shutil.copy2(src_img, val_out / f"{idx:05d}" / fname)
    else:
        # Some versions have per-class val folders
        for wnid, idx in wnid_to_idx.items():
            src = raw / "val" / wnid / "images"
            if src.exists():
                dst = val_out / f"{idx:05d}"
                for img in sorted(src.iterdir()):
                    if img.suffix.lower() in (".jpeg", ".jpg", ".png"):
                        shutil.copy2(img, dst / img.name)

    print("  Done.")


def prepare_imagenette():
    """Download ImageNette from fastai and remap to numbered folders."""
    print("\n=== Preparing ImageNette ===")
    out = DATA_ROOT / "imagenet-nette"
    if (out / "train" / "00009").exists():
        print("  Already prepared. Skipping.")
        return

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    cache = DATA_ROOT / "_cache"
    download_and_extract(url, cache, desc="ImageNette")

    raw = cache / "imagenette2"
    for split in ("train", "val"):
        remap_imagenet_folders(raw, out, IMAGENETTE_WNIDS, split)

    print("  Done.")


def prepare_imagewoof():
    """Download ImageWoof from fastai and remap to numbered folders."""
    print("\n=== Preparing ImageWoof ===")
    out = DATA_ROOT / "imagenet-woof"
    if (out / "train" / "00009").exists():
        print("  Already prepared. Skipping.")
        return

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz"
    cache = DATA_ROOT / "_cache"
    download_and_extract(url, cache, desc="ImageWoof")

    raw = cache / "imagewoof2"
    for split in ("train", "val"):
        remap_imagenet_folders(raw, out, IMAGEWOOF_WNIDS, split)

    print("  Done.")


def prepare_imagenet1k(imagenet_src: Path):
    """Remap standard ImageNet-1K (wnid folders) to numbered folders."""
    print("\n=== Preparing ImageNet-1K ===")
    out = DATA_ROOT / "imagenet-1k"
    if (out / "train" / "00999").exists():
        print("  Already prepared. Skipping.")
        return

    imagenet_src = Path(imagenet_src)
    wnids = build_imagenet1k_wnid_order(imagenet_src / "train")
    print(f"  Found {len(wnids)} classes in {imagenet_src / 'train'}")

    for split in ("train", "val"):
        remap_imagenet_folders(imagenet_src, out, wnids, split)

    # Save the wnid order for reference
    with open(out / "wnid_order.txt", "w") as f:
        for idx, wnid in enumerate(wnids):
            f.write(f"{idx:05d}: {wnid}\n")

    print("  Done.")


def prepare_imagenet100(imagenet_src: Path):
    """Extract ImageNet-100 subset from a full ImageNet-1K source."""
    print("\n=== Preparing ImageNet-100 ===")
    out = DATA_ROOT / "imagenet-100"
    if (out / "train" / "00099").exists():
        print("  Already prepared. Skipping.")
        return

    imagenet_src = Path(imagenet_src)
    if not (imagenet_src / "train").exists():
        print(f"  ERROR: ImageNet source not found at {imagenet_src}/train")
        return

    for split in ("train", "val"):
        remap_imagenet_folders(imagenet_src, out, IMAGENET100_WNIDS, split)

    print("  Done.")


def prepare_pretrained_models():
    """Download pretrained observer models from Google Drive."""
    print("\n=== Downloading Pretrained Observer Models ===")
    out = DATA_ROOT / "pretrain_models"
    out.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        print("  Installing gdown ...")
        os.system("pip install gdown")
        import gdown

    print(f"  Downloading entire folder from Google Drive ...")
    print(f"  Target: {out}")
    print(f"  Source:  https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}")
    gdown.download_folder(
        id=GDRIVE_FOLDER_ID,
        output=str(out),
        quiet=False,
    )
    print("  Done. Downloaded files:")
    for f in sorted(out.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for RDED Table 2 reproduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_datasets.py --all
  python prepare_datasets.py --cifar10 --cifar100
  python prepare_datasets.py --imagenet1k --imagenet1k-src /data/ILSVRC2012
  python prepare_datasets.py --pretrained-models
        """,
    )
    parser.add_argument("--all", action="store_true", help="Prepare all datasets")
    parser.add_argument("--cifar10", action="store_true")
    parser.add_argument("--cifar100", action="store_true")
    parser.add_argument("--tinyimagenet", action="store_true")
    parser.add_argument("--imagenette", action="store_true")
    parser.add_argument("--imagewoof", action="store_true")
    parser.add_argument("--imagenet1k", action="store_true")
    parser.add_argument("--imagenet100", action="store_true")
    parser.add_argument(
        "--imagenet1k-src", type=str, default=None,
        help="Path to existing ImageNet-1K with train/val dirs containing wnid folders",
    )
    parser.add_argument("--pretrained-models", action="store_true",
                        help="Download pretrained observer models from Google Drive")
    args = parser.parse_args()

    # If nothing selected, show help
    has_selection = any([
        args.all, args.cifar10, args.cifar100, args.tinyimagenet,
        args.imagenette, args.imagewoof, args.imagenet1k, args.imagenet100,
        args.pretrained_models,
    ])
    if not has_selection:
        parser.print_help()
        return

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if args.all or args.cifar10:
        prepare_cifar10()

    if args.all or args.cifar100:
        prepare_cifar100()

    if args.all or args.tinyimagenet:
        prepare_tinyimagenet()

    if args.all or args.imagenette:
        prepare_imagenette()

    if args.all or args.imagewoof:
        prepare_imagewoof()

    if args.all or args.imagenet1k:
        if args.imagenet1k_src is None:
            print("\n  ERROR: --imagenet1k requires --imagenet1k-src /path/to/ILSVRC2012")
            print("  ImageNet-1K must be downloaded manually from https://image-net.org")
        else:
            prepare_imagenet1k(args.imagenet1k_src)

    if args.all or args.imagenet100:
        if args.imagenet1k_src is None:
            print("\n  ERROR: --imagenet100 requires --imagenet1k-src /path/to/ILSVRC2012")
            print("  ImageNet-100 is extracted from ImageNet-1K.")
        else:
            prepare_imagenet100(args.imagenet1k_src)

    if args.all or args.pretrained_models:
        prepare_pretrained_models()

    print("\n" + "=" * 60)
    print("Preparation complete. Dataset status:")
    print("=" * 60)
    datasets = [
        ("cifar10", 10), ("cifar100", 100), ("tinyimagenet", 200),
        ("imagenet-nette", 10), ("imagenet-woof", 10),
        ("imagenet-100", 100), ("imagenet-1k", 1000),
    ]
    for name, nclass in datasets:
        train_dir = DATA_ROOT / name / "train"
        val_dir = DATA_ROOT / name / "val"
        train_ok = (train_dir / f"{nclass - 1:05d}").exists() if train_dir.exists() else False
        val_ok = (val_dir / f"{nclass - 1:05d}").exists() if val_dir.exists() else False
        status = "ready" if (train_ok and val_ok) else "MISSING"
        print(f"  {name:20s}  [{status}]  train={train_ok}  val={val_ok}")

    models_dir = DATA_ROOT / "pretrain_models"
    if models_dir.exists():
        n_models = len(list(models_dir.glob("*.pth")))
        print(f"  {'pretrained models':20s}  [{n_models} files]")
    else:
        print(f"  {'pretrained models':20s}  [MISSING]")


if __name__ == "__main__":
    main()
