"""
Single-file utility for getting the RDED teacher / observer setup off the
ground. Three operations:

    python scripts/teachers.py data    [--dataset D] [--copy]
    python scripts/teachers.py models  [--dataset D] [--arch A] [--out DIR]
    python scripts/teachers.py verify  [--dataset D] [--arch A] [--tol T]
                                       [--batch N] [--workers N] [--ckpt-dir DIR]

`data` downloads + reorganizes a dataset into RDED's 5-digit ImageFolder
layout under ./data/{subset}/{train,val}/{idx:05d}/...  Auto-downloadable:
cifar10, cifar100, tinyimagenet, imagenet-nette, imagenet-woof.
imagenet-10/-100/-1k require a local copy of full ImageNet and print
instructions instead.

`models` pulls the 15 released teacher checkpoints from the RDED Google
Drive folder into ./data/pretrain_models/{subset}_{arch}.pth.  Requires
`gdown`.

`verify` evaluates each teacher's top-1 on its real val split and compares
against the values in RDED/README.md.  Exits non-zero if any teacher drifts
by more than --tol.  imagenet-1k/resnet18 uses torchvision's pretrained
checkpoint rather than a local .pth.

A fourth, narrow subcommand `recover-mapping` exists to derive the WNID ->
class-index mapping for a manually-prepared ImageNet subset by reading off
the released teacher's predictions; needed only if you want to verify
imagenet-10/-100 with your own data layout.

Run from RDED/.
"""
import argparse
import collections
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile

import torch
import torch.nn as nn
import torchvision.models as thmodels
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets as tvdatasets

# Make `from synthesize.models import ConvNet` work when running this script
# from anywhere. `synthesize.models` is a flat module with no side effects.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from synthesize.models import ConvNet  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# (subset, arch, input_size, expected_top1, val_note) from RDED/README.md.
# val_note == "official-val" means the README value was measured on the
# official ImageNet validation set (50 images/class). When --data comes from
# the fastai imagenette2 / imagewoof2 tarballs the val split is drawn from
# ImageNet *train* data and is significantly easier, so positive drift
# (measured > expected) is expected and is NOT a model error.
TEACHERS = [
    ("cifar10",       "resnet18_modified", 32,  93.86, None),
    ("cifar10",       "conv3",             32,  82.24, None),
    ("cifar100",      "resnet18_modified", 32,  72.27, None),
    ("cifar100",      "conv3",             32,  61.27, None),
    ("tinyimagenet",  "resnet18_modified", 64,  61.98, None),
    ("tinyimagenet",  "conv4",             64,  49.73, None),
    ("imagenet-nette","resnet18",          224, 90.00, "official-val"),
    ("imagenet-nette","conv5",             128, 89.60, "official-val"),
    ("imagenet-woof", "resnet18",          224, 75.00, "official-val"),
    ("imagenet-woof", "conv5",             128, 67.40, "official-val"),
    ("imagenet-10",   "resnet18",          224, 87.40, "official-val"),
    ("imagenet-10",   "conv5",             128, 85.40, "official-val"),
    ("imagenet-100",  "resnet18",          224, 83.40, "official-val"),
    ("imagenet-100",  "conv6",             128, 72.82, "official-val"),
    ("imagenet-1k",   "conv4",             64,  43.60, None),
    # imagenet-1k resnet18 uses torchvision pretrained, ~69.76 top-1.
    ("imagenet-1k",   "resnet18",          224, 69.76, None),
]

NCLASS = {
    "cifar10": 10, "cifar100": 100, "tinyimagenet": 200,
    "imagenet-nette": 10, "imagenet-woof": 10, "imagenet-10": 10,
    "imagenet-100": 100, "imagenet-1k": 1000,
}

AUTO_DATASETS = ("cifar10", "cifar100", "tinyimagenet",
                 "imagenet-nette", "imagenet-woof")
MANUAL_DATASETS = ("imagenet-10", "imagenet-100", "imagenet-1k")

# RDED released models folder. URL is in the top-level README.md.
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1HmrheO6MgX453a5UPJdxPHK4UTv-4aVt"

# Recovered by predicting val images with the released teacher and reading
# off the modal predicted class (see `recover-mapping`). These are the only
# orderings that reproduce the README top-1 numbers.
IMAGENETTE_WNID_TO_IDX = {
    "n01440764": 0, "n02102040": 1, "n02979186": 2, "n03000684": 3,
    "n03028079": 4, "n03394916": 5, "n03417042": 6, "n03425413": 7,
    "n03445777": 8, "n03888257": 9,
}
IMAGEWOOF_WNID_TO_IDX = {
    "n02086240": 4, "n02087394": 6, "n02088364": 3, "n02089973": 5,
    "n02093754": 1, "n02096294": 0, "n02099601": 8, "n02105641": 9,
    "n02111889": 2, "n02115641": 7,
}

# Tiny-ImageNet WNID -> idx is parsed at runtime from prepare/tinyimagenet.md
# (the upstream-shipped mapping). 200 entries, lines like "00042: n02123394".
TINY_IMAGENET_MD = os.path.join(
    os.path.dirname(__file__), "..", "prepare", "tinyimagenet.md")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Model builder (replaces synthesize.utils.load_model for our 16 teachers)
# ─────────────────────────────────────────────────────────────────────────────

def build_teacher(arch, dataset, nclass, input_size):
    """Construct the architecture matching a released teacher checkpoint.

    The caller is responsible for loading the .pth state dict afterwards,
    except for imagenet-1k/resnet18 which returns a torchvision-pretrained
    model directly.
    """
    if dataset == "imagenet-1k" and arch == "resnet18":
        return thmodels.resnet18(weights="IMAGENET1K_V1")

    if arch.startswith("conv"):
        return ConvNet(
            num_classes=nclass, net_norm="batch", net_act="relu",
            net_pooling="avgpooling", net_depth=int(arch[-1]),
            net_width=128, channel=3, im_size=(input_size, input_size),
        )

    if arch == "resnet18_modified":
        # CIFAR/TinyImageNet variant: 3x3 stem, no maxpool. Build with the
        # right num_classes so the released .pth state dict loads cleanly.
        m = thmodels.resnet18(weights=None, num_classes=nclass)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m

    if arch == "resnet18":
        return thmodels.resnet18(weights=None, num_classes=nclass)

    raise ValueError(f"unknown arch {arch!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(url, dest):
    """Download `url` to `dest` with a single-line progress indicator."""
    print(f"  fetching {url}")
    last = [0]
    def _hook(blocks, bsz, total):
        if total <= 0:
            return
        done = min(100, int(100 * blocks * bsz / total))
        if done != last[0]:
            last[0] = done
            sys.stdout.write(f"\r  {done:3d}% ")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    sys.stdout.write("\n")


def _link_or_copy(src, dst, use_link):
    if os.path.exists(dst):
        return
    if use_link:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def _already_prepared(subset, data_root="./data"):
    return (os.path.isdir(f"{data_root}/{subset}/train") and
            os.path.isdir(f"{data_root}/{subset}/val"))


def _load_tinyimagenet_map():
    """Parse prepare/tinyimagenet.md for the WNID -> idx mapping."""
    import re
    pat = re.compile(r"^(\d{5}):\s*(n\d+)\s*$")
    mapping = {}
    with open(TINY_IMAGENET_MD) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                mapping[m.group(2)] = int(m.group(1))
    if len(mapping) != 200:
        raise SystemExit(
            f"expected 200 wnids in {TINY_IMAGENET_MD}, got {len(mapping)}")
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# `data` — download + prep datasets into 5-digit ImageFolder layout
# ─────────────────────────────────────────────────────────────────────────────

def _prep_cifar(subset, cache="./data/_torchvision_cache",
                data_root="./data"):
    if _already_prepared(subset, data_root):
        print(f"[{subset}] already prepared; skipping")
        return
    cls = tvdatasets.CIFAR10 if subset == "cifar10" else tvdatasets.CIFAR100
    n_classes = 10 if subset == "cifar10" else 100
    os.makedirs(cache, exist_ok=True)

    def _dump(ds, out_root):
        for c in range(n_classes):
            os.makedirs(f"{out_root}/{c:05d}", exist_ok=True)
        counts = [0] * n_classes
        for img, lab in ds:
            # CIFAR is 32x32. JPEG even at q=95 costs 1.6-3.1pp of teacher
            # top-1 (measured upstream); keep PNG.
            idx = counts[lab]
            img.save(f"{out_root}/{lab:05d}/img_{idx:05d}.png", optimize=False)
            counts[lab] += 1
        print(f"  wrote {sum(counts)} images to {out_root}")

    print(f"[{subset}] downloading + converting train split...")
    _dump(cls(cache, train=True, download=True), f"{data_root}/{subset}/train")
    print(f"[{subset}] downloading + converting val split...")
    _dump(cls(cache, train=False, download=True), f"{data_root}/{subset}/val")


def _prep_tinyimagenet(use_link, data_root="./data"):
    if _already_prepared("tinyimagenet", data_root):
        print("[tinyimagenet] already prepared; skipping")
        return
    raw_root = f"{data_root}/_raw"
    raw = f"{raw_root}/tiny-imagenet-200"
    os.makedirs(raw_root, exist_ok=True)
    if not os.path.isdir(raw):
        archive = f"{raw_root}/tiny-imagenet-200.zip"
        _fetch("http://cs231n.stanford.edu/tiny-imagenet-200.zip", archive)
        print("  extracting...")
        with zipfile.ZipFile(archive) as z:
            z.extractall(raw_root)
        os.remove(archive)

    wnid2idx = _load_tinyimagenet_map()
    out = f"{data_root}/tinyimagenet"
    os.makedirs(f"{out}/train", exist_ok=True)
    os.makedirs(f"{out}/val", exist_ok=True)

    n_train = 0
    for wnid in os.listdir(f"{raw}/train"):
        if wnid not in wnid2idx:
            raise SystemExit(f"unknown wnid {wnid} in train")
        dst = f"{out}/train/{wnid2idx[wnid]:05d}"
        os.makedirs(dst, exist_ok=True)
        img_dir = f"{raw}/train/{wnid}/images"
        for fn in os.listdir(img_dir):
            _link_or_copy(f"{img_dir}/{fn}", f"{dst}/{fn}", use_link)
            n_train += 1
    print(f"  train: {n_train} images -> {out}/train")

    for idx in range(200):
        os.makedirs(f"{out}/val/{idx:05d}", exist_ok=True)
    n_val = 0
    with open(f"{raw}/val/val_annotations.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            fn, wnid = parts[0], parts[1]
            if wnid not in wnid2idx:
                raise SystemExit(f"unknown wnid {wnid} in val_annotations")
            dst = f"{out}/val/{wnid2idx[wnid]:05d}/{fn}"
            _link_or_copy(f"{raw}/val/images/{fn}", dst, use_link)
            n_val += 1
    print(f"  val:   {n_val} images -> {out}/val")


def _prep_fastai_subset(subset, archive_basename, wnid2idx, use_link,
                        data_root="./data"):
    """Common path for imagenet-nette / imagenet-woof from fastai tarballs."""
    if _already_prepared(subset, data_root):
        print(f"[{subset}] already prepared; skipping")
        return
    raw_root = f"{data_root}/_raw"
    raw = f"{raw_root}/{archive_basename}"
    os.makedirs(raw_root, exist_ok=True)
    if not os.path.isdir(raw):
        archive = f"{raw_root}/{archive_basename}.tgz"
        _fetch(
            f"https://s3.amazonaws.com/fast-ai-imageclas/{archive_basename}.tgz",
            archive)
        print("  extracting...")
        with tarfile.open(archive) as t:
            t.extractall(raw_root)
        os.remove(archive)

    nclass = len(wnid2idx)
    out = f"{data_root}/{subset}"
    for split in ("train", "val"):
        src = f"{raw}/{split}"
        if not os.path.isdir(src):
            print(f"  {src} missing; skipping {split}")
            continue
        for idx in range(nclass):
            os.makedirs(f"{out}/{split}/{idx:05d}", exist_ok=True)
        n = 0
        for wnid in os.listdir(src):
            if wnid not in wnid2idx:
                continue
            dst_dir = f"{out}/{split}/{wnid2idx[wnid]:05d}"
            for fn in os.listdir(f"{src}/{wnid}"):
                _link_or_copy(f"{src}/{wnid}/{fn}", f"{dst_dir}/{fn}", use_link)
                n += 1
        print(f"  {split}: {n} images -> {out}/{split}")


def _manual_imagenet_instructions(subset):
    """Print setup instructions for the imagenet subsets that require a local
    copy of the full ImageNet archive, then return non-zero exit code."""
    nclass = NCLASS[subset]
    print(f"\n!!! {subset} cannot be auto-downloaded.", file=sys.stderr)
    if subset == "imagenet-1k":
        print(f"""\
1. Obtain ImageNet-1k (ILSVRC 2012) and arrange it as:
     ./data/_raw/imagenet-1k/{{train,val}}/<wnid>/<images>
2. Provide your own WNID -> 5-digit-index mapping (1000 entries) and copy /
   symlink into the 5-digit layout under ./data/imagenet-1k/.""",
              file=sys.stderr)
    else:
        print(f"""\
1. Obtain the raw {subset} split from full ImageNet, arranged as:
     ./data/_raw/{subset}/{{train,val}}/<wnid>/<images>
2. Recover the WNID -> class-index mapping from the released checkpoint:
     python scripts/teachers.py recover-mapping --subset {subset} \\
         --src ./data/_raw/{subset}/val --arch resnet18 \\
         --input-size 224 --nclass {nclass} \\
         --out /tmp/{subset}_mapping.json
3. Copy/symlink raw images into ./data/{subset}/{{train,val}}/<idx:05d>/
   using the mapping produced in step 2.""", file=sys.stderr)
    return 1


def cmd_data(args):
    use_link = not args.copy
    targets = [args.dataset] if args.dataset else list(AUTO_DATASETS)
    failed = 0
    for ds in targets:
        print(f"\n=== {ds} ===")
        if ds in ("cifar10", "cifar100"):
            _prep_cifar(ds)
        elif ds == "tinyimagenet":
            _prep_tinyimagenet(use_link)
        elif ds == "imagenet-nette":
            _prep_fastai_subset(ds, "imagenette2", IMAGENETTE_WNID_TO_IDX, use_link)
        elif ds == "imagenet-woof":
            _prep_fastai_subset(ds, "imagewoof2", IMAGEWOOF_WNID_TO_IDX, use_link)
        elif ds in MANUAL_DATASETS:
            failed += _manual_imagenet_instructions(ds)
        else:
            print(f"unknown dataset {ds!r}", file=sys.stderr)
            failed += 1

    if not args.dataset:
        # When running the default (all auto), summarize manual remainders.
        print()
        for ds in MANUAL_DATASETS:
            print(f"!!! {ds}: needs manual setup — "
                  f"`python scripts/teachers.py data --dataset {ds}` for instructions")
    return 1 if failed else 0


# ─────────────────────────────────────────────────────────────────────────────
# `models` — download teacher checkpoints from the RDED Google Drive folder
# ─────────────────────────────────────────────────────────────────────────────

def _expected_pth_files(dataset_filter=None, arch_filter=None):
    """Return the list of `{subset}_{arch}.pth` filenames we expect to have
    locally, optionally filtered by dataset/arch. imagenet-1k/resnet18 is
    skipped — it loads from torchvision, no .pth needed."""
    out = []
    for subset, arch, *_ in TEACHERS:
        if subset == "imagenet-1k" and arch == "resnet18":
            continue
        if dataset_filter and subset != dataset_filter:
            continue
        if arch_filter and arch != arch_filter:
            continue
        out.append(f"{subset}_{arch}.pth")
    return out


def cmd_models(args):
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    expected = _expected_pth_files(args.dataset, args.arch)
    missing = [fn for fn in expected if not os.path.exists(f"{out_dir}/{fn}")]
    if not missing:
        print(f"all {len(expected)} .pth files already present in {out_dir}, "
              "skipping")
        return 0

    try:
        import gdown
    except ImportError:
        print("error: `gdown` is required for model download. "
              "Install with: pip install gdown", file=sys.stderr)
        return 2

    # gdown.download_folder fetches the whole folder — there's no clean
    # per-file API for a public Drive folder without recording individual
    # file IDs. We download into a scratch dir, then move only the files we
    # need (and only those still missing).
    import tempfile
    with tempfile.TemporaryDirectory(prefix="rded_models_") as tmp:
        print(f"downloading folder -> {tmp}")
        paths = gdown.download_folder(GDRIVE_FOLDER_URL, output=tmp,
                                      quiet=False, use_cookies=False)
        if not paths:
            print("error: gdown returned no files", file=sys.stderr)
            return 2

        # gdown may put files in a subdir of `tmp`; walk to find them.
        found = {}
        for root, _, files in os.walk(tmp):
            for fn in files:
                if fn.endswith(".pth"):
                    found[fn] = os.path.join(root, fn)

        moved = skipped = absent = 0
        for fn in expected:
            dst = f"{out_dir}/{fn}"
            if os.path.exists(dst):
                skipped += 1
                continue
            if fn not in found:
                print(f"  !! {fn} not found in downloaded folder")
                absent += 1
                continue
            shutil.move(found[fn], dst)
            print(f"  moved {fn} -> {dst}")
            moved += 1

    print(f"\nmoved {moved} new, kept {skipped} existing, "
          f"missing {absent} from drive folder.")
    return 1 if absent else 0


# ─────────────────────────────────────────────────────────────────────────────
# `verify` — eval each teacher's top-1 and diff against README
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(model, val_dir, input_size, nclass, batch_size, workers):
    tfm = T.Compose([
        T.Resize(input_size * 8 // 7, antialias=True),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    # 5-digit folder names (00000, 00001, ...) sort lexicographically into
    # the same order as numeric class IDs, so ImageFolder gives the right
    # label mapping.
    ds = ImageFolder(root=val_dir, transform=tfm)
    if len(ds.classes) != nclass:
        raise SystemExit(
            f"{val_dir}: expected {nclass} classes, found {len(ds.classes)}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total else 0.0


def cmd_verify(args):
    rows = []
    for subset, arch, size, expected, val_note in TEACHERS:
        if args.dataset and subset != args.dataset:
            continue
        if args.arch and arch != args.arch:
            continue

        ckpt = f"{args.ckpt_dir}/{subset}_{arch}.pth"
        val = f"./data/{subset}/val"
        uses_torchvision = subset == "imagenet-1k" and arch == "resnet18"
        if not uses_torchvision and not os.path.exists(ckpt):
            rows.append((subset, arch, "no ckpt", expected, None, val_note))
            continue
        if not os.path.isdir(val):
            rows.append((subset, arch, "no val", expected, None, val_note))
            continue
        model = None
        try:
            model = build_teacher(arch, subset, NCLASS[subset], size)
            if not uses_torchvision:
                ck = torch.load(ckpt, map_location="cpu", weights_only=False)
                model.load_state_dict(ck["model"])
            model = nn.DataParallel(model).cuda()
            top1 = _evaluate(model, val, size, NCLASS[subset],
                             args.batch, args.workers)
            rows.append((subset, arch, "ok", expected, top1, val_note))
        except Exception as e:
            rows.append((subset, arch, f"err: {type(e).__name__}: {e}",
                         expected, None, val_note))
        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache()

    if not rows:
        print("no teachers matched filters; nothing to do")
        return 0

    width = max(len(s) + len(a) + 1 for s, a, *_ in rows) + 2
    print(f"{'teacher'.ljust(width)}  expected   measured     diff   status")
    print("-" * (width + 38))
    flagged = 0
    official_val_positive = 0
    for subset, arch, status, expected, top1, val_note in rows:
        tag = f"{subset}/{arch}".ljust(width)
        if top1 is None:
            print(f"{tag}  {expected:6.2f}    --       --       {status}")
            continue
        diff = top1 - expected
        # Positive drift on official-val datasets using fastai-derived val
        # is expected (fastai val is drawn from ImageNet *train*, easier
        # than the official val) — don't flag.
        val_mismatch = val_note == "official-val" and diff > 0
        bad = abs(diff) > args.tol and not val_mismatch
        if bad:
            flagged += 1
        if val_mismatch:
            official_val_positive += 1
        note = ("  !! drift" if bad
                else "  (fastai-val>official)" if val_mismatch
                else "")
        print(f"{tag}  {expected:6.2f}    {top1:6.2f}   {diff:+6.2f}   ok{note}")

    print()
    n_eval = sum(1 for r in rows if r[4] is not None)
    print(f"Evaluated {n_eval} teachers, flagged {flagged} at |diff|>{args.tol}.")
    if official_val_positive:
        print(f"Note: {official_val_positive} teacher(s) marked "
              "'(fastai-val>official)' — README values were measured on the "
              "official ImageNet val set; positive drift is expected when "
              "using fastai-derived data.")
    return 1 if flagged else 0


# ─────────────────────────────────────────────────────────────────────────────
# `recover-mapping` — produce a WNID -> idx JSON for a manual ImageNet subset
# ─────────────────────────────────────────────────────────────────────────────

def cmd_recover_mapping(args):
    """Predict every val image of each WNID with the released teacher and
    read off the modal class. Used to recover the ordering imagenet-10 /
    imagenet-100 teachers were trained with."""
    tfm = T.Compose([
        T.Resize(args.input_size * 8 // 7, antialias=True),
        T.CenterCrop(args.input_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = ImageFolder(args.src, transform=tfm)
    if len(ds.classes) != args.nclass:
        raise SystemExit(
            f"expected {args.nclass} classes, found {len(ds.classes)}")
    wnids = ds.classes  # alphabetically sorted

    ckpt = f"{args.ckpt_dir}/{args.subset}_{args.arch}.pth"
    model = build_teacher(args.arch, args.subset, args.nclass, args.input_size)
    if not (args.subset == "imagenet-1k" and args.arch == "resnet18"):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
    model = nn.DataParallel(model).cuda().eval()

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    votes = collections.defaultdict(collections.Counter)
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            preds = model(x).argmax(1).cpu().tolist()
            for wnid_idx, pred in zip(y.tolist(), preds):
                votes[wnid_idx][pred] += 1

    mapping = {}
    purity = {}
    print(f"{'wnid':16} -> {'idx':>4}   purity   votes")
    for wnid_idx in range(args.nclass):
        wnid = wnids[wnid_idx]
        c = votes[wnid_idx]
        top_idx, top_count = c.most_common(1)[0]
        total = sum(c.values())
        mapping[wnid] = top_idx
        purity[wnid] = top_count / total
        print(f"{wnid:16} -> {top_idx:4d}   {purity[wnid]:6.2%}   "
              f"{dict(c.most_common(3))}")

    if len(set(mapping.values())) != args.nclass:
        n_dupes = args.nclass - len(set(mapping.values()))
        print(f"\nWARN: {n_dupes} class-index collision(s); predictions for "
              "some WNIDs are ambiguous (probably wrong arch or subset).")
    else:
        print("\nbijective mapping recovered.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"wnid_to_idx": mapping, "purity": purity}, f, indent=2)
        print(f"wrote {args.out}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("data", help="download + prep dataset(s)")
    pd.add_argument("--dataset", choices=AUTO_DATASETS + MANUAL_DATASETS,
                    help="prepare a single dataset (default: all auto-downloadable)")
    pd.add_argument("--copy", action="store_true",
                    help="copy files instead of symlinking (more disk)")

    pm = sub.add_parser("models", help="download teacher .pth files")
    pm.add_argument("--dataset", choices=sorted(NCLASS.keys()),
                    help="only keep .pth files for this dataset")
    pm.add_argument("--arch",
                    help="only keep .pth files for this arch")
    pm.add_argument("--out", default="./data/pretrain_models",
                    help="output dir (default ./data/pretrain_models)")

    pv = sub.add_parser("verify", help="evaluate each teacher vs README")
    pv.add_argument("--dataset", choices=sorted(NCLASS.keys()),
                    help="only verify teachers on this dataset")
    pv.add_argument("--arch", help="only verify teachers with this arch")
    pv.add_argument("--tol", type=float, default=1.0,
                    help="flag if |measured - expected| > tol (default 1.0)")
    pv.add_argument("--batch", type=int, default=256)
    pv.add_argument("--workers", type=int, default=4)
    pv.add_argument("--ckpt-dir", default="./data/pretrain_models",
                    help="directory containing {subset}_{arch}.pth files")

    pr = sub.add_parser("recover-mapping",
                        help="recover a WNID -> idx JSON for a manual subset")
    pr.add_argument("--src", required=True,
                    help="val/ root with <wnid>/ subdirs")
    pr.add_argument("--subset", required=True,
                    choices=["imagenet-nette", "imagenet-woof",
                             "imagenet-10", "imagenet-100"])
    pr.add_argument("--arch", required=True)
    pr.add_argument("--input-size", type=int, required=True)
    pr.add_argument("--nclass", type=int, required=True)
    pr.add_argument("--ckpt-dir", default="./data/pretrain_models")
    pr.add_argument("--batch", type=int, default=128)
    pr.add_argument("--workers", type=int, default=4)
    pr.add_argument("--out", help="write JSON mapping to this path")

    return p


def main():
    args = _build_parser().parse_args()
    if args.cmd == "data":
        return cmd_data(args)
    if args.cmd == "models":
        return cmd_models(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    if args.cmd == "recover-mapping":
        return cmd_recover_mapping(args)
    raise AssertionError(f"unhandled cmd {args.cmd!r}")


if __name__ == "__main__":
    sys.exit(main() or 0)
