"""Teacher-side geometry diagnostic (RDED synthesis conditioning).

Runs in minutes on the pretrained teacher alone — no training. For one
(subset, arch, ipc) it answers the two motivating questions for the
geometry-targeted synthesis idea:

  H1 (synthesis geometry): is the teacher-induced feature geometry of the
      *distilled set* worse-conditioned than that of a matched *real* subset?
      -> compares NC1/NC2/NC3 of teacher features on syn_data vs real train.

  reference for H2: how calibrated / OOD-robust is the real-data teacher itself?
      -> ECE + OSCR/AUROC/FPR95 of the teacher on real val vs SVHN. This is the
         well-conditioned reference the RDED-trained student (logged separately
         via `main.py --diagnostics`) is contrasted against.

Usage:
    python tools/diagnose_geometry.py --subset cifar100 --arch-name resnet18_modified --ipc 10

Appends one JSON object per subject to logs/diagnostics.jsonl and prints a table.
"""

import argparse
import fcntl
import json
import os
import sys
import time

import torch
import torch.utils.data

# Allow running as `python tools/diagnose_geometry.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesize.utils import load_model
from validation.utils import (
    ImageFolder,
    seed_everything,
    make_loader_kwargs,
    eval_transform,
    DATASET_META,
)
from validation.diagnostics import (
    build_svhn_ood_loader,
    feature_geometry,
    run_diagnostics,
)


def build_args():
    p = argparse.ArgumentParser("diagnose_geometry")
    p.add_argument("--subset", required=True, choices=list(DATASET_META))
    p.add_argument("--arch-name", required=True, help="teacher arch (e.g. resnet18_modified, conv3)")
    p.add_argument("--ipc", type=int, required=True, help="images-per-class of the distilled set to inspect")
    # exp_name components — defaults match the distilled sets already on disk.
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--mipc", type=int, default=300)
    p.add_argument("--num-crop", type=int, default=5)
    p.add_argument("--syn-leaf", type=str, default="syn_data",
                   help="synth dir leaf under exp/<exp_name>/ (default 'syn_data'; "
                        "use 'syn_data_seed<N>' to inspect a sweep-produced set)")
    p.add_argument("--real-ipc", type=int, default=0, help="images/class for the real-train reference subset (0 = match --ipc)")
    p.add_argument("--select-method", type=str, default="stock",
                   choices=["stock", "random", "stratified", "covmatch"],
                   help="inspect a variance-aware distilled set (tags exp_name with _sel<method>, "
                        "matching argument.py); 'stock' is the plain RDED set")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--re-batch-size", type=int, default=256)
    p.add_argument("--ood-data-path", type=str, default="")
    p.add_argument("--ood-max", type=int, default=10000)
    p.add_argument("--results-file", type=str, default="./logs/diagnostics.jsonl")
    args = p.parse_args()

    args.nclass, args.input_size, args.val_ipc = DATASET_META[args.subset]
    args.classes = range(args.nclass)
    if args.real_ipc == 0:
        args.real_ipc = args.ipc
    args.exp_name = (
        f"{args.subset}_{args.arch_name}_f{args.factor}"
        f"_mipc{args.mipc}_ipc{args.ipc}_cr{args.num_crop}"
    )
    if args.select_method != "stock":
        args.exp_name += f"_sel{args.select_method}"
    args.syn_data_path = f"./exp/{args.exp_name}/{args.syn_leaf}"
    args.train_dir = f"./data/{args.subset}/train/"
    args.val_dir = f"./data/{args.subset}/val/"
    return args


def _loader(root, ipc, args):
    ds = ImageFolder(
        classes=range(args.nclass),
        ipc=ipc,
        mem=True,
        root=root,
        transform=eval_transform(args.input_size),
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=args.re_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        **make_loader_kwargs(args.seed),
    )


def _fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    args = build_args()
    seed_everything(args.seed)

    if not os.path.isdir(args.syn_data_path):
        raise SystemExit(
            f"distilled set not found: {args.syn_data_path}\n"
            "synthesize it first (python main.py ...) or check --factor/--mipc/--num-crop/--ipc."
        )

    print(f"=> loading teacher '{args.arch_name}' for {args.subset}")
    teacher = load_model(
        model_name=args.arch_name,
        dataset=args.subset,
        pretrained=True,
        classes=args.classes,
    ).cuda()
    teacher.eval()

    print("=> teacher-induced geometry on distilled set vs real-train subset")
    distilled_loader = _loader(args.syn_data_path, args.ipc, args)
    real_loader = _loader(args.train_dir, args.real_ipc, args)
    geo_distilled = feature_geometry(teacher, distilled_loader, args.nclass)
    geo_real = feature_geometry(teacher, real_loader, args.nclass)

    print("=> teacher reference: calibration + open-set on real val vs SVHN")
    val_loader = _loader(args.val_dir, args.val_ipc, args)
    ood_loader = build_svhn_ood_loader(args, max_samples=args.ood_max)
    ref = run_diagnostics(teacher, val_loader, ood_loader, args.nclass)

    rows = [
        ("distilled-set geometry (teacher feats)", geo_distilled),
        (f"real-train subset geometry (n/cls={args.real_ipc})", geo_real),
        ("teacher reference (real val vs SVHN)", ref),
    ]

    # --- table ---
    hdr = f"{'subject':40s} {'top1':>6s} {'NC1':>8s} {'NC2':>7s} {'NC3':>7s} {'ECE':>7s} {'OSCR':>7s} {'AUROC':>7s} {'FPR95':>7s}"
    print("\n" + "=" * len(hdr))
    print(f"{args.subset} / {args.arch_name} / ipc{args.ipc}")
    print(hdr)
    print("-" * len(hdr))
    for subject, m in rows:
        print(
            f"{subject:40s} {_fmt(m.get('top1'),2):>6s} "
            f"{_fmt(m.get('nc1')):>8s} {_fmt(m.get('nc2')):>7s} {_fmt(m.get('nc3')):>7s} "
            f"{_fmt(m.get('ece')):>7s} {_fmt(m.get('oscr_msp')):>7s} "
            f"{_fmt(m.get('auroc_msp')):>7s} {_fmt(m.get('fpr95_msp')):>7s}"
        )
    print("=" * len(hdr) + "\n")

    # --- append to logs/diagnostics.jsonl ---
    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    base = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "subset": args.subset,
        "arch": args.arch_name,
        "ipc": args.ipc,
        "real_ipc": args.real_ipc,
        "factor": args.factor,
        "mipc": args.mipc,
        "num_crop": args.num_crop,
        "seed": args.seed,
    }
    # Parallel diagnose.sh geom runs append concurrently; hold an exclusive lock
    # across the whole block so one run's rows can't interleave another's.
    payload = "".join(
        json.dumps({**base, "subject": subject, **m}) + "\n" for subject, m in rows
    )
    with open(args.results_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(payload)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"Logged {len(rows)} rows to {args.results_file}")


if __name__ == "__main__":
    main()
