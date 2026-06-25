import argparse
import os
import math

from validation.losses import LOSS_REGISTRY
from validation.utils import DATASET_META

parser = argparse.ArgumentParser("RDED")
"""Synthesis"""
parser.add_argument(
    "--arch-name",
    type=str,
    default="resnet18",
    help="arch name from pretrained torchvision models",
)
parser.add_argument(
    "--subset",
    type=str,
    default="imagenet-1k",
)
parser.add_argument(
    "--train-dir",
    type=str,
    default="../../data/imagenet-1k/train/",
    help="path to training dataset",
)
parser.add_argument(
    "--nclass",
    type=int,
    default=1000,
    help="number of classes for synthesis",
)
parser.add_argument(
    "--mipc",
    type=int,
    default=600,
    help="number of pre-loaded images per class",
)
parser.add_argument(
    "--ipc",
    type=int,
    default=50,
    help="number of images per class for synthesis",
)
parser.add_argument(
    "--num-crop",
    type=int,
    default=1,
    help="number of croped images for first scoring",
)
parser.add_argument(
    "--input-size",
    default=224,
    type=int,
    metavar="S",
)
parser.add_argument(
    "--factor",
    default=2,
    type=int,
)
"""Re Train"""
parser.add_argument("--re-batch-size", default=0, type=int, metavar="N")
parser.add_argument(
    "--re-accum-steps",
    type=int,
    default=1,
    help="gradient accumulation steps for small gpu memory",
)
parser.add_argument(
    "--mix-type",
    default="cutmix",
    type=str,
    choices=["mixup", "cutmix", None],
    help="mixup or cutmix or None",
)
parser.add_argument(
    "--stud-name",
    type=str,
    default="resnet18",
    help="arch name from torchvision models",
)
parser.add_argument(
    "--val-ipc",
    type=int,
    default=30,
)
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--classes",
    type=list,
    help="number of classes for synthesis",
)
parser.add_argument(
    "--temperature",
    type=float,
    help="temperature for distillation loss",
)
parser.add_argument(
    "--val-dir",
    type=str,
    default="../../data/imagenet-1k/val/",
    help="path to validation dataset",
)
parser.add_argument(
    "--min-scale-crops", type=float, default=0.08, help="argument in RandomResizedCrop"
)
parser.add_argument(
    "--max-scale-crops", type=float, default=1, help="argument in RandomResizedCrop"
)
parser.add_argument("--re-epochs", default=300, type=int)
parser.add_argument(
    "--syn-data-path",
    type=str,
    default="syn_data",
    help="where to store synthetic data",
)
parser.add_argument(
    "--seed", default=42, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--mixup",
    type=float,
    default=0.8,
    help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
)
parser.add_argument(
    "--cutmix",
    type=float,
    default=1.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
)
parser.add_argument("--cos", default=True, help="cosine lr scheduler")

# sgd
parser.add_argument("--sgd", default=False, action="store_true", help="sgd optimizer")
parser.add_argument(
    "-lr",
    "--learning-rate",
    type=float,
    default=0.1,
    help="sgd init learning rate",
)
parser.add_argument("--momentum", type=float, default=0.9, help="sgd momentum")
parser.add_argument("--weight-decay", type=float, default=1e-4, help="sgd weight decay")

# adamw
parser.add_argument("--adamw-lr", type=float, default=0, help="adamw learning rate")
parser.add_argument(
    "--adamw-weight-decay", type=float, default=0.01, help="adamw weight decay"
)
parser.add_argument(
    "--exp-name",
    type=str,
    help="name of the experiment, subfolder under syn_data_path",
)

# Student-loss weights: one --w-<name> flag per entry in LOSS_REGISTRY.
# Adding a new loss = adding a registry entry; argparse picks it up here.
for _name, (_, _default_w) in LOSS_REGISTRY.items():
    parser.add_argument(
        f"--w-{_name.replace('_', '-')}",
        type=float,
        default=_default_w,
        dest=f"w_{_name}",
        help=f"weight on the {_name.upper()} student-loss term (default {_default_w}; 0 disables)",
    )

# Per-loss hyperparameters (read by the corresponding term functions in losses.py).
parser.add_argument(
    "--gce-q",
    type=float,
    default=0.7,
    help="q for GCE; in (0, 1]. q->0 reduces to CE, q=1 is MAE (default 0.7)",
)
parser.add_argument(
    "--sce-log-floor",
    type=float,
    default=-4.0,
    help="log floor A for RCE: y is clamped to >= exp(A) before log (default -4.0, matches Wang et al.)",
)

parser.add_argument(
    "--skip-synth",
    action="store_true",
    help="skip synthesis and reuse existing syn_data (paired-protocol primitive: synth once, train N students)",
)

# Stage-1 selection (the variance-aware intervention). 'stock' = RDED top-confidence.
# The others inject within-class variance over a realism-floored pool at the same IPC.
parser.add_argument(
    "--select-method",
    type=str,
    default="stock",
    choices=["stock", "random", "stratified", "covmatch", "momentmatch", "qddpp"],
    help="crop selection: stock=top teacher-confidence (RDED); random/stratified/covmatch/"
         "momentmatch spread the selection across the class's feature spread at fixed IPC "
         "(covmatch maximizes feature volume; momentmatch matches the pool mean+covariance); "
         "qddpp = quality-diversity DPP that interpolates covmatch (--select-beta 0) and stock "
         "(--select-beta large) with one knob",
)
parser.add_argument(
    "--select-realism-floor",
    type=float,
    default=3.0,
    dest="select_realism_floor",
    help="eligible pool = ceil(this * n_select) most-confident candidates (>=1); a realism "
         "guard so variance-seeking selectors never pick teacher-garbage crops",
)
parser.add_argument(
    "--select-k",
    type=int,
    default=8,
    help="number of clusters for --select-method stratified",
)
parser.add_argument(
    "--momentmatch-mean-weight",
    type=float,
    default=1.0,
    dest="momentmatch_mean_weight",
    help="lambda weighting the mean term vs the covariance term in --select-method "
         "momentmatch's objective ||Sigma_S - Sigma_t||_F^2 + lambda*||mu_S - mu_t||^2",
)
parser.add_argument(
    "--select-beta",
    type=float,
    default=0.0,
    dest="select_beta",
    help="--select-method qddpp quality<->diversity knob: q_i = exp(beta * standardized "
         "confidence). beta=0 reproduces covmatch (pure feature volume); large beta collapses "
         "onto the top-quality crops (= stock when quality=confidence)",
)
parser.add_argument(
    "--select-quality",
    type=str,
    default="confidence",
    dest="select_quality",
    choices=["confidence", "margin"],
    help="--select-method qddpp quality score: confidence = teacher CE loss (far-OOD / "
         "calibration lever); margin = top1-top2 logit gap, up-weighting boundary crops "
         "(the near-OOD lever)",
)
parser.add_argument(
    "--synth-only",
    action="store_true",
    help="run synthesis and exit; skip training (used by diagnose.sh pre-pass)",
)

# Trustworthiness diagnostics (NC + calibration + open-set). Off by default so
# normal sweeps are untouched; when on, runs once post-training on the student.
parser.add_argument(
    "--diagnostics",
    action="store_true",
    help="after training, measure ECE + OSCR/AUROC/FPR95 (vs OOD) + NC on the student "
         "and log them under the 'diag' column of results.jsonl",
)
parser.add_argument(
    "--ood-dataset",
    type=str,
    default="svhn",
    choices=["svhn"],
    help="OOD negatives for open-set metrics (currently only 'svhn')",
)
parser.add_argument(
    "--ood-data-path",
    type=str,
    default="",
    help="override download/cache dir for the OOD dataset (default ./data/_torchvision_cache)",
)
parser.add_argument(
    "--fit-ipc",
    type=int,
    default=50,
    help="images/class drawn from the real TRAIN split (held out from the student) to fit "
         "the Mahalanobis class stats and the calibration temperature for --diagnostics",
)
parser.add_argument(
    "--ood-sets",
    type=str,
    default="",
    help="comma-separated OOD sets for --diagnostics (e.g. svhn,dtd,cifar10); empty falls "
         "back to --ood-dataset. Per-set open-set metrics are logged under diag.ood.<name>.*, "
         "with the first set mirrored at the top level of diag",
)
parser.add_argument(
    "--save-student",
    action="store_true",
    dest="save_student",
    help="save the trained student state_dict next to the distilled set "
         "(student_<syn_leaf>.pth), enabling --diagnostics-only re-evaluation",
)
parser.add_argument(
    "--diagnostics-only",
    action="store_true",
    dest="diagnostics_only",
    help="skip synthesis+training; load a --save-student checkpoint and run the trust panel "
         "only (training-free re-eval of new OOD sets/metrics; training is deterministic)",
)
parser.add_argument(
    "--results-file",
    type=str,
    default=None,
    help="override results JSONL path (default ./logs/results.jsonl); useful for smoke runs",
)

# Aim tracking (augments results.jsonl; does not replace it).
parser.add_argument(
    "--aim-repo",
    type=str,
    default="./logs/aim",
    help="Aim repo path (per run: experiment={exp_name}, run.name={run_tag})",
)
parser.add_argument(
    "--run-tag",
    type=str,
    default="",
    help="Aim run name. Empty -> derived from seed + nonzero loss weights",
)
parser.add_argument(
    "--sweep-name",
    type=str,
    default="",
    help="sweep identifier (YAML stem); recorded in Aim hparams for cross-run grouping",
)
parser.add_argument(
    "--cell-id",
    type=str,
    default="",
    help="sweep cell identifier (e.g. row index); recorded in Aim hparams",
)
parser.add_argument(
    "--disable-aim",
    action="store_true",
    help="skip Aim logging entirely (results.jsonl still written)",
)
parser.add_argument(
    "--monitor",
    type=str,
    default="",
    help="comma-separated LOSS_REGISTRY names to compute under no_grad and log "
         "without contributing to backprop. Names that are also active "
         "(w_<name> > 0) are deduplicated and treated as active only.",
)
args = parser.parse_args()

args.train_dir = f"./data/{args.subset}/train/"
args.val_dir = f"./data/{args.subset}/val/"

# set up dataset settings
# set smaller val_ipc only for quick validation
if args.subset in [
    "imagenet-a",
    "imagenet-b",
    "imagenet-c",
    "imagenet-d",
    "imagenet-e",
    "imagenet-birds",
    "imagenet-fruits",
    "imagenet-cats",
    "imagenet-10",
]:
    args.nclass = 10
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 224

elif args.subset == "imagenet-nette":
    args.nclass = 10
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 224
    if args.arch_name in ["conv5", "conv6"] or args.stud_name in ["conv5", "conv6"]:
        args.input_size = 128

elif args.subset == "imagenet-woof":
    args.nclass = 10
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 224
    if args.arch_name in ["conv5", "conv6"] or args.stud_name in ["conv5", "conv6"]:
        args.input_size = 128

elif args.subset == "imagenet-100":
    args.nclass = 100
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 224
    if args.arch_name in ["conv5", "conv6"] or args.stud_name in ["conv5", "conv6"]:
        args.input_size = 128

elif args.subset == "imagenet-1k":
    args.nclass = 1000
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 224

elif args.subset in DATASET_META:
    args.nclass, args.input_size, args.val_ipc = DATASET_META[args.subset]
    args.classes = range(args.nclass)

args.nclass = len(args.classes)

# set up batch size
if args.re_batch_size == 0:
    if args.ipc == 50:
        args.re_batch_size = 100
        args.workers = 4
    elif args.ipc == 10:
        args.re_batch_size = 50
        args.workers = 4
    elif args.ipc == 1:
        args.re_batch_size = 10
        args.workers = 0

    if args.nclass == 10:
        args.re_batch_size *= 1
    if args.nclass == 100:
        args.re_batch_size *= 2
    if args.nclass == 1000:
        args.re_batch_size *= 2

    # ! tinyimagenet
    if args.subset == "tinyimagenet":
        args.re_batch_size = 100

# reset batch size below ipc * nclass
if args.re_batch_size > args.ipc * args.nclass:
    args.re_batch_size = int(args.ipc * args.nclass)

# reset batch size with re_accum_steps
if args.re_accum_steps != 1:
    args.re_batch_size = int(args.re_batch_size / args.re_accum_steps)

# result dir for saving
args.exp_name = f"{args.subset}_{args.arch_name}_f{args.factor}_mipc{args.mipc}_ipc{args.ipc}_cr{args.num_crop}"
# Key the distilled-set path by the selection method so non-stock variants are stored
# (and synth-cached) separately and never clobber or get falsely reused as the stock set.
if getattr(args, "select_method", "stock") != "stock":
    args.exp_name += f"_sel{args.select_method}"
    # qddpp is a one-knob family: each beta (and a non-default quality score) is a distinct
    # distilled set, so path-key both to keep them separately synth-cached.
    if args.select_method == "qddpp":
        args.exp_name += f"_b{getattr(args, 'select_beta', 0.0):g}"
        if getattr(args, "select_quality", "confidence") != "confidence":
            args.exp_name += f"_q{args.select_quality}"
    # Path-key a non-default realism floor too, so floor variants of a selector are stored
    # (and synth-cached) separately and never clobber the floor=3.0 baseline set.
    _fl = getattr(args, "select_realism_floor", 3.0)
    if abs(_fl - 3.0) > 1e-9:
        args.exp_name += f"_fl{_fl:g}"
if not os.path.exists(f"./exp/{args.exp_name}"):
    os.makedirs(f"./exp/{args.exp_name}")
args.syn_data_path = os.path.join("./exp/" + args.exp_name, args.syn_data_path)

# temperature
if args.mix_type == "mixup":
    args.temperature = 4
elif args.mix_type == "cutmix":
    args.temperature = 20

# adamw learning rate
if args.stud_name == "vgg11":
    args.adamw_lr = 0.0005
elif args.stud_name == "conv3":
    args.adamw_lr = 0.001
elif args.stud_name == "conv4":
    args.adamw_lr = 0.001
elif args.stud_name == "conv5":
    args.adamw_lr = 0.001
elif args.stud_name == "conv6":
    args.adamw_lr = 0.001
elif args.stud_name == "resnet18":
    args.adamw_lr = 0.001
elif args.stud_name == "resnet18_modified":
    args.adamw_lr = 0.001
elif args.stud_name == "efficientnet_b0":
    args.adamw_lr = 0.002
elif args.stud_name == "mobilenet_v2":
    args.adamw_lr = 0.0025
elif args.stud_name == "alexnet":
    args.adamw_lr = 0.0001
elif args.stud_name == "resnet50":
    args.adamw_lr = 0.001
elif args.stud_name == "resnet101":
    args.adamw_lr = 0.001
elif args.stud_name == "resnet101_modified":
    args.adamw_lr = 0.001
elif args.stud_name == "vit_b_16":
    args.adamw_lr = 0.0001
elif args.stud_name == "swin_v2_t":
    args.adamw_lr = 0.0001

# special experiment
if (
    args.subset == "cifar100"
    and args.arch_name == "conv3"
    and args.stud_name == "conv3"
):
    args.re_batch_size = 25
    args.adamw_lr = 0.002

# Sanity: at least one student-loss term must be active.
if not any(getattr(args, f"w_{n}") > 0 for n in LOSS_REGISTRY):
    raise SystemExit(
        "no active student-loss term: set at least one --w-<name> > 0 "
        f"(registered losses: {', '.join(LOSS_REGISTRY)})"
    )
