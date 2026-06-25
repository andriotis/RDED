"""Why do covmatch's *crops* win beyond NC1? (the §2.4 momentmatch / §5 qddpp-frontier puzzle)

covmatch, momentmatch, and qddpp(beta>0) can reach near-identical aggregate within-class
geometry (NC1, effective rank) yet train students with different far-OOD detection. So the
selection's effect is not captured by the NC1 *summary* — the *specific crops* matter. This
tool measures, training-free, what distinguishes the crop sets the teacher actually selected,
along two axes the summary NC1 ignores:

  (1) DISCRIMINATIVE-DIRECTION SPREAD. The Mahalanobis far-OOD score reads within-class spread
      through the tied precision Sigma^{-1} fit on real train (the directions where ID is tight
      and OOD deviates). For a selected set with within-class covariance Sigma_W^{(S)} (on
      L2-normalized teacher features), we report
          trace(Sigma_W)                      — raw spread (the NC1 numerator, metric-blind),
          trace(Sigma_W . Sigma^{-1})         — spread measured in the *discriminative* metric,
          align = trace(Sigma_W Sigma^{-1}) / trace(Sigma_W)   — spread per unit raw variance.
      Two sets with equal raw spread (≈equal NC1) can place that spread along very different
      directions; `align` is high when the selected variance lies where the OOD score looks.

  (2) PER-CROP BOUNDARY PROPERTIES (means over the selected crops, teacher-side):
          entropy   H(softmax z)              — ambiguity,
          margin    z_(1) - z_(2)             — distance from the decision boundary,
          dist2mean ||phi~ - mu_y||           — distance from the real class prototype (tail-ness),
          ce        -log softmax(z)_y         — realism (teacher CE loss).
      Hypothesis: covmatch (pure volume) keeps more tail/boundary crops (higher entropy /
      dist2mean, smaller margin) than the confidence-biased qddpp(beta>0) or the prototype-
      seeking momentmatch — at matched NC1 — and that, not NC1, tracks far-OOD trust.

Usage:
    python tools/analyze_momentmatch_puzzle.py --subset tinyimagenet --arch-name conv4 --ipc 10 \
        --seed 42 --syn-leaf syn_data_seed42 \
        --methods stock,covmatch,momentmatch,qddpp:0.05,qddpp:0.1,qddpp:0.2,qddpp:0.3

Read-only; prints a comparison table (one row per selection). Compare the `align` / per-crop
columns against the students' far-OOD Maha-AUROC (logs/results_*) to see which property orders
with trust.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torch.utils.data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesize.utils import load_model
from validation.utils import (
    ImageFolder, seed_everything, make_loader_kwargs, eval_transform, DATASET_META,
)
from validation.diagnostics import collect_outputs, fit_mahalanobis, within_class_cov, _id_panel


def exp_name_for(spec, base):
    """Build the path-keyed exp_name for a method spec (matches argument.py).

    spec: 'stock' | 'covmatch' | 'momentmatch' | 'qddpp:<beta>[:<quality>]'.
    """
    parts = spec.split(":")
    method = parts[0]
    if method == "stock":
        return base
    e = f"{base}_sel{method}"
    if method == "qddpp":
        beta = float(parts[1]) if len(parts) > 1 else 0.0
        e += f"_b{beta:g}"
        quality = parts[2] if len(parts) > 2 else "confidence"
        if quality != "confidence":
            e += f"_q{quality}"
    return e


def build_args():
    p = argparse.ArgumentParser("analyze_momentmatch_puzzle")
    p.add_argument("--subset", required=True, choices=list(DATASET_META))
    p.add_argument("--arch-name", required=True)
    p.add_argument("--ipc", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--syn-leaf", type=str, default="syn_data_seed42")
    p.add_argument("--methods", type=str, default="stock,covmatch,momentmatch",
                   help="comma list; qddpp entries as qddpp:<beta>[:<quality>]")
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--mipc", type=int, default=300)
    p.add_argument("--num-crop", type=int, default=5)
    p.add_argument("--fit-ipc", type=int, default=50,
                   help="real-train images/class to fit the Mahalanobis precision (the metric)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--re-batch-size", type=int, default=256)
    args = p.parse_args()
    args.nclass, args.input_size, args.val_ipc = DATASET_META[args.subset]
    args.classes = range(args.nclass)
    args.base_exp = (f"{args.subset}_{args.arch_name}_f{args.factor}"
                     f"_mipc{args.mipc}_ipc{args.ipc}_cr{args.num_crop}")
    args.train_dir = f"./data/{args.subset}/train/"
    return args


def _loader(root, ipc, args):
    ds = ImageFolder(classes=range(args.nclass), ipc=ipc, mem=True, root=root,
                     transform=eval_transform(args.input_size))
    return torch.utils.data.DataLoader(
        ds, batch_size=args.re_batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, **make_loader_kwargs(args.seed))


def per_crop_stats(out, means_real):
    """Teacher-side per-crop means over the selected set (lower margin/CE, higher entropy/
    dist2mean = more boundary/tail)."""
    z = out["logits"].double()
    feats = F.normalize(out["features"].double(), dim=1)
    y = out["labels"].long()
    logp = torch.log_softmax(z, dim=1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=1)
    top2 = z.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    ce = -logp[torch.arange(len(y)), y]
    d2m = (feats - means_real[y]).norm(dim=1)
    return {"entropy": ent.mean().item(), "margin": margin.mean().item(),
            "ce": ce.mean().item(), "dist2mean": d2m.mean().item()}


def geom_stats(out, precision, nclass):
    """Aggregate within-class geometry: raw spread, discriminative-metric spread, alignment,
    eff-rank, NC1."""
    Sw = within_class_cov(out["features"], out["labels"], nclass)  # [D, D] float64, normalized
    tr_raw = torch.trace(Sw).item()
    tr_disc = torch.trace(Sw @ precision).item()
    fro2 = float((Sw * Sw).sum().item())
    effrank = tr_raw ** 2 / fro2 if fro2 > 0 else float("nan")
    nc1 = _id_panel(out, nclass).get("nc1")
    return {"nc1": nc1, "effrank": effrank, "tr_raw": tr_raw, "tr_disc": tr_disc,
            "align": tr_disc / tr_raw if tr_raw > 0 else float("nan")}


def main():
    args = build_args()
    seed_everything(args.seed)
    print(f"=> loading teacher '{args.arch_name}' for {args.subset}")
    teacher = load_model(model_name=args.arch_name, dataset=args.subset,
                         pretrained=True, classes=args.classes).cuda()
    teacher.eval()

    print(f"=> fitting Mahalanobis precision on real train (fit_ipc={args.fit_ipc})")
    fit_out = collect_outputs(teacher, _loader(args.train_dir, args.fit_ipc, args),
                              capture_features=True)
    means_real, precision = fit_mahalanobis(
        fit_out["features"], fit_out["labels"], args.nclass, normalize=True)

    specs = [s.strip() for s in args.methods.split(",") if s.strip()]
    rows = []
    for spec in specs:
        exp = exp_name_for(spec, args.base_exp)
        syn = f"./exp/{exp}/{args.syn_leaf}"
        if not os.path.isdir(syn):
            print(f"   [skip] {spec}: no distilled set at {syn}")
            continue
        out = collect_outputs(teacher, _loader(syn, args.ipc, args), capture_features=True)
        g = geom_stats(out, precision, args.nclass)
        c = per_crop_stats(out, means_real)
        rows.append((spec, {**g, **c}))

    hdr = (f"{'method':16s} {'NC1':>7s} {'effrank':>8s} {'tr(Sw)':>8s} {'tr(SwP)':>9s} "
           f"{'align':>7s} {'entropy':>8s} {'margin':>7s} {'d2mean':>7s} {'CE':>6s}")
    print("\n" + "=" * len(hdr))
    print(f"{args.subset}/{args.arch_name}/ipc{args.ipc}/seed{args.seed}  "
          f"(align = tr(Sw.Sigma^-1)/tr(Sw); higher => spread along discriminative dirs)")
    print(hdr)
    print("-" * len(hdr))
    for spec, m in rows:
        print(f"{spec:16s} {m['nc1']:7.3f} {m['effrank']:8.1f} {m['tr_raw']:8.4f} "
              f"{m['tr_disc']:9.2f} {m['align']:7.1f} {m['entropy']:8.3f} {m['margin']:7.3f} "
              f"{m['dist2mean']:7.3f} {m['ce']:6.3f}")
    print("=" * len(hdr) + "\n")


if __name__ == "__main__":
    main()
