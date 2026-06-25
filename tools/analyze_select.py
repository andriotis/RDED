"""Compare stock vs variance-aware selection (random/stratified/covmatch) — the H2 causal
readout for the variance-preserving synthesis intervention.

Reads one or more results JSONL files (default: every logs/results_select*.jsonl), recovers
the selection method from each row's exp_name (_sel<method> tag; untagged = stock), and prints,
per (dataset, arch, ipc), a method block with mean±std over seeds for the trust panel. The
question it answers: at FIXED ipc, do the trust metrics improve along the dose
stock -> random -> covmatch (which raises nc1_distilled toward real)?

Usage:
    python tools/analyze_select.py
    python tools/analyze_select.py logs/results_select_conv3.jsonl logs/results_select_conv4.jsonl
    python tools/analyze_select.py --arch conv3 --dataset cifar10 --ipc 10
"""

import argparse
import glob
import json
import sys
from collections import defaultdict

import numpy as np

METHOD_ORDER = ["stock", "random", "stratified", "covmatch", "momentmatch"]
# (key in diag, label, direction: +1 = higher better, -1 = lower better)
PANEL = [
    ("best_top1", "top1", +1),
    ("nc1", "nc1(stu/val)", 0),
    ("ece", "ece", -1),
    ("ece_ts", "ece_ts", -1),
    ("auroc_msp", "auroc_msp", +1),
    ("auroc_energy", "auroc_energy", +1),
    ("auroc_maha", "auroc_maha", +1),
    ("auroc_feat_norm", "auroc_featnorm", +1),
    ("oscr_msp", "oscr_msp", +1),
    ("fpr95_msp", "fpr95_msp", -1),
]


def method_of(exp_name):
    for m in ("covmatch", "stratified", "random", "momentmatch"):
        if f"_sel{m}" in (exp_name or ""):
            return m
    return "stock"


def load(paths):
    rows = []
    for p in paths:
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r.get("diag") and r.get("best_top1") is not None:
                        rows.append(r)
        except FileNotFoundError:
            pass
    return rows


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="results JSONL files (default: logs/results_select*.jsonl)")
    ap.add_argument("--arch", action="append", help="keep only these archs (repeatable)")
    ap.add_argument("--dataset", action="append", help="keep only these datasets (repeatable)")
    ap.add_argument("--ipc", action="append", type=int, help="keep only these ipc values (repeatable)")
    return ap.parse_args()


def main():
    args = parse_args()
    paths = args.paths or sorted(glob.glob("logs/results_select*.jsonl"))
    rows = load(paths)
    if not rows:
        sys.exit(f"no usable rows in {paths}")

    # Filter cells by the requested arch/dataset/ipc. If a value isn't present in the data,
    # report it (it's "not implemented yet") so the user knows the filter matched nothing.
    want_arch = set(args.arch) if args.arch else None
    want_dataset = set(args.dataset) if args.dataset else None
    want_ipc = set(args.ipc) if args.ipc else None
    if want_arch is not None:
        missing = want_arch - {r["arch"] for r in rows}
        if missing:
            print(f"# note: no rows for arch={sorted(missing)} — not implemented yet", file=sys.stderr)
    if want_dataset is not None:
        missing = want_dataset - {r["dataset"] for r in rows}
        if missing:
            print(f"# note: no rows for dataset={sorted(missing)} — not implemented yet", file=sys.stderr)
    if want_ipc is not None:
        missing = want_ipc - {r["ipc"] for r in rows}
        if missing:
            print(f"# note: no rows for ipc={sorted(missing)} — not implemented yet", file=sys.stderr)
    rows = [
        r for r in rows
        if (want_arch is None or r["arch"] in want_arch)
        and (want_dataset is None or r["dataset"] in want_dataset)
        and (want_ipc is None or r["ipc"] in want_ipc)
    ]
    if not rows:
        sys.exit("no rows match the requested --arch/--dataset/--ipc filters")
    # Dedup (dataset,arch,ipc,method,seed) keeping the last occurrence — shards (e.g. the
    # tiny/resnet18 _g*/_bf* files) sometimes log the same cell more than once, which would
    # otherwise double-count into the per-seed mean/std.
    dedup = {}
    for r in rows:
        key = (r["dataset"], r["arch"], r["ipc"], method_of(r.get("exp_name")), r.get("seed"))
        dedup[key] = r
    rows = list(dedup.values())

    # cell -> method -> metric -> [per-seed values]
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seeds = defaultdict(lambda: defaultdict(set))
    for r in rows:
        cell = (r["dataset"], r["arch"], r["ipc"])
        m = method_of(r.get("exp_name"))
        seeds[cell][m].add(r.get("seed"))
        d = dict(r["diag"])
        d.setdefault("best_top1", r["best_top1"])
        for key, _, _ in PANEL:
            if d.get(key) is not None:
                cells[cell][m][key].append(float(d[key]))

    def fmt(vals):
        if not vals:
            return "        -    "
        m = np.mean(vals)
        s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        return f"{m:7.3f}±{s:5.3f}"

    print(f"sources: {paths}\n")
    for cell in sorted(cells):
        dataset, arch, ipc = cell
        present = [m for m in METHOD_ORDER if cells[cell].get(m)]
        ns = {m: sorted(seeds[cell][m]) for m in present}
        print(f"### {dataset} / {arch} / ipc{ipc}   seeds={ {m: ns[m] for m in present} }")
        header = "metric           " + "".join(f"{m:>16s}" for m in present)
        print(header)
        for key, label, direction in PANEL:
            line = f"{label:16s} "
            means = []
            for m in present:
                vals = cells[cell][m].get(key, [])
                means.append(np.mean(vals) if vals else None)
                line += f"{fmt(vals):>16s}"
            # dose-response arrow: does the metric move monotonically the 'good' way
            # along stock->...->covmatch?
            if direction != 0 and all(v is not None for v in means) and len(means) >= 2:
                diffs = np.diff(means)
                good = np.all(diffs >= -1e-9) if direction > 0 else np.all(diffs <= 1e-9)
                delta = (means[-1] - means[0]) * direction
                tag = "  <== dose+" if (good and abs(means[-1] - means[0]) > 1e-6) else ""
                line += f"   Δ(last-first)*dir={delta:+.3f}{tag}"
            print(line)
        print()


if __name__ == "__main__":
    main()
