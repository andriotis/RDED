"""Paired (same-seed) selector comparison vs stock — the significance companion to
analyze_select.py.

analyze_select.py prints per-method mean±std over seeds; with 3 seeds those error bars
overlap even when a selector helps, because each seed carries its own synthesis+student-init
noise. Pairing on the seed cancels that shared nuisance: for each (dataset, arch, ipc) cell,
metric, and variant method, this reports the per-seed difference

    Delta_s = direction * ( metric(variant, seed=s) - metric(stock, seed=s) )

(direction flips lower-is-better metrics so +Delta always means "better than stock"), its
mean, a bootstrap 95% CI, and the sign count. A CI that excludes 0 is flagged with '*'.
With only ~3 seeds the bootstrap is illustrative, so the raw per-seed Deltas are printed too.

Rows are deduped by (cell, method, seed), keeping the last occurrence (shard-safe).

Usage:
    python tools/analyze_paired.py                                  # globs logs/results_select*.jsonl
    python tools/analyze_paired.py logs/results_momentmatch.jsonl logs/results_select_conv3.jsonl
"""

import glob
import json
import sys
from collections import defaultdict

import numpy as np

METHODS = ["random", "stratified", "covmatch", "momentmatch", "relmatch"]
# (key, label, direction: +1 higher-better, -1 lower-better)
PANEL = [
    ("best_top1", "top1", +1),
    ("ece", "ece", -1),
    ("ece_ts", "ece_ts", -1),
    ("auroc_maha", "auroc_maha", +1),
    ("auroc_energy", "auroc_energy", +1),
    ("auroc_feat_norm", "auroc_featnorm", +1),
    ("auroc_msp", "auroc_msp", +1),
    ("oscr_msp", "oscr_msp", +1),
    ("fpr95_msp", "fpr95_msp", -1),
]


def method_of(exp_name):
    for m in ("covmatch", "stratified", "random", "momentmatch", "relmatch"):
        if exp_name and f"_sel{m}" in exp_name:
            return m
    return "stock"


def load(paths):
    """Dedup (dataset,arch,ipc,method,seed) -> row, keeping the last occurrence."""
    rows = {}
    for p in paths:
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if not r.get("diag"):
                        continue
                    key = (r["dataset"], r["arch"], r["ipc"],
                           method_of(r.get("exp_name")), r.get("seed"))
                    rows[key] = r
        except FileNotFoundError:
            pass
    return rows


def metric_val(row, key):
    d = dict(row["diag"])
    d.setdefault("best_top1", row.get("best_top1"))
    return d.get(key)


def boot_ci(deltas, n_boot=10000, seed=0):
    d = np.asarray(deltas, dtype=float)
    if d.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    paths = sys.argv[1:] or sorted(glob.glob("logs/results_select*.jsonl"))
    rows = load(paths)
    if not rows:
        sys.exit(f"no usable rows in {paths}")

    idx = defaultdict(lambda: defaultdict(dict))  # cell -> method -> seed -> row
    for (ds, arch, ipc, m, seed), r in rows.items():
        idx[(ds, arch, ipc)][m][seed] = r

    print(f"sources: {paths}\n")
    for cell in sorted(idx):
        stock = idx[cell].get("stock", {})
        if not stock:
            continue
        print(f"### {cell[0]} / {cell[1]} / ipc{cell[2]}   (paired vs stock; stock seeds={sorted(stock)})")
        for m in METHODS:
            mr = idx[cell].get(m, {})
            seeds = sorted(set(stock) & set(mr))
            if not seeds:
                continue
            print(f"  -- {m}  (n={len(seeds)} paired seeds)")
            for key, label, direction in PANEL:
                deltas = []
                for s in seeds:
                    a, b = metric_val(stock[s], key), metric_val(mr[s], key)
                    if a is None or b is None:
                        continue
                    deltas.append(direction * (b - a))
                if not deltas:
                    continue
                mean = float(np.mean(deltas))
                lo, hi = boot_ci(deltas)
                npos = sum(1 for x in deltas if x > 0)
                sig = "*" if (lo > 0 or hi < 0) else " "
                raw = "[" + ", ".join(f"{x:+.4f}" for x in deltas) + "]"
                print(f"     {label:14s} meanDelta={mean:+.4f}  CI95[{lo:+.4f},{hi:+.4f}] {sig}"
                      f"  sign={npos}/{len(deltas)}  raw={raw}")
        print()


if __name__ == "__main__":
    main()
