"""Per-OOD-set selector comparison — answers "is the covmatch trust gain SVHN-specific?".

Reads runs that logged the multi-OOD block (diag.ood.<set>.* from main.py --ood-sets) and, for
each (dataset, arch, ipc) cell and OOD set, reports each method's auroc_maha / auroc_msp and the
paired (same-seed) covmatch-minus-stock delta. A gain that holds on SVHN (far-OOD) but not DTD
(near-OOD textures) means the benefit is shift-type-dependent, not a universal OOD improvement.

Rows are deduped by (cell, method, seed), last wins (shard-safe).

Usage:
    python tools/analyze_multiood.py                       # default globs
    python tools/analyze_multiood.py logs/results_baseline_multiood.jsonl logs/results_floor_sweep.jsonl
"""

import glob
import json
import sys
from collections import defaultdict

import numpy as np

SCORES = ["auroc_maha", "auroc_msp"]


def method_of(exp_name):
    for m in ("covmatch", "stratified", "random", "momentmatch", "relmatch"):
        if exp_name and f"_sel{m}" in exp_name:
            return m
    return "stock"


def load(paths):
    rows = {}
    for p in paths:
        try:
            for line in open(p):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                diag = r.get("diag") or {}
                if "ood" not in diag:
                    continue
                key = (r["dataset"], r["arch"], r["ipc"], method_of(r.get("exp_name")), r.get("seed"))
                rows[key] = r
        except FileNotFoundError:
            pass
    return rows


def main():
    paths = sys.argv[1:] or (
        sorted(glob.glob("logs/results_baseline_multiood.jsonl"))
        + sorted(glob.glob("logs/results_floor_sweep.jsonl"))
        + sorted(glob.glob("logs/results_momentmatch.jsonl"))
    )
    rows = load(paths)
    if not rows:
        sys.exit(f"no rows with a multi-OOD diag.ood block in {paths}")

    # cell -> method -> seed -> {ood_set: {score: val}}
    idx = defaultdict(lambda: defaultdict(dict))
    ood_sets = set()
    for (ds, arch, ipc, m, seed), r in rows.items():
        ood = r["diag"]["ood"]
        ood_sets |= set(ood)
        idx[(ds, arch, ipc)][m][seed] = ood
    ood_sets = sorted(ood_sets)

    print(f"sources: {paths}\nOOD sets: {ood_sets}\n")
    for cell in sorted(idx):
        methods = sorted(idx[cell])
        print(f"### {cell[0]}/{cell[1]}/ipc{cell[2]}")
        for score in SCORES:
            print(f"  [{score}]")
            header = "    " + "method".ljust(16) + "".join(f"{o:>10}" for o in ood_sets)
            print(header)
            for m in methods:
                seeds = sorted(idx[cell][m])
                line = "    " + m.ljust(16)
                for o in ood_sets:
                    vals = [idx[cell][m][s][o].get(score) for s in seeds
                            if o in idx[cell][m][s] and idx[cell][m][s][o].get(score) is not None]
                    line += f"{np.mean(vals):>10.3f}" if vals else f"{'-':>10}"
                print(line)
            # paired covmatch - stock per OOD set
            if "covmatch" in idx[cell] and "stock" in idx[cell]:
                line = "    " + "Δ(cov-stock)".ljust(16)
                for o in ood_sets:
                    seeds = sorted(set(idx[cell]["covmatch"]) & set(idx[cell]["stock"]))
                    ds_ = []
                    for s in seeds:
                        a = idx[cell]["stock"][s].get(o, {}).get(score)
                        b = idx[cell]["covmatch"][s].get(o, {}).get(score)
                        if a is not None and b is not None:
                            ds_.append(b - a)
                    line += f"{np.mean(ds_):>+10.3f}" if ds_ else f"{'-':>10}"
                print(line)
        print()


if __name__ == "__main__":
    main()
