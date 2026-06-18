"""Merge + dedup selector result JSONL shards into one clean file.

Runs of the same (dataset, arch, ipc, seed, method) cell were sometimes logged across
multiple GPU-shard files (e.g. logs/results_select_tiny_resnet_{g1,g2,bf*}.jsonl, produced
by launching run_select_variants.sh on several GPUs at once). This keeps the LAST occurrence
per cell so the analysis tools don't double-count into the per-seed mean/std.

The dedup key is (dataset, arch, ipc, seed, method); `method` is recovered from the exp_name
_sel<method> tag (untagged = stock). Note: it does NOT distinguish otherwise-identical cells
that differ only in a non-path-keyed hyperparameter (e.g. momentmatch_mean_weight) — keep
those in separate files if you sweep them.

Usage:
    python tools/consolidate_results.py -o logs/results_select_tiny_resnet_all.jsonl \
        logs/results_select_tiny_resnet*.jsonl
"""

import argparse
import glob
import json


def method_of(exp_name):
    for m in ("covmatch", "stratified", "random", "momentmatch"):
        if exp_name and f"_sel{m}" in exp_name:
            return m
    return "stock"


def main():
    ap = argparse.ArgumentParser("consolidate_results")
    ap.add_argument("-o", "--out", required=True, help="output JSONL path")
    ap.add_argument("inputs", nargs="+", help="input JSONL paths / globs")
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        paths.extend(sorted(glob.glob(pat)))
    paths = [p for p in paths if p != args.out]

    seen, order, n_in = {}, [], 0
    for p in paths:
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    n_in += 1
                    key = (r.get("dataset"), r.get("arch"), r.get("ipc"),
                           r.get("seed"), method_of(r.get("exp_name")))
                    if key not in seen:
                        order.append(key)
                    seen[key] = r  # keep last occurrence
        except FileNotFoundError:
            pass

    with open(args.out, "w") as f:
        for key in order:
            f.write(json.dumps(seen[key]) + "\n")
    print(f"read {n_in} rows from {len(paths)} files -> {len(order)} unique cells -> {args.out}")


if __name__ == "__main__":
    main()
