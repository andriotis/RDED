#!/usr/bin/env python
"""Phase B.3 cross-architecture analysis: pivot over loss for arch != stud cells.

Reads logs/results.jsonl, filters to rows where arch != stud, classifies each
row into a loss column (KL or KL+OCKL), aggregates by (dataset, arch, stud, ipc, loss),
and prints:
  1. Per-dataset best-top1 table (rows = (arch→stud, ipc), cols = loss settings)
  2. Per-dataset NC1 table
  3. Δ table: kl+ockl mean minus kl mean per cell — headline number for the
     architecture-invariance hypothesis (positive Δ supports the claim).

Treats student_loss="kl" or "kd" with w_ockl=0 as the KL baseline.
Treats student_loss="kl+ockl" or w_ockl>0 as KL+OCKL.

Usage:
  python tools/analyze_cross_arch.py
  python tools/analyze_cross_arch.py --results-file logs/results.jsonl
"""

import argparse
import json
import math
import os
import sys

import pandas as pd

LOSS_COLS = ["KL", "KL+OCKL"]
EXPECTED_SEEDS = 3
EXPECTED_PAIRS = [
    ("cifar100",     "conv3",              "resnet18_modified"),
    ("cifar100",     "resnet18_modified",  "conv3"),
    ("tinyimagenet", "conv4",              "resnet18_modified"),
    ("tinyimagenet", "resnet18_modified",  "conv4"),
]
EXPECTED_IPCS = [1, 10, 50]


def load_rows(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def classify_loss(row):
    student_loss = row.get("student_loss")
    w_ockl = row.get("w_ockl")
    if w_ockl is None or (isinstance(w_ockl, float) and math.isnan(w_ockl)):
        w_ockl = 0.0
    if student_loss in ("kd", "kl") and w_ockl == 0:
        return "KL"
    if student_loss == "kl+ockl" or w_ockl > 0:
        return "KL+OCKL"
    return None


def dedupe_latest(df):
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    return df.drop_duplicates(
        subset=["dataset", "arch", "stud", "ipc", "seed", "loss_col"], keep="last"
    )


def aggregate(df, column):
    if df.empty:
        return {}
    g = df.groupby(["dataset", "arch", "stud", "ipc", "loss_col"])[column].agg(
        ["mean", "std", "count"]
    )
    return g.to_dict("index")


def fmt_cell(stats):
    if stats is None or stats["count"] == 0:
        return "--"
    n = int(stats["count"])
    if n < EXPECTED_SEEDS:
        return f"{stats['mean']:.2f} (n={n})"
    std = 0.0 if math.isnan(stats["std"]) else stats["std"]
    return f"{stats['mean']:.2f} ± {std:.2f}"


def fmt_nc(stats):
    if stats is None or stats["count"] == 0:
        return "--"
    return f"{stats['mean']:.3f}"


def print_acc_tables(stats):
    for dataset in sorted({p[0] for p in EXPECTED_PAIRS}):
        print(f"\n## Cross-arch best top-1 — {dataset} (mean ± std over {EXPECTED_SEEDS} seeds)\n")
        rows = []
        for (d, a, s) in EXPECTED_PAIRS:
            if d != dataset:
                continue
            for ipc in EXPECTED_IPCS:
                row = {"Arch→Stud": f"{a} → {s}", "IPC": ipc}
                for loss in LOSS_COLS:
                    row[loss] = fmt_cell(stats.get((d, a, s, ipc, loss)))
                kl = stats.get((d, a, s, ipc, "KL"))
                ko = stats.get((d, a, s, ipc, "KL+OCKL"))
                if kl and ko and kl["count"] > 0 and ko["count"] > 0:
                    row["Δ (OCKL−KL)"] = f"{ko['mean'] - kl['mean']:+.2f}"
                else:
                    row["Δ (OCKL−KL)"] = "--"
                rows.append(row)
        print(pd.DataFrame(rows).to_markdown(index=False))


def print_nc_table(nc_stats):
    for dataset in sorted({p[0] for p in EXPECTED_PAIRS}):
        print(f"\n## Cross-arch NC1 — {dataset} (mean over {EXPECTED_SEEDS} seeds)\n")
        rows = []
        for (d, a, s) in EXPECTED_PAIRS:
            if d != dataset:
                continue
            for ipc in EXPECTED_IPCS:
                row = {"Arch→Stud": f"{a} → {s}", "IPC": ipc}
                for loss in LOSS_COLS:
                    row[loss] = fmt_nc(nc_stats["nc1"].get((d, a, s, ipc, loss)))
                rows.append(row)
        print(pd.DataFrame(rows).to_markdown(index=False))


def print_diagnostics(stats):
    print("\n## Diagnostics", file=sys.stderr)
    expected = [(d, a, s, ipc, loss)
                for (d, a, s) in EXPECTED_PAIRS
                for ipc in EXPECTED_IPCS
                for loss in LOSS_COLS]
    total = len(expected)
    missing = [k for k in expected if k not in stats or stats[k]["count"] == 0]
    partial = [k for k in expected
               if k in stats and 0 < stats[k]["count"] < EXPECTED_SEEDS]
    complete = total - len(missing) - len(partial)
    print(f"\n  Matrix coverage: {complete}/{total} cells complete, "
          f"{len(partial)} partial, {len(missing)} missing", file=sys.stderr)
    for k in missing:
        print(f"    [missing] {k[0]}/{k[1]}→{k[2]}/ipc={k[3]}/loss={k[4]}",
              file=sys.stderr)
    for k in partial:
        n = int(stats[k]["count"])
        print(f"    [partial] {k[0]}/{k[1]}→{k[2]}/ipc={k[3]}/loss={k[4]}: "
              f"n={n}/{EXPECTED_SEEDS}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-file", default="logs/results.jsonl")
    args = ap.parse_args()

    raw = load_rows(args.results_file)
    if raw.empty or "stud" not in raw.columns:
        print("No cross-arch results yet.")
        return

    raw["loss_col"] = raw.apply(classify_loss, axis=1)
    cross = raw[(raw["loss_col"].notna()) & (raw["arch"] != raw["stud"])].copy()
    cross = dedupe_latest(cross)

    top1_stats = aggregate(cross, "best_top1")
    nc_stats = {m: aggregate(cross, m) for m in ("nc1", "nc2", "nc3", "nc4")}

    print_acc_tables(top1_stats)
    print_nc_table(nc_stats)
    print_diagnostics(top1_stats)


if __name__ == "__main__":
    main()
