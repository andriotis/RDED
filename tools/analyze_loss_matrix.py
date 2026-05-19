#!/usr/bin/env python
"""Phase B loss-matrix analysis: pivot over loss settings, print tables + diagnostics.

Reads logs/results.jsonl, classifies each row into a loss column (KL or KL+OCKL),
aggregates by (dataset, arch, ipc, loss), and prints:
  1. Unified best-top1 table (stdout, markdown) — rows = (dataset, arch, ipc),
     cols = Paper (RDED Table 2 reference), KL, KL+OCKL
  2. Unified NC1-NC4 tables (stdout, markdown) — one table per metric, rows
     spanning all datasets
  3. Diagnostics: missing/partial cells, high-variance cells, schema (stderr)

Treats legacy Phase B.0 rows (student_loss="kd", missing w_ockl) as the KL baseline.
Treats rows with w_ockl > 0 as KL+OCKL.

Usage:
  python tools/analyze_loss_matrix.py
  python tools/analyze_loss_matrix.py --results-file logs/results.jsonl --variance-threshold 1.5
"""

import argparse
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_numbers import RDED_TABLE2, expected_cells

LOSS_COLS = ["KL", "KL+OCKL"]
EXPECTED_SEEDS = 3
DEFAULT_VARIANCE_THRESHOLD = 1.5  # pp


def load_rows(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def classify_loss(row):
    """Map a results row to one of LOSS_COLS, or None if it doesn't fit the matrix."""
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
    """Keep the most recent row per (dataset, arch, ipc, seed, loss_col) by timestamp."""
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    return df.drop_duplicates(
        subset=["dataset", "arch", "ipc", "seed", "loss_col"], keep="last"
    )


def aggregate(df, column):
    """Return dict {(dataset, arch, ipc, loss_col): {mean, std, count}} for `column`."""
    if df.empty:
        return {}
    g = df.groupby(["dataset", "arch", "ipc", "loss_col"])[column].agg(["mean", "std", "count"])
    return g.to_dict("index")


def fmt_cell(stats):
    if stats is None or stats["count"] == 0:
        return "--"
    n = int(stats["count"])
    if n < EXPECTED_SEEDS:
        return f"{stats['mean']:.2f} (n={n})"
    std = stats["std"] if not math.isnan(stats["std"]) else 0.0
    return f"{stats['mean']:.2f} ± {std:.2f}"


def fmt_nc(stats):
    if stats is None or stats["count"] == 0:
        return "--"
    return f"{stats['mean']:.3f}"


def print_acc_table(stats):
    """Single table across all datasets; rows = (dataset, arch, ipc); cols = Paper, loss settings."""
    print(f"\n## Best top-1 accuracy (mean ± std over {EXPECTED_SEEDS} seeds)\n")
    rows = []
    for key in expected_cells():
        d, arch, ipc = key
        paper_mean, paper_std = RDED_TABLE2[key]
        row = {
            "Dataset": d,
            "Arch": arch,
            "IPC": ipc,
            "Paper": f"{paper_mean:.1f} ± {paper_std:.1f}",
        }
        best_mean = None
        for loss in LOSS_COLS:
            s = stats.get((d, arch, ipc, loss))
            row[loss] = fmt_cell(s)
            if s is not None and s["count"] > 0:
                if best_mean is None or s["mean"] > best_mean:
                    best_mean = s["mean"]
        # Bold the best of {KL, KL+OCKL} per row; Paper is reference, not a competitor.
        if best_mean is not None:
            for loss in LOSS_COLS:
                s = stats.get((d, arch, ipc, loss))
                if s is not None and s["count"] > 0 and s["mean"] == best_mean:
                    row[loss] = f"**{row[loss]}**"
        rows.append(row)
    print(pd.DataFrame(rows).to_markdown(index=False))


def print_nc_tables(nc_stats):
    """One table per NC metric, rows spanning all datasets."""
    for metric in ("nc1", "nc2", "nc3", "nc4"):
        label = "NC1 (within-class variance collapse)" if metric == "nc1" else metric.upper()
        print(f"\n## {label} — (mean over {EXPECTED_SEEDS} seeds)\n")
        rows = []
        for key in expected_cells():
            d, arch, ipc = key
            row = {"Dataset": d, "Arch": arch, "IPC": ipc}
            for loss in LOSS_COLS:
                row[loss] = fmt_nc(nc_stats[metric].get((d, arch, ipc, loss)))
            rows.append(row)
        print(pd.DataFrame(rows).to_markdown(index=False))


def print_diagnostics(raw_df, matrix_df, stats, var_threshold):
    print("\n## Diagnostics", file=sys.stderr)

    expected_keys = [(d, a, i, loss) for (d, a, i) in expected_cells() for loss in LOSS_COLS]
    total = len(expected_keys)
    missing = [k for k in expected_keys if k not in stats or stats[k]["count"] == 0]
    partial = [k for k in expected_keys
               if k in stats and 0 < stats[k]["count"] < EXPECTED_SEEDS]
    complete = total - len(missing) - len(partial)

    print(f"\n  Matrix coverage: {complete}/{total} (dataset, arch, ipc, loss) cells complete, "
          f"{len(partial)} partial, {len(missing)} missing", file=sys.stderr)
    for k in missing:
        print(f"    [missing] {k[0]}/{k[1]}/ipc={k[2]}/loss={k[3]}", file=sys.stderr)
    for k in partial:
        n = int(stats[k]["count"])
        print(f"    [partial] {k[0]}/{k[1]}/ipc={k[2]}/loss={k[3]}: n={n}/{EXPECTED_SEEDS}",
              file=sys.stderr)

    hv = [
        (k, s["std"]) for k, s in stats.items()
        if s["count"] >= EXPECTED_SEEDS and not math.isnan(s["std"]) and s["std"] > var_threshold
    ]
    if hv:
        print(f"\n  High-variance cells (std > {var_threshold}pp):", file=sys.stderr)
        for k, std in hv:
            print(f"    - {k[0]}/{k[1]}/ipc={k[2]}/loss={k[3]}: std={std:.2f}pp", file=sys.stderr)
    else:
        print(f"\n  No high-variance cells (threshold {var_threshold}pp).", file=sys.stderr)

    if raw_df.empty:
        print("\n  ! results file is empty or missing.", file=sys.stderr)
        return

    issues = []
    needed = ["dataset", "arch", "ipc", "seed", "best_top1"]
    missing_fields = [f for f in needed if f not in raw_df.columns]
    if missing_fields:
        issues.append(f"missing fields: {missing_fields}")
    if "nc1" in raw_df.columns:
        nan_n = int(raw_df["nc1"].isna().sum())
        if nan_n:
            issues.append(f"{nan_n} rows with NaN NC1")
    if "best_top1" in raw_df.columns:
        nan_acc = int(raw_df["best_top1"].isna().sum())
        if nan_acc:
            issues.append(f"{nan_acc} rows with NaN best_top1")
    n_unclassified = int(len(raw_df) - len(matrix_df))
    if n_unclassified:
        issues.append(f"{n_unclassified} rows did not match a loss column "
                      "(student_loss not in {kd, kl, kl+ockl}, or schema mismatch)")

    if issues:
        print("\n  Schema notes:", file=sys.stderr)
        for issue in issues:
            print(f"    - {issue}", file=sys.stderr)
    else:
        print("\n  Schema OK.", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-file", default="logs/results.jsonl",
                    help="JSONL log to read (default: logs/results.jsonl)")
    ap.add_argument("--variance-threshold", type=float, default=DEFAULT_VARIANCE_THRESHOLD,
                    help=f"std (pp) above which a cell is flagged (default: {DEFAULT_VARIANCE_THRESHOLD})")
    args = ap.parse_args()

    raw_df = load_rows(args.results_file)
    if raw_df.empty:
        matrix_df = raw_df
    else:
        raw_df["loss_col"] = raw_df.apply(classify_loss, axis=1)
        matrix_df = dedupe_latest(raw_df[raw_df["loss_col"].notna()].copy())

    top1_stats = aggregate(matrix_df, "best_top1")
    nc_stats = {m: aggregate(matrix_df, m) for m in ("nc1", "nc2", "nc3", "nc4")}

    print_acc_table(top1_stats)
    print_nc_tables(nc_stats)
    print_diagnostics(raw_df, matrix_df, top1_stats, args.variance_threshold)


if __name__ == "__main__":
    main()
