#!/usr/bin/env python
"""Phase B.0 baseline analysis: reproduction + NC tables + diagnostics.

Reads logs/results.jsonl, filters to stock-RDED rows (student_loss=kd,
selector=ce, soft_label=teacher), aggregates by (dataset, arch, ipc), and prints:
  1. Reproduction table vs RDED paper Table 2 (stdout, markdown)
  2. NC1-NC4 baseline table (stdout, markdown)
  3. Diagnostics: missing cells, partial cells, high-variance cells, schema issues (stderr)

Re-runnable any time during the matrix; serves as a live progress dashboard.

Usage:
  python tools/analyze_baseline.py
  python tools/analyze_baseline.py --results-file logs/results.jsonl --variance-threshold 1.5
"""

import argparse
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_numbers import RDED_TABLE2, expected_cells

REQUIRED_FIELDS = [
    "dataset", "arch", "ipc", "seed", "student_loss", "selector",
    "soft_label", "best_top1", "nc1", "nc2", "nc3", "nc4",
]
EXPECTED_SEEDS = 3
DEFAULT_VARIANCE_THRESHOLD = 1.5  # pp


def load_rows(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def filter_baseline(df):
    if df.empty:
        return df
    mask = (
        (df["student_loss"] == "kd")
        & (df["selector"] == "ce")
        & (df["soft_label"] == "teacher")
    )
    return df[mask].copy()


def dedupe_latest(df):
    """Keep the most recent row per (dataset, arch, ipc, seed) by timestamp."""
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    return df.drop_duplicates(subset=["dataset", "arch", "ipc", "seed"], keep="last")


def aggregate(df, column):
    """Return dict {(dataset, arch, ipc): {mean, std, count}} for `column`."""
    if df.empty:
        return {}
    g = df.groupby(["dataset", "arch", "ipc"])[column].agg(["mean", "std", "count"])
    return g.to_dict("index")


def fmt_paper(p):
    mean, std = p
    return f"{mean:.1f} ± {std:.1f}"


def fmt_ours(stats):
    if stats is None or stats["count"] == 0:
        return "--"
    n = int(stats["count"])
    if n < EXPECTED_SEEDS:
        return f"{stats['mean']:.2f} (n={n})"
    std = stats["std"] if not math.isnan(stats["std"]) else 0.0
    return f"{stats['mean']:.2f} ± {std:.2f}"


def fmt_delta(paper, stats):
    if stats is None or stats["count"] == 0:
        return "--"
    return f"{stats['mean'] - paper[0]:+.2f}"


def status_glyph(paper, stats, var_threshold):
    if stats is None or stats["count"] == 0:
        return "⏳ n=0"
    n = int(stats["count"])
    suffix = ""
    std = stats.get("std", float("nan"))
    if not math.isnan(std) and std > var_threshold:
        suffix = "!"
    if n < EXPECTED_SEEDS:
        return f"⚠ n={n}{suffix}"
    delta = abs(stats["mean"] - paper[0])
    if delta <= 1.0:
        glyph = "✓"
    elif delta <= 2.0:
        glyph = "⚠"
    else:
        glyph = "✗"
    return f"{glyph}{suffix}"


def print_reproduction_table(top1_stats, var_threshold):
    print("## Reproduction vs RDED Table 2 (best top-1 %)\n")
    rows = []
    for key in expected_cells():
        dataset, arch, ipc = key
        paper = RDED_TABLE2[key]
        stats = top1_stats.get(key)
        rows.append({
            "Dataset": dataset,
            "Arch": arch,
            "IPC": ipc,
            "Paper": fmt_paper(paper),
            "Ours": fmt_ours(stats),
            "Δ": fmt_delta(paper, stats),
            "Status": status_glyph(paper, stats, var_threshold),
        })
    print(pd.DataFrame(rows).to_markdown(index=False))
    print(f"\nLegend: ✓ |Δ|≤1pp   ⚠ 1-2pp or n<{EXPECTED_SEEDS}   ✗ |Δ|>2pp   ⏳ no data")
    print(f"        ! suffix = std > {var_threshold:.1f}pp")


def print_nc_table(nc_stats):
    print("\n## NC metrics on stock-RDED student (mean, final epoch)\n")
    rows = []
    for key in expected_cells():
        dataset, arch, ipc = key
        any_stats = nc_stats["nc1"].get(key)
        n = 0 if any_stats is None else int(any_stats["count"])
        row = {"Dataset": dataset, "Arch": arch, "IPC": ipc}
        for metric in ("nc1", "nc2", "nc3", "nc4"):
            s = nc_stats[metric].get(key)
            row[metric.upper()] = "--" if (s is None or s["count"] == 0) else f"{s['mean']:.3f}"
        row["n"] = n
        rows.append(row)
    print(pd.DataFrame(rows).to_markdown(index=False))


def print_diagnostics(raw_df, baseline_df, top1_stats, var_threshold):
    print("\n## Diagnostics", file=sys.stderr)

    # Missing / partial cells
    missing = [k for k in expected_cells() if k not in top1_stats or top1_stats[k]["count"] == 0]
    partial = [k for k in expected_cells()
               if k in top1_stats and 0 < top1_stats[k]["count"] < EXPECTED_SEEDS]
    total = len(expected_cells())
    complete = total - len(missing) - len(partial)
    print(f"\n  Matrix coverage: {complete}/{total} cells complete, "
          f"{len(partial)} partial, {len(missing)} missing", file=sys.stderr)
    for k in missing:
        print(f"    [missing] {k[0]}/{k[1]}/ipc={k[2]}", file=sys.stderr)
    for k in partial:
        n = int(top1_stats[k]["count"])
        print(f"    [partial] {k[0]}/{k[1]}/ipc={k[2]}: n={n}/{EXPECTED_SEEDS}", file=sys.stderr)

    # High variance
    hv = [
        (k, s["std"]) for k, s in top1_stats.items()
        if s["count"] >= EXPECTED_SEEDS and not math.isnan(s["std"]) and s["std"] > var_threshold
    ]
    if hv:
        print(f"\n  High-variance cells (std > {var_threshold}pp):", file=sys.stderr)
        for k, std in hv:
            print(f"    - {k[0]}/{k[1]}/ipc={k[2]}: std={std:.2f}pp", file=sys.stderr)
    else:
        print(f"\n  No high-variance cells (threshold {var_threshold}pp).", file=sys.stderr)

    # Schema integrity
    if raw_df.empty:
        print("\n  ! results file is empty or missing.", file=sys.stderr)
        return
    issues = []
    missing_fields = [f for f in REQUIRED_FIELDS if f not in raw_df.columns]
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
    non_matrix = [
        (r["dataset"], r["arch"], int(r["ipc"]))
        for _, r in baseline_df.iterrows()
        if (r["dataset"], r["arch"], int(r["ipc"])) not in RDED_TABLE2
    ]
    if non_matrix:
        issues.append(f"{len(non_matrix)} baseline rows outside the expected matrix")
    n_dup = int(len(baseline_df)
                - len(baseline_df.drop_duplicates(["dataset", "arch", "ipc", "seed"])))
    if n_dup:
        issues.append(f"{n_dup} duplicate (dataset, arch, ipc, seed) rows (kept latest by timestamp)")

    n_non_baseline = int(len(raw_df) - len(filter_baseline(raw_df)))
    if n_non_baseline:
        print(f"\n  {n_non_baseline} non-baseline row(s) ignored "
              "(student_loss != kd or selector != ce or soft_label != teacher).",
              file=sys.stderr)

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
    baseline_df = dedupe_latest(filter_baseline(raw_df))

    top1_stats = aggregate(baseline_df, "best_top1")
    nc_stats = {m: aggregate(baseline_df, m) for m in ("nc1", "nc2", "nc3", "nc4")}

    print_reproduction_table(top1_stats, args.variance_threshold)
    print_nc_table(nc_stats)
    print_diagnostics(raw_df, baseline_df, top1_stats, args.variance_threshold)


if __name__ == "__main__":
    main()
