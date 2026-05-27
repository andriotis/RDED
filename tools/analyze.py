#!/usr/bin/env python
"""Best-top-1 analyzer driven by the logged weights dict.

Reads logs/results.jsonl. Each row's column key is derived from row["weights"]:
nonzero entries are sorted into a tuple ((name, w), ...) and rendered as a
human-readable label. Rows missing a "weights" dict (legacy schema) are
skipped.

Usage:
  python tools/analyze.py
  python tools/analyze.py --results-file logs/results.jsonl --variance-threshold 1.5
"""

import argparse
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_numbers import RDED_TABLE2

EXPECTED_SEEDS = 3
DEFAULT_VARIANCE_THRESHOLD = 1.5  # pp


def load_rows(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def loss_key(row):
    """Canonical key from the logged weights dict; ignores zero entries.

    Returns a sorted tuple of (name, weight) pairs, e.g. (("kl", 1.0),)
    or (("kl", 1.0), ("ockl", 0.3)). Rows without a `weights` dict (legacy
    schema) return None and are filtered out.
    """
    weights = row.get("weights")
    if not isinstance(weights, dict):
        return None
    active = tuple(sorted((name, float(w)) for name, w in weights.items() if float(w) > 0))
    return active if active else None


def col_label(key):
    """Display label for a sorted ((name, w), ...) tuple."""
    if not key:
        return "(none)"
    if key == (("kl", 1.0),):
        return "KL"
    if key == (("kl", 1.0), ("ockl", 1.0)):
        return "KL+OCKL"
    # Canonical gamma sweep: kl=1.0, ockl=<w> -- compact label.
    if len(key) == 2 and key[0] == ("kl", 1.0) and key[1][0] == "ockl":
        return f"KL+OCKL (g={key[1][1]:g})"
    # General fallback: NAME=w, joined by '+'. Render w==1.0 bare.
    parts = [n.upper() if w == 1.0 else f"{n.upper()}={w:g}" for n, w in key]
    return "+".join(parts)


def col_sort_key(key):
    """Order: KL first, KL+OCKL (default) next, then gamma-sweep variants by ockl, then others."""
    if key == (("kl", 1.0),):
        return (0,)
    if key == (("kl", 1.0), ("ockl", 1.0)):
        return (1,)
    if len(key) == 2 and key[0] == ("kl", 1.0) and key[1][0] == "ockl":
        return (2, key[1][1])
    return (3, key)


def dedupe_latest(df):
    """Keep the most recent row per (dataset, arch, stud, ipc, seed, loss_key)."""
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    return df.drop_duplicates(
        subset=["dataset", "arch", "stud", "ipc", "seed", "loss_key"], keep="last"
    )


def fmt_cell(mean, std, n):
    if n == 0 or mean is None:
        return "--"
    if n < EXPECTED_SEEDS:
        return f"{mean:.2f} (n={n})"
    std = 0.0 if std is None or math.isnan(std) else std
    return f"{mean:.2f} +/- {std:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-file", default="logs/results.jsonl",
                    help="JSONL log to read (default: logs/results.jsonl)")
    ap.add_argument("--variance-threshold", type=float, default=DEFAULT_VARIANCE_THRESHOLD,
                    help=f"std (pp) above which a cell is flagged (default: {DEFAULT_VARIANCE_THRESHOLD})")
    args = ap.parse_args()

    raw = load_rows(args.results_file)
    if raw.empty:
        print("No rows in results.jsonl.")
        return

    raw["loss_key"] = raw.apply(loss_key, axis=1)
    raw = raw[raw["loss_key"].notna()].copy()
    if raw.empty:
        print("No rows match a known weight configuration.")
        return

    df = dedupe_latest(raw)

    row_tuples = sorted(
        {(r["dataset"], r["arch"], r["stud"], int(r["ipc"])) for _, r in df.iterrows()},
        key=lambda t: (t[0], t[1], t[2], t[3]),
    )
    col_keys = sorted(set(df["loss_key"]), key=col_sort_key)

    agg = (
        df.groupby(["dataset", "arch", "stud", "ipc", "loss_key"])["best_top1"]
        .agg(["mean", "std", "count"])
        .to_dict("index")
    )

    rows_out = []
    for (d, a, s, ipc) in row_tuples:
        if a == s and (d, a, ipc) in RDED_TABLE2:
            pm, ps = RDED_TABLE2[(d, a, ipc)]
            paper = f"{pm:.1f} +/- {ps:.1f}"
        else:
            paper = "--"
        best_mean = None
        cell_means = {}
        for ck in col_keys:
            stats = agg.get((d, a, s, ipc, ck))
            if stats is None or stats["count"] == 0:
                cell_means[ck] = None
                continue
            cell_means[ck] = stats["mean"]
            if best_mean is None or stats["mean"] > best_mean:
                best_mean = stats["mean"]
        for ck in col_keys:
            stats = agg.get((d, a, s, ipc, ck))
            if stats is None:
                cell = "--"
            else:
                cell = fmt_cell(stats["mean"], stats["std"], int(stats["count"]))
            if (best_mean is not None and cell_means[ck] is not None
                    and cell_means[ck] == best_mean and cell != "--"):
                cell = f"**{cell}**"
            rows_out.append({
                "Dataset": d, "Arch": a, "Stud": s, "IPC": ipc,
                "Paper": paper, "Loss": col_label(ck), "Top-1": cell,
            })

    print(f"\n## Best top-1 accuracy (mean +/- std over {EXPECTED_SEEDS} seeds)\n")
    print(pd.DataFrame(rows_out).to_markdown(index=False))

    print("\n## Diagnostics", file=sys.stderr)
    total = len(row_tuples) * len(col_keys)
    complete = partial = missing = 0
    high_var = []
    for (d, a, s, ipc) in row_tuples:
        for ck in col_keys:
            stats = agg.get((d, a, s, ipc, ck))
            if stats is None or stats["count"] == 0:
                missing += 1
            elif stats["count"] < EXPECTED_SEEDS:
                partial += 1
            else:
                complete += 1
                std = 0.0 if math.isnan(stats["std"]) else stats["std"]
                if std > args.variance_threshold:
                    high_var.append(((d, a, s, ipc, ck), std))
    print(f"  Coverage: {complete}/{total} cells complete, {partial} partial, {missing} missing",
          file=sys.stderr)
    if high_var:
        print(f"  High-variance cells (std > {args.variance_threshold}pp):", file=sys.stderr)
        for (d, a, s, ipc, ck), std in high_var:
            print(f"    - {d}/{a}->{s}/ipc={ipc}/{col_label(ck)}: std={std:.2f}pp",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
