#!/usr/bin/env python
"""Phase B.2 gamma-sweep analysis: pivot over w_ockl values for one cell.

Reads logs/results.jsonl, filters to rows that look like a gamma-sweep cell
(student_loss="kl+ockl", w_kl=1.0, matching dataset/arch/ipc), groups by
gamma=w_ockl, and prints best-top1 mean ± std plus NC1 mean per gamma.

Picks the operating point by best top-1 mean; reports separately the gamma
that minimizes NC1 (stronger collapse) — they need not agree.

Usage:
  python tools/analyze_gamma_sweep.py
  python tools/analyze_gamma_sweep.py --dataset cifar100 --arch resnet18_modified --ipc 10
"""

import argparse
import json
import math
import os

import pandas as pd

EXPECTED_SEEDS = 3


def load_rows(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def filter_sweep(df, dataset, arch, ipc):
    if df.empty:
        return df
    mask = (
        (df["dataset"] == dataset)
        & (df["arch"] == arch)
        & (df["stud"] == arch)
        & (df["ipc"] == ipc)
        & (df["student_loss"] == "kl+ockl")
        & (df["w_kl"] == 1.0)
        & (df["w_ockl"] > 0)
    )
    return df[mask].copy()


def fmt_cell(mean, std, n):
    if n == 0:
        return "--"
    if n < EXPECTED_SEEDS:
        return f"{mean:.2f} (n={n})"
    std = 0.0 if math.isnan(std) else std
    return f"{mean:.2f} ± {std:.2f}"


def fmt_nc(mean, n):
    if n == 0:
        return "--"
    return f"{mean:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-file", default="logs/results.jsonl")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--arch", default="resnet18_modified")
    ap.add_argument("--ipc", type=int, default=10)
    args = ap.parse_args()

    raw = load_rows(args.results_file)
    df = filter_sweep(raw, args.dataset, args.arch, args.ipc)
    if df.empty:
        print(f"No rows match dataset={args.dataset} arch={args.arch} ipc={args.ipc} student_loss=kl+ockl.")
        return

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["w_ockl", "seed"], keep="last")

    grouped = df.groupby("w_ockl").agg(
        acc_mean=("best_top1", "mean"),
        acc_std=("best_top1", "std"),
        acc_n=("best_top1", "count"),
        nc1_mean=("nc1", "mean"),
        nc2_mean=("nc2", "mean"),
        nc3_mean=("nc3", "mean"),
    ).sort_index()

    print(f"\n## Gamma sweep — {args.dataset}/{args.arch}/ipc={args.ipc} "
          f"(mean ± std over {EXPECTED_SEEDS} seeds)\n")
    rows = []
    for gamma, r in grouped.iterrows():
        n = int(r["acc_n"])
        rows.append({
            "γ (w_ockl)": gamma,
            "Best top-1": fmt_cell(r["acc_mean"], r["acc_std"], n),
            "NC1": fmt_nc(r["nc1_mean"], n),
            "NC2": fmt_nc(r["nc2_mean"], n),
            "NC3": fmt_nc(r["nc3_mean"], n),
            "n": n,
        })
    print(pd.DataFrame(rows).to_markdown(index=False))

    complete = grouped[grouped["acc_n"] >= EXPECTED_SEEDS]
    if not complete.empty:
        best_acc_gamma = complete["acc_mean"].idxmax()
        best_nc1_gamma = complete["nc1_mean"].idxmin()
        print(f"\nBest accuracy: γ={best_acc_gamma} "
              f"(top-1 {complete.loc[best_acc_gamma, 'acc_mean']:.2f})")
        print(f"Strongest NC1:  γ={best_nc1_gamma} "
              f"(NC1 {complete.loc[best_nc1_gamma, 'nc1_mean']:.3f})")


if __name__ == "__main__":
    main()
