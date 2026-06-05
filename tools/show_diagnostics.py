"""Pretty-print the trustworthiness diagnostic results.

Pools the two logs the diagnostic writes:
  * logs/results.jsonl   -> student-on-RDED rows (H2), the ones with a "diag"
                            field; averaged over seeds.
  * logs/diagnostics.jsonl -> teacher geometry + reference rows (H1), from
                            tools/diagnose_geometry.py.

Usage (from repo root):
    python tools/show_diagnostics.py
    python tools/show_diagnostics.py --dataset cifar100 --ipc 10
    python tools/show_diagnostics.py --per-seed        # also list each seed
"""

import argparse
import os

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

# Compact, most-informative subset of the diag panel.
H2_COLS = ["best_top1", "ece", "overconf_gap", "oscr_msp", "auroc_msp", "fpr95_msp", "nc1", "nc2", "nc3"]


def _read_jsonl(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_json(path, lines=True)
    return df


def show_h2(path, args):
    df = _read_jsonl(path)
    if df.empty or "diag" not in df.columns:
        print(f"(no student diagnostics yet in {path} — run `bash scripts/diagnose.sh train`)\n")
        return
    df = df[df["diag"].notna()].copy()
    if args.dataset:
        df = df[df["dataset"] == args.dataset]
    if args.ipc:
        df = df[df["ipc"] == args.ipc]
    if df.empty:
        print("(no student diagnostic rows match the filter)\n")
        return

    diag = pd.json_normalize(df["diag"]).reset_index(drop=True)
    meta = df[["dataset", "arch", "stud", "ipc", "seed", "best_top1"]].reset_index(drop=True)
    tab = pd.concat([meta, diag], axis=1)
    cols = [c for c in H2_COLS if c in tab.columns]

    print("=" * 100)
    print("H2 — student trained on stock RDED (mean over seeds; higher OSCR/AUROC & lower ECE/FPR95 = better)")
    print("=" * 100)
    grp = tab.groupby(["dataset", "arch", "stud", "ipc"])
    mean = grp[cols].mean().round(4)
    nseeds = grp["seed"].nunique().rename("n_seeds")
    print(pd.concat([nseeds, mean], axis=1).to_string())

    if args.per_seed:
        print("\n-- per seed --")
        print(tab.sort_values(["dataset", "arch", "ipc", "seed"])[
            ["dataset", "arch", "ipc", "seed"] + cols
        ].round(4).to_string(index=False))
    print()


def show_h1(path, args):
    df = _read_jsonl(path)
    if df.empty:
        print(f"(no geometry rows yet in {path} — run `bash scripts/diagnose.sh geom`)\n")
        return
    if args.dataset:
        df = df[df["subset"] == args.dataset]
    if args.ipc:
        df = df[df["ipc"] == args.ipc]
    if df.empty:
        print("(no geometry rows match the filter)\n")
        return

    # keep the latest row per (subset, arch, ipc, subject)
    df = df.drop_duplicates(["subset", "arch", "ipc", "subject"], keep="last")
    cols = ["top1", "nc1", "nc2", "nc3", "ece", "oscr_msp", "auroc_msp", "fpr95_msp"]
    cols = [c for c in cols if c in df.columns]

    print("=" * 100)
    print("H1 — teacher-induced geometry: distilled set vs real subset, and teacher reference")
    print("=" * 100)
    for (subset, arch, ipc), g in df.groupby(["subset", "arch", "ipc"]):
        print(f"\n{subset} / {arch} / ipc{ipc}")
        print(g[["subject"] + cols].round(4).to_string(index=False))
    print()


def main():
    p = argparse.ArgumentParser("show_diagnostics")
    p.add_argument("--dataset", type=str, default=None, help="filter (cifar100 / tinyimagenet)")
    p.add_argument("--ipc", type=int, default=None, help="filter by ipc")
    p.add_argument("--per-seed", action="store_true", help="also list each seed's H2 row")
    p.add_argument("--results-file", type=str, default="logs/results.jsonl")
    p.add_argument("--diagnostics-file", type=str, default="logs/diagnostics.jsonl")
    args = p.parse_args()

    show_h2(args.results_file, args)
    show_h1(args.diagnostics_file, args)


if __name__ == "__main__":
    main()
