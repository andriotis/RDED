"""Pretty-print the trustworthiness diagnostic results.

Pools the two logs the diagnostic writes and reorganizes them *by experiment
cell* (dataset, arch, ipc) rather than by which file produced each row. Every
cell shows the same four views, ordered so the two comparisons that matter sit
side by side:

  input data (how the TEACHER sees the training set)
    distilled set (RDED)        ┐ compare these two: is the distilled set
    real subset  (n/cls=ipc)    ┘ over-collapsed vs an equal slice of real data?
  trained model (own feats on real val + SVHN OOD)
    student (trained on RDED)   ┐ compare these two: how far is the student's
    teacher (real-data ref)     ┘ trust below the real-data ceiling?

Sources pooled (the organizing principle is the cell, not the source):
  * logs/results.jsonl    -> student-on-RDED rows (the "diag" field; mean/seeds)
  * logs/diagnostics.jsonl -> teacher geometry + reference rows
                              (from tools/diagnose_geometry.py)

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

# (source key, header, decimals). Order = column order in every cell table.
DISPLAY = [
    ("top1", "top1", 2),
    ("nc1", "nc1", 3),
    ("nc2", "nc2", 3),
    ("nc3", "nc3", 3),
    ("ece", "ece", 3),
    ("overconf_gap", "o_gap", 3),
    ("oscr_msp", "oscr", 3),
    ("auroc_msp", "auroc", 3),
    ("fpr95_msp", "fpr95", 3),
]
# Columns meaningful for the two "input data" rows (no OOD; o_gap meaningless on
# the ~100%-accuracy training crops, so we dash it to avoid a misleading number).
DATA_COLS = {"top1", "nc1", "nc2", "nc3", "ece"}
MODEL_COLS = {k for k, _, _ in DISPLAY}

LABEL_W = 30
COL_W = 8


def _read_jsonl(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def _is_na(v):
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _fmt(v, nd):
    return "—" if _is_na(v) else f"{v:.{nd}f}"


def _header():
    cells = "".join(f"{h:>{COL_W}}" for _, h, _ in DISPLAY)
    return f"  {'subject':<{LABEL_W}}{cells}"


def _row(label, m, allowed):
    cells = []
    for key, _, nd in DISPLAY:
        v = m.get(key) if (m is not None and key in allowed) else None
        cells.append(f"{_fmt(v, nd):>{COL_W}}")
    return f"  {label:<{LABEL_W}}{''.join(cells)}"


# ----------------------------------------------------------------------------- loaders


def load_students(path, args):
    """Per-(dataset, arch, ipc) student rows, averaged over seeds.

    Returns (cells, per_seed_tab) where cells maps the key tuple to
    (n_seeds, metrics_dict) and per_seed_tab is the un-aggregated frame (or None).
    """
    df = _read_jsonl(path)
    if df.empty or "diag" not in df.columns:
        return {}, None
    df = df[df["diag"].notna()].copy()
    if args.dataset:
        df = df[df["dataset"] == args.dataset]
    if args.ipc:
        df = df[df["ipc"] == args.ipc]
    if df.empty:
        return {}, None

    diag = pd.json_normalize(df["diag"]).reset_index(drop=True)
    meta = df[["dataset", "arch", "ipc", "seed", "best_top1"]].reset_index(drop=True)
    tab = pd.concat([meta, diag], axis=1)
    # best_top1 is the accuracy we report as `top1` for the student.
    tab["top1"] = tab["best_top1"]
    metric_cols = [k for k, _, _ in DISPLAY if k in tab.columns]

    cells = {}
    for key, g in tab.groupby(["dataset", "arch", "ipc"]):
        cells[key] = (g["seed"].nunique(), g[metric_cols].mean().to_dict())
    return cells, tab


def load_geometry(path, args):
    """Per-(dataset, arch, ipc) teacher-side rows: distilled, real subset, teacher.

    Returns a dict: key -> {"distilled": m, "real": (label, m), "teacher": m}.
    """
    df = _read_jsonl(path)
    if df.empty:
        return {}
    df = df.rename(columns={"subset": "dataset"})
    if args.dataset:
        df = df[df["dataset"] == args.dataset]
    if args.ipc:
        df = df[df["ipc"] == args.ipc]
    if df.empty:
        return {}
    df = df.drop_duplicates(["dataset", "arch", "ipc", "subject"], keep="last")

    def pick(g, prefix):
        sub = g[g["subject"].str.startswith(prefix)]
        return sub.iloc[-1].to_dict() if not sub.empty else None

    out = {}
    for key, g in df.groupby(["dataset", "arch", "ipc"]):
        distilled = pick(g, "distilled-set")
        real = pick(g, "real-train subset")
        teacher = pick(g, "teacher reference")
        real_ipc = int(real["real_ipc"]) if real and not _is_na(real.get("real_ipc")) else key[2]
        out[key] = {
            "distilled": distilled,
            "real": (f"real subset (n/cls={real_ipc})", real),
            "teacher": teacher,
        }
    return out


# ----------------------------------------------------------------------------- printing

LEGEND = """\
Reading guide — each cell shows four views of one (dataset, arch, ipc), top-down:
  input data    : how the TEACHER sees the training set (NC on teacher feats)
  trained model : each model's OWN feats on real val + SVHN as OOD
  good direction:  top1 ↑   ece / o_gap ↓ (o_gap = conf−acc; >0 = overconfident)
                   oscr / auroc ↑   fpr95 ↓   nc1 / nc2 / nc3 ↓
  '—' = not applicable (OOD needs an eval set; o_gap is moot on ~100%-acc crops).
  Adjacent rows are the comparisons: distilled vs real subset (is the synthetic
  set over-collapsed?), and student vs teacher (how far below the real ceiling?).
  The teacher row is the fixed real-data reference (identical across ipc).\
"""


def print_overview(students, geometry):
    """At-a-glance student-vs-teacher gap across the whole grid."""
    rows = []
    for key in sorted(set(students) | set(geometry)):
        ds, arch, ipc = key
        s = students.get(key, (0, {}))[1]
        t = (geometry.get(key, {}).get("teacher")) or {}
        rows.append({
            "dataset": ds, "arch": arch, "ipc": ipc,
            "top1_s": s.get("top1"), "top1_t": t.get("top1"),
            "ece_s": s.get("ece"), "ece_t": t.get("ece"),
            "auroc_s": s.get("auroc_msp"), "auroc_t": t.get("auroc_msp"),
            "oscr_s": s.get("oscr_msp"), "oscr_t": t.get("oscr_msp"),
        })
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["dataset", "arch", "ipc"])
    num = [c for c in df.columns if c not in ("dataset", "arch", "ipc")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").round(3)
    print("=" * 92)
    print("OVERVIEW — student (s) vs teacher ceiling (t); gap = how much trust the distillation costs")
    print("=" * 92)
    print(df.to_string(index=False, na_rep="—"))
    print()


def print_cells(students, geometry):
    keys = sorted(set(students) | set(geometry))
    if not keys:
        return
    print("=" * 92)
    print("PER-CELL DETAIL — distilled vs real (geometry) and student vs teacher (trust)")
    print("=" * 92)
    print(LEGEND)

    last_group = None
    for key in keys:
        ds, arch, ipc = key
        if (ds, arch) != last_group:
            print(f"\n### {ds} / {arch}")
            last_group = (ds, arch)

        geo = geometry.get(key, {})
        n_seeds, student = students.get(key, (0, None))

        print(f"\n{ds} / {arch} / ipc{ipc}")
        print(_header())

        data_rows = []
        if geo.get("distilled") is not None:
            data_rows.append(("distilled set (RDED)", geo["distilled"]))
        if geo.get("real") and geo["real"][1] is not None:
            data_rows.append((geo["real"][0], geo["real"][1]))
        if data_rows:
            print("  input data (teacher's view):")
            for label, m in data_rows:
                print(_row(label, m, DATA_COLS))

        model_rows = []
        if student is not None:
            model_rows.append((f"student (RDED, mean/{n_seeds} seeds)", student))
        if geo.get("teacher") is not None:
            model_rows.append(("teacher (real-data reference)", geo["teacher"]))
        if model_rows:
            print("  trained model (real val + SVHN OOD):")
            for label, m in model_rows:
                print(_row(label, m, MODEL_COLS))
    print()


def print_per_seed(per_seed):
    if per_seed is None or per_seed.empty:
        return
    cols = [k for k, _, _ in DISPLAY if k in per_seed.columns]
    print("=" * 92)
    print("PER-SEED — student rows (un-averaged)")
    print("=" * 92)
    print(per_seed.sort_values(["dataset", "arch", "ipc", "seed"])[
        ["dataset", "arch", "ipc", "seed"] + cols
    ].round(4).to_string(index=False))
    print()


def main():
    p = argparse.ArgumentParser("show_diagnostics")
    p.add_argument("--dataset", type=str, default=None, help="filter (cifar100 / tinyimagenet)")
    p.add_argument("--ipc", type=int, default=None, help="filter by ipc")
    p.add_argument("--per-seed", action="store_true", help="also list each seed's student row")
    p.add_argument("--results-file", type=str, default="logs/results.jsonl")
    p.add_argument("--diagnostics-file", type=str, default="logs/diagnostics.jsonl")
    args = p.parse_args()

    students, per_seed = load_students(args.results_file, args)
    geometry = load_geometry(args.diagnostics_file, args)

    if not students and not geometry:
        print("(no rows — run `bash scripts/diagnose.sh geom` and `... train`)")
        return

    print_overview(students, geometry)
    print_cells(students, geometry)
    if args.per_seed:
        print_per_seed(per_seed)


if __name__ == "__main__":
    main()
