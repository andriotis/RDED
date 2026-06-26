"""Compare stock vs variance-aware selection (random/stratified/covmatch/relmatch) — the H2
causal readout for the variance-preserving synthesis intervention.

Reads one or more results JSONL files (default: every logs/results_select*.jsonl), recovers
the selection variant from each row's exp_name (_sel<method>[_extras] tag; untagged = stock),
and prints, per (dataset, arch, ipc), a table with:
  - one column per variant (name derived from exp_name, so relmatch vs relmatch_dw1 are separate)
  - a Δ column after each non-stock variant showing the difference vs stock
  - green / red colouring: green = better than stock, red = worse (direction-aware)
  - bold = best value across all variants for that metric

Usage:
    python tools/analyze_select.py
    python tools/analyze_select.py logs/results_select_conv3.jsonl logs/results_select_conv4.jsonl
    python tools/analyze_select.py --arch conv3 --dataset cifar10 --ipc 10
    python tools/analyze_select.py --no-color
"""

import argparse
import glob
import json
import sys
from collections import defaultdict

import numpy as np

# (key in diag/sl_stats, label, direction: +1 = higher better, -1 = lower better, 0 = no order)
PANEL = [
    ("best_top1",      "top1",           +1),
    ("nc1",            "nc1(stu/val)",    0),
    ("sl_entropy_mean","sl_entropy",     +1),   # higher = more diffuse soft-labels
    ("sl_entropy_std", "sl_ent_std",     +1),   # higher = more heterogeneous crop selection
    ("sl_conf_mean",   "sl_conf",        -1),   # lower = more off-diagonal mass
    ("sl_conf_std",    "sl_conf_std",    +1),   # higher = more spread in crop confidence
    ("sl_eff_n_mean",  "sl_eff_n",       +1),   # higher = richer inter-class signal
    ("sl_eff_n_std",   "sl_eff_n_std",   +1),   # higher = more spread in per-crop eff_n
    ("ece",            "ece",            -1),
    ("ece_ts",         "ece_ts",         -1),
    ("auroc_msp",      "auroc_msp",      +1),
    ("auroc_energy",   "auroc_energy",   +1),
    ("auroc_maha",     "auroc_maha",     +1),
    ("auroc_feat_norm","auroc_featnorm", +1),
    ("oscr_msp",       "oscr_msp",       +1),
    ("fpr95_msp",      "fpr95_msp",      -1),
]

# Will be toggled off by --no-color or when stdout is not a tty.
_COLOR = True


def _ansi(text, *, green=False, red=False, bold=False):
    if not _COLOR:
        return text
    codes = []
    if bold:
        codes.append("1")
    if green:
        codes.append("32")
    if red:
        codes.append("31")
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def variant_of(exp_name):
    """Return the selector variant key from exp_name.

    "stock" when no _sel tag; otherwise everything after "_sel", e.g.:
      "relmatch", "relmatch_dw1", "covmatch", "momentmatch".
    This lets relmatch_dw0 and relmatch_dw1 appear as separate columns.
    """
    s = exp_name or ""
    idx = s.find("_sel")
    if idx == -1:
        return "stock"
    return s[idx + 4:]  # strip the leading "_sel"


def load(paths):
    rows = []
    for p in paths:
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r.get("diag") and r.get("best_top1") is not None:
                        rows.append(r)
        except FileNotFoundError:
            pass
    return rows


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="*", help="results JSONL files (default: logs/results_select*.jsonl)")
    ap.add_argument("--arch",    action="append", help="keep only these archs (repeatable)")
    ap.add_argument("--dataset", action="append", help="keep only these datasets (repeatable)")
    ap.add_argument("--ipc",     action="append", type=int, help="keep only these ipc values (repeatable)")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colour output")
    return ap.parse_args()


def _fmt(vals):
    """Return (mean_or_None, formatted_string)."""
    if not vals:
        return None, "       -     "
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, f"{m:7.3f}±{s:5.3f}"


def main():
    global _COLOR
    args = parse_args()
    if args.no_color or not sys.stdout.isatty():
        _COLOR = False

    paths = args.paths or sorted(glob.glob("logs/results_select*.jsonl"))
    rows = load(paths)
    if not rows:
        sys.exit(f"no usable rows in {paths}")

    want_arch    = set(args.arch)    if args.arch    else None
    want_dataset = set(args.dataset) if args.dataset else None
    want_ipc     = set(args.ipc)     if args.ipc     else None
    if want_arch is not None:
        missing = want_arch - {r["arch"] for r in rows}
        if missing:
            print(f"# note: no rows for arch={sorted(missing)}", file=sys.stderr)
    if want_dataset is not None:
        missing = want_dataset - {r["dataset"] for r in rows}
        if missing:
            print(f"# note: no rows for dataset={sorted(missing)}", file=sys.stderr)
    if want_ipc is not None:
        missing = want_ipc - {r["ipc"] for r in rows}
        if missing:
            print(f"# note: no rows for ipc={sorted(missing)}", file=sys.stderr)

    rows = [
        r for r in rows
        if (want_arch    is None or r["arch"]    in want_arch)
        and (want_dataset is None or r["dataset"] in want_dataset)
        and (want_ipc     is None or r["ipc"]     in want_ipc)
    ]
    if not rows:
        sys.exit("no rows match the requested --arch/--dataset/--ipc filters")

    # Dedup (dataset,arch,ipc,variant,seed) keeping the last occurrence — shards sometimes
    # log the same cell more than once, which would double-count into mean/std.
    dedup = {}
    for r in rows:
        key = (r["dataset"], r["arch"], r["ipc"], variant_of(r.get("exp_name")), r.get("seed"))
        dedup[key] = r
    rows = list(dedup.values())

    # cell -> variant -> metric -> [per-seed values]
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seeds = defaultdict(lambda: defaultdict(set))
    for r in rows:
        cell = (r["dataset"], r["arch"], r["ipc"])
        v = variant_of(r.get("exp_name"))
        seeds[cell][v].add(r.get("seed"))
        d = dict(r["diag"])
        d.setdefault("best_top1", r["best_top1"])
        d.update(r.get("sl_stats") or {})  # sl_entropy_mean / sl_conf_mean / sl_eff_n_mean
        for key, _, _ in PANEL:
            if d.get(key) is not None:
                cells[cell][v][key].append(float(d[key]))

    print(f"sources: {paths}\n")

    for cell in sorted(cells):
        dataset, arch, ipc = cell

        # stock first, then the rest sorted alphabetically
        variants = sorted(cells[cell].keys(), key=lambda v: ("" if v == "stock" else v))
        non_stock = [v for v in variants if v != "stock"]
        has_stock = "stock" in variants

        ns = {v: sorted(seeds[cell][v]) for v in variants}
        print(f"### {dataset} / {arch} / ipc{ipc}   seeds={ {v: ns[v] for v in variants} }")

        # Column widths: value cell wide enough for the longest variant name or the value string
        # "  0.742±0.003" = 13 chars; pad to max(name_len, 13) + 2 breathing room.
        VAL_W   = max(15, max(len(v) for v in variants) + 2)
        DELTA_W = 9   # "+0.042" fits in 7; 9 gives room for label

        # Header row
        LABEL_W = 16
        hdr = f"{'metric':{LABEL_W}s}"
        for v in variants:
            hdr += _ansi(f"{v:>{VAL_W}s}", bold=True)
            if v != "stock":
                hdr += f"{'Δ vs stk':>{DELTA_W}s}"
        print(hdr)

        for mkey, label, direction in PANEL:
            # Collect mean + formatted string per variant
            means: dict[str, float | None] = {}
            fmts:  dict[str, str]          = {}
            for v in variants:
                vals = cells[cell][v].get(mkey, [])
                m, s = _fmt(vals)
                means[v] = m
                fmts[v]  = s

            # Best variant (for bold); only meaningful when direction is known
            best_v = None
            if direction != 0:
                valid = {v: m for v, m in means.items() if m is not None}
                if valid:
                    best_v = max(valid, key=lambda v: valid[v] * direction)

            stock_mean = means.get("stock") if has_stock else None

            line = f"{label:{LABEL_W}s}"
            for v in variants:
                m   = means[v]
                s   = fmts[v]
                is_best = v == best_v

                # Colour value cell vs stock (skip stock itself and direction=0 metrics)
                green = red = False
                if direction != 0 and v != "stock" and m is not None and stock_mean is not None:
                    improvement = (m - stock_mean) * direction
                    green = improvement >  1e-6
                    red   = improvement < -1e-6

                line += _ansi(f"{s:>{VAL_W}s}", green=green, red=red, bold=is_best)

                if v != "stock":
                    if m is not None and stock_mean is not None:
                        delta     = m - stock_mean
                        delta_str = f"{delta:+.3f}"
                        dg = direction != 0 and (delta * direction) >  1e-6
                        dr = direction != 0 and (delta * direction) < -1e-6
                        line += _ansi(f"{delta_str:>{DELTA_W}s}", green=dg, red=dr)
                    else:
                        line += f"{'':>{DELTA_W}s}"

            print(line)
        print()


if __name__ == "__main__":
    main()
