#!/usr/bin/env python
"""Plot training curves from logs/curves/*.jsonl.

Each run produces one JSONL file written by validation/curve_logger.py. The
filename encodes the canonical run key: {dataset}_{arch}_to_{stud}_ipc{ipc}_seed{seed}_w{tag}.
We group files by (dataset, arch, stud, ipc, tag) and aggregate seeds into a
mean line with a shaded +/-1 sigma band per config.

Usage:
  python tools/plot_curve.py                                 # window, all configs
  python tools/plot_curve.py --dataset cifar100 --arch conv3 # filter
  python tools/plot_curve.py --out curve.png                 # save PNG once
  python tools/plot_curve.py --watch 30                      # live window
  python tools/plot_curve.py --watch 30 --out curve.png      # refresh PNG every 30s
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

# Default to a non-interactive backend; the live/window paths upgrade to one
# that supports plt.show() at runtime.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Matches files named by canonical_run_key():
#   {dataset}_{arch}_to_{stud}_ipc{ipc}_seed{seed}_w{tag}.jsonl
FILENAME_RE = re.compile(
    r"^(?P<dataset>[^_]+)_(?P<arch>.+?)_to_(?P<stud>.+?)"
    r"_ipc(?P<ipc>\d+)_seed(?P<seed>\d+)_w(?P<tag>.+)\.jsonl$"
)

METRICS = ("train_loss", "top1_err", "top5_err")
METRIC_YLABEL = {
    "train_loss": "train loss",
    "top1_err": "train top-1 err (%)",
    "top5_err": "train top-5 err (%)",
}


def parse_filename(path):
    m = FILENAME_RE.match(os.path.basename(path))
    if not m:
        return None
    g = m.groupdict()
    g["ipc"] = int(g["ipc"])
    g["seed"] = int(g["seed"])
    return g


def load_curve(path):
    """Return list of rows (each a dict) sorted by epoch."""
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    rows.sort(key=lambda r: r.get("epoch", 0))
    return rows


def matches_filter(meta, args):
    if args.dataset and meta["dataset"] != args.dataset:
        return False
    if args.arch and meta["arch"] != args.arch:
        return False
    if args.stud and meta["stud"] != args.stud:
        return False
    if args.ipc is not None and meta["ipc"] != args.ipc:
        return False
    if args.weights and args.weights not in meta["tag"]:
        return False
    return True


def discover(args):
    """Yield (meta_dict, rows) for every curve file matching the filter."""
    pattern = os.path.join(args.curves_dir, "*.jsonl")
    for path in sorted(glob.glob(pattern)):
        meta = parse_filename(path)
        if meta is None:
            continue
        if not matches_filter(meta, args):
            continue
        rows = load_curve(path)
        if not rows:
            continue
        yield meta, rows


def config_label(meta):
    return f"{meta['dataset']}/{meta['arch']}->{meta['stud']}/ipc{meta['ipc']}/{meta['tag']}"


def aggregate(curves, metric):
    """Group curves by config, then mean+/-std per epoch across seeds.

    Returns dict {config_label: (epochs, mean, std, n_seeds)}.
    """
    by_config = {}
    for meta, rows in curves:
        key = (meta["dataset"], meta["arch"], meta["stud"], meta["ipc"], meta["tag"])
        by_config.setdefault(key, []).append(rows)

    out = {}
    for key, seed_rows in by_config.items():
        # Build epoch->list of values across seeds.
        per_epoch = {}
        for rows in seed_rows:
            for r in rows:
                e = r.get("epoch")
                v = r.get(metric)
                if e is None or v is None:
                    continue
                per_epoch.setdefault(int(e), []).append(float(v))
        if not per_epoch:
            continue
        epochs = sorted(per_epoch)
        vals = [per_epoch[e] for e in epochs]
        means = np.array([np.mean(v) for v in vals])
        stds = np.array([np.std(v, ddof=0) if len(v) > 1 else 0.0 for v in vals])
        n_seeds = max(len(v) for v in vals)
        meta = dict(zip(("dataset", "arch", "stud", "ipc", "tag"), key))
        out[config_label(meta)] = (np.array(epochs), means, stds, n_seeds)
    return out


def render(fig, axes, args):
    """(Re)draw the figure from current on-disk state. Returns True if any data."""
    curves = list(discover(args))
    metrics = args.y

    for ax in axes:
        ax.clear()

    if not curves:
        axes[0].set_title("no matching curve files")
        return False

    # Stable color across subplots: aggregate once per metric, but use the
    # same color cycle keyed by config_label.
    label_order = []
    for metric, ax in zip(metrics, axes):
        agg = aggregate(curves, metric)
        for label in agg:
            if label not in label_order:
                label_order.append(label)
    cmap = plt.get_cmap("tab10")
    label_color = {lbl: cmap(i % 10) for i, lbl in enumerate(label_order)}

    for metric, ax in zip(metrics, axes):
        agg = aggregate(curves, metric)
        for label, (epochs, means, stds, n) in agg.items():
            color = label_color[label]
            legend_label = f"{label} (n={n})"
            ax.plot(epochs, means, color=color, label=legend_label, linewidth=1.5)
            if n > 1:
                ax.fill_between(epochs, means - stds, means + stds,
                                color=color, alpha=0.18, linewidth=0)
        ax.set_ylabel(METRIC_YLABEL.get(metric, metric))
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("epoch")
    # Single legend on the top axis.
    if label_order:
        axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"training curves  ({time.strftime('%H:%M:%S')})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--curves-dir", default="logs/curves",
                    help="directory of per-run *.jsonl curve files (default: logs/curves)")
    ap.add_argument("--dataset", help="filter by dataset (e.g. cifar100)")
    ap.add_argument("--arch", help="filter by teacher arch (e.g. conv3)")
    ap.add_argument("--stud", help="filter by student arch (e.g. resnet18_modified)")
    ap.add_argument("--ipc", type=int, help="filter by ipc")
    ap.add_argument("--weights", help="substring filter on the weights tag (e.g. 'ockl0p3')")
    ap.add_argument("--y", default="train_loss,top1_err",
                    help=f"comma-separated metrics to plot, one subplot each. "
                         f"choices: {','.join(METRICS)}")
    ap.add_argument("--watch", type=int, metavar="SEC",
                    help="redraw every SEC seconds (live mode)")
    ap.add_argument("--out", metavar="FILE",
                    help="save figure to FILE.png (with --watch: overwrite every tick)")
    args = ap.parse_args()

    args.y = [m.strip() for m in args.y.split(",") if m.strip()]
    unknown = [m for m in args.y if m not in METRICS]
    if unknown:
        sys.exit(f"unknown --y metric(s): {unknown}; choose from {METRICS}")

    interactive = args.watch is not None and not args.out
    needs_window = args.watch is not None or not args.out
    if needs_window:
        # Try an interactive backend; fall back to Agg only if --out is set.
        for backend in ("TkAgg", "Qt5Agg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
        else:
            if not args.out:
                sys.exit("no interactive matplotlib backend available; "
                         "rerun with --out FILE.png")
        if interactive:
            plt.ion()

    fig, axes = plt.subplots(len(args.y), 1, sharex=True,
                             figsize=(9, 2.6 * len(args.y) + 0.6))
    if len(args.y) == 1:
        axes = [axes]

    def tick():
        had = render(fig, axes, args)
        if args.out:
            fig.savefig(args.out, dpi=130)
        return had

    if args.watch is None:
        had = tick()
        if not args.out:
            plt.show()
        elif had:
            print(f"wrote {args.out}")
        return

    # --watch loop
    try:
        while True:
            tick()
            if args.out:
                print(f"[{time.strftime('%H:%M:%S')}] refreshed {args.out}")
            if not args.out:
                fig.canvas.draw_idle()
                plt.pause(args.watch)
            else:
                time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
