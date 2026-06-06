"""Append one JSONL line per completed run to a shared results file.

A flat JSONL log keeps analysis trivial: pandas.read_json(lines=True) gives the
full table; group/filter columns produce the analysis tables.

Schema (one row per completed run):
  timestamp, dataset, arch, stud, ipc, mipc, factor, num_crop, re_epochs, seed,
  weights: {<loss-name>: <weight>, ...},   # one entry per LOSS_REGISTRY entry
  gce_q,                                   # non-weight hyperparam that
                                           # distinguishes otherwise-identical configs
  best_top1, final_top1, nc1, nc2, nc3, nc4, exp_name
"""

import fcntl
import json
import os
import time

from validation.losses import LOSS_REGISTRY


def log_run(args, best_top1, final_top1, nc_metrics, diagnostics=None):
    """Append one run's result to args.results_file (default: ./logs/results.jsonl).

    `diagnostics` (optional): the dict from validation.diagnostics.run_diagnostics
    (ECE / OSCR / AUROC / FPR95 / NC). Stored nested under "diag" so it never
    collides with the flat columns; None when --diagnostics is off.
    """
    path = getattr(args, "results_file", None) or os.path.join("logs", "results.jsonl")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.subset,
        "arch": args.arch_name,
        "stud": args.stud_name,
        "ipc": args.ipc,
        "mipc": args.mipc,
        "factor": args.factor,
        "num_crop": args.num_crop,
        "re_epochs": args.re_epochs,
        "seed": args.seed,
        "weights": {name: float(getattr(args, f"w_{name}")) for name in LOSS_REGISTRY},
        "gce_q": float(getattr(args, "gce_q", 0.7)),
        "best_top1": float(best_top1),
        "final_top1": float(final_top1),
        "nc1": None if nc_metrics is None else nc_metrics.get("nc1"),
        "nc2": None if nc_metrics is None else nc_metrics.get("nc2"),
        "nc3": None if nc_metrics is None else nc_metrics.get("nc3"),
        "nc4": None if nc_metrics is None else nc_metrics.get("nc4"),
        "diag": diagnostics,
        "exp_name": args.exp_name,
    }
    # Parallel sweep cells may finish near-simultaneously; take an exclusive
    # lock so concurrent appends can't interleave or truncate a JSONL line.
    line = json.dumps(row) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return path
