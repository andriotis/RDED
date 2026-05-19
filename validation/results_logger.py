"""Append one JSONL line per completed run to a shared results file.

A flat JSONL log keeps Phase B/C/D analysis trivial: pandas.read_json(lines=True)
gives the full table; group/filter columns produce the per-dataset accuracy/NC
tables in Phase D.
"""

import json
import os
import time


def log_run(args, best_top1, final_top1, nc_metrics):
    """Append one run's result to args.results_file (default: ./logs/results.jsonl)."""
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
        "student_loss": args.student_loss,
        "w_kl": args.w_kl,
        "w_ockl": args.w_ockl,
        "best_top1": float(best_top1),
        "final_top1": float(final_top1),
        "nc1": None if nc_metrics is None else nc_metrics.get("nc1"),
        "nc2": None if nc_metrics is None else nc_metrics.get("nc2"),
        "nc3": None if nc_metrics is None else nc_metrics.get("nc3"),
        "nc4": None if nc_metrics is None else nc_metrics.get("nc4"),
        "exp_name": args.exp_name,
    }
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return path
