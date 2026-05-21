"""Append one JSONL line per training epoch to logs/curves/{run_key}.jsonl.

Mirrors validation/results_logger.py but at per-epoch granularity, so a watcher
(tools/plot_curve.py) can render the training curve live or post-hoc.

Schema (one row per epoch):
  timestamp, epoch, train_loss, top1_err, top5_err, train_time
"""

import json
import os
import time

from validation.losses import LOSS_REGISTRY
from validation.run_key import canonical_run_key


def _user_weights(args):
    return {name: float(getattr(args, f"w_{name}")) for name in LOSS_REGISTRY}


def init_curve(args):
    """Compute the curve path on args, ensure parent dir, truncate any stale file.

    Resume is handled at the results.jsonl layer (experiment.sh skips a run if
    its result row already exists); when we reach here the run is fresh, so any
    pre-existing curve file is from an aborted attempt and is replaced.
    """
    key, _, _ = canonical_run_key(
        args.subset, args.arch_name, args.stud_name,
        args.ipc, args.seed, _user_weights(args),
    )
    args.curve_file = os.path.join("logs", "curves", f"{key}.jsonl")
    os.makedirs(os.path.dirname(args.curve_file), exist_ok=True)
    open(args.curve_file, "w").close()
    return args.curve_file


def log_epoch(args, epoch, train_loss, top1_err, top5_err, train_time):
    """Append one row to args.curve_file and flush, so live tailers see it."""
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "top1_err": float(top1_err),
        "top5_err": float(top5_err),
        "train_time": float(train_time),
    }
    with open(args.curve_file, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
