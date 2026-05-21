#!/usr/bin/env python
"""YAML-driven experiment sweep runner.

Reads a sweep config (see scripts/sweeps/*.yaml), expands the cartesian
product across list-valued fields per experiment block, and calls
scripts/experiment.sh for each cell.

Schema (per experiment block):
  Scalar/list fields handled identically: dataset, arch, stud_arch, seed, ipc,
  re_epochs, factor, num_crop, mipc, skip_synth.
  `weights:` may be either:
    - a dict, possibly with list-valued entries (cartesian-product across them):
        weights:
          kl: 1.0
          ockl: [0.1, 0.3, 1.0]    # 3 cells (3 ockl values)
    - a list of dicts (structural axis, one cell per dict):
        weights:
          - {kl: 1.0}
          - {kl: 1.0, ockl: 1.0}    # 2 cells

Top-level behavior:
  - Default: rotate logs/results.jsonl -> logs/results-YYYYMMDD[-N].jsonl
    once at the start, then loop and append fresh rows.
  - --resume: skip rotation; pass --resume through to experiment.sh.
  - --dry-run: pass --dry-run through; rotation message printed but not enacted.

Usage:
  python scripts/run_sweep.py scripts/sweeps/loss_matrix.yaml
  python scripts/run_sweep.py scripts/sweeps/gamma_sweep.yaml --resume
  python scripts/run_sweep.py scripts/sweeps/cross_arch.yaml --dry-run
"""

import argparse
import datetime as dt
import itertools
import os
import shutil
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Flag mapping: YAML key -> experiment.sh flag (for the simple scalar fields).
# `weights` is handled separately (dict / list-of-dicts).
SCALAR_FLAG_KEYS = [
    ("dataset",    "--dataset"),
    ("arch",       "--arch"),
    ("stud_arch",  "--stud-arch"),
    ("seed",       "--seed"),
    ("ipc",        "--ipc"),
    ("re_epochs",  "--re-epochs"),
    ("factor",     "--factor"),
    ("num_crop",   "--num-crop"),
    ("mipc",       "--mipc"),
]
BOOL_FLAG_KEYS = [
    ("skip_synth", "--skip-synth"),
]
ALL_KEYS = (
    {k for k, _ in SCALAR_FLAG_KEYS}
    | {k for k, _ in BOOL_FLAG_KEYS}
    | {"name", "weights"}
)


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        sys.exit(f"config must be a mapping at top level: {path}")
    defaults = cfg.get("defaults") or {}
    experiments = cfg.get("experiments") or []
    if not experiments:
        sys.exit(f"no experiments defined in {path}")
    for i, exp in enumerate(experiments):
        if not isinstance(exp, dict):
            sys.exit(f"experiments[{i}] must be a mapping in {path}")
        unknown = set(exp) - ALL_KEYS
        if unknown:
            sys.exit(f"experiments[{i}] has unknown keys: {sorted(unknown)}")
    return defaults, experiments


def weights_alternatives(spec):
    """Expand a `weights:` spec into a list of concrete dicts (one per cell).

    - dict shape: cartesian product over list-valued entries.
    - list-of-dicts shape: passthrough.
    - None / missing: single empty dict (no overrides; main.py uses registry defaults).
    """
    if spec is None:
        return [{}]
    if isinstance(spec, list):
        for i, d in enumerate(spec):
            if not isinstance(d, dict):
                sys.exit(f"weights[{i}] must be a mapping (list-of-dicts shape)")
        return [dict(d) for d in spec]
    if isinstance(spec, dict):
        keys = list(spec)
        value_lists = []
        for k in keys:
            v = spec[k]
            value_lists.append(v if isinstance(v, list) else [v])
        return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]
    sys.exit(f"weights must be a dict or list-of-dicts, got: {type(spec).__name__}")


def expand(experiment, defaults):
    """Yield parameter dicts from the cartesian product of list-valued fields.

    `weights` is merged specially when both defaults and experiment provide a dict
    (so an experiment can extend the default weight set without re-listing kl):
      defaults.weights = {kl: 1.0}, experiment.weights = {ockl: [0.1, 0.3, 1.0]}
        -> effective {kl: 1.0, ockl: [0.1, 0.3, 1.0]}
    For list-of-dicts shape on either side, the experiment value replaces
    defaults entirely (no element-wise merge — list-of-dicts is a structural axis).
    """
    merged = {**defaults, **experiment}
    d_w = defaults.get("weights")
    e_w = experiment.get("weights")
    if isinstance(d_w, dict) and isinstance(e_w, dict):
        merged["weights"] = {**d_w, **e_w}
    weights_spec = merged.pop("weights", None)
    weight_alts = weights_alternatives(weights_spec)

    list_keys, list_values = [], []
    fixed = {}
    for k, v in merged.items():
        if k == "name":
            continue
        if isinstance(v, list):
            list_keys.append(k)
            list_values.append(v)
        else:
            fixed[k] = v

    if not list_keys and len(weight_alts) == 1:
        yield {**fixed, "weights": weight_alts[0]}
        return

    combos = itertools.product(*list_values) if list_keys else [()]
    for combo in combos:
        base = dict(fixed)
        for k, v in zip(list_keys, combo):
            base[k] = v
        for w in weight_alts:
            cell = dict(base)
            cell["weights"] = w
            yield cell


def build_argv(params):
    argv = ["bash", "scripts/experiment.sh"]
    for key, flag in SCALAR_FLAG_KEYS:
        if key in params and params[key] is not None:
            argv += [flag, str(params[key])]
    for key, flag in BOOL_FLAG_KEYS:
        if params.get(key):
            argv.append(flag)
    for name, w in sorted(params.get("weights", {}).items()):
        flag = f"--w-{name.replace('_', '-')}"
        argv += [flag, str(w)]
    return argv


def rotate_results(dry_run):
    src = os.path.join(REPO_ROOT, "logs", "results.jsonl")
    if not os.path.exists(src):
        return
    stamp = dt.datetime.now().strftime("%Y%m%d")
    candidate = os.path.join(REPO_ROOT, "logs", f"results-{stamp}.jsonl")
    n = 1
    while os.path.exists(candidate):
        candidate = os.path.join(REPO_ROOT, "logs", f"results-{stamp}-{n}.jsonl")
        n += 1
    if dry_run:
        print(f"[dry-run] would rotate {src} -> {candidate}", flush=True)
    else:
        shutil.move(src, candidate)
        print(f"Rotated {src} -> {candidate}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="path to a sweep YAML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="print expanded commands without executing")
    ap.add_argument("--resume", action="store_true",
                    help="skip rotation; pass --resume to experiment.sh per cell")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)

    defaults, experiments = load_config(args.config)

    if args.resume:
        print("Resume mode: rotation skipped; cells already in results.jsonl will be skipped.",
              flush=True)
    else:
        rotate_results(args.dry_run)

    expanded = []
    for exp in experiments:
        name = exp.get("name", "<unnamed>")
        for params in expand(exp, defaults):
            expanded.append((name, params))

    print(f"Sweep: {args.config} | {len(experiments)} experiment blocks | {len(expanded)} cells",
          flush=True)

    n_ran = n_failed = 0
    for name, params in expanded:
        argv = build_argv(params)
        if args.resume:
            argv.append("--resume")
        if args.dry_run:
            argv.append("--dry-run")
        try:
            result = subprocess.run(argv, cwd=REPO_ROOT)
        except KeyboardInterrupt:
            print("\nInterrupted by user.", file=sys.stderr)
            sys.exit(130)
        rc = result.returncode
        if rc != 0:
            n_failed += 1
            print(f"[fail] {name}: experiment.sh exited {rc}", file=sys.stderr)
        else:
            n_ran += 1

    print(f"Sweep complete: {n_ran}/{len(expanded)} ok, {n_failed} failed")
    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
