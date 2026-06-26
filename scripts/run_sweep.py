#!/usr/bin/env python
"""YAML-driven experiment sweep runner.

Reads a sweep config (see scripts/sweeps/*.yaml), expands the cartesian
product across list-valued fields per experiment block, and calls
scripts/experiment.sh for each cell.

Schema (per experiment block):
  Scalar/list fields handled identically: dataset, arch, stud_arch, seed, ipc,
  re_epochs, factor, num_crop, mipc, skip_synth.
  Stage-1 selector + diagnostics (optional; omit => plain stock RDED, accuracy only):
  select_method, select_k, select_beta, select_quality,
  momentmatch_mean_weight, relmatch_diag_weight, diagnostics (bool), ood_sets,
  fit_ipc, results_file. All are list-expandable (e.g. relmatch_diag_weight: [0.0, 1.0]).
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
    once at the start, then run cells and append fresh rows.
  - --resume: skip rotation; pass --resume through to experiment.sh.
  - --dry-run: pass --dry-run through; rotation message printed but not enacted.

Execution (GPU-agnostic, throughput-first):
  Cells run in parallel, packed across whatever GPUs the server has. GPUs are
  auto-detected (CUDA_VISIBLE_DEVICES if set, else nvidia-smi, else torch);
  --gpus overrides. --per-gpu N runs N cells concurrently per GPU (these models
  underutilize a card, so >1 is usually a big win); --gpu-mem-mb caps cells per
  GPU to fit an estimated per-cell VRAM budget. Each cell is pinned to one GPU
  via CUDA_VISIBLE_DEVICES (CUDA_DEVICE_ORDER=PCI_BUS_ID), so no
  run fans a tiny batch across all cards. --shard i --num-shards N splits one
  sweep across several machines. Per-cell synth dirs (experiment.sh seed-scopes
  them) plus a synth lock keyed by RDED_SYNTH_RUN_ID make concurrent cells safe.

Usage:
  python scripts/run_sweep.py scripts/sweeps/loss_matrix.yaml --per-gpu 3
  python scripts/run_sweep.py scripts/sweeps/gamma_sweep.yaml --resume
  python scripts/run_sweep.py scripts/sweeps/cross_arch.yaml --dry-run
  python scripts/run_sweep.py scripts/sweeps/loss_matrix.yaml --gpus 1,2 --per-gpu 2
  # split across 2 servers (run one per machine):
  python scripts/run_sweep.py sweep.yaml --shard 0 --num-shards 2
  python scripts/run_sweep.py sweep.yaml --shard 1 --num-shards 2
"""

import argparse
import datetime as dt
import itertools
import os
import queue
import shutil
import subprocess
import sys
import threading

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
    ("monitor",    "--monitor"),
    ("gce_q",      "--gce-q"),
    # Stage-1 selector + trustworthiness-diagnostics knobs (omit => stock RDED, accuracy-only).
    ("select_method",           "--select-method"),
    ("select_k",                "--select-k"),
    ("select_beta",             "--select-beta"),
    ("select_quality",          "--select-quality"),
    ("momentmatch_mean_weight", "--momentmatch-mean-weight"),
    ("relmatch_diag_weight",    "--relmatch-diag-weight"),
    ("ood_sets",                "--ood-sets"),
    ("fit_ipc",                 "--fit-ipc"),
    ("results_file",            "--results-file"),
]
BOOL_FLAG_KEYS = [
    ("skip_synth",  "--skip-synth"),
    ("diagnostics", "--diagnostics"),
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


def detect_gpus():
    """Ordered list of GPU ids (as strings) this sweep may use.

    Server-agnostic: honors a pre-set CUDA_VISIBLE_DEVICES (respects a
    shared-server restriction), else queries nvidia-smi, else falls back to
    torch.cuda. Returns [] when none can be found.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip() != "":
        return [tok.strip() for tok in env.split(",") if tok.strip() != ""]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        ids = [ln.strip() for ln in out.splitlines() if ln.strip() != ""]
        if ids:
            return ids
    except Exception:
        pass
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def gpu_free_mem_mb(gpu_id):
    """Free memory (MiB) on one GPU via nvidia-smi; None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", str(gpu_id)],
            text=True, stderr=subprocess.DEVNULL,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="path to a sweep YAML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="print expanded commands without executing")
    ap.add_argument("--resume", action="store_true",
                    help="skip rotation; pass --resume to experiment.sh per cell")
    ap.add_argument("--gpus", default="",
                    help="comma-separated GPU ids to use "
                         "(default: auto-detect via CUDA_VISIBLE_DEVICES / nvidia-smi)")
    ap.add_argument("--per-gpu", type=int, default=1,
                    help="max concurrent cells per GPU (default 1). These models "
                         "underutilize a card, so 2-4 is usually a large speedup.")
    ap.add_argument("--gpu-mem-mb", type=int, default=0,
                    help="if >0, estimated peak VRAM (MiB) per cell; the scheduler "
                         "reads each GPU's free memory once and statically packs it "
                         "to fit (capped by --per-gpu), reserving the budget for each "
                         "cell's whole lifetime so deferred training peaks can't OOM.")
    ap.add_argument("--shard", type=int, default=0,
                    help="this server's shard index in [0, num-shards) for splitting "
                         "one sweep across machines (default 0)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="number of servers splitting this sweep (default 1 = no split)")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)

    if args.per_gpu < 1:
        sys.exit("--per-gpu must be >= 1")
    if args.num_shards < 1 or not (0 <= args.shard < args.num_shards):
        sys.exit("require --num-shards >= 1 and 0 <= --shard < --num-shards")

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

    # Assign stable global cell ids BEFORE sharding so --cell-id is identical no
    # matter how the sweep is split across servers, then keep only this shard.
    indexed = list(enumerate(expanded))
    if args.num_shards > 1:
        indexed = indexed[args.shard::args.num_shards]

    if args.gpus.strip():
        gpus = [t.strip() for t in args.gpus.split(",") if t.strip() != ""]
    else:
        gpus = detect_gpus()
    if not gpus:
        if args.dry_run:
            gpus = ["0"]
        else:
            sys.exit("no GPUs detected; pass --gpus or set CUDA_VISIBLE_DEVICES")

    # Slots per GPU. With --gpu-mem-mb, treat it as the *peak* VRAM one cell
    # needs and statically pack each GPU to fit — the reservation is held for the
    # cell's whole lifetime, so cells can't over-admit during their cheap synth
    # phase and then OOM once they all reach training. Free memory is read once,
    # here; there is no per-launch polling (which could hang or fail open).
    # Children run with CUDA_DEVICE_ORDER=PCI_BUS_ID so these nvidia-smi indices
    # address the same physical GPUs CUDA will pin to.
    HEADROOM_MB = 512
    if args.gpu_mem_mb and not args.dry_run:
        gpu_slots = {}
        for g in gpus:
            free = gpu_free_mem_mb(g)
            if free is None:
                gpu_slots[g] = args.per_gpu
                print(f"[warn] gpu {g}: could not read free memory; "
                      f"falling back to --per-gpu={args.per_gpu}",
                      file=sys.stderr, flush=True)
            else:
                cap = max(0, (free - HEADROOM_MB) // args.gpu_mem_mb)
                gpu_slots[g] = min(args.per_gpu, cap)
        gpus = [g for g in gpus if gpu_slots[g] > 0]
        if not gpus:
            sys.exit(f"no GPU can fit a {args.gpu_mem_mb} MiB cell "
                     f"(need {args.gpu_mem_mb}+{HEADROOM_MB} MiB free on some GPU)")
    else:
        gpu_slots = {g: args.per_gpu for g in gpus}

    slots = sum(gpu_slots[g] for g in gpus)
    sweep_name = os.path.splitext(os.path.basename(args.config))[0]
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"

    shard_note = "" if args.num_shards == 1 else f" | shard {args.shard}/{args.num_shards}"
    print(
        f"Sweep: {args.config} | {len(experiments)} blocks | {len(expanded)} cells"
        f"{shard_note} ({len(indexed)} on this server)",
        flush=True,
    )
    layout = " ".join(f"{g}:{gpu_slots[g]}" for g in gpus)
    print(
        f"GPUs(slots): {layout} | concurrent slots: {slots}"
        + (f" | per-cell budget: {args.gpu_mem_mb} MiB" if args.gpu_mem_mb else "")
        + f" | run_id: {run_id}",
        flush=True,
    )

    # --- dispatch ---------------------------------------------------------
    # `slots` worker threads pull cells from a shared queue; each cell is placed
    # on the least-loaded GPU that still has a free slot. Assigning by load
    # (rather than pinning threads to a GPU) spreads cells one-per-GPU first and
    # only stacks a second cell on a card once every card is busy — so a sweep
    # with fewer cells than slots still fans out across all GPUs. Threads are
    # fine: the real work is in the child processes.
    cell_q = queue.Queue()
    for item in indexed:
        cell_q.put(item)

    stats_lock = threading.Lock()
    procs_lock = threading.Lock()
    sched_lock = threading.Lock()    # guards inflight while choosing a GPU
    active_procs = set()             # live child Popens, for clean Ctrl-C teardown
    inflight = {g: 0 for g in gpus}  # cells currently running per GPU
    stop = threading.Event()
    counts = {"ran": 0, "failed": 0}

    def acquire_gpu():
        # Least-loaded GPU with a free slot. Total worker threads == sum(slots),
        # so a worker reaching here always finds at least one GPU under its cap.
        with sched_lock:
            g = min((x for x in gpus if inflight[x] < gpu_slots[x]),
                    key=lambda x: inflight[x])
            inflight[g] += 1
            return g

    def release_gpu(g):
        with sched_lock:
            inflight[g] -= 1

    def run_cell(cell_idx, name, params, gpu):
        argv = build_argv(params)
        argv += ["--sweep-name", sweep_name, "--cell-id", f"{name}:{cell_idx:04d}"]
        if args.resume:
            argv.append("--resume")
        if args.dry_run:
            argv.append("--dry-run")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "RDED_SYNTH_RUN_ID": run_id,
        }
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=env)
        with procs_lock:
            active_procs.add(proc)
            if stop.is_set():        # interrupt landed between dequeue and launch
                proc.terminate()
        try:
            return proc.wait()
        finally:
            with procs_lock:
                active_procs.discard(proc)

    def worker():
        while not stop.is_set():
            try:
                cell_idx, (name, params) = cell_q.get_nowait()
            except queue.Empty:
                return
            gpu = acquire_gpu()
            try:
                rc = run_cell(cell_idx, name, params, gpu)
            except Exception as e:   # a failed launch must not silently kill the slot
                with stats_lock:
                    counts["failed"] += 1
                print(f"[fail] {name} (gpu {gpu}): launch error: {e}",
                      file=sys.stderr, flush=True)
                continue
            finally:
                release_gpu(gpu)
            with stats_lock:
                if rc != 0:
                    counts["failed"] += 1
                    print(f"[fail] {name} (gpu {gpu}): experiment.sh exited {rc}",
                          file=sys.stderr, flush=True)
                else:
                    counts["ran"] += 1

    threads = []
    for _ in range(slots):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        stop.set()
        print("\nInterrupted by user; terminating running cells...", file=sys.stderr, flush=True)
        with procs_lock:
            for p in list(active_procs):
                p.terminate()
        sys.exit(130)

    done = counts["ran"] + counts["failed"]
    print(f"Sweep complete: {counts['ran']}/{done} ok, {counts['failed']} failed")
    sys.exit(0 if counts["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
