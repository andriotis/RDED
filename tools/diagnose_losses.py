#!/usr/bin/env python
"""Weight-aware loss diagnosis: does each student-loss term ever beat KL?

Pools every logs/results*.jsonl (plus all_losses.jsonl / baselines.jsonl),
dedups to the most-recent row per (dataset, arch, stud, ipc, seed, loss-key),
and for each auxiliary loss prints its full per-weight delta-vs-KL table with
n and seed-std, then a keep/deprecate verdict.

The verdict is *weight-aware* on purpose: a single w=1.0 datapoint is not a
verdict (see docs/LOSSES.md). A loss is flagged DEPRECATE only when its best
delta across every weight tested stays within the seed-noise band of KL.

Usage:
  python tools/diagnose_losses.py
  python tools/diagnose_losses.py --dataset cifar100 --arch conv3 --stud conv3
  python tools/diagnose_losses.py --noise-band 0.3
"""

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
try:
    from validation.losses import LOSS_REGISTRY
    CURRENT_LOSSES = set(LOSS_REGISTRY)
except Exception:  # tool stays usable even if imports break
    CURRENT_LOSSES = set()


def loss_key(weights):
    """Canonical sorted ((name, w), ...) over nonzero weights; mirrors tools/analyze.py."""
    if not isinstance(weights, dict):
        return None
    active = tuple(sorted((n, float(w)) for n, w in weights.items() if float(w) > 0))
    return active or None


def row_extra(r):
    """Non-weight hyperparams that make two same-weight configs distinct.

    gce_q only matters when gce is active; legacy rows (no field) normalize to the
    0.7 default so they don't double-count against new default-q rows.
    """
    weights = r.get("weights") or {}
    q = r.get("gce_q")
    q_norm = 0.7 if q is None else round(float(q), 6)
    gce_active = float(weights.get("gce", 0)) > 0
    return (q_norm if gce_active else None,)


def extra_label(r):
    (q_norm,) = row_extra(r)
    bits = []
    if q_norm is not None and q_norm != 0.7:
        bits.append(f"q{q_norm:g}")
    return ("," + ",".join(bits)) if bits else ""


def load_rows(patterns):
    """Latest row per (dataset, arch, stud, ipc, seed, loss-key), by timestamp."""
    seen = {}
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(REPO_ROOT, pat)))
    for path in sorted(set(files)):
        try:
            fh = open(path)
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("best_top1") is None:
                    continue
                key = loss_key(r.get("weights"))
                if key is None:
                    continue
                ident = (
                    r.get("dataset"), r.get("arch"), r.get("stud"),
                    r.get("ipc"), r.get("seed"), key, row_extra(r),
                )
                prev = seen.get(ident)
                if prev is None or r.get("timestamp", "") >= prev.get("timestamp", ""):
                    seen[ident] = r
    return list(seen.values())


def classify(key):
    """Return (auxname, auxweight, has_kl) for keys that isolate one aux term, else None.

    Recognizes  ((kl,1.0),)                      -> baseline (auxname='kl')
                ((kl,1.0), (aux,w))              -> aux on KL
                ((aux,w),)                       -> aux alone (has_kl=False)
    Multi-aux combinations are returned as None (reported separately).
    """
    names = dict(key)
    if key == (("kl", 1.0),):
        return ("kl", 1.0, True)
    if "kl" in names and names["kl"] == 1.0 and len(key) == 2:
        aux = [(n, w) for n, w in key if n != "kl"][0]
        return (aux[0], aux[1], True)
    if len(key) == 1 and key[0][0] != "kl":
        return (key[0][0], key[0][1], False)
    return None


def mean_std(vals):
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--arch", default="conv3")
    ap.add_argument("--stud", default="conv3")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--num-crop", type=int, default=5)
    ap.add_argument("--mipc", type=int, default=300)
    ap.add_argument("--re-epochs", type=int, default=300)
    ap.add_argument("--noise-band", type=float, default=0.3,
                    help="pp band around KL within which a delta counts as a tie (default 0.3)")
    ap.add_argument("--patterns", nargs="*",
                    default=["logs/results*.jsonl", "logs/all_losses.jsonl", "logs/baselines.jsonl"])
    args = ap.parse_args()

    rows = load_rows(args.patterns)
    scope = [
        r for r in rows
        if (r.get("dataset"), r.get("arch"), r.get("stud")) == (args.dataset, args.arch, args.stud)
        and r.get("factor") == args.factor and r.get("num_crop") == args.num_crop
        and r.get("mipc") == args.mipc and r.get("re_epochs") == args.re_epochs
    ]
    if not scope:
        print(f"No rows for {args.dataset}/{args.arch}->{args.stud} "
              f"(f{args.factor} mipc{args.mipc} cr{args.num_crop} re{args.re_epochs})")
        return

    print("=" * 78)
    print(f"Loss diagnosis: {args.dataset}/{args.arch}->{args.stud} "
          f"(f{args.factor} mipc{args.mipc} cr{args.num_crop} re{args.re_epochs})")
    print(f"Noise band: +/-{args.noise_band:g}pp  |  pooled {len(rows)} unique runs")
    print("=" * 78)

    ipcs = sorted({r["ipc"] for r in scope})

    # best_delta[auxname] tracks the single best delta-vs-KL across all ipc x weights.
    best_delta = defaultdict(lambda: float("-inf"))
    swept = defaultdict(set)  # auxname -> set of (ipc, weight, has_kl) cells seen

    for ipc in ipcs:
        cells = [r for r in scope if r["ipc"] == ipc]
        # group by (auxname, weight, has_kl, extra) -> [best_top1 per seed].
        # The plain-KL anchor is kl-only with no extras (default q).
        groups = defaultdict(list)
        baseline = []
        for r in cells:
            info = classify(loss_key(r["weights"]))
            if info is None:
                continue
            name, w, has_kl = info
            lab = extra_label(r)
            if name == "kl" and lab == "":
                baseline.append(r["best_top1"])
            else:
                groups[(name, w, has_kl, lab)].append(r["best_top1"])
        if not baseline:
            continue
        bm, bs = mean_std(baseline)
        print(f"\n----- ipc={ipc}   KL baseline = {bm:.2f} +/- {bs:.2f} (n={len(baseline)}) -----")
        for name in sorted({k[0] for k in groups}):
            pts = sorted((w, has_kl, lab, *mean_std(groups[(n2, w, has_kl, lab)]),
                          len(groups[(n2, w, has_kl, lab)]))
                         for (n2, w, has_kl, lab) in groups if n2 == name)
            tag = "WEIGHT-SWEPT" if len({p[0] for p in pts}) >= 3 else "single-w"
            print(f"  {name:16s} [{tag}]")
            for w, has_kl, lab, m, s, n in pts:
                d = m - bm
                best_delta[name] = max(best_delta[name], d)
                swept[name].add((ipc, w, has_kl))
                comp = "kl+" if has_kl else "alone:"
                print(f"      {comp}{name}={w:<8g}{lab:<8} best={m:6.2f} +/-{s:4.2f}  "
                      f"Δ={d:+6.2f}  (n={n})")

    print("\n" + "=" * 78)
    print("VERDICT (weight-aware; only current LOSS_REGISTRY terms are actionable)")
    print("=" * 78)
    band = args.noise_band
    actionable = [n for n in sorted(best_delta) if not CURRENT_LOSSES or n in CURRENT_LOSSES]
    legacy = [n for n in sorted(best_delta) if CURRENT_LOSSES and n not in CURRENT_LOSSES]
    for name in actionable:
        weights_tested = sorted({w for (_, w, _) in swept[name]})
        n_w = len(weights_tested)
        bd = best_delta[name]
        wlist = ", ".join(f"{w:g}" for w in weights_tested)
        if bd > band:
            verdict = f"KEEP/INVESTIGATE (best Δ={bd:+.2f} > +{band:g} band; beats KL somewhere)"
        elif n_w >= 3:
            verdict = f"DEPRECATE (best Δ={bd:+.2f} over {n_w} weights {{{wlist}}}; never beats KL)"
        elif bd >= -band:
            verdict = (f"SWEEP-FIRST (ties KL: best Δ={bd:+.2f} at w∈{{{wlist}}}; "
                       f"sweep to confirm it cannot win)")
        else:
            verdict = (f"SWEEP-FIRST (hurts at tested w∈{{{wlist}}}: best Δ={bd:+.2f}; "
                       f"sweep lower weights before any verdict)")
        print(f"  {name:16s} {verdict}")
    if legacy:
        print("\n  legacy (not in current LOSS_REGISTRY; shown for context only):")
        for name in legacy:
            print(f"    {name:16s} best Δ={best_delta[name]:+.2f}")


if __name__ == "__main__":
    main()
