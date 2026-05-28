#!/usr/bin/env python
"""Diagnose the existing sOCCE gamma sweep across all logs/results-*.jsonl files.

For every (dataset, arch, ipc) cell, produces:
  - best_top1 mean +/- std and delta vs matched KL baseline, across socce weights
  - final_top1 mean across socce weights (sanity check vs best_top1 selection)
  - NC2 mean across socce weights
  - Outcome classification (A/B/C/D) per the prompt's taxonomy:
        A monotone improvement up to some gamma*
        B even smallest gamma>0 degrades vs KL
        C inverted-U: peak at some interior gamma, regression at both ends
        D flat: no movement within +/-EPS_FLAT_PP of baseline

Reads every logs/results-*.jsonl, dedupes by (dataset, arch, stud, ipc, seed,
loss_key) keeping the most recent timestamp. Outputs:
  - markdown summary to stdout
  - logs/socce_diagnosis.md (same content)

Usage:
  python tools/analyze_socce_sweep.py
  python tools/analyze_socce_sweep.py --logs-dir logs --out logs/socce_diagnosis.md
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict


EPS_FLAT_PP = 0.3   # within +/-0.3pp of baseline = "flat" cell
EPS_MONO_PP = 0.1   # smallest meaningful difference between adjacent gammas


def load_all_rows(logs_dir):
    """Read every logs/results-*.jsonl in the directory, plus results.jsonl."""
    paths = sorted(glob.glob(os.path.join(logs_dir, "results-*.jsonl")))
    base = os.path.join(logs_dir, "results.jsonl")
    if os.path.exists(base):
        paths.append(base)
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def loss_key(row):
    """Sorted tuple of nonzero (name, weight) pairs."""
    weights = row.get("weights")
    if not isinstance(weights, dict):
        return None
    active = tuple(sorted((n, float(w)) for n, w in weights.items() if float(w) > 0))
    return active or None


def is_kl_baseline(key):
    return key == (("kl", 1.0),)


def socce_gamma(key):
    """If key is exactly KL=1.0 + socce=g, return g; else None."""
    if key is None or len(key) != 2:
        return None
    if key[0] != ("kl", 1.0):
        return None
    n, w = key[1]
    return float(w) if n == "socce" else None


def dedupe_latest(rows):
    """Keep most recent row per (dataset, arch, stud, ipc, seed, loss_key)."""
    best = {}
    for r in rows:
        k = loss_key(r)
        if k is None:
            continue
        cell = (r.get("dataset"), r.get("arch"), r.get("stud"),
                int(r.get("ipc", -1)), int(r.get("seed", -1)), k)
        ts = r.get("timestamp", "")
        if cell not in best or ts > best[cell].get("timestamp", ""):
            best[cell] = r
    return list(best.values())


def agg_mean_std(values):
    vs = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vs:
        return None, None, 0
    m = sum(vs) / len(vs)
    if len(vs) == 1:
        return m, 0.0, 1
    var = sum((v - m) ** 2 for v in vs) / (len(vs) - 1)
    return m, math.sqrt(var), len(vs)


def classify_outcome(baseline_mean, gamma_to_mean):
    """Return one of 'A', 'B', 'C', 'D', or 'unknown'.

    gamma_to_mean: dict gamma -> mean best_top1 (gamma > 0 only).
    """
    if baseline_mean is None or not gamma_to_mean:
        return "unknown"
    gammas = sorted(gamma_to_mean.keys())
    means = [gamma_to_mean[g] for g in gammas]
    deltas = [m - baseline_mean for m in means]

    # D: all within EPS_FLAT_PP of baseline
    if all(abs(d) < EPS_FLAT_PP for d in deltas):
        return "D"

    smallest_delta = deltas[0]
    # B: smallest gamma already hurts AND it's monotone non-improving after
    if smallest_delta < -EPS_FLAT_PP and all(deltas[i] <= deltas[i-1] + EPS_MONO_PP for i in range(1, len(deltas))):
        return "B"

    # A: monotone non-decreasing improvement (allow small dip at the tail) and best > baseline
    best_delta = max(deltas)
    if best_delta > EPS_FLAT_PP:
        # monotone-up to peak, then optional decline
        peak_idx = deltas.index(best_delta)
        up_phase = all(deltas[i] >= deltas[i-1] - EPS_MONO_PP for i in range(1, peak_idx + 1))
        if up_phase and peak_idx == len(deltas) - 1:
            return "A"
        if up_phase:
            return "C"  # rises then falls -> inverted-U

    # If we see a peak above baseline at some interior gamma, with regressions on
    # both sides, that's C; else B (mostly harm).
    if best_delta > 0:
        return "C"
    return "B"


def fmt_cell(mean, std, n):
    if n == 0 or mean is None:
        return "--"
    std = 0.0 if std is None or (isinstance(std, float) and math.isnan(std)) else std
    return f"{mean:.2f} +/- {std:.2f} (n={n})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs-dir", default="logs", help="directory holding results-*.jsonl")
    ap.add_argument("--out", default="logs/socce_diagnosis.md",
                    help="markdown writeup output path")
    args = ap.parse_args()

    rows = load_all_rows(args.logs_dir)
    if not rows:
        print(f"no rows found under {args.logs_dir}", file=sys.stderr)
        return 1
    rows = dedupe_latest(rows)

    # Bucket rows: per (dataset, arch, ipc), separate KL baseline rows and (gamma, row) pairs.
    cells = defaultdict(lambda: {"kl": [], "socce": defaultdict(list)})
    for r in rows:
        k = loss_key(r)
        cell = (r.get("dataset"), r.get("arch"), int(r.get("ipc", -1)))
        if is_kl_baseline(k):
            cells[cell]["kl"].append(r)
        else:
            g = socce_gamma(k)
            if g is not None:
                cells[cell]["socce"][g].append(r)

    # Drop cells with no socce coverage; sweep diagnosis is irrelevant there.
    cells = {c: v for c, v in cells.items() if v["socce"]}

    out_lines = []
    out_lines.append("# sOCCE sweep diagnosis")
    out_lines.append("")
    out_lines.append(f"Source: every `{args.logs_dir}/results-*.jsonl` "
                     "in this repo, deduped to the most-recent row per "
                     "(dataset, arch, stud, ipc, seed, loss_key).")
    out_lines.append("")
    out_lines.append(f"Outcome labels follow the Claude-Code prompt's taxonomy. Thresholds: "
                     f"flat within +/-{EPS_FLAT_PP}pp; monotone step tolerance {EPS_MONO_PP}pp.")
    out_lines.append("")

    cell_summaries = []   # (cell, outcome) for the final synthesis section

    for cell in sorted(cells.keys()):
        dataset, arch, ipc = cell
        kl_rows = cells[cell]["kl"]
        socce_buckets = cells[cell]["socce"]

        kl_top1 = [r["best_top1"] for r in kl_rows]
        kl_nc2 = [r.get("nc2") for r in kl_rows]
        kl_mean, kl_std, kl_n = agg_mean_std(kl_top1)
        kl_nc2_mean, _, _ = agg_mean_std(kl_nc2)

        out_lines.append(f"## {dataset}/{arch} IPC={ipc}")
        out_lines.append("")
        out_lines.append(f"- KL baseline best_top1: {fmt_cell(kl_mean, kl_std, kl_n)}; "
                         f"NC2 mean = {kl_nc2_mean if kl_nc2_mean is None else f'{kl_nc2_mean:.3f}'}")
        out_lines.append("")

        gammas_sorted = sorted(socce_buckets.keys())
        out_lines.append("| gamma | best_top1 | final_top1 | NC2 | delta vs KL (pp) |")
        out_lines.append("|------:|:----------|:-----------|:----|-----------------:|")
        gamma_to_mean = {}
        for g in gammas_sorted:
            bucket = socce_buckets[g]
            best = [r["best_top1"] for r in bucket]
            final = [r.get("final_top1") for r in bucket]
            nc2 = [r.get("nc2") for r in bucket]
            bm, bs, bn = agg_mean_std(best)
            fm, fs, fn = agg_mean_std(final)
            nm, ns, nn = agg_mean_std(nc2)
            delta = (bm - kl_mean) if (bm is not None and kl_mean is not None) else None
            gamma_to_mean[g] = bm
            out_lines.append(
                f"| {g:g} | {fmt_cell(bm, bs, bn)} | {fmt_cell(fm, fs, fn)} "
                f"| {('--' if nm is None else f'{nm:.3f}')} "
                f"| {('--' if delta is None else f'{delta:+.2f}')} |"
            )
        out_lines.append("")

        outcome = classify_outcome(kl_mean, gamma_to_mean)
        out_lines.append(f"**Outcome:** {outcome}")
        # NC2 trend descriptor (does NC2 drop monotonically with gamma?)
        nc2_means = []
        for g in gammas_sorted:
            nm, _, _ = agg_mean_std([r.get("nc2") for r in socce_buckets[g]])
            nc2_means.append(nm)
        if all(m is not None for m in nc2_means) and len(nc2_means) >= 2:
            # Robust trend: compare endpoints, allow small wobbles in the middle.
            tol = 0.01  # 0.01 of NC2 is the noise floor across a few seeds
            mostly_down = all(nc2_means[i] <= nc2_means[i-1] + tol for i in range(1, len(nc2_means)))
            mostly_up = all(nc2_means[i] >= nc2_means[i-1] - tol for i in range(1, len(nc2_means)))
            endpoint_delta = nc2_means[-1] - nc2_means[0]
            if mostly_down and endpoint_delta < -tol:
                trend = (f"DOWN with gamma ({nc2_means[0]:.3f} -> {nc2_means[-1]:.3f}); "
                         "OCCE mechanism firing as predicted")
            elif mostly_up and endpoint_delta > tol:
                trend = (f"UP with gamma ({nc2_means[0]:.3f} -> {nc2_means[-1]:.3f}); "
                         "OCCE pushing NC2 the WRONG way -- target structure conflicts with teacher")
            elif abs(endpoint_delta) <= tol:
                trend = f"flat ({nc2_means[0]:.3f} -> {nc2_means[-1]:.3f})"
            else:
                trend = (f"non-monotone ({nc2_means[0]:.3f} -> {nc2_means[-1]:.3f}, "
                         f"min {min(nc2_means):.3f}, max {max(nc2_means):.3f})")
            out_lines.append(f"**NC2 trend:** {trend}")
        out_lines.append("")
        cell_summaries.append((cell, outcome, gamma_to_mean, kl_mean))

    # Synthesis
    out_lines.append("## Synthesis")
    out_lines.append("")
    by_outcome = defaultdict(list)
    for cell, outcome, _gtm, _klm in cell_summaries:
        by_outcome[outcome].append(cell)
    for label in ["A", "B", "C", "D", "unknown"]:
        names = by_outcome.get(label, [])
        if not names:
            continue
        rendered = ", ".join(f"{d}/{a}/ipc={ipc}" for (d, a, ipc) in names)
        out_lines.append(f"- **{label}**: {rendered}")
    out_lines.append("")

    # Two questions the diagnosis must answer (per the plan)
    any_C = bool(by_outcome.get("C"))
    any_B = bool(by_outcome.get("B"))
    if any_C and not any_B:
        verdict = ("Pattern is **C** across cells: an interior small-gamma sweet spot exists but "
                   "larger gammas degrade. Suggests sOCCE acts as a weak regularizer at low strength "
                   "and overrides the teacher's similarity structure at high strength.")
    elif any_B and not any_C:
        verdict = ("Pattern is **B**: even the smallest gamma degrades top1. sOCCE's uniform-anticlass "
                   "target conflicts with the teacher's tempered soft-label structure at every weight tested.")
    elif any_B and any_C:
        verdict = ("Mixed: some cells are B (degrades at all gamma>0), others C (interior sweet spot). "
                   "Cell-dependent -- the IPC or architecture interacts with sOCCE.")
    else:
        verdict = "No clear B or C cells; check coverage."
    out_lines.append(f"**Q1 (failure mode):** {verdict}")
    out_lines.append("")

    # NC2 question: did the OCCE mechanism fire even when top1 didn't improve?
    tol = 0.01
    nc_down = nc_up = nc_flat = 0
    nc_total = 0
    for cell in cells:
        gammas_sorted = sorted(cells[cell]["socce"].keys())
        nc2_means = []
        for g in gammas_sorted:
            nm, _, _ = agg_mean_std([r.get("nc2") for r in cells[cell]["socce"][g]])
            nc2_means.append(nm)
        if all(m is not None for m in nc2_means) and len(nc2_means) >= 2:
            nc_total += 1
            endpoint = nc2_means[-1] - nc2_means[0]
            mostly_down = all(nc2_means[i] <= nc2_means[i-1] + tol for i in range(1, len(nc2_means)))
            mostly_up = all(nc2_means[i] >= nc2_means[i-1] - tol for i in range(1, len(nc2_means)))
            if mostly_up and endpoint > tol:
                nc_up += 1
            elif mostly_down and endpoint < -tol:
                nc_down += 1
            else:
                nc_flat += 1
    if nc_total > 0:
        out_lines.append(
            f"**Q2 (mechanism fired?):** NC2 trend across cells -- "
            f"DOWN (as OCCE predicts): {nc_down}/{nc_total}, "
            f"UP (anti-prediction): {nc_up}/{nc_total}, "
            f"flat/mixed: {nc_flat}/{nc_total}. "
        )
        if nc_up > nc_down:
            out_lines.append(
                "NC2 goes the WRONG direction in more cells than the right one. sOCCE on cutmix "
                "hard-label targets is **not** producing the geometric outcome the OCCE paper "
                "established for one-hot training -- the target structure is incompatible with "
                "the soft-label distillation regime. This is consistent with the memory's "
                "diagnosis (target choice is the bottleneck) and points the next experiment "
                "away from more anticlass logit-space penalties."
            )
    out_lines.append("")

    md = "\n".join(out_lines) + "\n"
    print(md)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
