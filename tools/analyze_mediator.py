"""Mediator analysis: does the distilled set's teacher-induced geometry predict the student's
downstream OOD robustness?

The causal chain under test is
    selection -> distilled-set within-class covariance recovery -> student feature geometry
    -> Mahalanobis OOD-AUROC.
This joins the *teacher-side* geometry of each distilled set (cov_gap_to_real and nc1, logged
by tools/diagnose_geometry.py) with the *student-side* Mahalanobis OOD-AUROC (logged by
main.py --diagnostics), matched on (dataset, arch, ipc, seed, select_method), and reports the
within-cell rank correlation between the mediator and the outcome.

cov_gap_to_real is exactly the objective the momentmatch selector minimizes, so a strong
negative cov_gap -> auroc_maha correlation both (a) explains the trust gain mechanistically and
(b) justifies optimizing it directly.

Usage:
    python tools/analyze_mediator.py                                   # default geom + results globs
    python tools/analyze_mediator.py --geom logs/mediator_geometry.jsonl \
        --results logs/results_select_conv3.jsonl logs/results_momentmatch.jsonl
"""

import argparse
import glob
import json
from collections import defaultdict

import numpy as np


def method_of(exp_name):
    for m in ("covmatch", "stratified", "random", "momentmatch", "relmatch"):
        if exp_name and f"_sel{m}" in exp_name:
            return m
    return "stock"


def spearman(x, y):
    """Spearman rho without scipy (Pearson on ranks)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def load_geometry(path):
    """(dataset,arch,ipc,seed,method) -> {cov_gap, nc1_distilled, nc1_real}."""
    geom = defaultdict(dict)
    try:
        rows = [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return geom
    for r in rows:
        if "cov_gap_to_real" not in r and not str(r.get("subject", "")).startswith("real-train"):
            continue
        key = (r.get("subset"), r.get("arch"), r.get("ipc"), r.get("seed"), r.get("select_method"))
        subj = str(r.get("subject", ""))
        if subj.startswith("distilled-set"):
            geom[key]["cov_gap"] = r.get("cov_gap_to_real")
            geom[key]["nc1_distilled"] = r.get("nc1")
        elif subj.startswith("real-train"):
            # real subset geometry is method-independent; attach to every method of the cell
            geom[key]["nc1_real"] = r.get("nc1")
    return geom


def load_results(paths):
    """(dataset,arch,ipc,seed,method) -> auroc_maha (deduped, last wins)."""
    out = {}
    for p in paths:
        try:
            for line in open(p):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not r.get("diag"):
                    continue
                key = (r["dataset"], r["arch"], r["ipc"], r.get("seed"), method_of(r.get("exp_name")))
                out[key] = r["diag"].get("auroc_maha")
        except FileNotFoundError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser("analyze_mediator")
    ap.add_argument("--geom", default="logs/mediator_geometry.jsonl")
    ap.add_argument("--results", nargs="*", default=None)
    args = ap.parse_args()
    results_paths = args.results or (sorted(glob.glob("logs/results_select*.jsonl"))
                                     + sorted(glob.glob("logs/results_momentmatch*.jsonl")))

    geom = load_geometry(args.geom)
    res = load_results(results_paths)

    # nc1_real is logged per (cell,seed,method); backfill a cell-level value so every method
    # has a real reference even if its own real-row is missing.
    real_by_cell = {}
    for (ds, arch, ipc, seed, m), g in geom.items():
        if g.get("nc1_real") is not None:
            real_by_cell[(ds, arch, ipc, seed)] = g["nc1_real"]

    # cell -> list of (method, seed, cov_gap, nc1_gap, auroc_maha)
    cells = defaultdict(list)
    for (ds, arch, ipc, seed, m), g in geom.items():
        cov_gap = g.get("cov_gap")
        amaha = res.get((ds, arch, ipc, seed, m))
        if cov_gap is None or amaha is None:
            continue
        nc1_d = g.get("nc1_distilled")
        nc1_r = g.get("nc1_real", real_by_cell.get((ds, arch, ipc, seed)))
        nc1_gap = abs(nc1_d - nc1_r) if (nc1_d is not None and nc1_r is not None) else None
        cells[(ds, arch, ipc)].append((m, seed, cov_gap, nc1_gap, amaha))

    if not cells:
        raise SystemExit(
            f"no joined (geometry x results) points. Geometry rows with cov_gap_to_real: "
            f"{sum(1 for g in geom.values() if g.get('cov_gap') is not None)}; "
            f"result keys: {len(res)}. Run tools/diagnose_geometry.py per method first."
        )

    print(f"geom={args.geom}  results={results_paths}\n")
    print(f"{'cell':32s} {'n':>3s} {'methods':>34s}  {'rho(covgap,maha)':>17s} {'rho(nc1gap,maha)':>17s}")
    pooled_cg, pooled_ng, pooled_y = [], [], []
    for cell in sorted(cells):
        pts = cells[cell]
        methods = ",".join(sorted({p[0] for p in pts}))
        cg = [p[2] for p in pts]
        ng = [p[3] for p in pts]
        y = [p[4] for p in pts]
        rho_cg = spearman(cg, y)
        have_ng = all(v is not None for v in ng)
        rho_ng = spearman(ng, y) if have_ng else float("nan")
        label = f"{cell[0]}/{cell[1]}/ipc{cell[2]}"
        print(f"{label:32s} {len(pts):3d} {methods:>34s}  {rho_cg:17.3f} {rho_ng:17.3f}")
        # pool with within-cell z-scoring so cells with different baselines are comparable
        if len(pts) >= 2 and np.std(cg) > 0:
            pooled_cg += list((np.array(cg) - np.mean(cg)) / (np.std(cg) + 1e-12))
            pooled_y += list((np.array(y) - np.mean(y)) / (np.std(y) + 1e-12))
            if have_ng and np.std(ng) > 0:
                pooled_ng += list((np.array(ng) - np.mean(ng)) / (np.std(ng) + 1e-12))

    print()
    if pooled_cg:
        print(f"POOLED (within-cell z-scored)  rho(covgap, maha) = {spearman(pooled_cg, pooled_y):.3f}  "
              f"(n={len(pooled_cg)});  expect NEGATIVE (smaller cov-gap -> higher Mahalanobis AUROC)")
    print("\nper-point detail (method, seed, cov_gap, nc1_gap, auroc_maha):")
    for cell in sorted(cells):
        print(f"  {cell[0]}/{cell[1]}/ipc{cell[2]}")
        for m, seed, cg, ng, y in sorted(cells[cell]):
            ngs = "  n/a" if ng is None else f"{ng:.3f}"
            print(f"    {m:12s} seed{seed}  cov_gap={cg:.4f}  nc1_gap={ngs}  auroc_maha={y:.4f}")


if __name__ == "__main__":
    main()
