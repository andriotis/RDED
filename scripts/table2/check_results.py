"""
Compare our Top-1 results in logs/table2/results.csv against the paper's
Table 2 RDED values (mean ± std). Flags any cell where |ours - mean|/std > 3.

Paper values are from Table 2 of arXiv:2312.03526 (RDED). Each tuple is
(mean, std) and refers to RDED ("Ours") in the corresponding column.

Run from RDED/:
    python scripts/table2/check_results.py
    python scripts/table2/check_results.py --z 2     # stricter threshold
"""
import argparse
import csv
import os
import re

# paper[subset][ipc][arch] = (mean, std)
# arch: "conv" = Conv-{3,4,5,6} per dataset; "rn18"; "rn101".
PAPER = {
    "cifar10": {
        1:  {"conv": (23.5, 0.3), "rn18": (22.9, 0.4), "rn101": (18.7, 0.1)},
        10: {"conv": (50.2, 0.3), "rn18": (37.1, 0.3), "rn101": (33.7, 0.3)},
        50: {"conv": (68.4, 0.1), "rn18": (62.1, 0.1), "rn101": (51.6, 0.4)},
    },
    "cifar100": {
        1:  {"conv": (19.6, 0.3), "rn18": (11.0, 0.3), "rn101": (10.8, 0.1)},
        10: {"conv": (48.1, 0.3), "rn18": (42.6, 0.2), "rn101": (41.1, 0.2)},
        50: {"conv": (57.0, 0.2), "rn18": (62.6, 0.1), "rn101": (63.4, 0.3)},
    },
    "tinyimagenet": {
        1:  {"conv": (12.0, 0.1), "rn18": (9.7,  0.4), "rn101": (3.8,  0.1)},
        10: {"conv": (39.6, 0.1), "rn18": (41.9, 0.2), "rn101": (22.9, 3.3)},
        50: {"conv": (47.6, 0.2), "rn18": (58.2, 0.1), "rn101": (41.2, 0.4)},
    },
    "imagenet-nette": {
        1:  {"conv": (33.8, 0.8), "rn18": (35.8, 1.2), "rn101": (25.1, 2.7)},
        10: {"conv": (63.2, 0.7), "rn18": (61.4, 0.4), "rn101": (54.0, 0.4)},
        50: {"conv": (83.8, 0.2), "rn18": (80.4, 0.4), "rn101": (75.0, 1.2)},
    },
    "imagenet-woof": {
        1:  {"conv": (18.5, 0.3), "rn18": (20.8, 1.2), "rn101": (19.6, 1.8)},
        10: {"conv": (40.6, 2.0), "rn18": (38.5, 2.1), "rn101": (31.3, 1.3)},
        50: {"conv": (68.5, 0.7), "rn18": (68.5, 0.7), "rn101": (59.1, 0.7)},
    },
    "imagenet-100": {
        1:  {"conv": (7.1,  0.2), "rn18": (8.1,  0.3), "rn101": (6.1,  0.8)},
        10: {"conv": (29.6, 0.2), "rn18": (36.0, 0.3), "rn101": (33.9, 0.7)},
        50: {"conv": (50.2, 0.2), "rn18": (61.6, 0.1), "rn101": (66.0, 0.6)},
    },
    "imagenet-1k": {
        1:  {"conv": (6.4,  0.1), "rn18": (6.6,  0.2), "rn101": (5.9,  0.4)},
        10: {"conv": (20.4, 0.2), "rn18": (42.0, 0.3), "rn101": (48.3, 0.4)},
        50: {"conv": (38.4, 0.2), "rn18": (56.5, 0.5), "rn101": (61.2, 0.4)},
    },
}

NAME_RE = re.compile(
    r"^(?P<subset>[a-z0-9\-]+)_ipc(?P<ipc>\d+)_(?P<arch>conv|rn18|rn101)$"
)


def parse_name(stem):
    m = NAME_RE.match(stem)
    if not m:
        return None
    return m.group("subset"), int(m.group("ipc")), m.group("arch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/table2/results.csv")
    ap.add_argument("--z", type=float, default=3.0,
                    help="flag cells with |z|>z (default 3)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"missing {args.csv}")

    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            stem = r["script"]
            try:
                ours = float(r["best_top1"])
            except (TypeError, ValueError):
                continue
            parsed = parse_name(stem)
            if not parsed:
                continue
            subset, ipc, arch = parsed
            cell = PAPER.get(subset, {}).get(ipc, {}).get(arch)
            if cell is None:
                continue
            mean, std = cell
            z = (ours - mean) / std if std > 0 else float("inf")
            rows.append((stem, ours, mean, std, z))

    if not rows:
        print("No comparable rows.")
        return

    rows.sort(key=lambda r: -abs(r[4]))
    flagged = [r for r in rows if abs(r[4]) > args.z]

    width = max(len(r[0]) for r in rows)
    hdr = f"{'cell'.ljust(width)}   ours    paper       diff    z"
    print(hdr)
    print("-" * len(hdr))
    for stem, ours, mean, std, z in rows:
        mark = "  !!" if abs(z) > args.z else ""
        print(f"{stem.ljust(width)}  {ours:5.2f}  {mean:5.2f}±{std:.2f}  "
              f"{ours-mean:+6.2f}  {z:+6.2f}{mark}")

    print()
    print(f"Compared {len(rows)} cells, flagged {len(flagged)} at |z|>{args.z}.")
    if flagged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
