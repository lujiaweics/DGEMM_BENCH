#!/usr/bin/env python3
"""DGEMM-Bench Plotting Script (Python3).

Reads CSV files produced by the driver (lines starting with "#" are metadata 
key-value pairs), plots GFLOPS curves for each implementation against 
matrix size n, and verifies numerical accuracy against OpenBLAS reference.

Usage::

    # Plot a single implementation (includes OpenBLAS reference)
    python3 scripts/plot_dgemm.py data/output_ijk.csv

    # Overlay multiple implementations and specify output
    python3 scripts/plot_dgemm.py data/output_*.csv -o data/dgemm.png

    # Relax tolerance / skip reference curve / limit y-axis
    python3 scripts/plot_dgemm.py data/output_ijk.csv --tol 1e-8 --no-ref --ymax 100

Note: Text in plots is intentionally in English as default matplotlib fonts 
often lack Chinese glyphs. To use Chinese, configure rcParams['font.sans-serif'].

Exit code: 0 if max|diff| for all data is within tolerance, otherwise 1.
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt

DEFAULT_TOL = 1e-10
META_PREFIX = "#"


def load_csv(path):
    """Read a driver output CSV, returning (meta, columns).

    meta    : dict[str, str], from "#" comment lines, e.g., {"implementation": "ijk"}
    columns : dict[str, list[float]], column name -> list of values
    """
    meta = {}
    data_lines = []

    with open(path, newline="") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(META_PREFIX):
                body = stripped.lstrip(META_PREFIX).strip()
                if "," in body:
                    key, value = body.split(",", 1)
                    meta[key.strip()] = value.strip()
                continue
            data_lines.append(line)

    if not data_lines:
        raise ValueError("%s: No data lines found in file" % path)

    reader = csv.DictReader(data_lines)
    names = reader.fieldnames or []
    columns = {name: [] for name in names}

    for row in reader:
        for name in names:
            columns[name].append(float(row[name]))

    return meta, columns


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot DGEMM-Bench performance curves and verify numerical accuracy")
    parser.add_argument("csv", nargs="+", metavar="CSV",
                        help="CSV files from driver (multiple files to overlay curves)")
    parser.add_argument("-o", "--output", default="data/dgemm_gflops.png",
                        help="Output image path (default: data/dgemm_gflops.png)")
    parser.add_argument("--title", default="DGEMM: C := A*B + C (row-major)",
                        help="Plot title")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL,
                        help="Correctness tolerance, default: %.0e" % DEFAULT_TOL)
    parser.add_argument("--no-ref", action="store_true",
                        help="Do not plot OpenBLAS reference curve")
    parser.add_argument("--ymax", type=float, default=None,
                        help="Y-axis upper limit (GFLOPS), default: auto")
    args = parser.parse_args(argv)

    figure, axes = plt.subplots(figsize=(8.0, 5.0))
    all_correct = True
    reference_drawn = False

    for path in args.csv:
        meta, columns = load_csv(path)
        label = meta.get("implementation",
                         os.path.splitext(os.path.basename(path))[0])

        # ---- Accuracy verification: max|diff| must be less than tolerance ----
        worst_diff = max(abs(value) for value in columns["diff"])
        passed = worst_diff < args.tol
        all_correct = all_correct and passed
        print("[%s] %-12s max|diff| = %.3e  (tol = %.1e)"
              % ("PASS" if passed else "FAIL", label, worst_diff, args.tol))

        # ---- GFLOPS Curve ----
        sizes = [int(value) for value in columns["n"]]
        mine = columns["my_gflops"]
        axes.plot(sizes, mine, marker="o", markersize=3,
                  linewidth=1.5, label="%s (tested)" % label)

        peak = max(mine)
        print("       %-12s peak = %8.2f GFLOPS @ n = %d"
              % (label, peak, sizes[mine.index(peak)]))

        # ---- Draw reference curve only once ----
        if not args.no_ref and not reference_drawn:
            ref = columns["ref_gflops"]
            axes.plot(sizes, ref, linestyle="--", color="black",
                      linewidth=1.5, label="OpenBLAS reference")
            print("       %-12s peak = %8.2f GFLOPS" % ("OpenBLAS", max(ref)))
            reference_drawn = True

    axes.set_xlabel("matrix size n  (m = n = k)")
    axes.set_ylabel("GFLOPS")
    axes.set_title(args.title)
    axes.grid(True, linestyle=":", alpha=0.6)
    axes.set_xlim(left=0)
    axes.set_ylim(bottom=0)
    if args.ymax is not None:
        axes.set_ylim(top=args.ymax)
    axes.legend()

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    figure.tight_layout()
    figure.savefig(args.output, dpi=150)
    print("[plot] saved %s" % args.output)

    return 0 if all_correct else 1


if __name__ == "__main__":
    sys.exit(main())
