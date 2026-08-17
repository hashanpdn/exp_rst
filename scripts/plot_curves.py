#!/usr/bin/env python
"""Plot mean +/- std accuracy curves across seeds for each sweep cell.

Expects the layout produced by scripts/sweep.py:
    <root>/<cell>/seed<S>/metrics.csv

Usage:
    python scripts/plot_curves.py runs/e1 --out fig_e1.png --metric test_acc
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="curves.png")
    ap.add_argument("--metric", default="test_acc")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    cells = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                   if os.path.isdir(d))
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for cell in cells:
        curves = []
        for csvf in sorted(glob.glob(os.path.join(cell, "seed*", "**",
                                                  "metrics.csv"),
                                     recursive=True)):
            df = pd.read_csv(csvf)
            if args.metric in df.columns:
                curves.append(df[args.metric].to_numpy())
        if not curves:
            continue
        L = min(map(len, curves))
        arr = np.stack([c[:L] for c in curves])
        mean, std = arr.mean(0), arr.std(0)
        x = np.arange(1, L + 1)
        label = os.path.basename(cell)
        (line,) = ax.plot(x, mean, label=f"{label} (n={len(curves)})")
        ax.fill_between(x, mean - std, mean + std,
                        color=line.get_color(), alpha=0.15)
    ax.set_xlabel("Global round")
    ax.set_ylabel(args.metric)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    if args.title:
        ax.set_title(args.title)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
