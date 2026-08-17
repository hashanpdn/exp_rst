"""Post-hoc statistical analysis over run directories.

Usage:
    python -m hdpba_hfl.stats runs/ --metric final_acc
    python -m hdpba_hfl.stats runs/ --compare e2_hdpba e2_random --metric final_acc

Aggregates summary.json files by run name (strips `_seed*`), reports
mean +/- std, and for --compare performs a paired Wilcoxon signed-rank test
(paired by seed) with Cliff's delta effect size. With 3-5 seeds, effect sizes
carry more weight than p-values; both are reported.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List


def collect(root: str) -> Dict[str, Dict[int, dict]]:
    """Group summaries by run name (flat layout) or sweep cell (sweep layout).

    Flat:   <root>/<name>_seed<S>/summary.json         -> group = <name>
    Sweep:  <root>/<cell>/seed<S>/<...>/summary.json   -> group = <cell>
    """
    groups: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for path in glob.glob(os.path.join(root, "**", "summary.json"),
                          recursive=True):
        with open(path) as f:
            s = json.load(f)
        rel = os.path.relpath(path, root).split(os.sep)
        seed_dirs = [p for p in rel if re.fullmatch(r"seed\d+", p)]
        if seed_dirs:                              # sweep layout
            name = rel[max(0, rel.index(seed_dirs[0]) - 1)]
            seed = int(seed_dirs[0][4:])
        else:                                      # flat layout
            run = os.path.basename(os.path.dirname(path))
            m = re.match(r"(.+)_seed(\d+)$", run)
            name, seed = (m.group(1), int(m.group(2))) if m else (run, 0)
        groups[name][seed] = s
    return groups


def cliffs_delta(a: List[float], b: List[float]) -> float:
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--metric", default="final_acc",
                    choices=["final_acc", "best_acc", "acc_auc", "eps_total",
                             "d_res", "final_worst_class_recall"])
    ap.add_argument("--compare", nargs=2, default=None,
                    help="two run-name prefixes to test (paired by seed)")
    args = ap.parse_args()

    groups = collect(args.root)
    if not groups:
        print(f"no summary.json found under {args.root}")
        return

    import numpy as np
    print(f"{'run':40s} {'n':>3s} {args.metric:>12s}")
    for name in sorted(groups):
        vals = [s[args.metric] for s in groups[name].values()]
        print(f"{name:40s} {len(vals):3d} {np.mean(vals):9.4f} "
              f"+/- {np.std(vals):.4f}")

    if args.compare:
        a_name, b_name = args.compare
        a_runs, b_runs = groups.get(a_name, {}), groups.get(b_name, {})
        seeds = sorted(set(a_runs) & set(b_runs))
        if len(seeds) < 2:
            print(f"\n[compare] need >=2 common seeds between {a_name} and "
                  f"{b_name}; found {len(seeds)}")
            return
        a = [a_runs[s][args.metric] for s in seeds]
        b = [b_runs[s][args.metric] for s in seeds]
        from scipy.stats import wilcoxon
        try:
            stat, p = wilcoxon(a, b)
        except ValueError:  # identical values
            stat, p = float("nan"), 1.0
        print(f"\n[compare] {a_name} vs {b_name} on {args.metric} "
              f"({len(seeds)} paired seeds)")
        print(f"  means: {np.mean(a):.4f} vs {np.mean(b):.4f} "
              f"(diff {np.mean(a) - np.mean(b):+.4f})")
        print(f"  Wilcoxon signed-rank: W={stat}, p={p:.4f}")
        print(f"  Cliff's delta: {cliffs_delta(a, b):+.3f}")
        print("  note: with few seeds, weigh the effect size over the p-value;"
              " apply Holm-Bonferroni across your pre-registered contrasts.")


if __name__ == "__main__":
    main()
