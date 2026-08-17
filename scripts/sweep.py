#!/usr/bin/env python
"""Grid sweep runner.

Reads a sweep YAML of the form:

    base: configs/e1_main.yaml        # base config (optional)
    seeds: [42, 123, 2024]
    grid:                             # cartesian product over these lists
      method: [bl1, bl2, escs, esca, auto]
      assignment: [random, hdpba]
    fixed:                            # applied to every run (optional)
      dataset: mnist
      global_rounds: 50

Each cell becomes one run named  <sweepname>/<k1-v1_k2-v2>/seed<S>  under
--outdir.  Runs execute sequentially in-process (single-GPU friendly).
Already-completed runs (summary.json exists) are skipped, so the sweep is
resumable after interruption.

Usage:
    python scripts/sweep.py sweeps/e1.yaml --outdir runs/e1
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hdpba_hfl.train import main as train_main  # noqa: E402


def cell_name(keys, values) -> str:
    return "_".join(f"{k}-{v}" for k, v in zip(keys, values))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_yaml")
    ap.add_argument("--outdir", default="runs/sweep")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    with open(args.sweep_yaml) as f:
        spec = yaml.safe_load(f)

    grid = spec.get("grid", {})
    keys = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys))) or [()]
    seeds = spec.get("seeds", [42])
    fixed = spec.get("fixed", {})
    base = spec.get("base")

    total = len(combos) * len(seeds)
    print(f"[sweep] {len(combos)} cells x {len(seeds)} seeds = {total} runs")

    done = failed = 0
    t0 = time.time()
    for values in combos:
        cname = cell_name(keys, values) if keys else "single"
        for seed in seeds:
            rdir = os.path.join(args.outdir, cname, f"seed{seed}")
            import glob as _glob
            if _glob.glob(os.path.join(rdir, "**", "summary.json"),
                          recursive=True):
                print(f"[skip] {rdir}")
                done += 1
                continue
            argv = []
            if base:
                argv += ["--config", base]
            for k, v in fixed.items():
                argv += [f"--{k}", str(v)]
            for k, v in zip(keys, values):
                argv += [f"--{k}", str(v)]
            argv += ["--seed", str(seed), "--outdir", rdir,
                     "--name", f"{cname}_s{seed}"]
            print(f"[run {done + failed + 1}/{total}] {' '.join(argv)}")
            if args.dry_run:
                continue
            try:
                train_main(argv)
                done += 1
            except Exception as exc:  # keep sweeping; record failure
                failed += 1
                os.makedirs(rdir, exist_ok=True)
                with open(os.path.join(rdir, "FAILED.json"), "w") as f:
                    json.dump({"error": repr(exc)}, f)
                print(f"[FAIL] {rdir}: {exc!r}")
    print(f"[sweep] finished: {done} ok, {failed} failed, "
          f"{(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
