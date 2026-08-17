#!/usr/bin/env python
"""Assignment-quality demo (runs in seconds, no training).

Compares client-to-edge assignment methods on a pathological non-IID
population and reports, per method:

  D_res_true   true max_j ||P_j - P_G||_1   (0 = perfectly edge-IID)
  Phi_true     true potential (sum of edge imbalances; lower = better)
  recovery     fraction of the oracle's Phi-improvement over random that
               the method achieves:  (Phi_rand - Phi_m) / (Phi_rand - Phi_orc)

Usage:
    python scripts/demo_assignment.py --clients 100 --edges 20 --k 2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hdpba_hfl.assignment import (  # noqa: E402
    hdpba_assignment, nonprivate_greedy, oracle_assignment,
    random_assignment, rr_presence_assignment, true_divergence)


def synth_hist(K: int, C: int, k: int, n_lo: int, n_hi: int,
               rng: np.random.Generator) -> np.ndarray:
    """Pathological k-class clients with random sizes."""
    hist = np.zeros((K, C), dtype=np.int64)
    for i in range(K):
        classes = rng.choice(C, size=k, replace=False)
        n = int(rng.integers(n_lo, n_hi + 1))
        counts = rng.multinomial(n, np.ones(k) / k)
        hist[i, classes] = counts
    return hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--edges", type=int, default=20)
    ap.add_argument("--k", type=int, default=2, help="classes per client")
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--n_lo", type=int, default=200)
    ap.add_argument("--n_hi", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    ap.add_argument("--eps_a", type=float, nargs="+",
                    default=[0.1, 0.2, 0.5, 1.0])
    ap.add_argument("--eps_g", type=float, default=None,
                    help="aggregate-release budget; default eps_a/2 per level "
                         "(choice budget and aggregate precision must be "
                         "balanced: confident choices on noisy aggregates "
                         "herd onto noise)")
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()

    rows: dict = {}

    def record(label, dres, phi):
        rows.setdefault(label, {"d": [], "p": []})
        rows[label]["d"].append(dres)
        rows[label]["p"].append(phi)

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        hist = synth_hist(args.clients, args.classes, args.k,
                          args.n_lo, args.n_hi, rng)

        a = random_assignment(args.clients, args.edges, None,
                              np.random.default_rng(seed))
        d, p = true_divergence(hist, a, args.edges)
        record("random", d, p)

        a = oracle_assignment(hist, args.edges, None,
                              np.random.default_rng(seed))
        d, p = true_divergence(hist, a, args.edges)
        record("oracle", d, p)

        r = nonprivate_greedy(hist, args.edges, np.random.default_rng(seed))
        record("nonprivate_greedy", r.d_res_true, r.phi_true)

        r = rr_presence_assignment(hist, args.edges, eps_rr=1.0,
                                   rng=np.random.default_rng(seed))
        record("rr_presence(eps=1.0)", r.d_res_true, r.phi_true)

        for ea in args.eps_a:
            eg = args.eps_g if args.eps_g is not None else ea / 2.0
            r = hdpba_assignment(hist, args.edges, eps_a=ea,
                                 eps_g=eg, delta_g=1e-6,
                                 passes=args.passes,
                                 rng=np.random.default_rng(seed))
            record(f"hdpba(eps_a={ea})", r.d_res_true, r.phi_true)

    phi_rand = float(np.mean(rows["random"]["p"]))
    phi_orc = float(np.mean(rows["oracle"]["p"]))
    span = max(phi_rand - phi_orc, 1e-12)

    print(f"\nK={args.clients} clients, J={args.edges} edges, "
          f"{args.k}-class non-IID, {len(args.seeds)} seeds")
    print(f"{'method':26s} {'D_res_true':>12s} {'Phi_true':>12s} "
          f"{'recovery':>9s}")
    order = ["random", "rr_presence(eps=1.0)"] + \
            [f"hdpba(eps_a={e})" for e in args.eps_a] + \
            ["nonprivate_greedy", "oracle"]
    for label in order:
        d = np.mean(rows[label]["d"])
        p = np.mean(rows[label]["p"])
        rec = (phi_rand - p) / span
        print(f"{label:26s} {d:12.4f} {p:12.4f} {rec:9.2f}")
    print("\nrecovery = share of the oracle's improvement over random that "
          "the method attains (1.00 = oracle-quality).")


if __name__ == "__main__":
    main()
