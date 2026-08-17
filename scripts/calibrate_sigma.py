#!/usr/bin/env python
"""Calibrate the training noise sigma for a target privacy budget.

Given the run configuration (rounds T, intermediate rounds U, client/edge
participation, clip) and a target training epsilon at delta, binary-search
the per-coordinate Gaussian sigma using the same RDP accountant the
training loop uses — so the printed sigma reproduces the target eps exactly.

Also prints a UTILITY FEASIBILITY diagnostic: the ratio of expected noise
L2 norm to the clipped-signal norm for a given model size. Rule of thumb:
per-upload noise-to-signal (sigma * sqrt(d) / clip) beyond ~2-3x, after
averaging over participating clients, usually prevents learning — if the
calibrated sigma fails this, the honest options are: larger eps target,
fewer uploads (smaller T*U), or lower participation q (better subsampling
amplification). Do NOT just lower sigma: that silently changes eps.

Usage:
    python scripts/calibrate_sigma.py --eps 5.0 --delta 1e-5 ^
        --rounds 50 --intermediate 2 --client_frac 0.5 --edge_frac 1.0 ^
        --clip 1.0 --model_dim 60000 --clients_per_edge 5
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hdpba_hfl.privacy import RDPAccountant  # noqa: E402


def eps_for_sigma(sigma: float, q: float, uploads: int, clip: float,
                  delta: float) -> float:
    acc = RDPAccountant()
    acc.add_sampled_gaussian(q=q, sigma=sigma, sensitivity=clip,
                             count=uploads, label="training_upload")
    return acc.get_epsilon(delta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, required=True,
                    help="target TRAINING epsilon (assignment budget is "
                         "separate; add it on top for eps_total)")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--intermediate", type=int, default=2)
    ap.add_argument("--client_frac", type=float, default=0.5)
    ap.add_argument("--edge_frac", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--model_dim", type=int, default=60000,
                    help="number of model parameters (LeNet~60k, "
                         "TanhLeNet~20k, AlexNetS~1M)")
    ap.add_argument("--clients_per_edge", type=int, default=5,
                    help="for the averaging-gain estimate in the "
                         "feasibility check")
    args = ap.parse_args()

    q = args.client_frac * args.edge_frac
    uploads = args.rounds * args.intermediate

    lo, hi = 1e-4, 1e5
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if eps_for_sigma(mid, q, uploads, args.clip, args.delta) > args.eps:
            lo = mid
        else:
            hi = mid
    sigma = hi
    achieved = eps_for_sigma(sigma, q, uploads, args.clip, args.delta)

    # feasibility diagnostic
    noise_l2 = sigma * math.sqrt(args.model_dim)
    n_avg = max(1, round(args.client_frac * args.clients_per_edge))
    eff_ratio = noise_l2 / (args.clip * math.sqrt(n_avg))

    print(f"target training eps : {args.eps}  (delta={args.delta})")
    print(f"uploads             : {uploads}  (T={args.rounds} x "
          f"U={args.intermediate}), sampling q={q}")
    print(f"CALIBRATED sigma    : {sigma:.4g}   (achieves eps="
          f"{achieved:.3f})")
    print(f"noise L2 per upload : ~{noise_l2:.1f}  vs clipped signal "
          f"{args.clip}")
    print(f"after edge avg (~{n_avg} clients): noise/signal ~ "
          f"{eff_ratio:.1f}x")
    if eff_ratio > 3:
        print("[FEASIBILITY WARNING] noise dominates the signal at this "
              "budget; expect little or no learning. Honest remedies: "
              "raise --eps, reduce --rounds/--intermediate, or lower "
              "--client_frac (never lower sigma directly).")
    else:
        print("[feasibility] OK: learning is plausible at this budget.")


if __name__ == "__main__":
    main()
