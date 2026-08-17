"""Client-to-edge assignment: H-DPBA (Algorithm 1) and baselines.

H-DPBA phases (see specification document, Sec. 3):
  0. Provisional random seeding within feasible sets (data-independent).
  1. Aggregate release: per-edge sums of normalized histograms + Gaussian noise
     (simulating secure aggregation with distributed noise); simplex-projected
     means published.
  2. R passes of exponential-mechanism self-assignment over feasible edges
     with capacity bookkeeping from the public transcript; utility = potential
     decrease if the client moves; 'stay' (u = 0) always in the support.
  3. Freeze; publish residual divergence D_res for the weighting selector.

Privacy events are registered into the provided RDPAccountant:
  - per pass + initial: one Gaussian release covering each client's record
  - per pass: one eps_a exponential-mechanism invocation per client
Sensitivities per Lemmas 1-2 of the specification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .privacy import (RDPAccountant, exponential_mechanism, gaussian_sigma,
                      project_simplex)


@dataclass
class AssignmentResult:
    assignment: np.ndarray            # client -> edge
    edge_dists: np.ndarray            # published P_hat_j  (J x C)
    global_dist: np.ndarray           # published P_hat_G  (C,)
    d_res: float                      # max_j ||P_hat_j - P_hat_G||_1 (public)
    phi: float                        # potential on published aggregates
    method: str
    d_res_true: float = float("nan")  # SIMULATION DIAGNOSTIC ONLY: true stats
    phi_true: float = float("nan")    # (never available to servers in deploy)


def true_divergence(hist: np.ndarray, assign: np.ndarray, num_edges: int):
    """Simulation-only diagnostics on TRUE histograms (not part of protocol)."""
    P = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1)
    PG = P.mean(axis=0)
    phi, dres = 0.0, 0.0
    for j in range(num_edges):
        mem = P[assign == j]
        if len(mem):
            pj = mem.mean(axis=0)
            phi += len(mem) * float(np.sum((pj - PG) ** 2))
            dres = max(dres, float(np.abs(pj - PG).sum()))
    return dres, phi


def _potential(edge_dists: np.ndarray, counts: np.ndarray,
               global_dist: np.ndarray) -> float:
    return float(sum(m * np.sum((p - global_dist) ** 2)
                     for m, p in zip(counts, edge_dists)))


def _d_res(edge_dists: np.ndarray, global_dist: np.ndarray) -> float:
    return float(max(np.abs(p - global_dist).sum() for p in edge_dists))


def _publish(P_norm: np.ndarray, assign: np.ndarray, num_edges: int,
             sigma_g: float, rng: np.random.Generator):
    """Simulate secure-aggregation release: noisy sums -> projected means."""
    C = P_norm.shape[1]
    G = np.zeros((num_edges, C))
    m = np.zeros(num_edges, dtype=int)
    for k, j in enumerate(assign):
        G[j] += P_norm[k]
        m[j] += 1
    G_noisy = G + rng.normal(0, sigma_g, size=G.shape)
    P_hat = np.stack([
        project_simplex(G_noisy[j] / max(m[j], 1)) for j in range(num_edges)
    ])
    P_G = project_simplex(G_noisy.sum(axis=0) / max(m.sum(), 1))
    return P_hat, P_G, m


def random_assignment(num_clients: int, num_edges: int,
                      feasible: Optional[np.ndarray],
                      rng: np.random.Generator) -> np.ndarray:
    """Capacity-balanced random assignment within feasible sets."""
    cap = int(np.ceil(num_clients / num_edges))
    load = np.zeros(num_edges, dtype=int)
    assign = np.full(num_clients, -1, dtype=int)
    for k in rng.permutation(num_clients):
        cands = np.arange(num_edges) if feasible is None else np.where(feasible[k])[0]
        open_ = [j for j in cands if load[j] < cap]
        j = int(rng.choice(open_ if open_ else cands))
        assign[k] = j
        load[j] += 1
    return assign


def oracle_assignment(hist: np.ndarray, num_edges: int,
                      feasible: Optional[np.ndarray],
                      rng: np.random.Generator) -> np.ndarray:
    """Privacy-free greedy balancing on TRUE histograms (utility upper bound).

    Greedy: iterate clients (largest first); give each to the feasible,
    non-full edge whose mean distribution moves closest to global. One swap
    pass refines. Equivalent role to `initialize_edges_iid` in the prior code.
    """
    return _greedy(hist, num_edges, feasible, rng, sigma_g=0.0,
                   eps_a=float("inf"), passes=2, accountant=None).assignment


def _greedy(hist: np.ndarray, num_edges: int, feasible: Optional[np.ndarray],
            rng: np.random.Generator, sigma_g: float, eps_a: float,
            passes: int, accountant: Optional[RDPAccountant],
            record_level: bool = True, batches: int = 4,
            refresh_per_batch: bool = True) -> AssignmentResult:
    K, C = hist.shape
    n_k = hist.sum(axis=1)
    P_norm = hist / np.maximum(n_k[:, None], 1)
    cap = int(np.ceil(K / num_edges))

    assign = random_assignment(K, num_edges, feasible, rng)   # Phase 0
    P_hat, P_G, m = _publish(P_norm, assign, num_edges, sigma_g, rng)  # Phase 1
    releases = 1
    cap_soft = cap + 1  # slack so moves exist from a balanced start; rebalanced
                        # data-independently after each pass (no privacy cost)

    for _ in range(passes):                                    # Phase 2
        order = rng.permutation(K)
        for batch in np.array_split(order, batches):
            for k in batch:
                cur = assign[k]
                cands = (np.arange(num_edges) if feasible is None
                         else np.where(feasible[k])[0])
                A_k = [j for j in cands if j == cur or m[j] < cap_soft]
                if cur not in A_k:
                    A_k.append(cur)
                base_cur = m[cur] * np.sum((P_hat[cur] - P_G) ** 2)
                # leave-effect on current edge (published stats + own p_k)
                if m[cur] > 1:
                    P_cur_minus = project_simplex(
                        (m[cur] * P_hat[cur] - P_norm[k]) / (m[cur] - 1))
                    left = (m[cur] - 1) * np.sum((P_cur_minus - P_G) ** 2)
                else:
                    left = 0.0
                utils = []
                for j in A_k:
                    if j == cur:
                        utils.append(0.0)
                        continue
                    P_j_plus = project_simplex(
                        (m[j] * P_hat[j] + P_norm[k]) / (m[j] + 1))
                    base_j = m[j] * np.sum((P_hat[j] - P_G) ** 2)
                    joined = (m[j] + 1) * np.sum((P_j_plus - P_G) ** 2)
                    utils.append((base_cur + base_j) - (left + joined))
                # Lemma 1 sensitivity (record- or client-level)
                m_min = max(1, int(m.min()))
                if record_level:
                    delta_u = 8 * np.sqrt(2) / (max(n_k[k], 1) * (m_min + 1))
                else:
                    delta_u = 8 * np.sqrt(2) / (m_min + 1)
                choice = A_k[exponential_mechanism(utils, eps_a, delta_u, rng)]
                if choice != cur:
                    m[cur] -= 1
                    m[choice] += 1
                    assign[k] = choice
            if refresh_per_batch:  # budget/accuracy knob: fresher aggregates,
                P_hat, P_G, m = _publish(P_norm, assign, num_edges, sigma_g,
                                         rng)  # ...one more release per batch
        # Rebalance to strict capacity via UNIFORMLY RANDOM moves. The choice of
        # which client moves / where is data-independent given the public
        # transcript, hence pure post-processing: no additional budget.
        while m.max() > cap:
            j_over = int(np.argmax(m))
            movers = [k for k in range(K) if assign[k] == j_over]

            def _under(k_: int):
                cands_ = (np.arange(num_edges) if feasible is None
                          else np.where(feasible[k_])[0])
                return [j for j in cands_ if m[j] < cap]

            ok = [k_ for k_ in movers if _under(k_)]
            k_mv = int(rng.choice(ok if ok else movers))
            under = _under(k_mv) or [j for j in range(num_edges) if m[j] < cap]
            j_to = int(rng.choice(under))
            assign[k_mv] = j_to
            m[j_over] -= 1
            m[j_to] += 1
        P_hat, P_G, m = _publish(P_norm, assign, num_edges, sigma_g, rng)
        releases += (batches + 1) if refresh_per_batch else 1

    if accountant is not None and np.isfinite(eps_a):
        n_min = max(1, int(n_k.min()))
        s2 = np.sqrt(2) / n_min if record_level else 1.0
        if sigma_g > 0:
            accountant.add_gaussian(sigma_g, s2, count=releases,
                                    label="assignment_aggregate_release")
        accountant.add_pure(eps_a, count=passes, label="assignment_exp_mech")

    phi = _potential(P_hat, m, P_G)
    dt, pt = true_divergence(hist, assign, num_edges)
    return AssignmentResult(assign, P_hat, P_G, _d_res(P_hat, P_G), phi,
                            method="greedy", d_res_true=dt, phi_true=pt)


def hdpba_assignment(hist: np.ndarray, num_edges: int, eps_a: float,
                     eps_g: float, delta_g: float, passes: int,
                     rng: np.random.Generator,
                     feasible: Optional[np.ndarray] = None,
                     accountant: Optional[RDPAccountant] = None,
                     record_level: bool = True,
                     refresh_per_batch: bool = True) -> AssignmentResult:
    """H-DPBA, Algorithm 1. `hist` = true per-client label counts (LOCAL)."""
    n_min = max(1, int(hist.sum(axis=1).min()))
    s2 = np.sqrt(2) / n_min if record_level else 1.0
    sigma_g = gaussian_sigma(s2, eps_g, delta_g)
    res = _greedy(hist, num_edges, feasible, rng, sigma_g, eps_a, passes,
                  accountant, record_level,
                  refresh_per_batch=refresh_per_batch)
    res.method = "hdpba"
    return res


def rr_presence_assignment(hist: np.ndarray, num_edges: int, eps_rr: float,
                           rng: np.random.Generator,
                           feasible: Optional[np.ndarray] = None,
                           accountant: Optional[RDPAccountant] = None
                           ) -> AssignmentResult:
    """A3 baseline (pp-CFL-style signal, adapted to balancing).

    Each client releases a BINARY label-presence vector b_k in {0,1}^C
    perturbed by randomized response with per-bit budget eps_rr / C
    (basic composition over the C bits => each client's release is
    eps_rr-LDP; across clients this is parallel composition, so the
    per-individual cost is eps_rr total).  The server then runs the SAME
    greedy balancer as H-DPBA on the debiased presence vectors.

    Purpose: shows that a coarse binary signal (prior art) is insufficient
    for edge-IID balancing compared with H-DPBA's aggregate+choice design.
    """
    K, C = hist.shape
    eps_bit = eps_rr / C
    p_true = np.exp(eps_bit) / (np.exp(eps_bit) + 1.0)   # keep-probability
    b = (hist > 0).astype(np.float64)
    flip = rng.random((K, C)) >= p_true
    b_noisy = np.where(flip, 1.0 - b, b)
    # unbiased estimate of presence, clipped to [0,1]
    b_hat = (b_noisy - (1.0 - p_true)) / (2.0 * p_true - 1.0 + 1e-12)
    b_hat = np.clip(b_hat, 0.0, 1.0)
    # pseudo-histogram: uniform mass over (estimated) present classes
    row = b_hat / np.maximum(b_hat.sum(axis=1, keepdims=True), 1e-9)
    pseudo = np.round(row * 100).astype(np.int64)          # scale-free counts
    pseudo[pseudo.sum(axis=1) == 0] = 1                    # degenerate rows
    if accountant is not None:
        accountant.add_pure(eps_rr, count=1, label="rr_presence(parallel)")
    res = _greedy(pseudo, num_edges, feasible, rng, sigma_g=0.0,
                  eps_a=float("inf"), passes=2, accountant=None)
    # recompute diagnostics against the TRUE histograms
    P_norm = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1)
    P_hat, P_G, m = _publish(P_norm, res.assignment, num_edges, 0.0, rng)
    dt, pt = true_divergence(hist, res.assignment, num_edges)
    return AssignmentResult(res.assignment, P_hat, P_G, _d_res(P_hat, P_G),
                            _potential(P_hat, m, P_G), "rr_presence",
                            d_res_true=dt, phi_true=pt)


def nonprivate_greedy(hist: np.ndarray, num_edges: int,
                      rng: np.random.Generator,
                      feasible: Optional[np.ndarray] = None
                      ) -> AssignmentResult:
    """Mhaisen-style non-private optimization baseline (raw histograms)."""
    res = _greedy(hist, num_edges, feasible, rng, sigma_g=0.0,
                  eps_a=float("inf"), passes=2, accountant=None)
    res.method = "nonprivate_greedy"
    return res


def build_assignment(cfg, hist: np.ndarray, rng: np.random.Generator,
                     accountant: Optional[RDPAccountant]) -> AssignmentResult:
    K = hist.shape[0]
    feasible = None
    if cfg.feasible_edges and cfg.feasible_edges < cfg.num_edges:
        feasible = np.zeros((K, cfg.num_edges), dtype=bool)
        for k in range(K):
            feasible[k, rng.choice(cfg.num_edges, cfg.feasible_edges,
                                   replace=False)] = True
    if cfg.assignment == "random":
        assign = random_assignment(K, cfg.num_edges, feasible, rng)
        P_norm = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1)
        P_hat, P_G, m = _publish(P_norm, assign, cfg.num_edges, 0.0, rng)
        dt, pt = true_divergence(hist, assign, cfg.num_edges)
        return AssignmentResult(assign, P_hat, P_G, _d_res(P_hat, P_G),
                                _potential(P_hat, m, P_G), "random",
                                d_res_true=dt, phi_true=pt)
    if cfg.assignment == "oracle":
        assign = oracle_assignment(hist, cfg.num_edges, feasible, rng)
        P_norm = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1)
        P_hat, P_G, m = _publish(P_norm, assign, cfg.num_edges, 0.0, rng)
        dt, pt = true_divergence(hist, assign, cfg.num_edges)
        return AssignmentResult(assign, P_hat, P_G, _d_res(P_hat, P_G),
                                _potential(P_hat, m, P_G), "oracle",
                                d_res_true=dt, phi_true=pt)
    if cfg.assignment == "nonprivate":
        return nonprivate_greedy(hist, cfg.num_edges, rng, feasible)
    if cfg.assignment == "rr":
        return rr_presence_assignment(hist, cfg.num_edges, cfg.eps_assign,
                                      rng, feasible, accountant)
    if cfg.assignment == "hdpba":
        return hdpba_assignment(hist, cfg.num_edges, cfg.eps_assign,
                                cfg.eps_agg, cfg.delta_agg, cfg.assign_passes,
                                rng, feasible, accountant,
                                cfg.assign_record_level)
    raise ValueError(f"unknown assignment {cfg.assignment}")
