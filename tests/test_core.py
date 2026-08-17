"""Unit tests for privacy primitives, the RDP accountant, and H-DPBA.

Torch-free: runs anywhere with numpy/scipy. Torch integration is covered by
the smoke test (tests/test_smoke.py).
"""
import math

import numpy as np
import pytest

from hdpba_hfl.privacy import (RDPAccountant, exponential_mechanism,
                               gaussian_sigma, project_simplex, rdp_gaussian,
                               rdp_sampled_gaussian)
from hdpba_hfl.assignment import (hdpba_assignment, oracle_assignment,
                                  random_assignment)


# ---------------------------------------------------------------- primitives

def test_project_simplex():
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = rng.normal(0, 2, size=10)
        p = project_simplex(v)
        assert abs(p.sum() - 1) < 1e-9 and (p >= -1e-12).all()
    # already-simplex vectors are fixed points
    p0 = np.array([0.2, 0.3, 0.5])
    assert np.allclose(project_simplex(p0), p0)


def test_exponential_mechanism_prefers_high_utility():
    rng = np.random.default_rng(1)
    utils = [0.0, 1.0, 0.2]
    picks = [exponential_mechanism(utils, eps=5.0, sensitivity=0.1, rng=rng)
             for _ in range(3000)]
    frac_best = np.mean(np.array(picks) == 1)
    assert frac_best > 0.95           # strongly concentrated on the best arm
    picks_weak = [exponential_mechanism(utils, eps=0.001, sensitivity=1.0,
                                        rng=rng) for _ in range(3000)]
    counts = np.bincount(picks_weak, minlength=3) / 3000
    assert counts.min() > 0.25        # near-uniform when eps ~ 0


def test_exponential_mechanism_distribution_matches_theory():
    rng = np.random.default_rng(2)
    utils = np.array([0.0, 0.5])
    eps, sens = 2.0, 0.5
    target = np.exp(eps * utils / (2 * sens))
    target = target / target.sum()
    picks = [exponential_mechanism(utils, eps, sens, rng) for _ in range(20000)]
    emp = np.bincount(picks, minlength=2) / 20000
    assert np.allclose(emp, target, atol=0.02)


# ---------------------------------------------------------------- accountant

def test_gaussian_rdp_analytic():
    got = rdp_gaussian(sigma=2.0, sensitivity=1.0, alphas=[2, 10])
    assert np.allclose(got, [2 / 8, 10 / 8])


def test_sampled_gaussian_limits():
    # q=1 recovers full Gaussian RDP; small q is much cheaper
    full = rdp_sampled_gaussian(1.0, sigma_over_s=1.0, alpha=8)
    assert abs(full - 8 / 2) < 1e-9
    sub = rdp_sampled_gaussian(0.05, sigma_over_s=1.0, alpha=8)
    assert 0 < sub < full / 5


def test_accountant_composition_and_conversion():
    acc = RDPAccountant()
    acc.add_gaussian(sigma=2.0, sensitivity=1.0, count=1)
    eps1 = acc.get_epsilon(1e-5)
    acc.add_gaussian(sigma=2.0, sensitivity=1.0, count=3)
    eps4 = acc.get_epsilon(1e-5)
    assert eps4 > eps1                                    # composition grows
    # single Gaussian at sigma=2: eps should be near classical calibration
    sigma_cls = gaussian_sigma(1.0, eps1, 1e-5)
    assert 0.3 * sigma_cls < 2.0 < 3.0 * sigma_cls        # same ballpark


def test_pure_dp_events_bounded():
    acc = RDPAccountant()
    acc.add_pure(0.2, count=2)
    assert acc.get_epsilon(1e-6) <= 0.4 + 1e-9            # never worse than basic


# ---------------------------------------------------------------- assignment

def _skewed_hists(K=100, C=10, k=2, n=600, seed=0):
    """Pathological k-class clients."""
    rng = np.random.default_rng(seed)
    H = np.zeros((K, C))
    for i in range(K):
        cls = rng.choice(C, size=k, replace=False)
        H[i, cls] = n // k
    return H, rng


def _phi_true(H, assign, J):
    P = H / H.sum(1, keepdims=True)
    PG = P.mean(0)
    phi = 0.0
    for j in range(J):
        members = P[assign == j]
        if len(members):
            phi += len(members) * np.sum((members.mean(0) - PG) ** 2)
    return phi


def test_capacity_and_feasibility_respected():
    H, rng = _skewed_hists()
    J, cap = 20, 5
    feas = np.zeros((100, J), dtype=bool)
    for kcl in range(100):
        feas[kcl, rng.choice(J, 3, replace=False)] = True
    res = hdpba_assignment(H, J, eps_a=0.5, eps_g=0.1, delta_g=1e-6, passes=2,
                           rng=rng, feasible=feas)
    counts = np.bincount(res.assignment, minlength=J)
    assert counts.max() <= cap         # strict after end-of-pass rebalance
    for kcl in range(100):
        assert feas[kcl, res.assignment[kcl]]


def test_hdpba_beats_random_and_tracks_oracle():
    J, recov = 20, []
    for seed in (3, 4, 5):
        H, _ = _skewed_hists(seed=seed)
        rand = random_assignment(100, J, None, np.random.default_rng(seed))
        orac = oracle_assignment(H, J, None, np.random.default_rng(seed))
        priv = hdpba_assignment(H, J, eps_a=0.5, eps_g=0.2, delta_g=1e-6,
                                passes=2, rng=np.random.default_rng(seed))
        pr, po, pp = (_phi_true(H, a, J) for a in
                      (rand, orac, priv.assignment))
        assert po < pr                 # oracle strictly improves on random
        assert pp < pr                 # private strictly improves on random
        recov.append((pr - pp) / max(pr - po, 1e-9))
    # at eps_a=0.5 the private mechanism recovers most of the oracle's gain
    assert np.mean(recov) > 0.5


def test_hdpba_registers_privacy_events():
    H, rng = _skewed_hists(seed=4)
    acc = RDPAccountant()
    hdpba_assignment(H, 20, eps_a=0.1, eps_g=0.05, delta_g=1e-6, passes=2,
                     rng=rng, accountant=acc)
    labels = " ".join(acc.events)
    assert "exp_mech" in labels and "aggregate_release" in labels
    # record-level assignment at the low-budget setting stays under 0.5
    assert acc.get_epsilon(1e-6) < 0.5


def test_noise_degrades_gracefully():
    lo, hi = [], []
    for seed in (5, 6, 7):
        H, _ = _skewed_hists(seed=seed)
        r_lo = hdpba_assignment(H, 20, eps_a=0.05, eps_g=0.5, delta_g=1e-6,
                                passes=2, rng=np.random.default_rng(seed))
        r_hi = hdpba_assignment(H, 20, eps_a=5.0, eps_g=0.5, delta_g=1e-6,
                                passes=2, rng=np.random.default_rng(seed))
        lo.append(_phi_true(H, r_lo.assignment, 20))
        hi.append(_phi_true(H, r_hi.assignment, 20))
    # on average, more budget yields better (lower-potential) assignments
    assert np.mean(hi) < np.mean(lo)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
