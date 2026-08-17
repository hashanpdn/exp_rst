"""Differential privacy primitives and RDP accounting for H-DPBA + DP-HFL.

Implements:
  - Gaussian mechanism calibration (analytic, per Dwork & Roth Thm A.1 form)
  - Exponential mechanism sampling (Gumbel-trick, numerically stable)
  - L2 clipping, Euclidean projection onto the probability simplex
  - RDP accountant:
      * Gaussian mechanism:            eps(alpha) = alpha * s^2 / (2 sigma^2)
      * Sampled Gaussian (Poisson q):  Mironov et al. 2019, integer alpha
      * Pure eps-DP (exp. mechanism):  eps(alpha) <= min(eps, alpha * eps^2 / 2)
      * RDP -> (eps, delta) conversion optimized over an alpha grid
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np

DEFAULT_ALPHAS: List[float] = [1 + x / 10.0 for x in range(1, 100)] + list(
    range(11, 64)
) + [128, 256, 512]


# ----------------------------------------------------------------------------- 
# Mechanisms
# -----------------------------------------------------------------------------

def gaussian_sigma(l2_sensitivity: float, eps: float, delta: float) -> float:
    """Classic analytic calibration: sigma = s * sqrt(2 ln(1.25/delta)) / eps.

    Valid for eps <= 1; conservative above. Used for the *assignment* aggregate
    releases where eps_g is small by design. Training noise is specified by
    sigma directly and accounted via RDP.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    return l2_sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / eps


def gaussian_noise(shape, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=sigma, size=shape)


def clip_l2(vec: np.ndarray, bound: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    scale = min(1.0, bound / (norm + 1e-12))
    return vec * scale


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of v onto the probability simplex (sort-based)."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError("project_simplex expects a 1-D vector")
    n = v.shape[0]
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - css / ind > 0
    if not np.any(cond):
        # Degenerate input; fall back to uniform.
        return np.full(n, 1.0 / n)
    rho = ind[cond][-1]
    theta = css[cond][-1] / float(rho)
    return np.maximum(v - theta, 0.0)


def exponential_mechanism(
    utilities: Sequence[float],
    eps: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> int:
    """Sample index i with prob proportional to exp(eps * u_i / (2 * sensitivity)).

    Uses the Gumbel-max trick for numerical stability. eps=inf -> argmax.
    """
    u = np.asarray(utilities, dtype=np.float64)
    if not math.isfinite(eps):
        return int(np.argmax(u))
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    scores = (eps * u) / (2.0 * sensitivity)
    gumbel = rng.gumbel(size=u.shape)
    return int(np.argmax(scores + gumbel))


# -----------------------------------------------------------------------------
# RDP accountant
# -----------------------------------------------------------------------------

def rdp_gaussian(sigma: float, sensitivity: float, alphas: Iterable[float]) -> np.ndarray:
    s = sensitivity / sigma
    return np.array([a * s * s / 2.0 for a in alphas])


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def rdp_sampled_gaussian(q: float, sigma_over_s: float, alpha: int) -> float:
    """RDP of the Poisson-sampled Gaussian mechanism at integer alpha >= 2.

    Mironov, Talwar, Zhang (2019), Eq. for integer orders:
      eps(alpha) = 1/(alpha-1) * log( sum_{k=0}^{alpha} C(alpha,k)
                    (1-q)^{alpha-k} q^k exp(k(k-1)/(2 sigma^2)) )
    sigma_over_s is the noise multiplier (sigma / sensitivity).
    """
    if q <= 0:
        return 0.0
    if q >= 1:
        return alpha / (2.0 * sigma_over_s**2)
    log_terms = []
    for k in range(alpha + 1):
        log_t = (
            _log_comb(alpha, k)
            + (alpha - k) * math.log(1 - q)
            + (k * math.log(q) if k > 0 else 0.0)
            + k * (k - 1) / (2.0 * sigma_over_s**2)
        )
        log_terms.append(log_t)
    m = max(log_terms)
    log_sum = m + math.log(sum(math.exp(t - m) for t in log_terms))
    return log_sum / (alpha - 1)


def rdp_pure(eps: float, alphas: Iterable[float]) -> np.ndarray:
    """eps-DP => (alpha, min(eps, alpha * eps^2 / 2))-RDP (Bun-Steinke)."""
    return np.array([min(eps, a * eps * eps / 2.0) for a in alphas])


@dataclass
class RDPAccountant:
    """Per-record worst-case RDP composition across heterogeneous mechanisms."""

    alphas: List[float] = field(default_factory=lambda: list(DEFAULT_ALPHAS))

    def __post_init__(self) -> None:
        self._rdp = np.zeros(len(self.alphas))          # all events
        self._rdp_nonpure = np.zeros(len(self.alphas))  # excludes pure-DP events
        self._pure_eps_sum = 0.0                        # basic composition track
        self.events: List[str] = []

    def add_gaussian(self, sigma: float, sensitivity: float, count: int = 1,
                     label: str = "gaussian") -> None:
        incr = count * rdp_gaussian(sigma, sensitivity, self.alphas)
        self._rdp += incr
        self._rdp_nonpure += incr
        self.events.append(f"{label} x{count} (sigma={sigma:.4g}, s={sensitivity:.4g})")

    def add_sampled_gaussian(self, q: float, sigma: float, sensitivity: float,
                             count: int = 1, label: str = "sampled_gaussian") -> None:
        sos = sigma / sensitivity
        incr = np.array([
            rdp_sampled_gaussian(q, sos, int(a)) if float(a).is_integer() and a >= 2
            else a / (2.0 * sos * sos)  # conservative for fractional alpha
            for a in self.alphas
        ])
        self._rdp += count * incr
        self._rdp_nonpure += count * incr
        self.events.append(
            f"{label} x{count} (q={q}, sigma={sigma:.4g}, s={sensitivity:.4g})"
        )

    def add_pure(self, eps: float, count: int = 1, label: str = "exp_mech") -> None:
        self._rdp += count * rdp_pure(eps, self.alphas)
        self._pure_eps_sum += count * eps
        self.events.append(f"{label} x{count} (eps={eps:.4g})")

    def get_epsilon(self, delta: float) -> float:
        """Composed RDP -> (eps, delta)-DP, optimized over alpha.

        Returns the tighter of (a) full-RDP conversion of all events and
        (b) hybrid bound: pure-DP events via basic composition (sum of eps,
        delta' = 0) plus the RDP conversion of the remaining events -- valid
        because (eps1, 0)-DP composed with (eps2, delta)-DP is
        (eps1 + eps2, delta)-DP.
        """
        def _convert(rdp_vec):
            vals = [r + math.log(1.0 / delta) / (a - 1.0)
                    for r, a in zip(rdp_vec, self.alphas) if a > 1.0]
            return min(vals) if vals else float("inf")

        full = _convert(self._rdp)
        hybrid = self._pure_eps_sum + (
            _convert(self._rdp_nonpure) if self._rdp_nonpure.any() else 0.0)
        return float(min(full, hybrid))

    def summary(self, delta: float) -> str:
        lines = [f"  - {e}" for e in self.events]
        lines.append(f"  => eps_total = {self.get_epsilon(delta):.4f} at delta = {delta:g}")
        return "\n".join(lines)
