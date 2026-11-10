"""
privacy.py:
Analytic Gaussian Mechanism (AGM) and differential privacy calibrations
for empirical second-moment statistics release.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def agm_equation(v: float, epsilon: float, delta: float) -> float:
    """
    Evaluates the privacy loss distribution equation for AGM.
    
    Parameters
    ----------
    v : float
        Normalized noise scale v = sigma / Delta_2.
    epsilon : float
        Privacy parameter epsilon > 0.
    delta : float
        Privacy parameter delta in (0, 1).
        
    Returns
    -------
    float
        Residual g(v) = Phi(1/(2v) - eps*v) - exp(eps) * Phi(-1/(2v) - eps*v) - delta.
    """
    if v <= 0:
        return 1.0 - delta
    arg1 = 1.0 / (2.0 * v) - epsilon * v
    arg2 = -1.0 / (2.0 * v) - epsilon * v
    term1 = norm.cdf(arg1)
    # Use log-space computation for exp(epsilon) * norm.cdf(arg2) if epsilon is large
    if epsilon > 50:
        log_term2 = epsilon + norm.logcdf(arg2)
        term2 = math.exp(log_term2) if log_term2 < 700 else float('inf')
    else:
        term2 = math.exp(epsilon) * norm.cdf(arg2)
    return float(term1 - term2 - delta)


def calibrate_agm(epsilon: float, delta: float, delta_2: float) -> float:
    """
    Calibrates exact noise scale sigma via the Analytic Gaussian Mechanism (AGM)
    (Balle & Wang, ICML 2018).
    
    Parameters
    ----------
    epsilon : float
        Differential privacy budget epsilon > 0.
    delta : float
        Differential privacy relaxation parameter delta in (0, 1).
    delta_2 : float
        Global L2-sensitivity Delta_2 > 0.
        
    Returns
    -------
    float
        Exact noise standard deviation sigma* > 0.
    """
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be strictly positive, got {epsilon}")
    if not (0 < delta < 1):
        raise ValueError(f"Delta must be in (0, 1), got {delta}")
    if delta_2 <= 0:
        raise ValueError(f"Sensitivity delta_2 must be positive, got {delta_2}")

    # Standard classical Gaussian upper bound provides an initial search bracket:
    # sigma_classical = delta_2 * sqrt(2 * ln(1.25 / delta)) / epsilon
    v_classical = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    
    # Establish root bracket: g(v_low) > 0, g(v_high) < 0
    v_low = 1e-7
    v_high = max(v_classical * 2.0, 10.0)
    
    while agm_equation(v_high, epsilon, delta) > 0:
        v_high *= 2.0
        if v_high > 1e6:
            raise RuntimeError("Failed to bracket AGM root; numerical limits exceeded.")

    v_star = brentq(agm_equation, v_low, v_high, args=(epsilon, delta), xtol=1e-12, rtol=1e-12)
    return float(v_star * delta_2)


def compute_l2_sensitivity(d: int, n: int) -> float:
    """
    Computes global L2-sensitivity for client-side empirical second-moment release:
    Delta_{2, i} = sqrt(2) * (d + 1) / n_i under unit-bounded augmented vectors.
    
    Parameters
    ----------
    d : int
        Feature dimension.
    n : int
        Local client sample size.
        
    Returns
    -------
    float
        Conservative global L2-sensitivity bound.
    """
    if n <= 0:
        raise ValueError(f"Sample size n must be positive, got {n}")
    return float(math.sqrt(2.0) * (d + 1) / n)


def sample_symmetric_gaussian_noise(dim: int, sigma: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Samples symmetric Gaussian perturbation matrix E for a (dim x dim) empirical second-moment release:
    E_jj ~ N(0, sigma^2)
    E_jk ~ N(0, sigma^2 / 2) for j != k
    This preserves ||svec(E)||_2 = ||E||_F.
    
    Parameters
    ----------
    dim : int
        Matrix dimension (d + 1).
    sigma : float
        AGM calibrated standard deviation.
    rng : np.random.Generator, optional
        Numpy random generator instance.
        
    Returns
    -------
    np.ndarray
        Symmetric noise matrix of shape (dim, dim).
    """
    if rng is None:
        rng = np.random.default_rng()
        
    # Generate upper triangle
    E = np.zeros((dim, dim), dtype=np.float64)
    # Off-diagonal: std = sigma / sqrt(2)
    triu_indices = np.triu_indices(dim, k=1)
    off_diag_noise = rng.normal(loc=0.0, scale=sigma / math.sqrt(2.0), size=len(triu_indices[0]))
    E[triu_indices] = off_diag_noise
    E[(triu_indices[1], triu_indices[0])] = off_diag_noise
    
    # Diagonal: std = sigma
    E[np.diag_indices(dim)] = rng.normal(loc=0.0, scale=sigma, size=dim)
    return E


@dataclass(frozen=True)
class PrivacyBudget:
    epsilon: float = 1.0
    delta: float = 1e-5
    
    def calibrate_noise(self, d: int, n: int) -> Tuple[float, float]:
        """
        Calibrates noise scale and returns (sensitivity, sigma).
        """
        delta_2 = compute_l2_sensitivity(d, n)
        sigma = calibrate_agm(self.epsilon, self.delta, delta_2)
        return delta_2, sigma
