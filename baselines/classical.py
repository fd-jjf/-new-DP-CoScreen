"""
classical.py:
Classical and non-private feature selection baselines:
1. mRMR-MID (Minimum Redundancy Maximum Relevance - Mutual Information Difference)
2. Centralized / Federated Chi-Square Feature Selection
3. Non-Private Greedy Reference (Zero Privacy Noise Benchmark)
4. Random Selection Baseline
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def select_random(d: int, k: int, seed: int = 42) -> np.ndarray:
    """Randomly selects k features from d candidates."""
    rng = np.random.default_rng(seed)
    return rng.choice(d, size=k, replace=False).astype(int)


def _discretize(values: np.ndarray, bins: int = 10) -> np.ndarray:
    unique_vals, inverse = np.unique(values, return_inverse=True)
    if len(unique_vals) <= bins:
        return inverse.astype(np.int32)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=np.int32)
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int32)


def _mutual_information(x_disc: np.ndarray, y_disc: np.ndarray) -> float:
    n = len(x_disc)
    joint_tab = np.zeros((np.max(x_disc) + 1, np.max(y_disc) + 1), dtype=float)
    np.add.at(joint_tab, (x_disc, y_disc), 1.0)
    p_xy = joint_tab / float(n)
    
    p_x = np.sum(p_xy, axis=1, keepdims=True)
    p_y = np.sum(p_xy, axis=0, keepdims=True)
    
    nonzero = p_xy > 0
    mi = np.sum(p_xy[nonzero] * np.log((p_xy[nonzero]) / (p_x @ p_y)[nonzero]))
    return max(0.0, float(mi))


def select_mrmr_mid(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    bins: int = 10
) -> np.ndarray:
    """
    mRMR-MID: Minimum Redundancy Maximum Relevance feature selection via Mutual Information.
    """
    n, d = X.shape
    y_disc = (y > 0).astype(np.int32)
    
    X_disc = np.column_stack([_discretize(X[:, j], bins) for j in range(d)])
    
    # 1. Relevance: MI(x_j, y)
    relevance = np.array([_mutual_information(X_disc[:, j], y_disc) for j in range(d)])
    
    selected: List[int] = []
    redundancy_table = np.zeros((d, d), dtype=float)
    computed = np.zeros((d, d), dtype=bool)

    for step in range(k):
        if step == 0:
            scores = relevance.copy()
        else:
            last = selected[-1]
            for j in range(d):
                if not computed[last, j]:
                    mi_pair = _mutual_information(X_disc[:, last], X_disc[:, j])
                    redundancy_table[last, j] = mi_pair
                    redundancy_table[j, last] = mi_pair
                    computed[last, j] = True
                    computed[j, last] = True
                    
            mean_red = np.mean(redundancy_table[selected, :], axis=0)
            scores = relevance - mean_red

        scores[selected] = -np.inf
        chosen = int(np.argmax(scores))
        selected.append(chosen)

    return np.array(selected, dtype=int)


def select_chi2(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    bins: int = 10
) -> np.ndarray:
    """
    Classical Chi-Square (chi2) feature selection.
    """
    n, d = X.shape
    y_disc = (y > 0).astype(np.int32)
    scores = np.zeros(d, dtype=float)
    
    for j in range(d):
        xj = _discretize(X[:, j], bins)
        observed = np.zeros((np.max(xj) + 1, 2), dtype=float)
        np.add.at(observed, (xj, y_disc), 1.0)
        
        row_totals = observed.sum(axis=1, keepdims=True)
        col_totals = observed.sum(axis=0, keepdims=True)
        expected = (row_totals @ col_totals) / float(n)
        
        mask = expected > 0
        stat = np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])
        scores[j] = stat

    return np.argsort(scores)[::-1][:k].astype(int)


def select_non_private_reference(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    rho: float = 1e-3
) -> np.ndarray:
    """
    Non-Private Reference:
    Computes exact sample second-moment matrix without privacy noise and performs
    orthogonal Schur-complement greedy screening.
    """
    n, d = X.shape
    y_col = y.reshape(-1, 1).astype(float)
    Z = np.hstack([X, y_col])
    H = (Z.T @ Z) / float(n)
    
    A = H[:d, :d]
    q = H[:d, d]
    
    from dp_coscreen.schur import project_psd_cone, schur_orthogonal_screening
    A_psd = project_psd_cone(A)
    return schur_orthogonal_screening(A_psd, q, K=k, rho=rho)
