"""
dp_baselines.py:
State-of-the-art differentially private feature selection baseline algorithms:
1. DPKendall (Srivastava et al., Google Research, 2023)
2. DP-Corr / Top-R (Swope et al., 2024; Wang et al., 2025)
3. Joint Top-k (Gillenwater et al., 2022)
4. FedSDG-FS (Zhang et al., 2024)
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import norm


def select_dp_corr(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    epsilon: float = 1.0,
    seed: int = 42
) -> np.ndarray:
    """
    DP-Corr / Top-R (Swope et al., 2024):
    Differentially private univariate Pearson correlation ranking.
    Perturbs individual feature-target correlations with Laplace noise.
    
    Parameters
    ----------
    X : np.ndarray of shape (n, d)
    y : np.ndarray of shape (n,)
    k : int
    epsilon : float
    seed : int
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    
    # Compute Pearson correlation with target
    corrs = np.zeros(d, dtype=float)
    y_std = np.std(y)
    if y_std < 1e-12:
        return rng.choice(d, size=k, replace=False)
        
    y_norm = (y - np.mean(y)) / (y_std + 1e-12)
    
    for j in range(d):
        xj = X[:, j]
        xj_std = np.std(xj)
        if xj_std < 1e-12:
            corrs[j] = 0.0
        else:
            xj_norm = (xj - np.mean(xj)) / xj_std
            corrs[j] = np.abs(np.mean(xj_norm * y_norm))
            
    # L1 Sensitivity of Pearson correlation: Delta = 2.0 / n
    sensitivity = 2.0 / float(n)
    laplace_scale = sensitivity / epsilon
    noisy_scores = corrs + rng.laplace(loc=0.0, scale=laplace_scale, size=d)
    
    # Select top-k
    selected = np.argsort(noisy_scores)[::-1][:k]
    return selected.astype(int)


def _scaled_abs_kendall(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, d = X.shape
    scores = np.zeros(d, dtype=float)
    for j in range(d):
        tau, _ = stats.kendalltau(X[:, j], y, method="asymptotic")
        scores[j] = 0.0 if np.isnan(tau) else np.abs((n / 2.0) * tau)
    return scores


def select_dpkendall(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    epsilon: float = 1.0,
    seed: int = 42,
    tie_jitter: float = 1e-6
) -> np.ndarray:
    """
    DPKendall feature selection (Srivastava et al., Google Research, 2023).
    Iteratively ranks Kendall tau concordance with target and pairwise inter-feature redundancy,
    calibrated with Gumbel noise under exponential peeling mechanism.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    
    if tie_jitter > 0:
        X_jitter = X + rng.uniform(-tie_jitter, tie_jitter, size=X.shape)
        y_jitter = y + rng.uniform(-tie_jitter, tie_jitter, size=y.shape)
    else:
        X_jitter, y_jitter = X, y

    label_corr = _scaled_abs_kendall(X_jitter, y_jitter)
    selected: list[int] = []
    redundancy_sum = np.zeros(d, dtype=float)
    split_eps = epsilon / float(k)

    for step in range(k):
        sensitivity = 1.5 if step == 0 else 3.0
        score = label_corr.copy()
        if selected:
            last_sel = selected[-1]
            tau_last = np.zeros(d, dtype=float)
            for j in range(d):
                t, _ = stats.kendalltau(X_jitter[:, last_sel], X_jitter[:, j], method="asymptotic")
                tau_last[j] = 0.0 if np.isnan(t) else np.abs((n / 2.0) * t)
            redundancy_sum += tau_last
            score -= redundancy_sum / float(len(selected))
            
        score[selected] = -np.inf
        noise_scale = sensitivity / (split_eps / 2.0)
        noisy_score = score + rng.gumbel(loc=0.0, scale=noise_scale, size=d)
        best = int(np.argmax(noisy_score))
        selected.append(best)

    return np.array(selected, dtype=int)


def select_joint_topk(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    epsilon: float = 1.0,
    seed: int = 42
) -> np.ndarray:
    """
    Joint Top-k Selection (Gillenwater et al., 2022):
    Base scoring with Kendall correlation and inter-feature covariance redundancy penalty,
    selected via Exponential Mechanism (Gumbel noise).
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    base_scores = _scaled_abs_kendall(X, y)
    
    corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    selected = []
    split_eps = epsilon / float(k)
    beta = 2.0 / split_eps

    for _ in range(k):
        if len(selected) == 0:
            scores = base_scores.copy()
        else:
            penalty = np.sum(corr_matrix[selected, :], axis=0) / float(len(selected))
            scores = base_scores - penalty

        scores[selected] = -np.inf
        noise = rng.gumbel(loc=0.0, scale=beta, size=d)
        chosen = int(np.argmax(scores + noise))
        selected.append(chosen)

    return np.array(selected, dtype=int)


def select_fedsdg(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    epsilon: float = 1.0,
    n_iters: int = 50,
    seed: int = 42
) -> np.ndarray:
    """
    FedSDG-FS: Federated Stochastic Dual-Gating Feature Selection (Zhang et al., 2024).
    Iteratively optimizes gating parameters via federated DP-SGD.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # Initial gating mean
    mu = np.full(d, 0.5, dtype=float)
    sigma_gate = 0.3
    # DP-SGD Gaussian noise per step under fixed composition
    sigma_dp = np.sqrt(2.0 * np.log(1.25 / 1e-5)) / (epsilon / np.sqrt(n_iters))
    lr = 0.05
    lambda_reg = 0.05

    for _ in range(n_iters):
        rho = rng.normal(0, sigma_gate, d)
        s = np.clip(mu + rho, 0.0, 1.0)

        phi = norm.pdf(mu / sigma_gate) / sigma_gate
        Phi_val = norm.cdf(mu / sigma_gate) + 1e-8
        grad_reg = lambda_reg * phi / Phi_val

        X_sel = X * s.reshape(1, -1)
        w, _, _, _ = np.linalg.lstsq(X_sel, y, rcond=None)
        residual = y - X_sel @ w
        grad_importance = np.abs(X_sel.T @ residual) / float(n)
        max_imp = np.max(grad_importance)
        if max_imp > 1e-12:
            grad_importance /= max_imp

        # DP-SGD noise addition
        grad = grad_importance - grad_reg + rng.normal(0, sigma_dp, d)
        mu = np.clip(mu + lr * grad, 0.0, 1.0)

    selected = np.argsort(mu)[::-1][:k]
    return selected.astype(int)
