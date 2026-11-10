"""
schur.py:
Metric PSD cone projection and closed-form greedy Schur-complement orthogonal feature screening.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def project_psd_cone(A: np.ndarray, eps_floor: float = 0.0) -> np.ndarray:
    """
    Computes metric projection onto the positive semidefinite (PSD) cone S_+^d:
    Pi_{>= 0}(A) = V max(Lambda, eps_floor) V^T
    
    Parameters
    ----------
    A : np.ndarray of shape (d, d)
        Symmetric matrix (or perturbed feature block).
    eps_floor : float, default=0.0
        Minimum eigenvalue threshold (0.0 for pure projection).
        
    Returns
    -------
    np.ndarray of shape (d, d)
        Symmetric positive semidefinite matrix.
    """
    A_sym = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    clipped_eigvals = np.maximum(eigvals, eps_floor)
    A_psd = (eigvecs * clipped_eigvals) @ eigvecs.T
    return (A_psd + A_psd.T) / 2.0


def schur_orthogonal_screening(
    A: np.ndarray,
    q: np.ndarray,
    K: int,
    rho: float = 1e-3,
    return_trace: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, List[Dict[str, Union[int, float]]]]]:
    """
    Executes closed-form greedy orthogonal feature screening over regularized quadratic objective:
    Delta F_rho(j | S) = (r_j^{(S)})^2 / max(t_j^{(S)}, rho)
    
    Conditioning candidate features on the orthogonal subspace of currently selected attributes
    suppresses multicollinearity in O(|S| d) time with zero additional privacy expenditure.
    
    Parameters
    ----------
    A : np.ndarray of shape (d, d)
        PSD projected feature second-moment matrix.
    q : np.ndarray of shape (d,)
        Aggregated cross-moment vector between features and target.
    K : int
        Feature selection budget (number of features to select).
    rho : float, default=1e-3
        Tikhonov ridge regularizer ensuring strictly positive denominator.
    return_trace : bool, default=False
        Whether to return greedy selection metrics per step.
        
    Returns
    -------
    selected : np.ndarray of shape (K,)
        Indices of selected features.
    trace : list of dict (optional)
        Step-by-step selection gains and statistics.
    """
    d = len(q)
    if A.shape != (d, d):
        raise ValueError(f"Shape mismatch: A {A.shape} vs q {q.shape}")
    if not (1 <= K <= d):
        raise ValueError(f"Feature budget K must be in [1, {d}], got {K}")
    if rho <= 0:
        raise ValueError(f"Ridge parameter rho must be positive, got {rho}")

    selected: List[int] = []
    selected_set = set()
    trace: List[Dict[str, Union[int, float]]] = []

    # Initialize residuals and conditional variances
    residual_q = q.copy()
    diag_A = np.diag(A)
    conditional_var = diag_A + rho

    for step in range(K):
        # Calculate greedy score for all unselected candidates
        if step == 0:
            scores = (residual_q ** 2) / np.maximum(conditional_var, rho)
        else:
            # Block Schur complement update
            # S is current selected set
            idx_S = np.array(selected, dtype=int)
            A_SS = A[np.ix_(idx_S, idx_S)] + rho * np.eye(len(selected))
            A_all_S = A[:, idx_S]  # (d, |S|)
            
            # Solve (A_SS + rho I) * beta_S = q_S
            beta_S = np.linalg.solve(A_SS, q[idx_S])
            residual_q = q - A_all_S @ beta_S
            
            # Solve (A_SS + rho I) * W = A_{S, :} -> W^T: (d, |S|)
            # conditional_var_j = A_jj + rho - A_{j, S} (A_SS + rho I)^{-1} A_{S, j}
            W = np.linalg.solve(A_SS, A_all_S.T)  # (|S|, d)
            quad_term = np.sum(A_all_S * W.T, axis=1)  # (d,)
            conditional_var = diag_A + rho - quad_term
            
            scores = (residual_q ** 2) / np.maximum(conditional_var, rho)

        # Mask out already selected features
        for s in selected_set:
            scores[s] = -np.inf

        best_candidate = int(np.argmax(scores))
        best_gain = float(scores[best_candidate])
        
        selected.append(best_candidate)
        selected_set.add(best_candidate)
        
        if return_trace:
            trace.append({
                "step": step,
                "selected_feature": best_candidate,
                "schur_gain": best_gain,
                "residual_correlation": float(residual_q[best_candidate]),
                "conditional_variance": float(conditional_var[best_candidate])
            })

    result = np.array(selected, dtype=int)
    if return_trace:
        return result, trace
    return result


def compute_spectral_suboptimality_bound(
    A_noisy: np.ndarray,
    q_noisy: np.ndarray,
    A_clean: np.ndarray,
    q_clean: np.ndarray,
    rho: float = 1e-3
) -> Dict[str, float]:
    """
    Computes spectral perturbation bounds on the feature screening objective:
    ||A_noisy - A_clean||_2 and ||q_noisy - q_clean||_2.
    """
    delta_A = float(np.linalg.norm(A_noisy - A_clean, ord=2))
    delta_q = float(np.linalg.norm(q_noisy - q_clean, ord=2))
    
    # Suboptimality gap derived in manuscript Section 4
    # Gap <= (2 * ||q_clean||_2 * delta_q + ||q_clean||_2^2 * delta_A / rho) / rho
    norm_q = float(np.linalg.norm(q_clean, ord=2))
    theoretical_gap_bound = (2.0 * norm_q * delta_q + (norm_q ** 2) * delta_A / rho) / rho
    
    return {
        "spectral_norm_error_A": delta_A,
        "euclidean_norm_error_q": delta_q,
        "theoretical_suboptimality_gap": theoretical_gap_bound
    }
