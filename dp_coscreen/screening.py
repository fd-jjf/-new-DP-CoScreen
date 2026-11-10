"""
screening.py:
Main DP-CoScreen framework: One-shot differentially private collaborative orthogonal feature screening.
Formalizes Algorithm 1 from the manuscript.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .privacy import (
    calibrate_agm,
    compute_l2_sensitivity,
    sample_symmetric_gaussian_noise,
)
from .preprocessing import clip_augmented_vectors
from .schur import project_psd_cone, schur_orthogonal_screening


class DPCoScreen:
    """
    DP-CoScreen: Differentially Private Collaborative Orthogonal Feature Screening
    
    Parameters
    ----------
    K : int, default=10
        Target feature selection budget.
    epsilon : float, default=1.0
        Differential privacy budget epsilon > 0.
    delta : float, default=1e-5
        Differential privacy relaxation parameter delta in (0, 1).
    rho : float, default=1e-3
        Tikhonov ridge regularization parameter for Schur-complement conditioning.
    seed : int, optional
        Simulation random seed for reproducible benchmark runs.
    """
    def __init__(
        self,
        K: int = 10,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        rho: float = 1e-3,
        adaptive_rho: bool = True,
        seed: Optional[int] = 42
    ) -> None:
        self.K = K
        self.epsilon = epsilon
        self.delta = delta
        self.rho = rho
        self.adaptive_rho = adaptive_rho
        self.seed = seed
        
        # State populated during fit
        self.selected_indices_: Optional[np.ndarray] = None
        self.screening_trace_: Optional[List[Dict[str, Any]]] = None
        self.diagnostics_: Dict[str, Any] = {}
        self.rng_ = np.random.default_rng(seed)

    def fit(
        self,
        client_datasets: List[Tuple[np.ndarray, np.ndarray]],
        record_timings: bool = True
    ) -> "DPCoScreen":
        """
        Executes DP-CoScreen end-to-end distributed screening across M edge clients.
        
        Parameters
        ----------
        client_datasets : list of (X_i, y_i) tuples
            Distributed telemetry datasets residing on M edge gateways.
            X_i: (n_i, d) with attributes in [0, 1]^d.
            y_i: (n_i,) binary intrusion labels in {0, 1}.
        record_timings : bool, default=True
            Whether to record detailed latency profiling for edge gateways and coordinator.
            
        Returns
        -------
        self : DPCoScreen
            Fitted screener with selected_indices_ populated.
        """
        M = len(client_datasets)
        if M == 0:
            raise ValueError("client_datasets cannot be empty.")
            
        d = client_datasets[0][0].shape[1]
        dim = d + 1
        
        t0_total = time.perf_counter()
        
        # =========================================================================
        # Phase 1: Client-Side Local Private Statistics Release (Parallelizable)
        # =========================================================================
        t0_client = time.perf_counter()
        sanitized_matrices: List[np.ndarray] = []
        client_sizes: List[int] = []
        client_sigmas: List[float] = []
        
        for i, (X_i, y_i) in enumerate(client_datasets):
            n_i = len(y_i)
            client_sizes.append(n_i)
            
            # Step 1.1: Physical RFC/Modbus bounding and augmented vector formation
            Z_i = clip_augmented_vectors(X_i, y_i)
            
            # Step 1.2: Local empirical second moment
            H_emp_i = (Z_i.T @ Z_i) / float(n_i)
            
            # Step 1.3: Analytic Gaussian Mechanism calibration
            delta_2_i = compute_l2_sensitivity(d, n_i)
            sigma_i = calibrate_agm(self.epsilon, self.delta, delta_2_i)
            client_sigmas.append(sigma_i)
            
            # Step 1.4: Symmetric Gaussian perturbation
            E_i = sample_symmetric_gaussian_noise(dim, sigma_i, self.rng_)
            H_sanitized_i = H_emp_i + E_i
            
            sanitized_matrices.append(H_sanitized_i)
            
        t_client = (time.perf_counter() - t0_client) * 1000.0 / float(M)  # Average per-gateway ms
        
        # =========================================================================
        # Phase 2: Server-Side Sample-Weighted Aggregation & PSD Projection
        # =========================================================================
        t0_server = time.perf_counter()
        total_N = sum(client_sizes)
        weights = [float(n_i) / float(total_N) for n_i in client_sizes]
        
        # Collaborative aggregation: entrywise variance suppressed by 1/M
        H_agg = np.zeros((dim, dim), dtype=np.float64)
        for w_i, H_s_i in zip(weights, sanitized_matrices):
            H_agg += w_i * H_s_i
            
        # Partition into feature second-moment block A and feature-target vector q
        A_raw = H_agg[:d, :d]
        q_raw = H_agg[:d, d]
        
        # Step 2.2: Metric projection onto PSD cone S_+^d
        A_psd = project_psd_cone(A_raw)
        q_final = q_raw.copy()

        # Step 2.3: Calibrate ridge parameter
        if self.adaptive_rho:
            eff_sigma = float(np.sqrt(sum((w_i * s_i)**2 for w_i, s_i in zip(weights, client_sigmas))))
            actual_rho = max(self.rho, eff_sigma * np.sqrt(float(d)))
        else:
            actual_rho = self.rho
        
        # =========================================================================
        # Phase 3: Closed-Form Greedy Schur-Complement Orthogonal Screening
        # =========================================================================
        selected, trace = schur_orthogonal_screening(
            A=A_psd,
            q=q_final,
            K=self.K,
            rho=actual_rho,
            return_trace=True
        )
        t_server = (time.perf_counter() - t0_server) * 1000.0  # Server ms
        t_total = (time.perf_counter() - t0_total) * 1000.0
        
        # Store results
        self.selected_indices_ = selected
        self.screening_trace_ = trace
        
        # Uplink communication payload: upper triangle of symmetric (d+1) x (d+1) matrix
        n_elements = (dim * (dim + 1)) // 2
        payload_bytes_float32 = n_elements * 4
        payload_bytes_float64 = n_elements * 8
        
        self.diagnostics_ = {
            "n_clients": M,
            "total_samples": total_N,
            "feature_dim": d,
            "selected_k": self.K,
            "privacy_epsilon": self.epsilon,
            "privacy_delta": self.delta,
            "ridge_rho": actual_rho,
            "mean_sigma": float(np.mean(client_sigmas)),
            "uplink_payload_kb_float32": float(payload_bytes_float32) / 1024.0,
            "uplink_payload_kb_float64": float(payload_bytes_float64) / 1024.0,
            "latency_edge_gateway_ms": t_client,
            "latency_coordinator_screening_ms": t_server,
            "latency_total_pipeline_ms": t_total,
            "min_eigval_before_psd": float(np.min(np.linalg.eigvalsh((A_raw + A_raw.T) / 2.0))),
            "min_eigval_after_psd": float(np.min(np.linalg.eigvalsh(A_psd))),
        }
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projects high-dimensional telemetry onto the screened K-dimensional subspace.
        
        Parameters
        ----------
        X : np.ndarray of shape (N, d)
            Input telemetry matrix.
            
        Returns
        -------
        np.ndarray of shape (N, K)
            Subspace-screened telemetry matrix.
        """
        if self.selected_indices_ is None:
            raise RuntimeError("DPCoScreen must be fitted before calling transform().")
        return X[:, self.selected_indices_]

    def fit_transform(
        self,
        client_datasets: List[Tuple[np.ndarray, np.ndarray]],
        X_eval: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits on distributed edge clients and transforms evaluation set.
        """
        self.fit(client_datasets)
        if X_eval is not None:
            return self.selected_indices_, self.transform(X_eval)
        return self.selected_indices_, None
