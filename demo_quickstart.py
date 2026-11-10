"""
demo_quickstart.py:
1-Minute Quickstart Demonstration for DP-CoScreen.
Requires zero external dataset downloads; runs out-of-the-box on synthetic IIoT telemetry.

Demonstrates:
1. Multi-client non-IID edge telemetry generation with multicollinear attributes.
2. Single-shot private empirical second-moment release under Analytic Gaussian Mechanism (AGM).
3. Sample-weighted aggregation and metric PSD cone projection.
4. Closed-form greedy Schur-complement orthogonal feature screening.
5. Suppression of redundant attributes and downstream NIDS intrusion detection evaluation.
"""

import sys
import time
from pathlib import Path

# Add package root to Python path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
from dp_coscreen import (
    DPCoScreen,
    evaluate_classifier,
    partition_dirichlet_noniid,
)
from baselines import select_dp_corr, select_dpkendall


def run_quickstart():
    print("=" * 75)
    print("  DP-CoScreen: Differentially Private Orthogonal Feature Screening")
    print("  Journal of Information Security and Applications (JISA) Artifact")
    print("=" * 75)
    
    # ---------------------------------------------------------
    # 1. Synthesize Distributed Industrial IoT Telemetry
    # ---------------------------------------------------------
    n_samples = 25000
    n_base_features = 20
    n_clients = 10
    K = 6
    epsilon = 1.0
    delta = 1e-5
    
    print(f"\n[1] Generating distributed telemetry across M={n_clients} non-IID edge gateways...")
    print(f"    - Total samples: {n_samples:,}")
    print(f"    - Privacy budget: eps={epsilon}, delta={delta}")
    print(f"    - Selection target: K={K} features")
    
    rng = np.random.default_rng(2026)
    
    # Base independent telemetry signals
    X_base = rng.uniform(0.0, 1.0, size=(n_samples, n_base_features))
    
    # Inject 3 collinear duplicate features:
    # Feature 20 is nearly identical to Feature 0 (e.g. redundant packet/byte counter)
    # Feature 21 is nearly identical to Feature 1 (e.g. correlated flow duration)
    # Feature 22 is nearly identical to Feature 2
    f_dup0 = np.clip(X_base[:, 0] + rng.normal(0, 0.01, size=n_samples), 0.0, 1.0)
    f_dup1 = np.clip(X_base[:, 1] + rng.normal(0, 0.01, size=n_samples), 0.0, 1.0)
    f_dup2 = np.clip(X_base[:, 2] + rng.normal(0, 0.01, size=n_samples), 0.0, 1.0)
    
    X = np.column_stack([X_base, f_dup0, f_dup1, f_dup2])
    d = X.shape[1]
    
    # Ground-truth attack probability determined by features 0, 1, 2, and 3
    logits = 3.5 * X[:, 0] - 3.0 * X[:, 1] + 2.5 * X[:, 2] - 2.0 * X[:, 3] - 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(0.0, 1.0, size=n_samples) < probs).astype(int)
    
    # Train / Test Split
    train_size = 20000
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    # Partition train data across M edge clients under Dirichlet non-IID split
    client_datasets = partition_dirichlet_noniid(
        X_train, y_train, n_clients=n_clients, alpha=0.5, seed=42
    )
    print(f"    - Client partition complete (gateway sample counts: {[len(cd[1]) for cd in client_datasets]})")
    
    # ---------------------------------------------------------
    # 2. Execute DP-CoScreen Framework
    # ---------------------------------------------------------
    print("\n[2] Executing DP-CoScreen Pipeline...")
    screener = DPCoScreen(K=K, epsilon=epsilon, delta=delta, rho=1e-3, seed=42)
    
    t0 = time.perf_counter()
    screener.fit(client_datasets)
    t_elapsed = (time.perf_counter() - t0) * 1000.0
    
    diag = screener.diagnostics_
    selected_dp = screener.selected_indices_
    
    print(f"    * Pipeline Completed in: {t_elapsed:.2f} ms")
    print(f"    * Edge Gateway Release Latency: {diag['latency_edge_gateway_ms']:.2f} ms")
    print(f"    * Coordinator Screening Latency: {diag['latency_coordinator_screening_ms']:.2f} ms")
    print(f"    * Uplink Payload per Gateway: {diag['uplink_payload_kb_float32']:.2f} KB (float32)")
    print(f"    * Min Eigenvalue before PSD: {diag['min_eigval_before_psd']:.4f} -> after PSD: {diag['min_eigval_after_psd']:.4e}")
    print(f"    * Selected Feature Indices: {selected_dp}")
    
    # ---------------------------------------------------------
    # 3. Multicollinearity Analysis
    # ---------------------------------------------------------
    print("\n[3] Multicollinearity Suppression Verification:")
    # Check if collinear duplicates (20, 21, 22) were picked alongside their original parents (0, 1, 2)
    has_dup0 = (0 in selected_dp and 20 in selected_dp)
    has_dup1 = (1 in selected_dp and 21 in selected_dp)
    has_dup2 = (2 in selected_dp and 22 in selected_dp)
    
    if not (has_dup0 or has_dup1 or has_dup2):
        print("    [SUCCESS] Schur-complement conditioning successfully eliminated all collinear duplicates!")
    else:
        print("    [WARNING] Duplicate feature admitted.")
        
    # ---------------------------------------------------------
    # 4. Compare with Baseline Selectors
    # ---------------------------------------------------------
    print("\n[4] Comparative Downstream Intrusion Detection Evaluation:")
    
    # 4.1 DP-CoScreen
    m_dp = evaluate_classifier(X_train, y_train, X_test, y_test, selected_indices=selected_dp)
    
    # 4.2 DP-Corr / Top-R (Univariate Ranking)
    sel_corr = select_dp_corr(X_train, y_train, k=K, epsilon=epsilon, seed=42)
    m_corr = evaluate_classifier(X_train, y_train, X_test, y_test, selected_indices=sel_corr)
    
    # 4.3 DPKendall
    sel_kendall = select_dpkendall(X_train[:2000], y_train[:2000], k=K, epsilon=epsilon, seed=42)
    m_kendall = evaluate_classifier(X_train, y_train, X_test, y_test, selected_indices=sel_kendall)
    
    # 4.4 All-Features (Non-Private Reference Bound)
    m_all = evaluate_classifier(X_train, y_train, X_test, y_test, selected_indices=None)

    print("-" * 75)
    print(f"{'Method':<25} | {'Selected Indices':<18} | {'Macro-F1':<10} | {'Accuracy':<10}")
    print("-" * 75)
    print(f"{'DP-CoScreen (Ours)':<25} | {str(selected_dp[:4].tolist()) + '...':<18} | {m_dp['macro_f1']:.4f}     | {m_dp['accuracy']:.4f}")
    print(f"{'DPKendall':<25} | {str(sel_kendall[:4].tolist()) + '...':<18} | {m_kendall['macro_f1']:.4f}     | {m_kendall['accuracy']:.4f}")
    print(f"{'DP-Corr / Top-R':<25} | {str(sel_corr[:4].tolist()) + '...':<18} | {m_corr['macro_f1']:.4f}     | {m_corr['accuracy']:.4f}")
    print(f"{'Full Data (All d={d})':<25} | {'[All ' + str(d) + ' features]':<18} | {m_all['macro_f1']:.4f}     | {m_all['accuracy']:.4f}")
    print("-" * 75)
    print("\nQuickstart demo completed successfully!")


if __name__ == "__main__":
    run_quickstart()
