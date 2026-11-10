"""
evaluation.py:
Downstream intrusion detection classifier evaluation and statistical aggregation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def evaluate_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_indices: Optional[np.ndarray] = None,
    classifier_type: str = "logistic_regression",
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluates downstream intrusion detection performance on test partition.
    
    Parameters
    ----------
    X_train : np.ndarray
        Full or transformed training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Testing features.
    y_test : np.ndarray
        Testing labels.
    selected_indices : np.ndarray, optional
        Feature subset indices (if None, all features are used).
    classifier_type : str, default="logistic_regression"
        Classifier model ('logistic_regression', 'random_forest', 'mlp').
    seed : int, default=42
        Random seed.
        
    Returns
    -------
    dict
        Dictionary containing Macro-F1, Accuracy, AUROC, and Macro-Recall.
    """
    if selected_indices is not None:
        X_tr = X_train[:, selected_indices]
        X_te = X_test[:, selected_indices]
    else:
        X_tr = X_train
        X_te = X_test
        
    # Standardize selected subspace
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    if classifier_type == "logistic_regression":
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=seed
        )
    elif classifier_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    elif classifier_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    clf.fit(X_tr_scaled, y_train)
    
    y_pred = clf.predict(X_te_scaled)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_te_scaled)[:, 1]
    else:
        probs = y_pred.astype(float)
        
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    
    try:
        auroc = float(roc_auc_score(y_test, probs))
    except Exception:
        auroc = float(balanced_accuracy_score(y_test, y_pred))
        
    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "auroc": auroc,
        "macro_recall": recall
    }


def aggregate_split_statistics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregates metrics across multiple experimental splits: mean, std, and 95% confidence intervals.
    """
    keys = metrics_list[0].keys()
    results = {}
    n = len(metrics_list)
    
    for k in keys:
        vals = np.array([m[k] for m in metrics_list], dtype=float)
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        ci_95 = 1.96 * (std_val / np.sqrt(n)) if n > 1 else 0.0
        results[k] = {
            "mean": mean_val,
            "std": std_val,
            "ci95": ci_95,
            "formatted": f"{mean_val:.4f} +/- {ci_95:.4f}"
        }
    return results
