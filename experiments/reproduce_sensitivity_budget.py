"""
reproduce_sensitivity_budget.py:
Reproduces parameter sensitivity experiments (Figures 2 & 3 in manuscript):
1. Feature budget sensitivity K in [2, 4, 6, 8, 10] at epsilon = 1.0.
2. Privacy budget sensitivity epsilon in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0] at K = 10.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from dp_coscreen import (
    DPCoScreen,
    FixedFeatureMap,
    evaluate_classifier,
    partition_dirichlet_noniid,
)
from baselines import select_dpkendall, select_dp_corr, select_fedsdg

from reproduce_table3_benchmark import DATASET_CONFIGS, load_dataset


def run_sensitivity(
    dataset_key: str = "ton",
    n_splits: int = 3,
    output_dir: Path | None = None
):
    df, cfg = load_dataset(dataset_key)
    output_dir = output_dir or (PACKAGE_ROOT / "results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Stratified Group Split
    group_fields = [c for c in (cfg["numeric"] + cfg["categorical"]) if c in df.columns]
    group_ids = df.groupby(group_fields, dropna=False, sort=False).ngroup().to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dev_idx, res_idx = next(splitter.split(df, groups=group_ids))
    
    mapper = FixedFeatureMap(cfg["numeric"], cfg["categorical"], hash_buckets=4)
    X_dev = mapper.transform(df.iloc[dev_idx].reset_index(drop=True))
    y_dev = df.iloc[dev_idx][cfg["label"]].to_numpy(dtype=int)
    X_res = mapper.transform(df.iloc[res_idx].reset_index(drop=True))
    y_res = df.iloc[res_idx][cfg["label"]].to_numpy(dtype=int)
    
    K_VALUES = [2, 4, 6, 8, 10]
    EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    seeds = [3101 + i for i in range(n_splits)]
    
    # -------------------------------------------------------------
    # 1. Feature Budget Sensitivity (K)
    # -------------------------------------------------------------
    print(f"\n[1] Evaluating Feature Budget Sensitivity K in {K_VALUES} (eps=1.0)...")
    k_rows = []
    for k in K_VALUES:
        for seed in seeds:
            client_datasets = partition_dirichlet_noniid(X_dev, y_dev, n_clients=10, alpha=0.5, seed=seed)
            screener = DPCoScreen(K=k, epsilon=1.0, delta=1e-5, rho=1e-3, seed=seed)
            screener.fit(client_datasets)
            m = evaluate_classifier(X_dev, y_dev, X_res, y_res, selected_indices=screener.selected_indices_)
            k_rows.append({
                "Dataset": cfg["name"],
                "K": k,
                "Epsilon": 1.0,
                "Seed": seed,
                "Macro_F1": m["macro_f1"],
                "Accuracy": m["accuracy"],
                "AUROC": m["auroc"],
            })
        mean_f1 = np.mean([r["Macro_F1"] for r in k_rows if r["K"] == k])
        print(f"  K={k:<2} | Mean Macro-F1: {mean_f1:.4f}")
        
    df_k = pd.DataFrame(k_rows)
    df_k.to_csv(output_dir / f"sensitivity_k_{dataset_key}.csv", index=False)
    
    # -------------------------------------------------------------
    # 2. Privacy Budget Sensitivity (Epsilon)
    # -------------------------------------------------------------
    print(f"\n[2] Evaluating Privacy Budget Sensitivity Epsilon in {EPSILON_VALUES} (K=10)...")
    eps_rows = []
    for eps in EPSILON_VALUES:
        for seed in seeds:
            client_datasets = partition_dirichlet_noniid(X_dev, y_dev, n_clients=10, alpha=0.5, seed=seed)
            screener = DPCoScreen(K=10, epsilon=eps, delta=1e-5, rho=1e-3, seed=seed)
            screener.fit(client_datasets)
            m = evaluate_classifier(X_dev, y_dev, X_res, y_res, selected_indices=screener.selected_indices_)
            eps_rows.append({
                "Dataset": cfg["name"],
                "K": 10,
                "Epsilon": eps,
                "Seed": seed,
                "Macro_F1": m["macro_f1"],
                "Accuracy": m["accuracy"],
                "AUROC": m["auroc"],
            })
        mean_f1 = np.mean([r["Macro_F1"] for r in eps_rows if r["Epsilon"] == eps])
        print(f"  Eps={eps:<4} | Mean Macro-F1: {mean_f1:.4f}")
        
    df_eps = pd.DataFrame(eps_rows)
    df_eps.to_csv(output_dir / f"sensitivity_epsilon_{dataset_key}.csv", index=False)
    print(f"\n[OK] Sensitivity results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Sensitivity Experiments")
    parser.add_argument("--dataset", choices=["ton", "edge"], default="ton")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    
    run_sensitivity(args.dataset, args.splits, args.output_dir)
