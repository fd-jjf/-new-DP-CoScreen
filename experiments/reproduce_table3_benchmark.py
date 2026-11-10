"""
reproduce_table3_benchmark.py:
Reproduces Table 3 from the JISA manuscript:
Multi-metric intrusion detection performance across 6 methods on ToN-IoT (d=94) and Edge-IIoTset (d=72).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PACKAGE_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from dp_coscreen import (
    DPCoScreen,
    FixedFeatureMap,
    evaluate_classifier,
    aggregate_split_statistics,
    partition_dirichlet_noniid,
)
from baselines import (
    select_dp_corr,
    select_dpkendall,
    select_joint_topk,
    select_fedsdg,
    select_non_private_reference,
)

DATASET_CONFIGS = {
    "ton": {
        "name": "ToN-IoT",
        "default_path": Path("data/ToN-IoT/train_test_network.csv"),
        "label": "label",
        "aux": "type",
        "numeric": [
            "src_port", "dst_port", "duration", "src_bytes", "dst_bytes",
            "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts",
            "dst_ip_bytes", "dns_qclass", "dns_qtype", "dns_rcode",
            "http_trans_depth", "http_request_body_len", "http_response_body_len",
            "http_status_code",
        ],
        "categorical": [
            "proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA",
            "dns_rejected", "ssl_version", "ssl_cipher", "ssl_resumed",
            "ssl_established", "http_method", "http_version",
            "http_orig_mime_types", "http_resp_mime_types",
        ],
        "k": 10,
        "n_clients": 10,
    },
    "edge": {
        "name": "Edge-IIoTset",
        "default_path": Path("data/Edge-IIoTset/ML-EdgeIIoT-dataset.csv"),
        "label": "Attack_label",
        "aux": "Attack_type",
        "numeric": [
            "arp.opcode", "arp.hw.size", "icmp.seq_le", "http.content_length",
            "http.response", "tcp.connection.fin", "tcp.connection.rst",
            "tcp.connection.syn", "tcp.connection.synack", "tcp.dstport",
            "tcp.len", "tcp.srcport", "udp.port", "udp.time_delta",
            "dns.qry.name.len", "dns.qry.qu", "dns.qry.type",
            "dns.retransmission", "dns.retransmit_request",
            "dns.retransmit_request_in", "mqtt.conflag.cleansess",
            "mqtt.conflags", "mqtt.hdrflags", "mqtt.len", "mqtt.msgtype",
            "mqtt.proto_len", "mqtt.topic_len", "mqtt.ver",
        ],
        "categorical": [
            "http.request.method", "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic",
        ],
        "k": 10,
        "n_clients": 10,
    }
}


def load_dataset(dataset_key: str, custom_path: Path | None = None) -> Tuple[pd.DataFrame, dict]:
    cfg = DATASET_CONFIGS[dataset_key]
    data_path = custom_path or cfg["default_path"]
    
    # Try finding the data file in common locations
    candidates = [
        data_path,
        PACKAGE_ROOT / data_path,
        PACKAGE_ROOT.parent / data_path,
    ]
    actual_file = None
    for c in candidates:
        if c.exists():
            actual_file = c
            break
            
    if actual_file is None:
        raise FileNotFoundError(
            f"Dataset file not found at {data_path}. Please place the dataset in data/ or pass --data-path."
        )
        
    print(f"Loading {cfg['name']} from: {actual_file}")
    df = pd.read_csv(actual_file, low_memory=False).drop_duplicates(keep="first").reset_index(drop=True)
    return df, cfg


def run_benchmark(
    dataset_key: str = "ton",
    n_splits: int = 5,
    epsilon: float = 1.0,
    data_path: Path | None = None,
    output_dir: Path | None = None,
):
    df, cfg = load_dataset(dataset_key, data_path)
    output_dir = output_dir or (PACKAGE_ROOT / "results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    K = cfg["k"]
    n_clients = cfg["n_clients"]
    
    # Stratified Group Split
    group_fields = [c for c in (cfg["numeric"] + cfg["categorical"]) if c in df.columns]
    group_ids = df.groupby(group_fields, dropna=False, sort=False).ngroup().to_numpy()
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dev_idx, res_idx = next(splitter.split(df, groups=group_ids))
    
    dev_frame = df.iloc[dev_idx].reset_index(drop=True)
    res_frame = df.iloc[res_idx].reset_index(drop=True)
    
    # Preprocess
    mapper = FixedFeatureMap(cfg["numeric"], cfg["categorical"], hash_buckets=4)
    X_dev = mapper.transform(dev_frame)
    y_dev = dev_frame[cfg["label"]].to_numpy(dtype=int)
    
    X_res = mapper.transform(res_frame)
    y_res = res_frame[cfg["label"]].to_numpy(dtype=int)
    
    print(f"\nBenchmark Setup: {cfg['name']}")
    print(f"  * Feature Dimension: d={X_dev.shape[1]}")
    print(f"  * Development Set: {len(X_dev):,} samples")
    print(f"  * Reserved Test Set: {len(X_res):,} samples")
    print(f"  * Privacy Budget: eps={epsilon}, delta=1e-5, K={K}")
    print(f"  * Evaluating {n_splits} independent non-IID client splits...\n")
    
    methods = [
        "Non-Private Reference",
        "DP-CoScreen (Ours)",
        "DPKendall",
        "DP-Corr / Top-R",
        "Joint Top-k",
        "FedSDG-FS",
    ]
    
    results = {m: [] for m in methods}
    split_seeds = [3101 + i for i in range(n_splits)]
    
    for split_idx, seed in enumerate(split_seeds):
        print(f"--- Split {split_idx + 1}/{n_splits} (Seed={seed}) ---")
        client_datasets = partition_dirichlet_noniid(
            X_dev, y_dev, n_clients=n_clients, alpha=0.5, seed=seed
        )
        
        # 1. Non-Private Reference
        sel_np = select_non_private_reference(X_dev, y_dev, k=K, rho=1e-3)
        m_np = evaluate_classifier(X_dev, y_dev, X_res, y_res, selected_indices=sel_np)
        results["Non-Private Reference"].append(m_np)
        
        # 2. DP-CoScreen (Ours)
        screener = DPCoScreen(K=K, epsilon=epsilon, delta=1e-5, rho=1e-3, seed=seed)
        screener.fit(client_datasets)
        m_coscreen = evaluate_classifier(X_dev, y_dev, X_res, y_res, selected_indices=screener.selected_indices_)
        results["DP-CoScreen (Ours)"].append(m_coscreen)
        
        # 3. DPKendall
        # Use sub-sample for Kendall quadratic loop edge feasibility
        sub_n = min(3000, len(X_dev))
        sel_kendall = select_dpkendall(X_dev[:sub_n], y_dev[:sub_n], k=K, epsilon=epsilon, seed=seed)
        m_kendall = evaluate_classifier(X_dev, y_dev, X_res, y_res, selected_indices=sel_kendall)
        results["DPKendall"].append(m_kendall)
        
        # 4. DP-Corr / Top-R
        sel_corr = select_dp_corr(X_dev, y_dev, k=K, epsilon=epsilon, seed=seed)
        m_corr = evaluate_classifier(X_dev, y_dev, X_res, y_res, selected_indices=sel_corr)
        results["DP-Corr / Top-R"].append(m_corr)
        
        # 5. Joint Top-k
        sel_joint = select_joint_topk(X_dev[:sub_n], y_dev[:sub_n], k=K, epsilon=epsilon, seed=seed)
        m_joint = evaluate_classifier(X_dev, y_dev, X_res, y_res, selected_indices=sel_joint)
        results["Joint Top-k"].append(m_joint)
        
        # 6. FedSDG-FS
        sel_fedsdg = select_fedsdg(X_dev[:sub_n], y_dev[:sub_n], k=K, epsilon=epsilon, seed=seed)
        m_fedsdg = evaluate_classifier(X_dev, y_dev, X_res, y_res, selected_indices=sel_fedsdg)
        results["FedSDG-FS"].append(m_fedsdg)
        
        print(f"  DP-CoScreen F1: {m_coscreen['macro_f1']:.4f} | DPKendall F1: {m_kendall['macro_f1']:.4f} | FedSDG F1: {m_fedsdg['macro_f1']:.4f}")

    # Summary
    print("\n" + "=" * 90)
    print(f"  Table 3 Benchmark Results Summary: {cfg['name']} (eps={epsilon}, K={K})")
    print("=" * 90)
    print(f"{'Method':<25} | {'Macro-F1 (95% CI)':<20} | {'Accuracy (95% CI)':<20} | {'AUROC (95% CI)':<20}")
    print("-" * 90)
    
    summary_rows = []
    for m in methods:
        stats = aggregate_split_statistics(results[m])
        f1_str = stats["macro_f1"]["formatted"]
        acc_str = stats["accuracy"]["formatted"]
        auc_str = stats["auroc"]["formatted"]
        print(f"{m:<25} | {f1_str:<20} | {acc_str:<20} | {auc_str:<20}")
        summary_rows.append({
            "Method": m,
            "Macro-F1": stats["macro_f1"]["mean"],
            "Macro-F1_CI95": stats["macro_f1"]["ci95"],
            "Accuracy": stats["accuracy"]["mean"],
            "Accuracy_CI95": stats["accuracy"]["ci95"],
            "AUROC": stats["auroc"]["mean"],
            "AUROC_CI95": stats["auroc"]["ci95"],
            "Macro-Recall": stats["macro_recall"]["mean"],
            "Macro-Recall_CI95": stats["macro_recall"]["ci95"],
        })
        
    out_csv = output_dir / f"table3_benchmark_{dataset_key}.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print("-" * 90)
    print(f"[OK] Saved Table 3 benchmark results to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Table 3 Benchmark")
    parser.add_argument("--dataset", choices=["ton", "edge"], default="ton")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    
    run_benchmark(args.dataset, args.splits, args.epsilon, args.data_path, args.output_dir)
