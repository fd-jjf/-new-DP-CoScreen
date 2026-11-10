"""
preprocessing.py:
Data-independent physical bounding based on RFC 793 and Modbus V1.1b specifications,
fixed feature schemas, and Dirichlet non-IID federated data partitioning.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# RFC 793 (TCP) and Modbus Application Protocol Specification V1.1b Bounding Dictionary
PHYSICAL_DOMAIN_BOUNDS: Dict[str, Tuple[float, float]] = {
    # Network Ports & Transport Identifiers (RFC 793 / IANA)
    "src_port": (0.0, 65535.0),
    "dst_port": (0.0, 65535.0),
    "tcp.srcport": (0.0, 65535.0),
    "tcp.dstport": (0.0, 65535.0),
    "udp.port": (0.0, 65535.0),
    "dns_qclass": (0.0, 65535.0),
    "dns_qtype": (0.0, 65535.0),
    "dns.qry.type": (0.0, 65535.0),
    "icmp.seq_le": (0.0, 65535.0),
    
    # Protocol Flags & Binary Control Signals (RFC 793 standard unasserted = 0)
    "tcp.connection.syn": (0.0, 1.0),
    "tcp.connection.rst": (0.0, 1.0),
    "tcp.connection.fin": (0.0, 1.0),
    "tcp.connection.ack": (0.0, 1.0),
    "http.response": (0.0, 1.0),
    "mqtt.conflag.cleansess": (0.0, 1.0),
    "dns.qry.qu": (0.0, 1.0),
    "dns.retransmission": (0.0, 1.0),
    "dns.retransmit_request": (0.0, 1.0),
    "dns.retransmit_request_in": (0.0, 1.0),
    
    # Byte-wide Protocol Fields (0 - 255)
    "arp.opcode": (0.0, 255.0),
    "arp.hw.size": (0.0, 255.0),
    "mqtt.conflags": (0.0, 255.0),
    "mqtt.hdrflags": (0.0, 255.0),
    "mqtt.msgtype": (0.0, 255.0),
    "mqtt.ver": (0.0, 255.0),
    "dns_rcode": (0.0, 255.0),
    "http_status_code": (0.0, 999.0),
    
    # Durations & Flow Deltas (seconds, max bounded by 24h = 86400s)
    "duration": (0.0, 86400.0),
    "udp.time_delta": (0.0, 86400.0),
}


def normalize_attribute(col_name: str, values: np.ndarray) -> np.ndarray:
    """
    Normalizes numeric attribute using data-independent protocol bounds and clamps to [0, 1].
    Eliminates auxiliary distribution leakage from private datasets.
    """
    arr = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    
    if col_name in PHYSICAL_DOMAIN_BOUNDS:
        low, high = PHYSICAL_DOMAIN_BOUNDS[col_name]
        scaled = (arr - low) / (high - low)
    else:
        # Generic flow counters (packet counts, byte lengths): signed log1p scaled to max 32-bit int
        max_bound = np.log1p(float(2**32 - 1))
        scaled = np.sign(arr) * np.log1p(np.abs(arr)) / max_bound

    # Strict elementwise clamping onto unit interval [0, 1]
    return np.clip(scaled, 0.0, 1.0)


class FixedFeatureMap:
    """
    Fixed-schema feature transformer using deterministic hashing for categoricals
    and RFC/Modbus physical bounding for numerics.
    Zero statistics fitted to protected client records.
    """
    def __init__(
        self,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str],
        hash_buckets: int = 4
    ) -> None:
        self.numeric_cols = tuple(numeric_cols)
        self.categorical_cols = tuple(categorical_cols)
        self.hash_buckets = max(2, hash_buckets)
        
        feature_names: List[str] = []
        for col in self.numeric_cols:
            feature_names.append(f"num::{col}")
            feature_names.append(f"missing::{col}")
        for col in self.categorical_cols:
            for b in range(self.hash_buckets):
                feature_names.append(f"hash::{col}::{b}")
                
        self.feature_names = tuple(feature_names)
        self.dimension = len(self.feature_names)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms raw dataframe to float64 matrix in [0, 1]^d.
        """
        n = len(df)
        cols_output = []
        
        # 1. Numeric features
        for col in self.numeric_cols:
            if col in df.columns:
                series = df[col]
                missing = series.isna().to_numpy().astype(float)
                vals = pd.to_numeric(series, errors='coerce').fillna(0.0).to_numpy()
                norm_vals = normalize_attribute(col, vals)
            else:
                missing = np.ones(n, dtype=float)
                norm_vals = np.zeros(n, dtype=float)
                
            cols_output.append(norm_vals)
            cols_output.append(missing)
            
        # 2. Categorical features with fixed BLAKE2b hashing
        for col in self.categorical_cols:
            buckets_onehot = np.zeros((n, self.hash_buckets), dtype=float)
            if col in df.columns:
                series = df[col].astype(str).str.lower().str.strip()
                for i, val in enumerate(series):
                    if pd.notna(val) and val != "" and val != "nan":
                        h = int(hashlib.blake2b(val.encode("utf-8"), digest_size=8).hexdigest(), 16)
                        b_idx = h % self.hash_buckets
                        buckets_onehot[i, b_idx] = 1.0
            for b in range(self.hash_buckets):
                cols_output.append(buckets_onehot[:, b])
                
        X = np.column_stack(cols_output).astype(np.float64)
        return np.clip(X, 0.0, 1.0)


def clip_augmented_vectors(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Forms augmented vectors z_it = [x_it^T, y_it]^T in R^{d+1} and clips
    to physical radius C_z = sqrt(d + 1).
    Since x in [0, 1]^d and y in {0, 1}, ||z||_2 <= sqrt(d + 1) holds naturally,
    while clipping eliminates out-of-spec physical corruption spikes.
    """
    n, d = X.shape
    y_col = y.reshape(-1, 1).astype(np.float64)
    Z = np.hstack([X, y_col])  # (n, d + 1)
    
    C_z = np.sqrt(float(d + 1))
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    scale_factors = np.minimum(1.0, C_z / np.maximum(norms, 1e-12))
    return Z * scale_factors


def partition_dirichlet_noniid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int = 10,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples_per_client: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partitions data across distributed edge clients under Dirichlet Non-IID distribution Dir(alpha).
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (N, d).
    y : np.ndarray
        Labels (N,).
    n_clients : int, default=10
        Number of edge clients M.
    alpha : float, default=0.5
        Dirichlet concentration parameter (smaller alpha = higher non-IID heterogeneity).
    seed : int, default=42
        Random seed.
    min_samples_per_client : int, default=5
        Minimum samples guaranteed per client to ensure well-defined local empirical statistics.
        
    Returns
    -------
    List of (X_i, y_i) client tuples.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y)
    classes = np.unique(y)
    
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]
    
    for c in classes:
        indices = np.where(y == c)[0]
        rng.shuffle(indices)
        
        # Draw proportions from Dirichlet distribution
        proportions = rng.dirichlet(np.full(n_clients, alpha))
        counts = rng.multinomial(len(indices), proportions)
        
        start = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(indices[start : start + count].tolist())
            start += count

    # Guarantee each client has at least min_samples_per_client by borrowing from the largest client if needed
    for i in range(n_clients):
        while len(client_indices[i]) < min_samples_per_client:
            # Find client with maximum samples
            donor_client = int(np.argmax([len(lst) for lst in client_indices]))
            if len(client_indices[donor_client]) <= min_samples_per_client:
                break
            borrowed_idx = client_indices[donor_client].pop()
            client_indices[i].append(borrowed_idx)

    client_datasets = []
    for i in range(n_clients):
        idx = np.array(client_indices[i], dtype=int)
        rng.shuffle(idx)
        client_datasets.append((X[idx], y[idx]))
        
    return client_datasets
