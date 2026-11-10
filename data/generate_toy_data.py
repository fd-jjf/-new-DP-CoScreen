"""
generate_toy_data.py:
Generates a small synthetic IIoT telemetry CSV matching the schema of ToN-IoT.
Allows running the experiment scripts out-of-the-box without downloading large CSV files.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_toy_ton_dataset(n_samples: int = 5000, output_path: Path | None = None):
    output_path = output_path or Path(__file__).parent / "toy_ton_iot.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(42)
    
    # Numeric features
    data = {
        "src_port": rng.integers(1024, 65535, size=n_samples),
        "dst_port": rng.choice([80, 443, 22, 502, 1883, 8080], size=n_samples),
        "duration": rng.exponential(scale=2.5, size=n_samples),
        "src_bytes": rng.integers(64, 150000, size=n_samples),
        "dst_bytes": rng.integers(64, 500000, size=n_samples),
        "missed_bytes": rng.choice([0, 100, 500], p=[0.9, 0.08, 0.02], size=n_samples),
        "src_pkts": rng.integers(1, 200, size=n_samples),
        "src_ip_bytes": rng.integers(64, 20000, size=n_samples),
        "dst_pkts": rng.integers(1, 300, size=n_samples),
        "dst_ip_bytes": rng.integers(64, 40000, size=n_samples),
        "dns_qclass": rng.choice([0, 1], p=[0.8, 0.2], size=n_samples),
        "dns_qtype": rng.choice([0, 1, 28], p=[0.8, 0.15, 0.05], size=n_samples),
        "dns_rcode": rng.choice([0, 3], p=[0.95, 0.05], size=n_samples),
        "http_trans_depth": rng.choice([0, 1, 2], p=[0.7, 0.25, 0.05], size=n_samples),
        "http_request_body_len": rng.integers(0, 1000, size=n_samples),
        "http_response_body_len": rng.integers(0, 5000, size=n_samples),
        "http_status_code": rng.choice([0, 200, 404, 500], p=[0.6, 0.35, 0.04, 0.01], size=n_samples),
        # Categorical features
        "proto": rng.choice(["tcp", "udp", "icmp"], p=[0.85, 0.12, 0.03], size=n_samples),
        "service": rng.choice(["http", "dns", "modbus", "mqtt", "ssl", "none"], size=n_samples),
        "conn_state": rng.choice(["SF", "S0", "REJ", "RSTO", "RSTR"], size=n_samples),
        "dns_AA": rng.choice(["-", "T", "F"], size=n_samples),
        "dns_RD": rng.choice(["-", "T", "F"], size=n_samples),
        "dns_RA": rng.choice(["-", "T", "F"], size=n_samples),
        "dns_rejected": rng.choice(["-", "T", "F"], size=n_samples),
        "ssl_version": rng.choice(["-", "TLSv12", "TLSv13"], size=n_samples),
        "ssl_cipher": rng.choice(["-", "ECDHE-RSA-AES128-GCM-SHA256", "AES256-SHA"], size=n_samples),
        "ssl_resumed": rng.choice(["-", "T", "F"], size=n_samples),
        "ssl_established": rng.choice(["-", "T", "F"], size=n_samples),
        "http_method": rng.choice(["-", "GET", "POST", "HEAD"], size=n_samples),
        "http_version": rng.choice(["-", "1.1", "2.0"], size=n_samples),
        "http_orig_mime_types": rng.choice(["-", "text/html", "application/json"], size=n_samples),
        "http_resp_mime_types": rng.choice(["-", "text/html", "image/png"], size=n_samples),
        # Label & Type
        "label": rng.choice([0, 1], p=[0.6, 0.4], size=n_samples),
        "type": rng.choice(["normal", "dos", "ddos", "xss", "scanning", "injection"], size=n_samples),
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"[OK] Generated toy dataset at: {output_path} ({n_samples} samples)")


if __name__ == "__main__":
    generate_toy_ton_dataset()
