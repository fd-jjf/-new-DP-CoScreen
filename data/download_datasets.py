"""
download_datasets.py:
Dataset verification and download assistance script.
Checks SHA-256 integrity of ToN-IoT and Edge-IIoTset datasets against the reported manifests.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

CHECKSUMS = {
    "ToN-IoT": {
        "file": Path("data/ToN-IoT/train_test_network.csv"),
        "sha256": "26DDC513552DE36DE6428B2E578EFAED2B57504C716DFBA847CC0109A64E1974",
        "url": "https://research.unsw.edu.au/projects/toniot-datasets",
    },
    "Edge-IIoTset": {
        "file": Path("data/Edge-IIoTset/ML-EdgeIIoT-dataset.csv"),
        "sha256": "53101FAD091AF20BEE815860962A5016B802EE45D456A141042F6183094C3C1A",
        "url": "https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications",
    },
}


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def verify_datasets():
    print("=" * 70)
    print("  Dataset Cryptographic SHA-256 Integrity Verification")
    print("=" * 70)
    
    root_dir = Path(__file__).parent.parent
    all_ok = True
    
    for name, info in CHECKSUMS.items():
        rel_path = info["file"]
        candidates = [root_dir / rel_path, root_dir.parent / rel_path]
        target = None
        for c in candidates:
            if c.exists():
                target = c
                break
                
        if target is None:
            print(f"[-] {name}: File not found at {rel_path}")
            print(f"    Download source: {info['url']}")
            all_ok = False
            continue
            
        print(f"[*] Verifying {name} ({target.stat().st_size / (1024*1024):.2f} MB)...")
        actual_hash = compute_sha256(target)
        expected_hash = info["sha256"]
        
        if actual_hash == expected_hash:
            print(f"    [PASS] Hash matches: {actual_hash[:16]}...")
        else:
            print(f"    [FAIL] Hash mismatch!")
            print(f"      Actual:   {actual_hash}")
            print(f"      Expected: {expected_hash}")
            all_ok = False
            
    print("=" * 70)
    if all_ok:
        print("[SUCCESS] All available datasets verified bit-exact with the manuscript.")
    else:
        print("[NOTICE] Some datasets missing or modified. Follow instructions in data/README.md.")


if __name__ == "__main__":
    verify_datasets()
