# Dataset Guidelines and Integrity Verification

This directory houses the industrial IoT telemetry datasets used for the empirical evaluation in the manuscript.

Due to GitHub repository size limits (>100 MB per file, 1 GB repository recommendation) and original data licensing policies, raw benchmark datasets are not checked into version control. Instead, they can be obtained from their official public repositories and verified using the cryptographic SHA-256 checksums below.

---

## Benchmark Datasets

### 1. ToN-IoT Network Intrusion Dataset
- **Paper**: Al-Khafaji et al., *"A new realistic cyber dataset of IoT and IIoT network traffic"*, Computer Networks, 2020.
- **Official Source**: [UNSW Canberra Cyber ToN-IoT Repository](https://research.unsw.edu.au/projects/toniot-datasets) / [Kaggle ToN-IoT](https://www.kaggle.com/datasets)
- **Target File**: `train_test_network.csv`
- **Expected Path**: `data/ToN-IoT/train_test_network.csv`
- **Features**: $d=94$ ($17$ continuous flow metrics + $77$ one-hot/bucketed protocol indicators)
- **Expected SHA-256**:
  ```
  26DDC513552DE36DE6428B2E578EFAED2B57504C716DFBA847CC0109A64E1974
  ```

### 2. Edge-IIoTset Cybersecurity Benchmark
- **Paper**: Ferrag et al., *"Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications"*, IEEE Access, 2022.
- **Official Source**: [IEEE Dataport Edge-IIoTset](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)
- **Target File**: `ML-EdgeIIoT-dataset.csv`
- **Expected Path**: `data/Edge-IIoTset/ML-EdgeIIoT-dataset.csv`
- **Features**: $d=72$ ($28$ Modbus/MQTT/TCP sensor metrics + $44$ categorical indicators)
- **Expected SHA-256**:
  ```
  53101FAD091AF20BEE815860962A5016B802EE45D456A141042F6183094C3C1A
  ```

---

## Verification Script

To verify that your downloaded datasets are bit-exact with the ones used in our reported results:

```bash
python data/download_datasets.py --verify-only
```

---

## Out-of-the-Box Quickstart (No Large Downloads Required)

If you only wish to test the pipeline, inspect the algorithms, or run unit tests, you can execute:

```bash
python demo_quickstart.py
```

or generate a lightweight local synthetic dataset:

```bash
python data/generate_toy_data.py
```
