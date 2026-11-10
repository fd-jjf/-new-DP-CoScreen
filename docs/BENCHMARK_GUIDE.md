# DP-CoScreen: Benchmark Reproduction Guide

This guide details the step-by-step procedures to replicate the complete empirical findings presented in our manuscript submitted to the *Journal of Information Security and Applications* (Elsevier).

---

## 1. Quick Verification (1-Minute Dry Run)

To verify the installation and core mechanics on a lightweight synthetic industrial IoT federation:

```bash
python demo_quickstart.py
```

Expected output:
- Pipeline completion in $<40$ ms
- Gateway release latency $\approx 3.5$ ms
- Coordinator screening latency $\approx 2.2$ ms
- Automatic suppression of multicollinear duplicates
- Multi-metric downstream intrusion detection scores

---

## 2. Unit and Integration Test Suite

Run the full automated test suite verifying AGM calibrations, PSD projections, Schur complements, and non-IID partitioning:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected status: All 9 test cases pass (`OK`).

---

## 3. Reproducing Main Manuscript Tables & Figures

### Table 3: Main Benchmark Comparison (6 Methods)
Runs across 5 macro-independent non-IID client splits:
```bash
# On ToN-IoT (d=94)
python experiments/reproduce_table3_benchmark.py --dataset ton --splits 5

# On Edge-IIoTset (d=72)
python experiments/reproduce_table3_benchmark.py --dataset edge --splits 5
```
Results will be exported to `results/table3_benchmark_ton.csv` and `results/table3_benchmark_edge.csv`.

---

### Figures 2 & 3: Parameter Sensitivity ($K$ and $\epsilon$)
Evaluates feature budget $K \in \{2, 4, 6, 8, 10\}$ and privacy budget $\epsilon \in \{0.1, 0.5, 1.0, 2.0, 3.0, 5.0\}$:
```bash
python experiments/reproduce_sensitivity_budget.py --dataset ton --splits 3
python experiments/reproduce_sensitivity_budget.py --dataset edge --splits 3
```

---

### Table 5: Systematic Component Ablations
Isolates the individual contributions of PSD cone projection, Schur-complement conditioning, and collaborative multi-client aggregation:
```bash
python experiments/reproduce_ablation_study.py --dataset ton --splits 5
python experiments/reproduce_ablation_study.py --dataset edge --splits 5
```

---

### Figure 6: Latency and Embedded Hardware Profiling
Benchmarks computational runtime, uplink communication payload, and ARM Cortex-A72 hardware costs:
```bash
python experiments/reproduce_latency_profile.py
```

---

### Figure Generation
To generate publication-ready vector PDFs and high-resolution PNGs:
```bash
python experiments/plot_figures.py
```
Generated plots are placed in `results/figures/`:
- `fig2_k_sensitivity.pdf` / `png`
- `fig3_eps_sensitivity.pdf` / `png`
- `fig6_latency_comparison.pdf` / `png`
- `fig8_ablation_study.pdf` / `png`
