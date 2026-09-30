# Business Entity Resolution — Amazon ML Challenge 2026

## Overview
This repository contains the engineering and machine learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution**.
The goal is to resolve deduplicated reference business entities (**Source 1**) to noisy, unlinked multi-source fragments (**Source 2** and **Source 3**) across multiple jurisdictions (US, India, France) evaluated under macro $F_{0.5}$.

---

## Directory Structure

```text
code/business_entity_resolution/
├── src/                    # Production source modules (data loading, blocking, feature extraction, scoring)
├── scripts/                # Standalone reproduction & profiling scripts
│   ├── profile_dataset.py  # High-throughput streaming dataset profiler
│   └── inspect_ground_truth.py # Ground truth distribution & cross-source integrity auditor
├── tests/                  # Unit and integration test suites
├── configs/                # Configuration files (paths, blocking hyperparameters, thresholds)
├── reports/                # Machine-readable (JSON) and human-readable (MD) audit reports
│   ├── dataset_profile.json
│   ├── dataset_profile.md
│   └── ground_truth_report.json
├── experiments/            # Experiment logs and ablation records
├── requirements.txt        # Pinned runtime dependencies
└── README.md               # Pipeline documentation & reproduction guide
```

---

## Environment & Requirements

- **Python Version:** 3.12+ (tested on Python 3.12.7 64-bit AMD64)
- **OS:** Windows / Linux / macOS
- **Hardware Profile:** 20 Cores / 28 Threads, NVIDIA GeForce RTX 4050 (6 GB VRAM, CUDA 13.2 / PyTorch cu121)
- **Key Dependencies:**
  - `pandas>=2.2.0`
  - `numpy>=1.26.0`
  - `scikit-learn>=1.4.0`
  - `rapidfuzz>=3.9.0`
  - `xgboost>=2.1.0`
  - `pyarrow>=15.0.0`
  - `scipy>=1.11.0`
  - `tqdm>=4.66.0`

Install dependencies:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## Data Profiling & Audit Reproduction

To reproduce all dataset profiles and integrity audits from scratch:

```bash
# Profile all source files and ground truth distributions
python code/business_entity_resolution/scripts/profile_dataset.py

# Perform cross-table integrity audits and country-level match breakdown
python code/business_entity_resolution/scripts/inspect_ground_truth.py
```

Generated outputs will be saved to:
- `code/business_entity_resolution/reports/dataset_profile.json`
- `code/business_entity_resolution/reports/dataset_profile.md`
- `code/business_entity_resolution/reports/ground_truth_report.json`
