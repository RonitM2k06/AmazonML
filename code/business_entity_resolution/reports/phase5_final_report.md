# Phase 5: Candidate-Pair Feature Engineering & Matching Model Report

## 1. Executive Summary

- **Best Matcher Architecture:** `XGBOOST`
- **Optimal Decision Threshold:** `0.96`
- **Validation Entity-Level Macro F0.5:** **`0.9501`**
- **Validation Macro Precision:** **`0.9738`**
- **Validation Macro Recall:** **`0.9028`**
- **Singleton Accuracy (Zero False Merges):** **`96.20%`**
- **Validation Population:** 25,000 Source 1 entities (3,658,629 candidate pairs)
- **Total Pipeline Execution Time:** 118.6 seconds

## 2. Multi-Model Benchmark Comparison (Task 7)

| Model | Entity Macro F0.5 | Macro Precision | Macro Recall | Singleton Acc | Train Time (s) | Throughput (pairs/s) | RAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `LIGHTGBM` | **0.9500** | 0.9727 | 0.9056 | 96.13% | 3.71s | 872,328 | 130.95 MB |
| `CATBOOST` | **0.9485** | 0.9751 | 0.8941 | 97.06% | 4.87s | 5,851,934 | 91.29 MB |
| `XGBOOST` | **0.9501** | 0.9738 | 0.9028 | 96.20% | 4.72s | 984,932 | 139.31 MB |

## 3. Threshold Sweep Analysis (Task 12)

The Macro F0.5 objective heavily rewards precision ($F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$), penalizing false positive singleton merges with an immediate 0.0 score. Sweeping from 0.50 to 0.99 demonstrates the precision-recall trade-off:

- At threshold **0.50**: Macro F0.5 is penalized by false merges on singletons.
- Peak Macro F0.5 is reached at threshold **`0.96`**, where singleton accuracy reaches **`96.20%`** while preserving high recall on true matches.

## 4. Entity Decision Strategy Comparison (Task 10)

| Decision Strategy | Macro F0.5 | Macro Precision | Macro Recall | Exact Set Acc |
| :--- | :---: | :---: | :---: | :---: |
| Global Threshold | **0.9501** | 0.9738 | 0.9028 | 72.37% |
| Source-Specific (S2 vs S3) | **0.9490** | 0.9759 | 0.8933 | 70.32% |
| Margin Relative (delta=0.08) | **0.9501** | 0.9738 | 0.9028 | 72.37% |
| Platt Calibrated Threshold | **0.9501** | 0.9739 | 0.9027 | 72.34% |

## 5. Controlled Feature Ablation Study (Task 14)

| Feature Subset | Dimensions | Pair Precision | Pair Recall | Pair F0.5 | Entity Macro F0.5 | Macro Prec | Macro Rec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_Name_Only` | 27 | 0.8220 | 0.3578 | 0.6527 | **0.4908** | 0.5787 | 0.3637 |
| `2_Address_Only` | 17 | 0.9867 | 0.8759 | 0.9624 | **0.9130** | 0.9592 | 0.8229 |
| `3_Name_Plus_Address` | 37 | 0.9878 | 0.9572 | 0.9815 | **0.9475** | 0.9738 | 0.8941 |
| `4_Plus_Postal_House` | 43 | 0.9877 | 0.9589 | 0.9818 | **0.9482** | 0.9740 | 0.8957 |
| `5_Plus_Rare_Tokens` | 45 | 0.9878 | 0.9583 | 0.9818 | **0.9482** | 0.9741 | 0.8954 |
| `6_Plus_Transliteration` | 47 | 0.9877 | 0.9605 | 0.9821 | **0.9488** | 0.9743 | 0.8971 |
| `7_Full_Model` | 54 | 0.9877 | 0.9582 | 0.9817 | **0.9480** | 0.9740 | 0.8951 |

## 6. Error Diagnostics & Remaining Failure Modes (Task 13)

- **Total False Positives:** 1,097 candidate pairs accepted above threshold that were non-matches.
- **Total False Negatives:** 8,635 true candidate pairs missed or assigned probability < threshold.
- **Primary False Positive Cause:** High token overlap between sibling franchise branches or common legal suffixes across different suites at same street.
- **Primary False Negative Cause:** Asymmetric address noise (e.g. S1 has landmark only, S2 has door number only) with slight trade name variation.
- Full error samples logged in `reports/phase5_false_positive_analysis.csv` and `reports/phase5_false_negative_analysis.csv`.
