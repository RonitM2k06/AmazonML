# EXP_N008 Comprehensive Error Forensics & Blocker Recovery Report

## 1. High-Level Performance Metrics
- **Metric A (Decision Layer on Captured Candidates)**: Macro F0.5 = 0.986154
- **Metric B (Complete End-to-End System)**: Macro F0.5 = 0.972440
- **Target**: Complete System Macro F0.5 >= 0.9950 (Gap: +0.02256)
- **Total S1 Evaluated**: 25,000
- **Total True Ground Truth Matches**: 86,232
- **Total False Positives**: 699
- **Total False Negatives**: 4,854
  - Model missed from candidates: 1,788 (36.8% of FN)
  - Blocker never retrieved: 3,066 (63.2% of FN)

## 2. False Positive (FP) Breakdown (Total: 699)
- **I_other_semantic_error**: 395 (56.5%)
- **F_branch_or_chain_collision**: 171 (24.5%)
- **B_same_addr_diff_biz**: 57 (8.2%)
- **A_same_name_diff_biz**: 54 (7.7%)
- **H_marginal_threshold_error**: 18 (2.6%)
- **C_short_name_collision**: 4 (0.6%)

## 3. False Negative — Model Miss Breakdown (Total: 1,788)
- **G_missing_address_penalty**: 478 (26.7%)
- **B_low_model_score**: 477 (26.7%)
- **E_name_variation**: 330 (18.5%)
- **C_margin_pruned**: 309 (17.3%)
- **D_bridge_subthreshold**: 170 (9.5%)
- **F_address_divergence**: 24 (1.3%)

## 4. False Negative — Blocker Miss Breakdown (Total: 3,066)
- **B_non_latin_script**: 1,622 (52.9%)
- **C_zero_name_overlap_address_match**: 768 (25.0%)
- **F_complex_variant**: 671 (21.9%)
- **E_short_name_unmatched**: 4 (0.1%)
- **D_no_surface_overlap_severe_noise**: 1 (0.0%)

## 5. Potential Recovery Pathways for 3,066 Blocker Misses
- **address_token_pair**: 2,750 (89.7%)
- **char_ngram4_shared**: 1,156 (37.7%)
- **phonetic_skeleton_match**: 710 (23.2%)
- **shared_token_len4plus**: 594 (19.4%)
- **shared_token_short**: 158 (5.2%)
- **single_distinctive_address_token**: 60 (2.0%)
- **UNRECOVERABLE_LEXICAL**: 1 (0.0%)

