# Phase 5: Pipeline & Architecture Audit

## 1. Overview
Before implementing the candidate-pair feature extraction and machine learning matching models, this audit inspects the existing codebase to ensure strict adherence to DRY (Don't Repeat Yourself), architectural consistency, backward compatibility, and zero data leakage.

---

## 2. Inventory of Existing Reusable Components

### A. Text Normalization Module (`src/normalization.py`)
All feature extraction routines will directly import and reuse the canonical normalization logic:
- `normalize_unicode(text)`: NFC/NFKD normalization stripping Latin diacritics while strictly preserving Indic scripts (Devanagari, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam).
- `normalize_whitespace(text)`: Collapses repeated whitespace.
- `normalize_lowercase(text)`: Lowercases and strips whitespace.
- `strip_punctuation(text)`: Strips non-alphanumeric punctuation while maintaining Unicode words.
- `normalize_alphanumeric(text)`: ASCII alphanumeric cleaner.
- `strip_legal_suffixes(name)`: Canonical multi-country regex for legal business entity suffixes (US, India, France).
- `normalize_business_name(name, strip_legal=True)`: Master business name standardizer.
- `expand_address_abbreviations(address)`: Street, landmark, and building abbreviation expander.
- `normalize_address(address, expand_abbr=True)`: Master address standardizer.
- `extract_postal_codes(address)`: Regex extractor for 5-digit and 6-digit postal/PIN codes.
- `extract_numbers(text)`: Numeric sequence extractor (house numbers, suite numbers).
- `tokenize(text)`: Standardized whitespace tokenizer.
- `token_jaccard(t1, t2)`, `token_overlap(t1, t2)`: High-efficiency token set overlap metrics.
- `character_similarity(s1, s2)`: RapidFuzz normalized Levenshtein similarity ($[0.0, 1.0]$).
- `token_set_similarity(s1, s2)`: RapidFuzz token set ratio ($[0.0, 1.0]$).
- `token_sort_similarity(s1, s2)`: RapidFuzz token sort ratio ($[0.0, 1.0]$).

*Policy:* **No secondary or conflicting normalization functions will be created.** All feature extractors in `src/pair_features.py` must derive their inputs directly from these functions.

---

### B. Blocking & Candidate Generation Module (`src/blocking.py`)
The candidate pool generator will reuse the battle-tested **Union E** configuration:
- `transliterate_indic(text)`: Universal Indic Unicode phonetic transliterator across 8 Indian scripts.
- `clean_business_name(name)`: Strips domain extensions (`.com`, `.in`), legal suffixes, and stop words.
- `extract_blocking_keys(name, address, country, strategies)`:
  - Exact & fused name (`exact_norm`, `fused_name`)
  - Sorted tokens (`sorted_tokens`)
  - Postal + name tokens (`post_name_tok`)
  - House number + name tokens (`house_name_tok`, `addr_name_tok`)
  - Multi-token unordered signatures (`two_tokens`)
  - Address signatures (`house_addr_tok`, `post_house_num`, `post_addr_tok`)
  - Cleaned character n-grams (`pfx4_lbucket`, `post_pfx4`)
  - Distinctive rare name tokens (`rare_tok`)
- `BlockIndex`: Capped inverted index with `max_bucket_size=1000` protection against high-frequency bucket explosion.
- `generate_candidates()`: Efficient candidate generation with `max_candidates=200` per S1 entity.

*Recall Ceiling Established in Phase 4:* **92.45%** on the validation benchmark.

---

### C. Evaluation & Metric Engine (`src/evaluation.py`)
Evaluation for model selection and threshold tuning will strictly use the official challenge objective:
- `compute_entity_metrics(true_ids, pred_ids)`: Computes exact entity-level $F_{0.5}$, precision, recall, and TP/FP/FN.
  - Correctly identified singletons ($|true|=0$ and $|pred|=0$) receive exact score $1.0$.
  - False positive predictions on singletons ($|true|=0$ and $|pred|>0$) receive penalty $0.0$.
  - Missed matches on true entities receive penalty $0.0$.
- `Evaluator`: Macro-level streaming aggregator reporting:
  - `macro_f_05`
  - `macro_precision`
  - `macro_recall`
  - `singleton_accuracy`
  - `singleton_false_positive_rate`
  - `exact_set_accuracy`

*Policy:* Models will **not** be chosen based on pairwise accuracy or standard pairwise F1. All model comparisons and threshold optimizations must optimize entity-level `macro_f_05`.

---

### D. Validation Protocol & Splits (`experiments/validation_s1_ids.txt`)
- Frozen 80/20 entity-disjoint split:
  - **441,363 validation S1 entities** listed in `experiments/validation_s1_ids.txt`.
  - **1,765,458 training S1 entities** (the exact complement in `train_source1.tsv`).
- Zero entity leakage: An S1 entity in train will NEVER appear in validation.
- Target records ($S_2$ and $S_3$) are shared references, but labeled pairs are partitioned strictly by $S_1$ entity ID.

---

## 3. Architecture for Phase 5 Implementation

```
code/business_entity_resolution/
├── src/
│   ├── normalization.py          (REUSED - Canonical text normalization)
│   ├── blocking.py               (REUSED - Candidate generation Union E)
│   ├── evaluation.py             (REUSED - Entity-level macro F0.5 evaluator)
│   ├── pair_features.py          [NEW] - Modular, vectorized feature extractor
│   ├── build_pair_dataset.py     [NEW] - Candidate pair builder with hard negative mining
│   ├── train_matcher.py          [NEW] - Training LightGBM, CatBoost, XGBoost
│   ├── calibrate_matcher.py      [NEW] - Probability calibration (Platt / Isotonic)
│   └── entity_decision.py        [NEW] - Macro F0.5 threshold & decision logic
├── models/                       [NEW] - Trained model artifacts
└── reports/
    ├── phase5_pipeline_audit.md  [NEW] - This document
    ├── phase5_feature_benchmark.csv
    ├── phase5_model_benchmark.csv
    ├── phase5_threshold_sweep.csv
    ├── phase5_false_positive_analysis.csv
    ├── phase5_false_negative_analysis.csv
    ├── phase5_feature_ablation.csv
    └── phase5_final_report.md
```

All new modules will directly import and reuse `src/normalization.py`, `src/blocking.py`, and `src/evaluation.py`.
