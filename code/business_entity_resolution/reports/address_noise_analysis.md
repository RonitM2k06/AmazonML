# Address Noise Characterization Report

**Sample Size:** 173,033 ground-truth pairs.

## 1. Address Completeness & Equality

- **Missing Address on S2/S3:** 7,565 (4.37%)
- **Exact Address Match:** 3,763 (2.17%)
- **Normalized Address Match:** 20,227 (11.69%)

## 2. Address Similarity Distributions (Non-Missing)

| Metric | Mean ± Std | Median | P25 | P75 | P90 | P95 |
|---|---|---|---|---|---|---|
| `character_similarity` | 0.4846 ± 0.2374 | 0.4559 | 0.3051 | 0.6765 | 0.8219 | 0.8837 |
| `token_overlap` | 0.8434 ± 0.1534 | 0.8571 | 0.8 | 1.0 | 1.0 | 1.0 |
| `numeric_token_overlap` | 0.8262 ± 0.3701 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `postal_code_overlap` | 0.9647 ± 0.1845 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `length_ratio` | 0.8553 ± 0.1565 | 0.8989 | 0.8087 | 0.9667 | 1.0 | 1.0 |

## 3. Critical Address Matching Regimes

### `regime_a_address_missing`: 7,565 (4.37%)
Address completely missing in S2/S3; entity match relies 100% on business name and country

### `regime_b_address_different_name_similar`: 23,332 (13.48%)
Address similarity < 0.40 but Name similarity >= 0.80 (e.g. partial address, city only, relocated branch)

### `regime_c_address_similar_name_different`: 8,302 (4.8%)
Address similarity >= 0.80 but Name similarity < 0.60 (e.g. severe abbreviation, acronyms, trade names)

