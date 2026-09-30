# Name Noise Characterization Report

**Sample Size:** 173,033 ground-truth pairs.

## 1. Exact & Normalized Match Hierarchy

| Normalization Level | Cumulative True Matches | % of All True Matches |
|---|---|---|
| Raw Exact String Match | 7,962 | 4.6% |
| Lowercase Match | 18,630 | 10.77% |
| Punctuation-Normalized Match | 37,811 | 21.85% |
| Alphanumeric-Normalized Match | 38,297 | 22.13% |
| Legal-Suffix-Stripped Match | 67,543 | 39.03% |

## 2. Quantitative Similarity Distributions

| Metric | Mean ± Std | Median | P25 | P75 | P90 | P95 |
|---|---|---|---|---|---|---|
| `character_similarity` | 0.6224 ± 0.2982 | 0.7 | 0.3929 | 0.878 | 0.9583 | 0.9744 |
| `token_jaccard` | 0.6421 ± 0.3349 | 0.6667 | 0.5 | 1.0 | 1.0 | 1.0 |
| `token_overlap` | 0.775 ± 0.347 | 1.0 | 0.6667 | 1.0 | 1.0 | 1.0 |
| `token_set_similarity` | 0.7556 ± 0.3161 | 0.9189 | 0.5652 | 1.0 | 1.0 | 1.0 |
| `token_sort_similarity` | 0.6937 ± 0.2956 | 0.8108 | 0.5 | 0.9275 | 1.0 | 1.0 |
| `length_ratio` | 0.86 ± 0.1627 | 0.9143 | 0.7907 | 1.0 | 1.0 | 1.0 |

## 3. Discovered Real Noise Categories

| Noise Category | Frequency | Percentage |
|---|---|---|
| `token_insertion_or_deletion` | 46,166 | 26.68% |
| `minor_typo_or_spelling` | 31,040 | 17.94% |
| `legal_suffix_difference` | 29,246 | 16.9% |
| `punctuation_difference_only` | 19,181 | 11.09% |
| `word_order_transposition` | 13,194 | 7.63% |
| `case_difference_only` | 10,668 | 6.17% |
| `exact_match` | 7,962 | 4.6% |
| `script_transliteration_mismatch` | 6,978 | 4.03% |
| `accent_or_unicode_only` | 6,274 | 3.63% |
| `alphanumeric_normalized_only` | 486 | 0.28% |
