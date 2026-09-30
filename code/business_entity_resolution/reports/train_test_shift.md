# Train vs Test Distribution Shift Report

## 1. Feature Distribution Comparison (S1)

| Metric | Train S1 | Test S1 | Shift / Difference |
|---|---|---|---|
| **Country: US** | 59.89% | 38.42% | **-21.7%** |
| **Country: India** | 40.11% | 46.6% | **+6.7%** |
| **Country: France** | **0.0%** | **14.98%** | **+15.0% (NEW)** |
| **Mean Name Length** | 24.04 chars | 23.83 chars | Minor (-0.21) |
| **Mean Address Length** | 52.08 chars | 57.1 chars | Minor (5.02) |
| **Accented Characters** | 0.0% | 2.32% | **+2.32%** |
| **Devanagari Script** | 0.0% | 0.0% | 0.0% |

## 2. Strategic Modeling Implications

1. **Language-Agnostic Normalization:** The pipeline must strip Latin combining accents (so French `Café` matches `Cafe`) without stripping Indian vowel diacritics.
2. **Open-Set Categoricals:** Do not hard-code `{US, India}` anywhere in feature pipelines.
3. **Cross-Lingual Matching:** India records frequently have English Latin transliterations in S1 and Devanagari Hindi in S2/S3 (or vice-versa).
