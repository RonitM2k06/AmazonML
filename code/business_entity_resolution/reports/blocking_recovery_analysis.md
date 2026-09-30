# Blocking Incremental Recovery Analysis

This report quantifies which blocking strategies recover *new, previously uncaptured* ground-truth matches, and why.

## 1. Incremental Recovery Table

| Stage / Strategy | New True Pairs Recovered | % of Ground Truth | Why It Recovers Missed Matches |
|---|---|---|---|
| **Strategy A (All Name Norm)** | **41,319** | **47.92%** | Clean name variations, legal suffixes, sorted tokens |
| **Strategy B (Postal + Name)** | **1,736** | **2.01%** | Severe name truncation/abbreviation anchored by postal code |
| **Strategy C (Address + Name)** | **26,733** | **31.0%** | Address number and street tokens with partial name |
| **Strategy D (Char N-Gram)** | **2,015** | **2.34%** | Minor spelling typos and prefix matches |
| **Strategy E (Transliteration/Phonetic)** | **17** | **0.02%** | Devanagari cross-script Hindi in India partition |
| **Strategy F (Address-First)** | **2,000** | **2.32%** | Radical name changes / acronyms at identical addresses |

## 2. Recovery Case Studies

- **Why Postal+Name Matters:** Addresses frequently contain minor name truncations (e.g. `American Diagnostic Center` -> `American Diagnostic`). The exact name block fails, but the PIN/ZIP code + token `american` cleanly links them.
- **Why Address+Name Matters:** In India, business records often have identical house/plot numbers and city names, with minor spelling differences in the name.
- **Why Devanagari Transliteration Matters:** Over 10% of India pairs link Hindi script to English Latin transliteration. Without local transliteration, these matches are 100% missed.
