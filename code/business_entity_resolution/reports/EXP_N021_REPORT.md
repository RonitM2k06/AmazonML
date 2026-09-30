# EXP_N021 — Forensic Analysis & Plateau Diagnosis Report

## 1. Executive Summary & Authoritative Results

In **EXP_N021**, we investigated whether a corruption-aware cross-script rescue classifier could separate the remaining 2,869 in-pool false negatives from candidate negatives under strict anti-collision constraints.

### Authoritative Results Comparison
- **N020 Champion Baseline:**
  - Complete-System Macro $F_{0.5}$: **0.984281**
  - Precision: **0.993552** | Recall: **0.962537**
  - FP: **103** | In-pool FN: **2,869** | Blocker FN: **404**
- **EXP_N021 Evaluation:**
  - Complete-System Macro $F_{0.5}$: **0.984264**
  - Precision: **0.993524** | Recall: **0.962567**
  - FP: **105** (+2 FPs) | In-pool FN: **2,866** (3 TPs recovered) | Blocker FN: **404**

Because $F_{0.5}$ penalizes precision 4× more heavily than recall ($\beta = 0.5$), rescuing 3 true matches while acquiring 2 false positives resulted in a net penalty of $-0.000017$. Therefore, the champion remains **N020 (`0.984281`)**.

---

## 2. Deep Forensic Diagnosis: Why the System Plateaued at ~0.9843

Per user instructions:
> *"If <0.990: DO NOT perform a cosmetic threshold sweep. Instead provide a forensic explanation of: (1) how many remaining FNs are actually separable, (2) which information is missing, (3) which architecture component failed, (4) whether retrieval or decision remains the bottleneck, (5) the exact evidence needed for the next architectural experiment."*

### 1. How Many Remaining FNs are Actually Separable?
From the Phase 0 Oracle Audit across all 25,000 entities:
- **Total In-Pool False Negatives:** **2,869**
- **Separable within candidate pool:** **1,846 (64.3%)**
  *(For 1,846 FNs, their model score is strictly higher than any negative candidate within the same S1 pool)*
- **Unseparable within candidate pool:** **1,023 (35.7%)**
  *(For 1,023 FNs, at least one false positive candidate in the same pool has a strictly higher score than the true match)*

### 2. The Global Threshold Bottleneck
Why can't the 1,846 separable FNs simply be admitted?
Because while they outscore their *local* pool negatives, their *global* probability values are low:
- **55.8% (1,600 / 2,869)** of in-pool FNs have global model probability $p < 0.70$.
- **35.8% (1,027 / 2,869)** have global model probability $p < 0.30$.
- Across the entire 25,000 entities, there are **5,032 negative candidates** sitting in the range $p \in [0.70, 0.9320)$, and **17,765 negative candidates** with $p \ge 0.30$.
- Therefore, any attempt to globally lower the decision threshold or loosen rescue rules instantly admits dozens of false positives from non-matching entities across the dataset.

### 3. Which Information is Fundamentally Missing?
1. **Pincode / Postal Geometry:**
   - 99.6% of in-pool FNs have `addr_postal_exact == 0` (only 13 out of 2,869 share an exact postal code).
   - In Indian business listings, postal codes are either missing, partially formatted (e.g. 3 digits vs 6 digits), or absent.
2. **True Cross-Lingual Transliteration Dictionaries:**
   - 64.0% (1,835 / 2,869) of in-pool FNs have zero token overlap (`name_token_jaccard == 0.000`).
   - Neural cross-encoders give these pairs moderate cosine similarities ($\sim 0.65–0.75$), but cannot distinguish them from co-located commercial neighbors without explicit character-level phonetic alignment.
3. **Branch Disambiguation:**
   - 71.8% of false positives are identical-name branches. Without sequential street numbers or lat/long coordinates, branch collision remains an intractable boundary for pairwise classifiers.

### 4. Is Retrieval or Decision the Bottleneck?
- **Decision is the dominant bottleneck for 2,869 pairs (87.6% of all errors).**
- **Retrieval is the hard ceiling for 404 pairs (12.4% of all errors).**
- Even with a theoretical 100% recall on the 404 blocker misses, the system cannot reach $\ge 0.990$ without resolving at least 1,500 of the 2,869 in-pool false negatives.

---

## 3. Champion Preservation

In accordance with user rules, **N020 is preserved as the active champion**:
- **Baseline Champion:** `selective_collective_graph_decoder_EXP_N020`
- **Complete-System Macro $F_{0.5}$ (Metric B):** `0.984281`
- **Precision:** `0.993552`
- **Recall:** `0.962537`
- **FP:** `103`
- **In-Pool FN:** `2,869`
- **Blocker Misses:** `404`

[`reports/champion.json`](file:///c:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/code/business_entity_resolution/reports/champion.json) maintains the `0.984281` record.
