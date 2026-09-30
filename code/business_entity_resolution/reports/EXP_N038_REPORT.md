# EXP_N038 Final Forensic Report: Reciprocal Entity-Set Decoding

## 1. Executive Summary & New Champion Status
- **Experiment ID:** EXP_N038
- **Champion Status:** **NEW CHAMPION OFFICIALLY DECLARED!**
- **Champion Architecture:** Reciprocal Entity-Set Decoding Engine (`TestTimeEntitySetDecoder`)
- **Complete-System Macro $F_{0.5}$ (Metric B):** **0.993311** (Improving over EXP_N037 at 0.993289 by **+0.000022**).
- **Macro Precision:** **0.993207** (Maintained at tournament peak).
- **Macro Recall:** **0.994559** (Surged from 0.994499).
- **False Positives:** **191** (**EXACTLY ZERO NEW FALSE POSITIVES INTRODUCED**).
- **Blocker FN Misses:** Maintained at **42** (**99.9513%** candidate pool retrieval recall ceiling).
- **In-Pool FNs:** Slashed from 169 down to **165** (**4 additional true matches rescued**).
- **Total False Negatives:** Decreased from **211 down to 207** (-4 false negatives eliminated).
- **Rescue Efficiency:** **4 TP : 0 FP** ($\infty$ precision ratio).

---

## 2. Phase 0 Freeze & Verification
- Baseline EXP_N037 exact reproduction verified:
  - Complete Macro $F_{0.5} = 0.993289$, Precision $= 0.993207$, Recall $= 0.994499$.
  - $\text{FP} = 191$, $\text{FN}_{\text{in-pool}} = 169$, $\text{FN}_{\text{blocker}} = 42$, $\text{Total FN} = 211$.
  - Exactly 0 prediction bit changes during baseline reproduction.

---

## 3. Comprehensive Authoritative Ablation Matrix Ladder

Evaluated on the canonical 25,000 entity-disjoint validation set (86,232 ground truth matches) with zero data leakage:

| Variant | Architectural Component | Metric B ($F_{0.5}$) | Precision | Recall | FP | In-Pool FN | Total FN | Changed Bits | $\Delta \text{TP}$ | $\Delta \text{FP}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **N038-A** | N037 Exact Reproduction Baseline | 0.993289 | 0.993207 | 0.994499 | 191 | 169 | 211 | 0 | 0 | 0 |
| **N038-B** | Pairwise Reciprocal Only | 0.993289 | 0.993207 | 0.994499 | 191 | 169 | 211 | 0 | 0 | 0 |
| **N038-C** | Set Support ($\text{conn} \ge 1, \cos \ge 0.65$) | 0.993309 | 0.993202 | 0.994565 | 192 | 165 | 207 | 5 | +4 | +1 |
| **N038-D** | $\ge 2$ Independent Reciprocal Partners | 0.993302 | 0.993207 | 0.994547 | 191 | 165 | 207 | 4 | +4 | 0 |
| **N038-E** | Bipartite Matching | 0.993302 | 0.993207 | 0.994547 | 191 | 165 | 207 | 4 | +4 | 0 |
| **N038-F** | Matching + Neural Guard | 0.993309 | 0.993202 | 0.994565 | 192 | 165 | 207 | 5 | +4 | +1 |
| **N038-G** | Matching + Address Guard | 0.993307 | 0.993207 | 0.994545 | 191 | 166 | 208 | 3 | +3 | 0 |
| **N038-H** | Matching + Contradiction Guard ($\text{bs} == 0$) | 0.993307 | 0.993207 | 0.994545 | 191 | 166 | 208 | 3 | +3 | 0 |
| **N038-I** | Full Entity-Set Decoder Engine | **0.993311** | **0.993207** | **0.994559** | **191** | **165** | **207** | **4** | **+4** | **0** 🏆 |
| **N038-J** | **FINAL EXP_N038 Unified System** | **0.993311** | **0.993207** | **0.994559** | **191** | **165** | **207** | **4** | **+4** | **0** 🏆 |

---

## 4. Phase 2 Forensic Categorization of Remaining FNs

Exhaustive categorization of the remaining 165 in-pool false negatives across 123 distinct query entities:

| Category Code | Description | Count | Percentage |
| :--- | :--- | :--- | :--- |
| **A** | Blocker miss (outside 80-candidate pool ceiling) | 42 | 20.3% of total FN |
| **B** | Isolated single candidate (no graph edges to any pool candidate) | 92 | 55.8% of in-pool FN |
| **C** | Weak direct probability ($p < 0.10$) due to heavy lexical drift / abbreviation | 91 | 55.2% of in-pool FN |
| **D** | Zero name token overlap | 115 | 69.7% of in-pool FN |
| **E** | High direct probability ($p \ge 0.70$) near-threshold boundary cases | 64 | 38.8% of in-pool FN |
| **F** | Deep rank ($r > 20$) | 34 | 20.6% of in-pool FN |
| **G** | Address mismatch / missing street number | 48 | 29.1% of in-pool FN |

---

## 5. Test-Time Safety & Zero-Leakage Invariants
- Comprehensive audit documented in [`reports/EXP_N038_TEST_TIME_AUDIT.md`](file:///c:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/code/business_entity_resolution/reports/EXP_N038_TEST_TIME_AUDIT.md).
- The inference engine ([`src/test_time_entity_set.py`](file:///c:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/code/business_entity_resolution/src/test_time_entity_set.py)) operates strictly on observable features:
  - Neural text embeddings (dense two-tower cosine similarity $\cos$).
  - Normalized token Jaccard similarity.
  - Postal code prefix match ratios.
  - Negative branch classification flags.
  - Zero validation or test ground truth labels are accessed during inference.

---

## 6. Official Tournament Championship Status
- **New Champion:** **EXP_N038**
- **Official Score:** **0.993311**
- **State File:** [`reports/champion.json`](file:///c:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/code/business_entity_resolution/reports/champion.json) updated.
- **Audits Generated:**
  - [`reports/EXP_N038_RESULTS.json`](file:///c:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/code/business_entity_resolution/reports/EXP_N038_RESULTS.json)
  - [`reports/EXP_N038_FP_AUDIT.csv`](file:///c:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/code/business_entity_resolution/reports/EXP_N038_FP_AUDIT.csv) (191 rows)
  - [`reports/EXP_N038_FN_AUDIT.csv`](file:///c:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/code/business_entity_resolution/reports/EXP_N038_FN_AUDIT.csv) (165 rows)
  - [`reports/EXP_N038_SET_AUDIT.csv`](file:///c:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/code/business_entity_resolution/reports/EXP_N038_SET_AUDIT.csv) (4 rows)
