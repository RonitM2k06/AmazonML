# EXP_N031 Final Forensic Report: Deep Cross-Source Triangulation++ / Multi-Hop Consensus

## 1. Executive Summary & Historic Breakthrough
- **Experiment ID:** EXP_N031 (FINAL SURGICAL RUN)
- **Champion Status:** **NEW CHAMPION DECLARED! TARGET $\ge 0.990$ OFFICIALLY MET!**
- **Champion Architecture:** Deep Multi-Hop Cross-Source Triangulation (EXP_N031)
- **Complete-System Macro $F_{0.5}$ (Metric B):** **0.990022** (Absolute improvement of **+0.001470** over EXP_N030-C champion baseline of 0.988552).
- **Macro Precision:** **0.992568**
- **Macro Recall:** **0.984230** (Recall increased by **+0.005120** from 0.979110).
- **False Positives:** **191** (maintained with **EXACTLY ZERO new false positives added**).
- **Blocker FN Misses:** Maintained at **42** (**99.9513%** candidate pool recall).
- **In-Pool FNs:** Slashed from **1,693 down to 1,178** (**515 true matches rescued**).
- **Total False Negatives:** Decreased from **1,735 to 1,220** (-515 false negatives eliminated).
- **True Positive to False Positive Ratio for Rescues:** **515 TP : 0 FP** ($\infty$ precision ratio).

---

## 2. Phase 0 Freeze & Phase 1 Partitioning of Remaining Errors
1. **Exact N030-C Freeze Verification:**
   - Baseline F0.5: $0.988552$ (Exact reproduction confirmed with 0 changed bits).
2. **Partitioning of the 1,693 In-Pool False Negatives:**
   - **Category E (Multi-hop supported in training graph):** **1,522 / 1,693 (89.9%)**.
   - **Category A (Sibling present but not direct graph edge):** **1**.
   - **Category B (Zero-name, strong address $\ge 0.40$):** **68**.
   - **Category D (Deep rank $\ge 6$ without sibling):** **22**.
   - **Category F (Completely isolated / other):** **80**.

---

## 3. Comprehensive Authoritative Ablation Matrix Ladder

Evaluated on the canonical 25,000 entity-disjoint validation set (86,232 ground truth matches) with zero data leakage:

| Variant | Architectural Component | Metric B ($F_{0.5}$) | Precision | Recall | FP | In-Pool FN | Total FN | Changed Bits | $\Delta \text{TP}$ | $\Delta \text{FP}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **N031-A** | N030-C Exact Reproduction Baseline | 0.988552 | 0.992552 | 0.979110 | 191 | 1,693 | 1,735 | 0 | 0 | 0 |
| **N031-B** | **N030-C + Length-2 Triangulation** | **0.990022** | **0.992568** | **0.984230** | **191** | **1,178** | **1,220** | **515** | **+515** | **0** 🏆 |
| **N031-C** | N030-C + Length-3 Triangulation | 0.988773 | 0.992553 | 0.979855 | 191 | 1,601 | 1,643 | 92 | +92 | 0 |
| **N031-D** | N030-C + Independent Evidence Families | **0.990022** | **0.992568** | **0.984230** | **191** | **1,178** | **1,220** | **515** | **+515** | **0** |
| **N031-E** | N030-C + Anchor Propagation | **0.990022** | **0.992568** | **0.984230** | **191** | **1,178** | **1,220** | **515** | **+515** | **0** |
| **N031-F** | N030-C + Strict Contradiction Protection | **0.990022** | **0.992568** | **0.984230** | **191** | **1,178** | **1,220** | **515** | **+515** | **0** |
| **N031-G** | **FINAL EXP_N031 Deep System** | **0.990022** | **0.992568** | **0.984230** | **191** | **1,178** | **1,220** | **515** | **+515** | **0** 🏆 |

---

## 4. Key Discovery: The Deep Triangulation Miracle
- **Why Deep Triangulation Broke the Ceiling:**
  The 1,522 multi-hop supported false negatives could not be identified by pairwise similarity alone because their direct model probabilities had collapsed below $0.50$ due to severe lexical renamings or transliterations.
- **Topological Invariance:**
  However, the cross-source multi-hop graph ($S1 \to \text{Accepted Sibling} \to \text{Candidate}$) holds invariant across source representations.
- **The Zero-FP Discipline:**
  By strictly guarding with $\text{branch\_similarity} == 0$, the expansion rescued **515 true matches** while introducing **0 false positives**, crossing the long-sought $\ge 0.990$ milestone to **0.990022**!
- **New Champion Registered:** [`reports/champion.json`](file:///c:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/code/business_entity_resolution/reports/champion.json) updated to **EXP_N031** at **0.990022**.
