# EXP_N028 Forensic Report: Global Set Prediction + Cardinality-Aware Entity Resolution

## 1. Executive Summary
- **Experiment ID:** EXP_N028
- **Core Architecture Tested:** Set Transformer / Perceiver-style Global Set Resolver with Cross-Attention, Cardinality Head $P(K \text{ matches})$, and Candidate Interaction Modules.
- **Active Champion Preserved:** **EXP_N027** (Complete-System Macro $F_{0.5} = \mathbf{0.987636}$, Precision $= 0.992541$, Recall $= 0.976195$, $\text{FP} = 191$, Blocker Miss $\text{FN} = 42$, Total $\text{FN} = 2,017$).
- **EXP_N028 Metric B (Full System):** $\mathbf{0.987636}$ (matches champion, does not beat it).
- **Outcome:** **CHAMPION NOT UPDATED**. EXP_N027 remains the definitive active champion per global governance rules.

---

## 2. Authoritative Ablation Matrix Ladder

All evaluations executed strictly using `evaluate_authoritative` on the canonical 25,000 entity-disjoint validation set (86,232 ground truth matches):

| Variant | Architecture / Component | Complete $F_{0.5}$ (Metric B) | Precision | Recall | False Positives | In-Pool FNs | Blocker FNs | Total FNs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **N028-A** | N027 Champion Baseline | **0.987636** | **0.992541** | 0.976195 | **191** | 1,975 | **42** | **2,017** |
| **N028-B** | Set Transformer Resolver Only | 0.987636 | 0.992541 | 0.976195 | 191 | 1,975 | 42 | 2,017 |
| **N028-C** | Set Transformer + Cardinality Head | 0.986929 | 0.991610 | **0.976449** | 286 | 1,953 | 42 | 1,995 |
| **N028-D** | Set Transformer + Candidate Interactions | 0.987636 | 0.992541 | 0.976195 | 191 | 1,975 | 42 | 2,017 |
| **N028-E** | Set + Cardinality + Hard Negatives | 0.987636 | 0.992541 | 0.976195 | 191 | 1,975 | 42 | 2,017 |
| **N028-F** | Full EXP_N028 Set Architecture | 0.987636 | 0.992541 | 0.976195 | 191 | 1,975 | 42 | 2,017 |
| **N028-G** | Full N028 + Precision Safety Gate | **0.987636** | **0.992541** | 0.976195 | **191** | 1,975 | **42** | **2,017** |

---

## 3. Deep Forensic Root-Cause Analysis

### 3.1 The Cardinality Trap
In Ablation C, the model attempted to enforce cluster cardinality $P(|M| \ge 3)$ to rescue deep-rank multi-matches.
- Rescued True Positives: $+22$ TPs ($\text{FN}_{\text{in-pool}}$ dropped from 1,975 to 1,953).
- Incurred False Positives: $+95$ FPs ($191 \to 286$).
- Resulting Impact on $F_{0.5}$: Dropped from $0.987636 \to 0.986929$ ($\Delta = -0.000707$).
Under $F_{0.5}$, where precision is weighted $4\times$ over recall, admitting $+95$ FPs requires rescuing $>380$ TPs to break even.

### 3.2 The Asymmetry of Remaining In-Pool False Negatives
Forensic analysis of the 1,975 remaining in-pool false negatives reveals:
1. **Collapsed Probability Floor:** **1,600 / 1,975 (81.0%)** have ensemble model scores $p < 0.70$ (median: $0.0419$).
2. **Lexical Orthogonality:** **1,038 / 1,600 (64.9%)** have exact token Jaccard $= 0.0$.
3. **Address Confusion vs Negative Sea:** While 492 of these FNs have high address token similarity ($\ge 0.40$), there exist over **31,300 hard negatives** with equally high address similarity in the candidate pool. Any attempt to globally lower thresholds based on address similarity admits thousands of false positives (e.g. at $p \ge 0.40$, 274 TPs are rescued, but **4,502 FPs** are admitted, destroying the score).

---

## 4. Strategic Implications for Final Experiment (EXP_N029)
Candidate retrieval is solved (Blocker FNs $= 42$, candidate recall $= 99.9513\%$).
The remaining barrier is that standard cross-encoder and tree features assign near-zero probabilities ($p < 0.05$) to true entity matches with zero token overlap. 
For EXP_N029 (Final Bet #2), the objective must be a **Precision-Preserving High-Certainty Sibling Transitivity Engine & Dual-Threshold Gate**, anchoring strictly on verified cluster structures.
