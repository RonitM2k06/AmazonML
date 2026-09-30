# EXP_N023 — Learned Entity Prefix Boundary Model Report

## 1. Executive Summary & Authoritative Milestone

In **EXP_N023**, we transitioned from handcrafted local boundary gap heuristics to a **Learned Entity Prefix Boundary Model**.

### The Breakthrough Result
- **Previous Champion (EXP_N022):** Complete-System Macro $F_{0.5} = \mathbf{0.984299}$
- **New Champion (EXP_N023):** Complete-System Macro $F_{0.5} = \mathbf{0.985939}$
- **Absolute Gain:** $\mathbf{+0.001640}$ (a significant leap over the previous plateau)
- **False Negatives Model (in pool):** Reduced from $2,865 \to \mathbf{2,121}$ (**744 true matches rescued!**)
- **Total False Negatives:** Reduced from $3,269 \to \mathbf{2,525}$
- **Multi-Match Recall:** Increased from $96.2085\% \to \mathbf{97.0719\%}$

---

## 2. Complete Authoritative Trajectory

| Experiment | Architecture / Strategy | Complete-System Macro $F_{0.5}$ (Metric B) | Decision Layer $F_{0.5}$ (Metric A) | Macro Precision | Macro Recall | False Positives (FP) | In-Pool Model FN | Blocker Miss FN | Total FN |
|---|---|---|---|---|---|---|---|---|---|
| **N018** | Entity Graph Bridge | 0.983070 | 0.979616 | 0.993181 | 0.959085 | 102 | 3,220 | 404 | 3,624 |
| **N019-F** | HierGAT+ Collective Alignment | 0.984094 | 0.980655 | 0.993438 | 0.962092 | 103 | 2,925 | 404 | 3,329 |
| **N020** | Selective Graph Decoder | 0.984281 | 0.980845 | 0.993552 | 0.962537 | 103 | 2,869 | 404 | 3,273 |
| **N021** | Corruption-Aware Rescue | 0.984264 | 0.980829 | 0.993524 | 0.962567 | 105 | 2,866 | 404 | 3,270 |
| **N022** | Listwise Local Gap Decoder | 0.984299 | 0.980863 | 0.993552 | 0.962585 | 103 | 2,865 | 404 | 3,269 |
| **EXP_N023** | **Learned Prefix Boundary Model (NEW CHAMPION)** | **0.985939** | **0.982555** | **0.992453** | **0.970758** | **198** | **2,121** | **404** | **2,525** |

---

## 3. Mandated Ablation Matrix Results

| Ablation | Configuration | Metric B ($F_{0.5}$) | Precision | Recall | False Positives | Total FN |
|---|---|---|---|---|---|---|
| **N023-A** | N022 Baseline Champion | 0.984299 | 0.993552 | 0.962585 | 103 | 3,269 |
| **N023-B** | Learned Boundary Only ($\tau = 0.85$) | 0.985238 | 0.993262 | 0.966481 | 129 | 2,902 |
| **N023-C** | Boundary + Precision Guard | 0.984601 | 0.993512 | 0.963570 | 106 | 3,179 |
| **N023-D** | Contextual Cross-Source Counts | 0.984711 | 0.993465 | 0.963977 | 108 | 3,151 |
| **N023-E** | Hard-Negative Drop Guard ($\le 0.065$) | 0.984309 | 0.993552 | 0.962621 | 103 | 3,265 |
| **N023-F** | Strict Zero-Added-FP Rescue | 0.984309 | 0.993552 | 0.962621 | 103 | 3,265 |
| **N023-G** | **Full N023 System (Optimal Calibration $\tau = 0.60$)** | **0.985939** | **0.992453** | **0.970758** | **198** | **2,525** |

---

## 4. Key Architectural Discoveries

1. **Why Learned Prefix Boundary Outperformed Fixed Gap Rules:**
   - Fixed gap heuristics were constrained to $\Delta p \le 0.050$. For true matches where the model score had dropped to $p \approx 0.60 - 0.85$, the drop from the anchor was often $0.10 - 0.25$, which fixed gap rules completely rejected.
   - The learned boundary model uses contextual features (candidate rank, preceding drop, subsequent drop to next candidate, address Jaccard, house number matching, token-set similarity, and current accepted source counts) to recognize that a candidate at rank $r$ is still an entity member because the drop to rank $r+1$ is massive ($\Delta p_{r \to r+1} \gg 0$).

2. **The Precision-Recall Frontier:**
   - Under strict zero added false positives, the model recovered 4 matches ($F_{0.5} = 0.984309$).
   - Allowing high-confidence boundary extensions with the learned transition model recovered **744 in-pool false negatives** while admitting only 95 false positives across 25,000 entities, pushing Macro Recall up by $+0.82\%$ and driving Metric B to **0.985939**.

---

## 5. Remaining Error Analysis & Next Bottlenecks

1. **Remaining 2,121 In-Pool FNs:**
   - 1,280 have zero address overlap and $p < 0.40$ (severe cross-script/transliteration gap).
   - 841 are interleaved behind high-scoring negative candidates (ranking inversions).
2. **Remaining 404 Blocker Miss FNs:**
   - These 404 true matches were never retrieved by the candidate generation phase, setting the candidate ceiling at $99.5315\%$.
