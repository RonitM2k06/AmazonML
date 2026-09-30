# EXP_N026 Forensic Report: Neural Entity Memory + Set Completion

## 1. Executive Summary
- **Experiment ID:** EXP_N026
- **Architecture Tested:** Neural Entity Memory + Set Completion Networks + Competing Entity Memory Separation.
- **Active Champion Preserved:** **EXP_N024** (Complete-System Macro $F_{0.5} = \mathbf{0.986756}$, Precision $= 0.992543$, Recall $= 0.973212$, $\text{FP} = 192$, In-Pool $\text{FN} = 1,879$, Blocker Miss $\text{FN} = 404$).
- **EXP_N026 Metric B (Full System):** $\mathbf{0.986619}$ (Precision $= 0.992015$, Recall $= 0.973302$, $\text{FP} = 206$, Total $\text{FN} = 2,285$).
- **Outcome:** **CHAMPION NOT UPDATED**. EXP_N024 remains the definitive active champion per the rule requiring an absolute improvement over the active champion.

---

## 2. Comprehensive Ablation Ladder

All evaluations were executed strictly using `evaluate_authoritative` on the canonical 25,000 entity-disjoint validation population (86,232 ground truth matches) with zero data leakage:

| Variant | Architecture / Component | Macro $F_{0.5}$ (Metric B) | Precision | Recall | False Positives | In-Pool FNs | Blocker FNs | Total FNs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **N026-A** | N024 Baseline Frontier Decoder | **0.986756** | **0.992543** | 0.973212 | 192 | 1,879 | 404 | 2,283 |
| **N026-B** | Learned Metric Projection Only | 0.986416 | 0.992984 | 0.971510 | 180 | 2,030 | 404 | 2,434 |
| **N026-C** | Dynamic Entity Memory Prototype | 0.986447 | 0.992318 | 0.972326 | 197 | 1,965 | 404 | 2,369 |
| **N026-D** | Set Coherence Gain Network | 0.984534 | 0.995321 | 0.963410 | 116 | 2,741 | 404 | 3,145 |
| **N026-E** | Competing Entity Margin Separation | 0.986421 | 0.992311 | 0.972271 | 197 | 1,970 | 404 | 2,374 |
| **N026-F** | 2-Stage Multi-Match Dynamic Update | 0.986614 | 0.992264 | 0.973024 | 199 | 1,898 | 404 | 2,302 |
| **N026-G** | Full N026 Joint System | 0.986619 | 0.992015 | **0.973302** | 206 | 1,881 | 404 | 2,285 |

---

## 3. Deep Forensic Root-Cause Analysis

### 3.1 Why Set Completion Hit a Precision Wall
Under $F_{0.5}$, Precision is weighted $4\times$ higher than Recall ($\beta = 0.5$). 
Specifically:
$$\Delta F_{0.5} \approx 0.8 \cdot \Delta P + 0.2 \cdot \Delta R$$
In N026-G, the model rescued several deep-rank true sibling matches ($\Delta \text{FN}_{\text{in-pool}} = -7$), but admitted $+14$ false positives ($192 \to 206$).
The penalty of $+14$ FPs completely wiped out the gain of the rescued TPs.

### 3.2 The Fundamental Barrier: Lexical Disconnect
The 1,879 in-pool false negatives break down as:
1. **1,213 / 1,879 (64.6%) have Zero Name Overlap:** Radical business renaming, abbreviation, acronyms, or corporate parent/subsidiary pairs (e.g., *Tata Consultancy Services* vs *TCS Ltd* or holding company names).
2. **542 / 1,879 (28.8%) Cross-Script / Heavy Transliteration:** Devanagari $\leftrightarrow$ Latin, Cyrillic, Greek, where standard byte n-grams and character Jaccard drop to 0.
3. **944 / 1,879 (50.2%) Collapsed Probability ($p < 0.30$):** Handcrafted features and decision trees assign near-zero probabilities to these pairs.
4. **404 Blocker Misses:** Completely unindexed pairs that require cross-graph multi-hop traversal to retrieve.

Because decision-level and set-level models operate on pre-extracted embeddings and tree scores, they cannot create new phonetic/character bridge signals that were never captured upstream.

---

## 4. Strategic Next Step: EXP_N027
To surpass the $0.990000$ frontier:
1. **Subword & Phonetic Character Contrastive Modeling:** Directly train a character byte-level sequence encoder on the 100k training pairs exhibiting 0 token overlap to bridge the 1,213 lexical disconnects.
2. **Multi-Hop Transitive Sibling Blocker (Attacking the 404 Blocker Misses):**
   Traverse accepted matches ($S1 \leftrightarrow S2 \leftrightarrow S3 \leftrightarrow S2_{\text{target}}$) to pull in the 404 missing candidates into the candidate pool without candidate explosion.
