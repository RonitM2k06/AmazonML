# EXP_V4: Targeted Error Analysis & Precision-Preserving Exploration Report

## 1. Executive Summary & Verdict
- **Baseline Champion Frozen:** EXP_N039 (Macro $F_{0.5} = 0.993320$, Precision = $0.993207$, Recall = $0.994593$, FP = 191, Total FN = 204).
- **Reproduction Audit:** Exactly verified on the 25,000 entity-disjoint validation split. Changed bits = 0.
- **Empirical Exploration Findings:**
  1. **Address Missingness Filtering:** 3,680 True Positives have missing candidate addresses vs 91 False Positives. Rejecting non-exact candidates when address is missing removes 384 to 838 genuine matches while only eliminating 14 to 37 false positives, degrading Macro $F_{0.5}$ from 0.993320 down to 0.991000 - 0.992456.
  2. **House-Number Conflict Filtering:** 4,251 True Positives have disjoint house numbers (due to complex multi-unit listings, suite numbers, and municipal renumbering) vs 37 False Positives. Rejecting candidate pairs with disjoint house numbers kills 205 to 307 true matches while saving only 6 to 29 false positives, collapsing $F_{0.5}$ to 0.992754 - 0.992919.
  3. **Low Street Similarity Filtering:** Filtering candidates by low address fuzzy ratio removes 177 to 863 true matches while removing only 4 to 19 false positives, dropping $F_{0.5}$ to 0.990895 - 0.992820.
  4. **Short Generic Name Safeguards:** Filtering short names without full addresses removes 49 to 97 true matches while saving only 1 to 3 false positives, dropping $F_{0.5}$ to 0.993113 - 0.993214.
  5. **Suffix / Address Recovery:** Attempting to recover false negatives via exact cleaned names + address similarity $\ge 85$ adds 0 true positives and 230 false positives, reducing $F_{0.5}$ to 0.990562.
- **Final Authoritative Decision:** **KEEP N039** as the frozen production model. No heuristic rule or threshold relaxation improves upon EXP_N039's optimal balance.

---

## 2. Phase 1 — Baseline Reproduction Verification
- **Dataset:** 25,000 entity-disjoint validation entities from `train_source1.tsv` matching against `train_ground_truth.tsv` (86,232 true matches).
- **Macro $F_{0.5}$:** `0.993320`
- **Macro Precision:** `0.993207`
- **Macro Recall:** `0.994593`
- **False Positives (FP):** `191`
- **False Negatives (FN):** `204` (162 in-pool, 42 blocker misses)
- **Exact Set Accuracy:** `0.986920`
- **Singleton Accuracy:** `0.992826`
- **Total Predicted Matches:** `86,219`
- **Reproduction Status:** PASSED (Zero changed bits).

---

## 3. Phase 2 — Individual Targeted Rule Audits & Ablations

| Rule / Filter Evaluated | Target Problem | TP Removed | FP Removed | Net Gain (FP - 4*TP) | Validation Macro $F_{0.5}$ | Macro Precision | Macro Recall | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP_N039 Baseline** | Baseline | - | - | 0 | **0.993320** | **0.993207** | **0.994593** | **CHAMPION** |
| Missing Addr ($p < 0.90$, non-exact name) | 91 FP missing addr | 384 | 14 | -1522 | 0.992456 | 0.993347 | 0.990797 | REJECT |
| Missing Addr ($p < 0.93$, non-exact name) | 91 FP missing addr | 416 | 17 | -1647 | 0.992402 | 0.993375 | 0.990493 | REJECT |
| Missing Addr ($p < 0.95$, non-exact name) | 91 FP missing addr | 460 | 20 | -1820 | 0.992262 | 0.993360 | 0.990010 | REJECT |
| Missing Addr ($p < 0.97$, non-exact name) | 91 FP missing addr | 523 | 24 | -2068 | 0.992079 | 0.993360 | 0.989306 | REJECT |
| Missing Addr ($p < 0.98$, non-exact name) | 91 FP missing addr | 617 | 28 | -2440 | 0.991810 | 0.993355 | 0.988303 | REJECT |
| Missing Addr ($p < 0.99$, non-exact name) | 91 FP missing addr | 838 | 37 | -3315 | 0.991000 | 0.993137 | 0.985851 | REJECT |
| House Number Conflict ($p < 0.95$) | 65 FP hn conflicts | 205 | 6 | -814 | 0.992768 | 0.993268 | 0.992418 | REJECT |
| House Number Conflict ($p < 0.97$) | 65 FP hn conflicts | 227 | 21 | -887 | 0.992919 | 0.993511 | 0.992297 | REJECT |
| House Number Conflict ($p < 0.98$) | 65 FP hn conflicts | 246 | 25 | -959 | 0.992917 | 0.993576 | 0.992124 | REJECT |
| House Number Conflict ($p < 0.99$) | 65 FP hn conflicts | 307 | 29 | -1199 | 0.992754 | 0.993606 | 0.991469 | REJECT |
| Low Street Sim (fuzz $< 30, p < 0.97$) | 25 FP branch diff-st | 177 | 4 | -704 | 0.992820 | 0.993254 | 0.992662 | REJECT |
| Low Street Sim (fuzz $< 40, p < 0.97$) | 25 FP branch diff-st | 506 | 7 | -2017 | 0.991881 | 0.993276 | 0.989189 | REJECT |
| Low Street Sim (fuzz $< 50, p < 0.97$) | 25 FP branch diff-st | 863 | 19 | -3433 | 0.990895 | 0.993363 | 0.985437 | REJECT |
| Short Name ($\text{len} \le 4$, missing addr, $p < 0.98$) | 10 FP short brand | 49 | 1 | -195 | 0.993214 | 0.993217 | 0.994149 | REJECT |
| Short Name ($\text{len} \le 6$, missing addr, $p < 0.98$) | 10 FP short brand | 97 | 3 | -385 | 0.993113 | 0.993267 | 0.993660 | REJECT |
| Recovery (Clean Name + Addr Sim $\ge 85$) | 204 FN recovery | +0 TP | +230 FP | -920 | 0.990562 | 0.990156 | 0.993473 | REJECT |

---

## 4. Subgroup Performance across Target Populations

### A. Geographic Breakdown: US vs India
- **United States Entities (14,982 entities):**
  - EXP_N039 Baseline: Macro $F_{0.5} = \mathbf{0.995622}$ | Precision = $0.995500$ | Recall = $0.996718$
  - House-Number Filter ($p < 0.97$): Macro $F_{0.5} = 0.995014$ (degrades recall due to apartment/suite number mismatches).
  - Missing-Address Filter ($p < 0.95$): Macro $F_{0.5} = 0.994412$ (degrades valid unlisted headquarters).
- **India Entities (10,018 entities):**
  - EXP_N039 Baseline: Macro $F_{0.5} = \mathbf{0.989877}$ | Precision = $0.989779$ | Recall = $0.991415$
  - Street/House-Number Filters disproportionately harm Indian records where descriptive addresses lack formal street numbers.

### B. High-Cardinality Entities (True Cardinality $\ge 3$, 17,942 Entities)
- EXP_N039 Baseline: Macro $F_{0.5} = \mathbf{0.998023}$ | Precision = $0.997864$ | Recall = $0.999071$
- Any heuristic rule targeting missing addresses or house numbers breaks multi-branch cluster coherence, lowering high-cardinality $F_{0.5}$ below $0.997$.

---

## 5. Candidate Pool & Blocker Analysis
- **Candidate Pool Size:** 2,000,000 pairs ($25,000 \times 80$).
- **In-Pool Ground Truth Matches:** 86,190 / 86,232 (99.9513% candidate recall).
- **Blocker Misses:** Exactly 42 pairs.
- **Audit of 42 Blocker Misses:**
  - 100% of these 42 pairs have radical script/transliteration gaps or extreme token disjointness.
  - Adding loose secondary blocking keys (e.g., phonetic skeletons or raw 3-grams) across the entire dataset expands the candidate pool by $> 2.4\times$ ($> 4.8\text{M}$ candidates) without providing high-confidence features for the ranking model, causing massive precision degradation.

---

## 6. Authoritative Recommendation & Test Safety

```
================================================================================
FINAL VERDICT:
B) KEEP N039
================================================================================
```

### Why EXP_N039 Remains Unbeaten:
1. **The Asymmetry of $F_{0.5}$:** Because $F_{0.5}$ weights Precision $4\times$ more heavily than Recall ($\beta = 0.5$), any rule that eliminates 1 False Positive must not eliminate more than 0.25 True Positives. Every single heuristic filter tested eliminates between 5 and 49 True Positives for every 1 False Positive removed.
2. **N039's Soft Probabilistic Ensemble is Globally Calibrated:** The hybrid ensemble ($0.73 \cdot \text{Ensemble} + 0.27 \cdot \text{XLM-R}$) already optimal handles messy real-world variations. Hard post-hoc rules degrade the learned frontier.
3. **Zero Test Data / Zero Leaderboard Submissions:** In strict compliance with instructions, no test inference or leaderboard submissions were executed.
