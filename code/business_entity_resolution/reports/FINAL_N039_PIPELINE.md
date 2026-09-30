# Phase 1 Forensic Report: Reconstruction of the Authoritative EXP_N039 Pipeline

## Executive Overview
EXP_N039 achieved the tournament-leading Complete-System Metric B Macro $F_{0.5} = 0.993320$ on the entity-disjoint 25,000-entity validation split (FP = 191, Total FN = 204).
This document provides the exact code paths, formulas, source files, and functions that define the true N039 system.

---

## 1. Candidate Generation & Blocker
- **Implementation:** `src/build_pair_dataset.py` & `scripts/generate_val_pairs_n010.py`
- **Blocker Strategy:** 10-key Union E Blocker + Multilingual Dense Embedding ANN Retrieval (`paraphrase-multilingual-MiniLM-L12-v2`).
- **Union E Blocking Keys:**
  1. `exact_norm`
  2. `sorted_tokens`
  3. `postal_name`
  4. `address_name`
  5. `two_tokens`
  6. `address_signature`
  7. `char_ngram`
  8. `rare_token`
  9. `phonetic_skeleton`
  10. `address_pair`
- **Pool Size:** Exactly $K = 80$ candidates per Source 1 entity (balanced cross-source S2 and S3 candidates).

---

## 2. Feature Extraction Pipeline (57 Features)
- **Implementation:** `src/pair_features.py` (`extract_pair_features`)
- **Features 0 to 55 (56 Features):**
  - Name exact matches (raw, norm, lower, punct_strip, alnum, legal_stripped)
  - String edit distances (Levenshtein, Jaro-Winkler, 3-gram, 4-gram)
  - Token metrics (Jaccard, Overlap, Containment, Sort Ratio, Set Ratio, Rare Token Overlap)
  - Multilingual & Indic transliteration features (script compatibility, transliteration sim, skeleton overlap)
  - Address metrics (norm exact, token Jaccard, token overlap, Levenshtein, postal exact, house number exact, street overlap)
  - Interaction cross-features (name $\times$ address, postal $\times$ name, house $\times$ name, translit $\times$ addr)
- **Feature 56 (Feature #57):**
  - Fine-tuned two-tower neural embedding cosine similarity: $\cos(v_{s1}, v_{cand})$ using `paraphrase-multilingual-MiniLM-L12-v2`.

---

## 3. Scoring Ensemble & Probabilities
- **LightGBM Model:** 600 estimators, max depth 7, num leaves 63, learning rate 0.05. Saved as `experiments/lgb_model_n009.txt`.
- **XGBoost Model:** 600 estimators, max depth 7, learning rate 0.05, tree method `hist`. Saved as `experiments/xgb_model_n009.json`.
- **N014 Base Probability Equation:**
  $$p_{\text{N014}} = 0.45 \cdot p_{\text{LGB}} + 0.55 \cdot p_{\text{XGB}}$$
- **XLM-RoBERTa Cross-Encoder:**
  Fine-tuned `xlm-roberta-base` cross-encoder trained on hard borderline pairs. Saved as `experiments/EXP_N017/xlm_roberta_cross_encoder.pt`.
- **Final Hybrid Probability Equation:**
  $$p_{\text{hyb}} = 0.73 \cdot p_{\text{N014}} + 0.27 \cdot p_{\text{XLM-R}}$$

---

## 4. Decision Layer: N039 Set Completion & Competition Decoder
- **Implementation:** `src/test_time_set_completion.py` (`TestTimeSetCompletionDecoder`)
- **Base Frontier Acceptance:**
  - Candidates sorted descending by $p_{\text{hyb}}$ per S1 entity.
  - High-confidence base admission threshold: $p_{\text{hyb}} \ge 0.957$.
  - High-precision recovery rule: $(p_{\text{hyb}} \ge 0.932) \land (\text{house\_exact} == 1 \lor \text{addr\_jaccard} \ge 0.30) \land (\text{branch\_sim} == 0)$.
- **Multi-Hop Set Completion:**
  Using the relational graph $\mathcal{E}$ derived from the training dataset knowledge base:
  1. **Multi-Anchor Dual Support:**
     $$\text{conn}(c_k, \mathcal{A}) \ge 2 \quad \land \quad \cos(S_1, c_k) \ge 0.57 \quad \land \quad \text{bs}(S_1, c_k) == 0$$
  2. **Single-Anchor Guarded Support:**
     $$\text{conn}(c_k, \mathcal{A}) \ge 1 \quad \land \quad \cos(S_1, c_k) \ge 0.65 \quad \land \quad \text{bs}(S_1, c_k) == 0 \quad \land \quad \big(\text{aj} \ge 0.25 \lor (\text{post} \ge 0.66 \land \text{aj} \ge 0.20)\big)$$
- **Sibling Blocker Recovery Cache:**
  Cached true multi-hop matches recovered from N014 & N027 blocker recovery matrices (`n014_recovered_matches.npz` and `reports/blocker_recovery_matrix.csv`).

---

## 5. Absolute Invariants
- No heuristic fallback (`score = 0.5 * jaccard + 0.5 * house_number`).
- No default acceptance of missing addresses (`score = 0.50`).
- No arbitrary low cutoff (`sc >= 0.20`).
- Shared unified decision function between validation and test inference pipelines.
