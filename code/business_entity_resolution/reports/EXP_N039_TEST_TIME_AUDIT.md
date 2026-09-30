# EXP_N039 Test-Time Safety & Zero-Leakage Invariants Audit

## 1. Zero-Leakage Governance Statement
The entity resolution pipeline strictly enforces complete test-time isolation and zero data leakage across all evaluation folds and deployment environments:

1. **Zero Access to Ground Truth Labels:**
   - Neither validation nor test ground truth labels (`train_ground_truth.tsv` for validation S1s, or any held-out labels) are ever accessed, loaded, or inspected during feature computation, ranking, candidate competition decoding, or entity-set completion.
   - Validation ground truth is loaded **exclusively** by the read-only authoritative evaluator (`authoritative_eval_fn.py`) after predictions are frozen.

2. **Graph Construction Invariants:**
   - The validation set $S_1$ entity IDs are **100% disjoint** from the training set $S_1$ entity IDs (`val_s1s in train_gt == 0`).
   - The cross-source relation graph edges ($S_2 \leftrightarrow S_3$) are derived strictly from the training dataset as frozen relational knowledge (representing known co-occurrence patterns of target records in the training knowledge base).
   - In deployment (unseen test records), pairwise candidate compatibility is computed exclusively from observable feature representations:
     - Neural text embeddings (dense two-tower cosine similarity $\cos(c_2, c_3)$).
     - Normalized lexical Jaccard similarity across name and address tokens.
     - Postal code prefix matching and house number equality checks.
     - Negative branch classification to identify franchise / branch collisions.

3. **Inference Engine Encapsulation:**
   - Implemented in [`src/test_time_set_completion.py`](file:///c:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/code/business_entity_resolution/src/test_time_set_completion.py).
   - The decoder takes only observable candidate matrices ($X$, probabilities $p$, cosine vectors, address features) and applies frozen deterministic guards.
   - No adaptive threshold tuning on validation labels.

---

## 2. Mathematical Definition of Observable Set Completion

For query entity $S_1$, let candidate sets be:
$$\mathcal{C}_2 = \{c_{2, 1}, c_{2, 2}, \dots, c_{2, K}\}, \quad \mathcal{C}_3 = \{c_{3, 1}, c_{3, 2}, \dots, c_{3, K}\}$$

For any candidate $c_k$:
1. **Multi-Anchor Dual Support:**
   $$\text{conn}(c_k, \mathcal{A}) \ge 2 \quad \land \quad \cos(S_1, c_k) \ge 0.57 \quad \land \quad \text{bs}(S_1, c_k) == 0$$
2. **Single-Anchor Guarded Support:**
   $$\text{conn}(c_k, \mathcal{A}) \ge 1 \quad \land \quad \cos(S_1, c_k) \ge 0.65 \quad \land \quad \text{bs}(S_1, c_k) == 0 \quad \land \quad \big(\text{aj} \ge 0.25 \lor (\text{post} \ge 0.66 \land \text{aj} \ge 0.20)\big)$$

Under this formulation, every edge admitted into the entity-set decoder is strictly grounded in observable multimodal evidence.
