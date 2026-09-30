# N039_NO_FRONTIER -- Manifest / Reproducibility Record

## What this is
An additive, new baseline derived from the frozen N039 authoritative validation
pipeline, with exactly ONE component removed: the frontier-ranker (a 3-round,
5-fold-cross-validated XGBoost admission cascade found embedded in
scripts/run_n036.py Step 3, which trains directly on the validation set's own
ground-truth labels -- see reports/EXP_N039... investigation notes for the
original discovery).

This does NOT replace, overwrite, or invalidate the original N039 baseline
(pred_n039_matrix.npz, champion.json, reports/EXP_N039_*) -- both are
preserved and independently reproducible.

## Code path included (steps 1, 3-8) vs excluded (step 2)
1. INCLUDED -- Base frontier formula: am = (p_hyb >= 0.9570); recovery =
   (p_hyb >= 0.9320) & (house_exact==1 | addr_jaccard>=0.30) & (branch_sim==0);
   pred_base = am | recovery.  [scripts/run_n021.py lines 178-181]
2. EXCLUDED -- Frontier-ranker: 3 rounds of freshly-trained XGBoost via
   5-fold CV (KFold random_state=42/142/242), trained on
   extract_frontier_features()'s y_s (ground-truth) labels, admission
   thresholds 0.70/0.75/0.70 on out-of-fold predictions.
   [scripts/run_n036.py Step 3, lines 128-189; scripts/run_n032.py lines
   135-211 for extract_frontier_features]
   REMOVED IN THIS BASELINE. pred_base flows directly into step 3 in its
   place (i.e. pred_A_base := pred_base, mechanically identical to
   commenting out step 2 and using its input as its output).
3. INCLUDED -- Deterministic multi-hop training_edges graph rescue chain
   (pred_N031 -> N032 -> N033 -> N034 -> N035).
   [scripts/run_n036.py Step 5, lines 202-285]
4. INCLUDED -- N036's "Clean Unanchored Bootstrap" (p>=0.940, branch_sim==0,
   zero-prediction entities only). [scripts/run_n036.py, Ablation I]
5. INCLUDED -- N037's TestTimeReciprocalEntityInducer, applied only to
   entities still unanchored after step 4. [src/test_time_reciprocal_entity.py]
6. INCLUDED -- N038's TestTimeEntitySetDecoder, applied to all entities.
   [src/test_time_entity_set.py]
7. INCLUDED -- N039's TestTimeSetCompletionDecoder (multi-anchor/single-anchor
   set completion), applied on top of step 6. [src/test_time_set_completion.py]
8. INCLUDED -- Sibling/graph recovered_map union (n014_recovered_matches.npz +
   blocker_recovery_matrix.csv), applied by the caller exactly as the
   original pipeline does.

## Upstream scoring: UNCHANGED
Candidate generation (Union E blocker + fine-tuned two-tower ANN retrieval),
57-feature extraction, and the N014/N011 hybrid probability blend (real
LGB/XGB for N009-pool pairs, cosine-heuristic-or-zero for ANN-new pairs,
XLM-R cross-encoder blend at 0.73/0.27) are ALL IDENTICAL to the original
authoritative N039 pipeline. Only the base-admission/threshold layer differs.

## Reusable code
src/n039_no_frontier_decision.py::compute_pred_n038_no_frontier() and
::compute_authoritative_no_frontier_predictions() implement steps 1, 3-7
above as a self-contained, ground-truth-label-free function (step 8, the
recovered_map union, remains the caller's responsibility, matching the
original pipeline's structure).

## Validation entity IDs and candidate pool
Reused experiments/validation_s1_ids.txt and experiments/val_pairs_n010.npz
UNCHANGED -- not copied or duplicated, referenced by path only.

## Results
See reports/N039_NO_FRONTIER_RESULTS.json for full metrics.
