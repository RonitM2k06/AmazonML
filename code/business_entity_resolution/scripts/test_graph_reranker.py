#!/usr/bin/env python3
"""
Test Graph Reranker & Set-Level Prediction on N010 validation pairs.
"""

import sys
import numpy as np

sys.path.insert(0, "scripts")
from authoritative_eval_fn import run_authoritative_eval

va = np.load("experiments/val_pairs_n010.npz", allow_pickle=True)
pl09 = np.load("experiments/probs_n011_optimal_lgb.npy")
px09 = np.load("experiments/probs_n011_optimal_xgb.npy")
probs_orig = 0.45 * pl09 + 0.55 * px09

s1_ids = va["s1_ids"].astype(str)
cand_ids = va["cand_ids"].astype(str)
sources = va["sources"].astype(str)
y = va["y"]
X = va["X"]

# Feature indices
addr_jac_idx = 30
name_jac_idx = 11
house_exact_idx = 40
addr_exact_idx = 29
name_exact_idx = 1
cos_idx = 56

print("Optimizing graph set-level decision rule...")

# Group by S1
unique_s1, s1_codes = np.unique(s1_ids, return_inverse=True)
sort_key = np.lexsort((-probs_orig, s1_codes))
s1_codes_s = s1_codes[sort_key]
probs_s = probs_orig[sort_key]
cand_ids_s = cand_ids[sort_key]
sources_s = sources[sort_key]
y_s = y[sort_key]
X_s = X[sort_key]

boundaries = np.where(np.diff(s1_codes_s) != 0)[0] + 1
starts = np.concatenate([[0], boundaries])
ends = np.concatenate([boundaries, [len(probs_s)]])

# We can test an enhanced decision_fn directly!
# A decision function: (ps, cids, is_s3) -> Set[str]
# But wait, in run_authoritative_eval, decide only takes (ps, cids, is_s3)!
# Can we construct updated probabilities `probs_graph_lgb` and `probs_graph_xgb`?
# YES! If we update probs per pair based on graph support and contradiction,
# then run_authoritative_eval evaluates the complete system seamlessly!

probs_graph = probs_orig.copy()
boosted_count = 0
penalized_count = 0

for k in range(len(unique_s1)):
    sl = slice(starts[k], ends[k])
    orig_indices = sort_key[sl]
    ps = probs_s[sl]
    xs = X_s[sl]
    srcs = sources_s[sl]
    N = len(ps)
    if N == 0:
        continue
    top = ps[0]
    
    # Check if there is a very high confidence anchor (p >= 0.96)
    if top >= 0.96:
        anchor_src = srcs[0]
        anchor_hn_exact = xs[0, house_exact_idx]
        anchor_addr_jac = xs[0, addr_jac_idx]
        anchor_name_jac = xs[0, name_jac_idx]
        
        for i in range(1, N):
            p = ps[i]
            pair_idx = orig_indices[i]
            hn_exact = xs[i, house_exact_idx]
            addr_jac = xs[i, addr_jac_idx]
            name_jac = xs[i, name_jac_idx]
            cos_sim = xs[i, cos_idx]
            is_cross = (srcs[i] != anchor_src)
            
            # 1. Transitive Support: candidate is in the OTHER source, has borderline probability (0.85 <= p < 0.94),
            # and agrees strongly on address/house-number and name with the anchor!
            if is_cross and (0.85 <= p < 0.94):
                if (xs[i, addr_exact_idx] == 1.0 and name_jac >= 0.25) or \
                   (hn_exact == 1.0 and addr_jac >= 0.40 and name_jac >= 0.25 and cos_sim >= 0.60):
                    # Boost to bridge threshold
                    probs_graph[pair_idx] = max(p, 0.945)
                    boosted_count += 1
            
            # 2. Contradiction Penalty:
            # If candidate contradicts the anchor's verified address:
            # e.g. anchor has high address agreement (>=0.40) & house exact == 1.0,
            # but candidate has house_exact == 0.0 AND addr_jac == 0.0 AND name_jac < 0.50
            if anchor_addr_jac >= 0.40 and anchor_hn_exact == 1.0:
                if hn_exact == 0.0 and addr_jac == 0.0 and name_jac < 0.50:
                    probs_graph[pair_idx] = min(p, 0.50)
                    penalized_count += 1

print(f"Graph Reranker Applied: Boosted {boosted_count} candidates, Penalized {penalized_count} candidates.")

np.save("experiments/probs_n012_graph_lgb.npy", probs_graph)
np.save("experiments/probs_n012_graph_xgb.npy", probs_graph)

# Now run authoritative evaluation
print("\nRunning Authoritative Evaluation for EXP_N012...")
res = run_authoritative_eval(
    val_npz_path="experiments/val_pairs_n010.npz",
    probs_lgb_path="experiments/probs_n012_graph_lgb.npy",
    probs_xgb_path="experiments/probs_n012_graph_xgb.npy",
    w_lgb=0.50,
    label="EXP_N012_Graph",
    base_th=0.940,
    dense_th=0.950,
    margin=0.025,
    anchor=0.960,
    bridge=0.920,
    density_cutoff=75,
    verbose=True
)

mB = res["B"]["macro_f_05"]
mA = res["A"]["macro_f_05"]
pB = res["B"]["macro_precision"]
rB = res["B"]["macro_recall"]
errs = res["errors"]

print("=" * 60)
print(f"EXP_N012 GRAPH-AUGMENTED RESULTS:")
print(f"  Complete System F0.5 (Metric B): {mB:.6f}")
print(f"  Decision Layer F0.5 (Metric A):  {mA:.6f}")
print(f"  Macro Precision:                {pB:.6f}")
print(f"  Macro Recall:                   {rB:.6f}")
print(f"  False Positives (FP):           {errs['fp']}")
print(f"  False Negatives Model:          {errs['fn_model']}")
print(f"  False Negatives Blocker:        {errs['fn_blocker']}")
print("=" * 60)
