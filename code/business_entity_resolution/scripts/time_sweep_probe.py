#!/usr/bin/env python3
"""Timing probe: measure how fast one bridge config evaluates across 25k entities."""
import sys, time, random, json
from collections import defaultdict
import numpy as np

sys.path.insert(0, 'src')

va = np.load('experiments/val_pairs_n008.npz', allow_pickle=True)
val_s1_ids   = va['s1_ids']
val_cand_ids = va['cand_ids']
val_sources  = va['sources']

probs_lgb = np.load('experiments/probs_n008_lgb.npy')
probs_xgb = np.load('experiments/probs_n008_xgb.npy')
probs = (0.45 * probs_lgb + 0.55 * probs_xgb).astype(np.float32)

random.seed(42)
with open('experiments/validation_s1_ids.txt') as f:
    val_pool = [l.strip() for l in f if l.strip()]
selected = set(random.sample(val_pool, 25000))
val_matches = {}
train_gt = r'C:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/dataset/train/train_ground_truth.tsv'
with open(train_gt) as f:
    next(f)
    for line in f:
        parts = line.rstrip('\r\n').split('\t')
        s1 = parts[0]
        if s1 not in selected:
            continue
        mids = parts[1].split(',') if len(parts) > 1 and parts[1].strip() else []
        val_matches[s1] = mids
        if len(val_matches) >= 25000:
            break

val_true_sets = {s1: set(mids) for s1, mids in val_matches.items()}

# Build entity structs
s1_idx_map = defaultdict(list)
for i, s1 in enumerate(val_s1_ids):
    s1_idx_map[str(s1)].append(i)

entities = []
for s1_id, indices in s1_idx_map.items():
    idx_arr = np.array(indices, dtype=np.int32)
    p = probs[idx_arr]
    order = np.argsort(-p)
    entities.append({
        's1_id': s1_id,
        'probs': p[order],
        'cand_ids': [str(val_cand_ids[indices[j]]) for j in order],
        'is_s3': np.array([str(val_sources[indices[j]]) == 'S3' for j in order], dtype=np.bool_),
        'true_set': val_true_sets.get(s1_id, set()),
    })

avg_cands = sum(len(e['probs']) for e in entities) / len(entities)
print(f'Entities: {len(entities):,}, avg cands per entity: {avg_cands:.1f}')

# Time 20 config evaluations
N_TRIALS = 20
t0 = time.time()

for trial in range(N_TRIALS):
    base_th = 0.960 + trial * 0.001
    dense_th = base_th + 0.005
    margin = 0.030
    anchor = 0.985
    bridge = 0.935
    dc = 100
    sg = base_th * 0.85

    sum_f05 = 0.0
    total = 0
    for ent in entities:
        ps = ent['probs']
        cids = ent['cand_ids']
        is_s3 = ent['is_s3']
        ts = ent['true_set']
        n = len(ps)
        total += 1

        if n == 0 or ps[0] < sg:
            if not ts:
                sum_f05 += 1.0
            continue

        top = ps[0]
        eff_th = dense_th if n > dc else base_th
        accept_mask = (ps >= eff_th) & ((top - ps) <= margin)
        anchor_mask = ps >= anchor

        if np.any(anchor_mask & accept_mask):
            acc_s3 = bool(np.any(accept_mask & is_s3))
            acc_s2 = bool(np.any(accept_mask & ~is_s3))
            if not acc_s3:
                accept_mask = accept_mask | ((~accept_mask) & is_s3 & (ps >= bridge))
            if not acc_s2:
                accept_mask = accept_mask | ((~accept_mask) & ~is_s3 & (ps >= bridge))

        accepted = {cids[j] for j in range(n) if accept_mask[j]}
        np_set = len(accepted)

        if not ts:
            if not accepted:
                sum_f05 += 1.0
        elif np_set:
            tp = len(ts & accepted)
            if tp:
                p2 = tp / np_set
                r = tp / len(ts)
                d = 0.25 * p2 + r
                if d:
                    sum_f05 += 1.25 * p2 * r / d

    macro = sum_f05 / total

elapsed = time.time() - t0
per_config = elapsed / N_TRIALS

print(f'20 configs evaluated in {elapsed:.1f}s')
print(f'Per config: {per_config:.3f}s')
print(f'Sample F0.5 (last config): {macro:.6f}')
print(f'For  500 configs: {500*per_config:.0f}s = {500*per_config/60:.1f}min')
print(f'For 1000 configs: {1000*per_config:.0f}s = {1000*per_config/60:.1f}min')
print(f'For 2400 configs: {2400*per_config:.0f}s = {2400*per_config/60:.1f}min')
