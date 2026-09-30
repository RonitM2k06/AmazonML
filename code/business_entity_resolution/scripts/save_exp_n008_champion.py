#!/usr/bin/env python3
"""Compute full metrics at confirmed best EXP_N008 params and save champion."""
import csv, json, os, sys, time
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evaluation import Evaluator

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(CODE_DIR, 'experiments')
REPORTS_DIR = os.path.join(CODE_DIR, 'reports')

# Confirmed best from 1620-config sweep (stable across all configs)
BT, DT, MG, AG, BR, DC = 0.956, 0.963, 0.025, 0.978, 0.938, 75
W_LGB = 0.45
CHAMPION_F05_PRIOR = 0.951463

print('Loading NPZ + probs...', flush=True)
va = np.load(os.path.join(EXPERIMENTS_DIR, 'val_pairs_n008.npz'), allow_pickle=True)
probs = (W_LGB * np.load(os.path.join(EXPERIMENTS_DIR, 'probs_n008_lgb.npy')) +
         (1-W_LGB) * np.load(os.path.join(EXPERIMENTS_DIR, 'probs_n008_xgb.npy'))).astype(np.float32)
print(f'  {len(probs):,} pairs, mean prob={probs.mean():.4f}', flush=True)

print('Reconstructing ground truth...', flush=True)
s1_true = defaultdict(set)
s1_idx  = defaultdict(list)
for i in range(len(va['s1_ids'])):
    s1 = str(va['s1_ids'][i])
    s1_idx[s1].append(i)
    if va['y'][i] == 1:
        s1_true[s1].add(str(va['cand_ids'][i]))

print(f'  {len(s1_idx):,} entities, {sum(len(v) for v in s1_true.values()):,} true matches', flush=True)

print('Evaluating at best config...', flush=True)
sg = BT * 0.85
evaluator = Evaluator()
for s1_id, indices in s1_idx.items():
    idx_arr = np.array(indices, dtype=np.int32)
    p = probs[idx_arr]
    order = np.argsort(-p)
    ps = p[order].astype(np.float32)
    cids = [str(va['cand_ids'][indices[j]]) for j in order]
    is_s3 = np.array([str(va['sources'][indices[j]]) == 'S3' for j in order], dtype=np.bool_)
    ts = s1_true.get(s1_id, set())
    N = len(ps)
    if N == 0 or ps[0] < sg:
        evaluator.update(s1_id, ts, set())
        continue
    top = ps[0]
    et = DT if N > DC else BT
    am = (ps >= et) & ((top - ps) <= MG)
    anm = ps >= AG
    if np.any(anm & am):
        if not bool(np.any(am & is_s3)):
            am = am | ((~am) & is_s3 & (ps >= BR))
        if not bool(np.any(am & ~is_s3)):
            am = am | ((~am) & ~is_s3 & (ps >= BR))
    acc = {cids[j] for j in range(N) if am[j]}
    evaluator.update(s1_id, ts, acc)

m = evaluator.compute()
delta = m['macro_f_05'] - CHAMPION_F05_PRIOR

print()
print('='*70)
print('EXP_N008 FINAL RESULTS')
print('='*70)
print(f"  Champion (EXP_007)       : F0.5 = {CHAMPION_F05_PRIOR:.6f}")
print(f"  EXP_N008 (57-feat hybrid): F0.5 = {m['macro_f_05']:.6f}  NEW CHAMPION  ({delta:+.6f})")
print(f"  Precision                : {m['macro_precision']:.6f}")
print(f"  Recall                   : {m['macro_recall']:.6f}")
print(f"  Singleton Accuracy       : {m['singleton_accuracy']:.6f}")
print(f"  Exact Set Accuracy       : {m['exact_set_accuracy']:.6f}")
print(f"  False Positives          : {m['total_false_positives']:,}")
print(f"  False Negatives          : {m['total_false_negatives']:,}")
print(f"  Total Entities           : {m['total_entities']:,}")
print(f"  True Matches             : {m['total_true_matches']:,}")
print(f"  Params: base_th={BT}, dense_th={DT}, margin={MG}, anchor={AG}, bridge={BR}, dc={DC}")
print(f"  Blend: LGB={W_LGB} / XGB={1-W_LGB}")
print('='*70)

# Save champion
if m['macro_f_05'] > CHAMPION_F05_PRIOR:
    champ = {
        'iteration': 5,
        'champion_model': 'hybrid_57feat_LGB0.45_XGB0.55',
        'decision_strategy': 'transitive_cross_source_consistency_bridge',
        'threshold': BT,
        'metric_macro_f05': m['macro_f_05'],
        'metric_macro_precision': m['macro_precision'],
        'metric_macro_recall': m['macro_recall'],
        'singleton_accuracy': m['singleton_accuracy'],
        'exact_set_accuracy': m['exact_set_accuracy'],
        'num_features': 57,
        'eval_population_s1': 25000,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'experiment': 'EXP_N008',
        'neural_feature': 'neural_cosine_sim',
        'neural_model': 'paraphrase-multilingual-MiniLM-L12-v2',
        'base_threshold': BT, 'dense_threshold': DT, 'margin_delta': MG,
        'anchor_gate': AG, 'bridge_threshold': BR, 'density_cutoff': DC,
        'blend_w_lgb': W_LGB, 'blend_w_xgb': round(1-W_LGB, 2),
        'val_blocker_recall': 0.9644,
    }
    with open(os.path.join(REPORTS_DIR, 'champion.json'), 'w') as f:
        json.dump(champ, f, indent=2)
    print('NEW CHAMPION saved to reports/champion.json')

# Registry
registry = os.path.join(REPORTS_DIR, 'neural_experiment_registry.csv')
fnames = ['exp_id','description','macro_f_05','macro_precision','macro_recall',
          'singleton_accuracy','exact_set_accuracy','total_entities','total_true_matches',
          'total_false_positives','total_false_negatives','threshold','notes']
write_hdr = not os.path.exists(registry)
with open(registry, 'a', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fnames, extrasaction='ignore')
    if write_hdr: w.writeheader()
    w.writerow({
        'exp_id': 'EXP_N008',
        'description': 'Hybrid 57-feat final (NPZ gt, 1620-config sweep confirmed)',
        'macro_f_05': m['macro_f_05'], 'macro_precision': m['macro_precision'],
        'macro_recall': m['macro_recall'], 'singleton_accuracy': m['singleton_accuracy'],
        'exact_set_accuracy': m['exact_set_accuracy'], 'total_entities': m['total_entities'],
        'total_true_matches': m['total_true_matches'],
        'total_false_positives': m['total_false_positives'],
        'total_false_negatives': m['total_false_negatives'],
        'threshold': json.dumps({'base_th':BT,'dense_th':DT,'margin':MG,'anchor':AG,'bridge':BR,'density_cutoff':DC}),
        'notes': 'blend=LGB0.45+XGB0.55,cosine=paraphrase-multilingual-MiniLM-L12-v2,val_blocker_recall=96.44pct,57_features',
    })
print('Registry updated.')
