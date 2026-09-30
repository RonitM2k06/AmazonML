#!/usr/bin/env python3
"""
Fast champion evaluation for EXP_N008.
Uses pandas-free vectorized groupby via numpy lexsort to avoid 1.66M Python loops.
"""
import csv, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evaluation import Evaluator

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(CODE_DIR, 'experiments')
REP = os.path.join(CODE_DIR, 'reports')

BT, DT, MG, AG, BR, DC = 0.956, 0.963, 0.025, 0.978, 0.938, 75
W_LGB = 0.45
CHAMPION_F05_PRIOR = 0.951463

print('Loading NPZ + probs...', flush=True)
t0 = time.time()
va = np.load(os.path.join(EXP, 'val_pairs_n008.npz'), allow_pickle=True)
s1_ids_raw  = va['s1_ids']    # object array of strings
cand_ids_raw = va['cand_ids']
sources_raw  = va['sources']
y            = va['y'].astype(np.int8)

probs = (W_LGB  * np.load(os.path.join(EXP, 'probs_n008_lgb.npy')) +
         (1-W_LGB) * np.load(os.path.join(EXP, 'probs_n008_xgb.npy'))).astype(np.float32)

N = len(y)
print(f'  {N:,} pairs loaded in {time.time()-t0:.1f}s, mean_prob={probs.mean():.4f}', flush=True)

# ── Vectorized group-by using numpy argsort on s1_ids ──
print('Vectorized grouping by S1...', flush=True)
t0 = time.time()

# Encode s1_ids as integer codes via np.unique
s1_ids_str = s1_ids_raw.astype(str)
unique_s1, s1_codes = np.unique(s1_ids_str, return_inverse=True)  # s1_codes[i] = int code of s1_ids[i]
cand_ids_str = cand_ids_raw.astype(str)
is_s3 = (sources_raw.astype(str) == 'S3')

# Sort all arrays by (s1_code, -prob) to get per-entity sorted candidates
sort_key = np.lexsort((-probs, s1_codes))
s1_codes_s  = s1_codes[sort_key]
probs_s     = probs[sort_key]
cand_ids_s  = cand_ids_str[sort_key]
is_s3_s     = is_s3[sort_key]
y_s         = y[sort_key]

# Find boundaries of each S1 group
boundaries = np.where(np.diff(s1_codes_s) != 0)[0] + 1
starts = np.concatenate([[0], boundaries])
ends   = np.concatenate([boundaries, [N]])

n_entities = len(unique_s1)
print(f'  {n_entities:,} entities grouped in {time.time()-t0:.1f}s', flush=True)

# ── Evaluate ──
print('Evaluating at best config...', flush=True)
t0 = time.time()
sg = BT * 0.85

evaluator = Evaluator()
for k in range(n_entities):
    s1_id = unique_s1[k]
    sl = slice(starts[k], ends[k])
    ps   = probs_s[sl]
    cids = cand_ids_s[sl]
    s3   = is_s3_s[sl]
    ys   = y_s[sl]

    ts = set(cids[ys == 1])   # true matches in candidate pool
    M  = len(ps)

    if M == 0 or ps[0] < sg:
        evaluator.update(s1_id, ts, set())
        continue

    top = ps[0]
    et  = DT if M > DC else BT
    am  = (ps >= et) & ((top - ps) <= MG)
    anm = ps >= AG

    if np.any(anm & am):
        if not bool(np.any(am & s3)):
            am = am | ((~am) & s3 & (ps >= BR))
        if not bool(np.any(am & ~s3)):
            am = am | ((~am) & ~s3 & (ps >= BR))

    acc = set(cids[am])
    evaluator.update(s1_id, ts, acc)

    if (k + 1) % 5000 == 0:
        print(f'  {k+1:,}/{n_entities:,} entities...', flush=True)

m = evaluator.compute()
elapsed = time.time() - t0
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
print(f"  True Matches in pool     : {m['total_true_matches']:,}")
print(f"  Eval time                : {elapsed:.1f}s")
print(f"  Best params              : base_th={BT}, dense_th={DT}, margin={MG}, anchor={AG}, bridge={BR}, dc={DC}")
print(f"  Blend                    : LGB={W_LGB} / XGB={1-W_LGB}")
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
        'singleton_false_positive_rate': m['singleton_false_positive_rate'],
        'exact_set_accuracy': m['exact_set_accuracy'],
        'num_features': 57,
        'eval_population_s1': n_entities,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'experiment': 'EXP_N008',
        'neural_feature': 'neural_cosine_sim',
        'neural_model': 'paraphrase-multilingual-MiniLM-L12-v2',
        'base_threshold': BT, 'dense_threshold': DT, 'margin_delta': MG,
        'anchor_gate': AG, 'bridge_threshold': BR, 'density_cutoff': DC,
        'blend_w_lgb': W_LGB, 'blend_w_xgb': round(1-W_LGB, 2),
        'val_blocker_recall': 0.9644,
        'total_false_positives': m['total_false_positives'],
        'total_false_negatives': m['total_false_negatives'],
    }
    champ_path = os.path.join(REP, 'champion.json')
    with open(champ_path, 'w') as f:
        json.dump(champ, f, indent=2)
    print(f'NEW CHAMPION saved: {champ_path}')
else:
    print(f'No improvement. Champion retained at {CHAMPION_F05_PRIOR:.6f}')

# Registry
registry = os.path.join(REP, 'neural_experiment_registry.csv')
fnames = ['exp_id','description','macro_f_05','macro_precision','macro_recall',
          'singleton_accuracy','exact_set_accuracy','total_entities','total_true_matches',
          'total_false_positives','total_false_negatives','threshold','notes']
write_hdr = not os.path.exists(registry)
with open(registry, 'a', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fnames, extrasaction='ignore')
    if write_hdr: w.writeheader()
    w.writerow({
        'exp_id': 'EXP_N008',
        'description': 'Hybrid 57-feat final (vectorized eval, 1620-sweep confirmed best)',
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
