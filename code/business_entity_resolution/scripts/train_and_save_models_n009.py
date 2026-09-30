import os, sys, numpy as np
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path

script_dir = r'c:\Users\ronit\Videos\hackathons\AmazonML\amazon-ml-challenge\code\business_entity_resolution\scripts'
project_root = r'c:\Users\ronit\Videos\hackathons\AmazonML\amazon-ml-challenge\code\business_entity_resolution'
sys.path.insert(0, os.path.join(project_root, 'src'))
from neural_encoder import load_embeddings

EXPERIMENTS_DIR = os.path.join(project_root, 'experiments')
TRAIN_NPZ = os.path.join(EXPERIMENTS_DIR, 'train_pairs_exp008.npz')
S1_EMB_PATH = os.path.join(EXPERIMENTS_DIR, 's1_embeddings.npz')
TARGET_EMB_PATH = os.path.join(EXPERIMENTS_DIR, 'target_embeddings.npz')

tr_data = np.load(TRAIN_NPZ, allow_pickle=True)
X_train_56 = tr_data['X'].astype(np.float32)
y_train = tr_data['y'].astype(np.int8)
train_s1_ids = tr_data['s1_ids']
train_cand_ids = tr_data['cand_ids']
N_tr = len(y_train)

s1_ids_emb, s1_vecs = load_embeddings(S1_EMB_PATH)
target_ids_emb, tgt_vecs = load_embeddings(TARGET_EMB_PATH)
s1_emb_map = {id_: i for i, id_ in enumerate(s1_ids_emb)}
tgt_emb_map = {id_: i for i, id_ in enumerate(target_ids_emb)}

train_cosine = np.zeros(N_tr, dtype=np.float32)
for i in range(N_tr):
    idx1 = s1_emb_map.get(train_s1_ids[i])
    idx2 = tgt_emb_map.get(train_cand_ids[i])
    if idx1 is not None and idx2 is not None:
        train_cosine[i] = float(np.dot(s1_vecs[idx1], tgt_vecs[idx2]))

X_train_57 = np.hstack([X_train_56, train_cosine.reshape(-1, 1)]).astype(np.float32)

pos = int(np.sum(y_train == 1))
neg = int(np.sum(y_train == 0))
scale = neg / max(1, pos)

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 600,
    'max_depth': 7,
    'num_leaves': 63,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'scale_pos_weight': scale,
    'n_jobs': 4,
    'random_state': 42,
    'verbose': -1,
}
print('Training LightGBM...')
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_train_57, y_train)
lgb_model.booster_.save_model(os.path.join(EXPERIMENTS_DIR, 'lgb_model_n009.txt'))
print('Saved lgb_model_n009.txt')

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 600,
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'scale_pos_weight': scale,
    'n_jobs': 4,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cuda',
    'verbosity': 0,
}
print('Training XGBoost...')
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train_57, y_train)
xgb_model.save_model(os.path.join(EXPERIMENTS_DIR, 'xgb_model_n009.json'))
print('Saved xgb_model_n009.json')
