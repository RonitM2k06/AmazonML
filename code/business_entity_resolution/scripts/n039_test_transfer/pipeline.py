#!/usr/bin/env python3
"""
N039 test-transfer pipeline (requirement A).

Orchestrates, for a bounded set of S1 entities and a bounded target pool
(S2+S3 records), the full chain:

  normalization/blocking (reuse) -> N009-equivalent keyword blocker (reuse
  Union E, same params as generate_val_pairs_n009.py) -> two-tower ANN
  retrieval (reuse two_tower_encoder.py via device_utils.py) -> exact
  N009-vs-ANN-new provenance split (per-pair boolean) -> 57-feature
  extraction (reuse pair_features.py) -> N039 scoring EXACTLY as validated
  (heuristic/zero for ANN-new, real LGB/XGB for N009-pool pairs) -> XLM-R
  blend (sparse selection + real inference) -> threshold/admission via
  src/n039_no_frontier_decision.py (no frontier-ranker, zero ground-truth
  dependency) -> TestTimeSetCompletionDecoder -> sibling recovery explicitly
  disabled (logged) -> matching_results.tsv.

Streaming/chunking: process_country() handles one country at a time and one
batch of S1 entities at a time (batch_size param) so memory stays bounded
regardless of scale; a checkpoint file records which (country, batch_index)
pairs have completed, enabling resume (requirement G).

This module is deliberately NOT run at full test scale in this phase --
only smoke_test.py and validation_transfer_smoke.py invoke it, on small
bounded inputs.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_CODE_DIR = _THIS_DIR.parents[1]
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_CODE_DIR / "src"))
sys.path.insert(0, str(_CODE_DIR / "scripts"))

import config
from device_utils import encode_records_fused_auto_device
from blocking import build_target_index, generate_candidates, clean_business_name
from build_pair_dataset import UNION_E_KEYS
from pair_features import extract_pair_features, FEATURE_NAMES
from normalization import extract_numbers, tokenize
from two_tower_encoder import load_two_tower, RecordTokenizer
from n039_no_frontier_decision import compute_authoritative_no_frontier_predictions

import faiss
import lightgbm as lgb
import xgboost as xgb


def load_records(path: Path, needed_ids: Set[str] = None, limit: int = None) -> Dict[str, Tuple]:
    """Stream a source TSV into {id: (eid, name, addr, country)}."""
    recs: Dict[str, Tuple] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = (row.get("entity_id") or "").strip()
            if not eid:
                continue
            if needed_ids is not None and eid not in needed_ids:
                continue
            nm = (row.get("business_name") or "").strip()
            addr = (row.get("business_address") or "").strip()
            ctry = (row.get("country") or "").strip()
            recs[eid] = (eid, nm, addr, ctry)
            if limit is not None and len(recs) >= limit:
                break
    return recs


def build_kw_index(target_records: Dict[str, Tuple]):
    """N009-equivalent keyword blocker index (Union E keys, same params)."""
    return build_target_index(
        list(target_records.values()), UNION_E_KEYS, max_bucket_size=config.UNION_E_MAX_BUCKET_SIZE
    )


def keyword_candidates(s1_records: Dict[str, Tuple], kw_index) -> Dict[str, Set[str]]:
    return generate_candidates(
        s1_records.values(), kw_index, UNION_E_KEYS, max_candidates_per_s1=config.KEYWORD_MAX_CANDIDATES_PER_S1
    )


def compute_two_tower_embeddings(records: Dict[str, Tuple], device_log: list) -> Tuple[List[str], np.ndarray]:
    device = config.select_device()
    device_log.append(device)
    two_tower = load_two_tower(str(config.TWO_TOWER_CHECKPOINT), device=device)
    tok = RecordTokenizer()
    ids, vecs = encode_records_fused_auto_device(two_tower, tok, records, batch_size=64)
    return ids, vecs


def ann_candidates_per_country(
    s1_records: Dict[str, Tuple],
    target_records: Dict[str, Tuple],
    s1_vecs_map: Dict[str, np.ndarray],
    tgt_ids: List[str],
    tgt_vecs: np.ndarray,
    top_k: int = config.ANN_TOP_K,
) -> Dict[str, Set[str]]:
    """
    Exact top-k cosine (inner product on L2-normalised vectors) retrieval,
    mathematically identical to faiss.IndexFlatIP -- implemented in plain
    numpy instead of FAISS.

    NOTE (environment finding, not a design preference): faiss-cpu (pip
    wheel) and torch both statically link their own copy of libomp.dylib on
    this machine; importing/using both in the same process reliably raises
    "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
    already initialized" and, if suppressed via KMP_DUPLICATE_LIB_OK=TRUE,
    segfaults (SIGSEGV, exit code 139) inside faiss.IndexFlatIP.add()/
    search() the moment torch has already been imported/used in-process
    (confirmed by direct reproduction: `import torch; import faiss; ...`
    segfaults regardless of KMP_DUPLICATE_LIB_OK or faiss.omp_set_num_threads
    settings). Since this pipeline necessarily uses torch (two-tower
    embeddings, XLM-R) in the same process as candidate retrieval, FAISS is
    not usable here without a cross-process split. This numpy fallback is
    exact (brute-force, not approximate) so retrieval CORRECTNESS is
    unaffected -- only throughput is (no SIMD/BLAS-specialized ANN
    acceleration), which matters for the resource/scale estimate (see
    resource_estimator.py output and the final report's throughput numbers).
    """
    if len(tgt_ids) == 0:
        return {s1: set() for s1 in s1_records}

    tgt_vecs_f32 = np.ascontiguousarray(tgt_vecs, dtype=np.float32)
    s1_list = [s1 for s1 in s1_records if s1 in s1_vecs_map]
    if not s1_list:
        return {s1: set() for s1 in s1_records}
    q = np.vstack([s1_vecs_map[s1] for s1 in s1_list]).astype(np.float32)
    k = min(top_k, len(tgt_ids))

    sims = q @ tgt_vecs_f32.T  # [n_queries, n_targets], inner product == cosine (vectors already L2-normalised)
    if k < sims.shape[1]:
        top_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    else:
        top_idx = np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))

    out: Dict[str, Set[str]] = {s1: set() for s1 in s1_records}
    for row_i, s1 in enumerate(s1_list):
        out[s1] = {tgt_ids[j] for j in top_idx[row_i]}
    return out


def build_candidate_pool(
    s1_records: Dict[str, Tuple],
    kw_cands: Dict[str, Set[str]],
    ann_cands: Dict[str, Set[str]],
) -> Dict[str, List[Tuple[str, bool]]]:
    """
    Union kw + ann candidates per S1, with provenance flag (True = found by
    keyword blocker = "N009-equivalent pool"; False = ANN-only = "ANN-new").
    Caps to FINAL_POOL_CAP_K, prioritizing keyword-blocker-found candidates
    first (documented design choice, since there is no ground truth to
    prioritize by here, unlike the original label-aware capping).
    """
    pool: Dict[str, List[Tuple[str, bool]]] = {}
    for s1 in s1_records:
        kw = kw_cands.get(s1, set())
        ann = ann_cands.get(s1, set())
        kw_only_list = sorted(kw)
        ann_only_list = sorted(ann - kw)
        combined = [(c, True) for c in kw_only_list] + [(c, False) for c in ann_only_list]
        pool[s1] = combined[: config.FINAL_POOL_CAP_K]
    return pool


def extract_features_for_pool(
    s1_records: Dict[str, Tuple],
    target_records: Dict[str, Tuple],
    pool: Dict[str, List[Tuple[str, bool]]],
    tt_s1_vecs: Dict[str, np.ndarray],
    tt_tgt_vecs: Dict[str, np.ndarray],
    zs_s1_vecs: Dict[str, np.ndarray],
    zs_tgt_vecs: Dict[str, np.ndarray],
):
    """
    Returns flat arrays: s1_ids, cand_ids, is_kw_pool (bool),
    X57_twotower (feature 57 = two-tower cosine -- feeds decoder/threshold),
    feat57_zeroshot (feature 57 = zero-shot MiniLM cosine -- feeds the real
    LGB/XGB model ONLY for is_kw_pool pairs, matching the model's actual
    training feature space).
    """
    s1_out, cand_out, is_kw_out = [], [], []
    X56_list = []
    cos_tt_list, cos_zs_list = [], []

    for s1, cands in pool.items():
        if not cands:
            continue
        s1_rec = s1_records[s1]
        for cid, is_kw in cands:
            cand_rec = target_records.get(cid)
            if cand_rec is None:
                continue
            feats56 = extract_pair_features(
                s1_rec[1], s1_rec[2], s1_rec[3], cand_rec[1], cand_rec[2], cand_rec[3]
            )
            v1_tt = tt_s1_vecs.get(s1)
            v2_tt = tt_tgt_vecs.get(cid)
            cos_tt = float(np.dot(v1_tt, v2_tt)) if v1_tt is not None and v2_tt is not None else 0.0

            v1_zs = zs_s1_vecs.get(s1)
            v2_zs = zs_tgt_vecs.get(cid)
            cos_zs = float(np.dot(v1_zs, v2_zs)) if v1_zs is not None and v2_zs is not None else 0.0

            s1_out.append(s1)
            cand_out.append(cid)
            is_kw_out.append(is_kw)
            X56_list.append(feats56)
            cos_tt_list.append(cos_tt)
            cos_zs_list.append(cos_zs)

    if not s1_out:
        return (np.array([], dtype=object), np.array([], dtype=object), np.array([], dtype=bool),
                np.zeros((0, 57), dtype=np.float32), np.zeros((0, 57), dtype=np.float32))

    X56 = np.array(X56_list, dtype=np.float32)
    cos_tt = np.array(cos_tt_list, dtype=np.float32).reshape(-1, 1)
    cos_zs = np.array(cos_zs_list, dtype=np.float32).reshape(-1, 1)
    X57_tt = np.hstack([X56, cos_tt]).astype(np.float32)
    X57_zs = np.hstack([X56, cos_zs]).astype(np.float32)

    assert not np.any(np.isnan(X57_tt)), "NaN in two-tower feature matrix"
    assert not np.any(np.isinf(X57_tt)), "Inf in two-tower feature matrix"
    assert not np.any(np.isnan(X57_zs)), "NaN in zero-shot feature matrix"
    assert not np.any(np.isinf(X57_zs)), "Inf in zero-shot feature matrix"

    return (
        np.array(s1_out, dtype=object), np.array(cand_out, dtype=object), np.array(is_kw_out, dtype=bool),
        X57_tt, X57_zs,
    )


def heuristic_or_zero_prob(X57_tt: np.ndarray) -> np.ndarray:
    """
    Pure function reproducing scripts/run_optimal_eval_n011.py's heuristic
    (lines 38-47), UNCHANGED. Applies to ANN-new pairs only (caller's
    responsibility to mask). Uses two-tower cosine (feature index 56) --
    matches the original pipeline's usage exactly.
    """
    addr_jac = X57_tt[:, FEATURE_NAMES.index("addr_token_jaccard")]
    name_jac = X57_tt[:, FEATURE_NAMES.index("name_token_jaccard")]
    name_lev = X57_tt[:, FEATURE_NAMES.index("name_levenshtein_sim")]
    house_exact = X57_tt[:, FEATURE_NAMES.index("addr_house_num_exact")]
    neural_cos = X57_tt[:, 56]

    is_supp = (
        (house_exact == 1.0) & (addr_jac >= config.HEURISTIC_HOUSE_ADDR_JAC_MIN) & (neural_cos >= config.HEURISTIC_HOUSE_COS_MIN)
    ) | (
        (name_jac >= config.HEURISTIC_NAMEJAC_MIN) & (addr_jac >= config.HEURISTIC_NAMEJAC_ADDR_MIN) & (neural_cos >= config.HEURISTIC_NAMEJAC_COS_MIN)
    ) | (
        (name_lev >= config.HEURISTIC_NAMELEV_MIN) & (addr_jac >= config.HEURISTIC_NAMELEV_ADDR_MIN) & (neural_cos >= config.HEURISTIC_NAMELEV_COS_MIN)
    ) | (
        (addr_jac >= config.HEURISTIC_ADDRJAC_MIN) & (neural_cos >= config.HEURISTIC_ADDRJAC_COS_MIN)
    )
    prob = np.where(is_supp, np.minimum(config.HEURISTIC_PROB_CAP, neural_cos), 0.0).astype(np.float32)
    return prob


def score_pool(is_kw_pool: np.ndarray, X57_tt: np.ndarray, X57_zs: np.ndarray, lgb_model, xgb_model) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (p_lgb, p_xgb): real model inference (on zero-shot feature space,
    matching training) for is_kw_pool==True pairs; heuristic-or-zero
    (on two-tower feature space, matching the original pipeline) for
    is_kw_pool==False (ANN-new) pairs. UNCHANGED behavior vs authoritative
    N039 -- no real-everywhere scoring, no new heuristics.
    """
    n = len(is_kw_pool)
    p_lgb = np.zeros(n, dtype=np.float32)
    p_xgb = np.zeros(n, dtype=np.float32)

    if np.any(is_kw_pool):
        Xk = X57_zs[is_kw_pool]
        p_lgb[is_kw_pool] = lgb_model.predict(Xk)
        p_xgb[is_kw_pool] = xgb_model.predict(xgb.DMatrix(Xk))

    if np.any(~is_kw_pool):
        heur = heuristic_or_zero_prob(X57_tt[~is_kw_pool])
        p_lgb[~is_kw_pool] = heur
        p_xgb[~is_kw_pool] = heur

    return p_lgb, p_xgb


def xlmr_select_and_score(
    s1_ids: np.ndarray, cand_ids: np.ndarray, p_n014: np.ndarray,
    s1_records: Dict[str, Tuple], target_records: Dict[str, Tuple],
    device_log: list,
) -> np.ndarray:
    """
    Sparse selection rule (prepare_data_n017.py lines 184-199, UNCHANGED) +
    real XLM-R inference (predict_cross_encoder_n017.py, UNCHANGED) for the
    selected subset only. All other pairs get p_xlmr = 0.0 (matches the
    original pipeline's construction exactly).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    n = len(s1_ids)
    p_xlmr = np.zeros(n, dtype=np.float32)
    if n == 0:
        return p_xlmr

    unique_s1, s1_codes = np.unique(s1_ids, return_inverse=True)
    sort_key = np.lexsort((-p_n014, s1_codes))
    s1_codes_s = s1_codes[sort_key]
    p_s = p_n014[sort_key]

    boundaries = np.where(np.diff(s1_codes_s) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(p_s)]])

    select_mask_sorted = np.zeros(n, dtype=bool)
    for k in range(len(unique_s1)):
        sl = slice(starts[k], ends[k])
        ps = p_s[sl]
        if len(ps) > 0 and ps[0] >= config.XLMR_SELECT_TOP_GATE:
            c_mask = ps >= config.XLMR_SELECT_PAIR_THRESHOLD
            c_mask[0] = True
            select_mask_sorted[starts[k]:ends[k]][c_mask] = True

    select_mask = np.zeros(n, dtype=bool)
    select_mask[sort_key] = select_mask_sorted
    sel_idx = np.where(select_mask)[0]
    if len(sel_idx) == 0:
        return p_xlmr

    device = "cuda" if torch.cuda.is_available() else "cpu"  # matches predict_cross_encoder_n017.py's own fallback
    device_log.append(f"xlmr:{device}")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=1)
    model.load_state_dict(torch.load(config.XLMR_CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()

    def fmt(rec):
        return f"{rec[1]} [SEP] {rec[2]} [SEP] {rec[3]}"

    texts_a = [fmt(s1_records[s1_ids[i]]) for i in sel_idx]
    texts_b = [fmt(target_records[cand_ids[i]]) for i in sel_idx]

    with torch.no_grad():
        enc = tokenizer(texts_a, texts_b, max_length=config.XLMR_MAX_LENGTH, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        probs = torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy().astype(np.float32)

    p_xlmr[sel_idx] = probs
    return p_xlmr


def run_pipeline_for_entities(
    s1_records: Dict[str, Tuple],
    target_records: Dict[str, Tuple],
    training_edges: Dict[str, Set[str]],
    log: dict,
) -> Dict[str, Set[str]]:
    """
    Full pipeline for a bounded set of S1 entities against a bounded target
    pool. Returns {s1_id: set(matched_candidate_ids)}. Sibling recovery is
    explicitly NOT applied (logged via `log["sibling_recovery"]`).
    """
    t0 = time.time()
    device_log = []

    print(f"[pipeline] Building N009-equivalent keyword blocker index over {len(target_records):,} targets...", flush=True)
    kw_index = build_kw_index(target_records)
    kw_cands = keyword_candidates(s1_records, kw_index)
    log["kw_candidate_counts"] = {s1: len(c) for s1, c in kw_cands.items()}

    print("[pipeline] Computing two-tower embeddings (S1 + targets)...", flush=True)
    all_records_for_tt = {**s1_records, **target_records}
    tt_ids, tt_vecs = compute_two_tower_embeddings(all_records_for_tt, device_log)
    tt_map = {i: tt_vecs[k] for k, i in enumerate(tt_ids)}
    tt_s1_vecs = {s1: tt_map[s1] for s1 in s1_records if s1 in tt_map}
    tt_tgt_vecs = {c: tt_map[c] for c in target_records if c in tt_map}

    print("[pipeline] Running ANN retrieval (two-tower FAISS)...", flush=True)
    tgt_ids_list = [c for c in target_records if c in tt_tgt_vecs]
    tgt_vecs_arr = np.vstack([tt_tgt_vecs[c] for c in tgt_ids_list]) if tgt_ids_list else np.zeros((0, 256), dtype=np.float32)
    ann_cands = ann_candidates_per_country(s1_records, target_records, tt_s1_vecs, tgt_ids_list, tgt_vecs_arr)
    log["ann_candidate_counts"] = {s1: len(c) for s1, c in ann_cands.items()}

    pool = build_candidate_pool(s1_records, kw_cands, ann_cands)
    n_kw_pairs = sum(1 for cands in pool.values() for _, is_kw in cands if is_kw)
    n_ann_pairs = sum(1 for cands in pool.values() for _, is_kw in cands if not is_kw)
    log["n_kw_pool_pairs"] = n_kw_pairs
    log["n_ann_new_pairs"] = n_ann_pairs
    print(f"[pipeline] Candidate pool: {n_kw_pairs:,} N009-equivalent (keyword) pairs, {n_ann_pairs:,} ANN-new pairs", flush=True)

    print("[pipeline] Computing zero-shot MiniLM embeddings (for real LGB/XGB feature space)...", flush=True)
    from neural_encoder import NeuralEncoder
    zs_encoder = NeuralEncoder(device=config.select_device())
    zs_ids, zs_vecs = zs_encoder.encode_records(all_records_for_tt)
    zs_map = {i: zs_vecs[k] for k, i in enumerate(zs_ids)}
    zs_s1_vecs = {s1: zs_map[s1] for s1 in s1_records if s1 in zs_map}
    zs_tgt_vecs = {c: zs_map[c] for c in target_records if c in zs_map}

    print("[pipeline] Extracting 57-dim features...", flush=True)
    s1_ids, cand_ids, is_kw_pool, X57_tt, X57_zs = extract_features_for_pool(
        s1_records, target_records, pool, tt_s1_vecs, tt_tgt_vecs, zs_s1_vecs, zs_tgt_vecs
    )
    log["n_feature_rows"] = int(len(s1_ids))
    log["nan_check"] = "PASS (asserted inside extract_features_for_pool)"

    if len(s1_ids) == 0:
        log["sibling_recovery"] = "DISABLED (no-op) -- permanently excluded, oracle-dependent, non-transferable"
        return {s1: set() for s1 in s1_records}

    print("[pipeline] Scoring: real LGB/XGB for N009-pool pairs, heuristic/zero for ANN-new pairs...", flush=True)
    lgb_model = lgb.Booster(model_file=str(config.LGB_MODEL_PATH))
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(config.XGB_MODEL_PATH))
    p_lgb, p_xgb = score_pool(is_kw_pool, X57_tt, X57_zs, lgb_model, xgb_model)
    p_n014 = (config.W_LGB * p_lgb + config.W_XGB * p_xgb).astype(np.float32)
    log["p_n014_distribution"] = {
        "min": float(p_n014.min()), "p25": float(np.percentile(p_n014, 25)),
        "p50": float(np.percentile(p_n014, 50)), "p75": float(np.percentile(p_n014, 75)),
        "max": float(p_n014.max()),
    }

    print("[pipeline] XLM-R sparse selection + real inference...", flush=True)
    p_xlmr = xlmr_select_and_score(s1_ids, cand_ids, p_n014, s1_records, target_records, device_log)
    p_hyb = (config.W_N014 * p_n014 + config.W_XLMR * p_xlmr).astype(np.float32)
    log["p_hyb_distribution"] = {
        "min": float(p_hyb.min()), "p25": float(np.percentile(p_hyb, 25)),
        "p50": float(np.percentile(p_hyb, 50)), "p75": float(np.percentile(p_hyb, 75)),
        "max": float(p_hyb.max()),
    }
    log["xlmr_selected_count"] = int(np.sum(p_xlmr != 0.0))
    log["device_log"] = device_log

    print("[pipeline] Building sorted per-entity arrays for threshold/decoder...", flush=True)
    unique_s1 = np.array(sorted(s1_records.keys()))
    K = config.FINAL_POOL_CAP_K
    N_ent = len(unique_s1)
    s1_to_row = {s1: i for i, s1 in enumerate(unique_s1)}

    cand_mat = np.full((N_ent, K), "", dtype=object)
    src_mat = np.full((N_ent, K), "", dtype=object)
    X_mat = np.zeros((N_ent, K, 57), dtype=np.float32)
    p_mat = np.zeros((N_ent, K), dtype=np.float32)
    valid_mat = np.zeros((N_ent, K), dtype=bool)

    row_fill_count = defaultdict(int)
    for i in range(len(s1_ids)):
        s1 = s1_ids[i]
        row = s1_to_row[s1]
        col = row_fill_count[s1]
        if col >= K:
            continue
        cand_mat[row, col] = cand_ids[i]
        src_mat[row, col] = "S2" if cand_ids[i].startswith("S2-") else "S3"
        X_mat[row, col] = X57_tt[i]
        p_mat[row, col] = p_hyb[i]
        valid_mat[row, col] = True
        row_fill_count[s1] += 1

    # Pad probabilities of unfilled slots to -inf so they sort last and are never admitted.
    p_mat_for_sort = np.where(valid_mat, p_mat, -np.inf)
    sort_idx = np.argsort(-p_mat_for_sort, axis=1)
    row_idx = np.arange(N_ent)[:, None]
    p_s = p_mat[row_idx, sort_idx]
    X_s = X_mat[row_idx, sort_idx]
    c_s = cand_mat[row_idx, sort_idx]
    src_s = src_mat[row_idx, sort_idx]
    valid_s = valid_mat[row_idx, sort_idx]
    p_s = np.where(valid_s, p_s, 0.0)

    print("[pipeline] Applying threshold/admission (n039_no_frontier_decision, zero ground-truth dependency)...", flush=True)
    pred_final = compute_authoritative_no_frontier_predictions(p_s, X_s, c_s, src_s, training_edges)
    pred_final = pred_final & valid_s  # never admit a padding slot

    log["sibling_recovery"] = "DISABLED (no-op) -- permanently excluded, oracle-dependent, non-transferable (see reports/N039_NO_FRONTIER_MANIFEST.md and sibling-recovery investigation)"

    results: Dict[str, Set[str]] = {}
    for i, s1 in enumerate(unique_s1):
        matched = set(c_s[i][pred_final[i]])
        matched.discard("")
        results[s1] = matched

    log["pipeline_seconds"] = time.time() - t0
    return results


def write_matching_results_tsv(results: Dict[str, Set[str]], out_path: Path, all_s1_order: List[str]):
    """Writes matching_results.tsv in the required format, preserving S1 order,
    including empty rows for entities with zero matches."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in all_s1_order:
            matched = results.get(s1, set())
            f.write(f"{s1}\t{','.join(sorted(matched))}\n")
