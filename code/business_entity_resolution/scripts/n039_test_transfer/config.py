#!/usr/bin/env python3
"""
Configuration for the N039 test-transfer pipeline.

Layout of this directory (all new files, additive, none overwrite anything
existing):
  config.py                    -- this file: paths, params, device selection
  device_utils.py               -- MPS/CPU-preferring device wrapper for
                                    src/two_tower_encoder.py (does not modify
                                    that file's own hardcoded "cuda" defaults)
  provenance_guard.py           -- runtime file-open guard (requirement F)
  pipeline.py                   -- main orchestrator (requirement A)
  output_validator.py           -- requirement E
  resource_estimator.py         -- requirement D
  smoke_test.py                 -- requirement C
  validation_transfer_smoke.py  -- validation-transfer parity smoke check
"""

from __future__ import annotations

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_DIR.parents[1]
SRC_DIR = CODE_DIR / "src"
SCRIPTS_DIR = CODE_DIR / "scripts"
EXP_DIR = CODE_DIR / "experiments"
MODELS_DIR = CODE_DIR / "models"
REPORTS_DIR = CODE_DIR / "reports"

# This module's own scratch/output area (new, additive)
TRANSFER_DIR = Path(__file__).resolve().parent
TRANSFER_OUT_DIR = EXP_DIR / "N039_TEST_TRANSFER"

TRAIN_DATA_DIR = PROJECT_ROOT / "dataset" / "train"
TEST_DATA_DIR = PROJECT_ROOT / "dataset" / "test"

# Files that must NEVER be opened during a test-mode run (enforced by
# provenance_guard.py, not just documented here).
FORBIDDEN_LABEL_FILES = {
    str(TRAIN_DATA_DIR / "train_ground_truth.tsv"),
}

# ---------------------------------------------------------------------------
# Model / checkpoint paths (all existing, frozen artifacts -- read-only reuse)
# ---------------------------------------------------------------------------
LGB_MODEL_PATH = EXP_DIR / "lgb_model_n009.txt"
XGB_MODEL_PATH = EXP_DIR / "xgb_model_n009.json"
TWO_TOWER_CHECKPOINT = MODELS_DIR / "two_tower_stageC.pt"
XLMR_CHECKPOINT = EXP_DIR / "EXP_N017" / "xlm_roberta_cross_encoder.pt"

# ---------------------------------------------------------------------------
# Candidate generation params (as identified in the Test Transfer Spec /
# generate_val_pairs_n009.py / generate_val_pairs_n010.py -- UNCHANGED values)
# ---------------------------------------------------------------------------
UNION_E_MAX_BUCKET_SIZE = 1000
KEYWORD_MAX_CANDIDATES_PER_S1 = 200   # matches generate_val_pairs_n009.py's cap
ANN_TOP_K = 120                        # matches generate_val_pairs_n010.py's per-country FAISS search
FINAL_POOL_CAP_K = 80                  # matches the frozen 80-candidate/entity pool shape

# ---------------------------------------------------------------------------
# Scoring blend weights (UNCHANGED, frozen)
# ---------------------------------------------------------------------------
W_LGB = 0.45
W_XGB = 0.55
W_N014 = 0.73
W_XLMR = 0.27

# XLM-R sparse-selection rule (prepare_data_n017.py, UNCHANGED)
XLMR_SELECT_TOP_GATE = 0.50
XLMR_SELECT_PAIR_THRESHOLD = 0.70
XLMR_MAX_LENGTH = 96

# N011 heuristic thresholds (run_optimal_eval_n011.py, UNCHANGED)
HEURISTIC_HOUSE_ADDR_JAC_MIN = 0.20
HEURISTIC_HOUSE_COS_MIN = 0.60
HEURISTIC_NAMEJAC_MIN = 0.45
HEURISTIC_NAMEJAC_ADDR_MIN = 0.20
HEURISTIC_NAMEJAC_COS_MIN = 0.60
HEURISTIC_NAMELEV_MIN = 0.75
HEURISTIC_NAMELEV_ADDR_MIN = 0.15
HEURISTIC_NAMELEV_COS_MIN = 0.55
HEURISTIC_ADDRJAC_MIN = 0.50
HEURISTIC_ADDRJAC_COS_MIN = 0.55
HEURISTIC_PROB_CAP = 0.99

# Threshold/admission logic: reuse src/n039_no_frontier_decision.py as-is
# (no frontier-ranker, zero ground-truth dependency). Not duplicated here.

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def select_device() -> str:
    """MPS-then-CPU-preferring device selection (requirement: patch the
    CUDA-hardcoded default without touching two_tower_encoder.py itself)."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Country partitions to process independently (streaming/chunking boundary)
COUNTRIES = ["US", "India", "France"]

# Sibling recovery: PERMANENTLY DISABLED for test transfer (oracle-dependent,
# non-transferable -- see reports/N039_NO_FRONTIER_MANIFEST.md and the
# sibling-recovery investigation). This is a fixed, documented decision, not
# a per-run toggle.
SIBLING_RECOVERY_ENABLED = False
