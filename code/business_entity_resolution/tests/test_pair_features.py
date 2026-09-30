"""
Unit tests for Pairwise Feature Extraction Engine.
"""

import sys
import os
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pair_features import (
    FEATURE_NAMES,
    FEATURE_GROUPS,
    extract_pair_features,
)


def test_feature_names_integrity():
    """Verify feature names list has unique identifiers and no empty names."""
    assert len(FEATURE_NAMES) > 40
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    for name in FEATURE_NAMES:
        assert isinstance(name, str) and len(name) > 0


def test_exact_match_pair():
    """Verify feature extraction for an identical business pair."""
    name = "Acme Global Solutions LLC"
    addr = "123 Main Street, Suite 400, Springfield, IL 62701"
    country = "US"

    feats = extract_pair_features(name, addr, country, name, addr, country)
    assert len(feats) == len(FEATURE_NAMES)

    # Check for no NaNs or Infs
    arr = np.array(feats, dtype=np.float32)
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()

    feat_map = dict(zip(FEATURE_NAMES, feats))
    assert feat_map["name_raw_exact"] == 1.0
    assert feat_map["name_norm_exact"] == 1.0
    assert feat_map["name_levenshtein_sim"] == 1.0
    assert feat_map["addr_norm_exact"] == 1.0
    assert feat_map["addr_postal_exact"] == 1.0
    assert feat_map["addr_house_num_exact"] == 1.0
    assert feat_map["inter_both_strong"] == 1.0
    assert feat_map["is_s1_addr_missing"] == 0.0
    assert feat_map["is_cand_addr_missing"] == 0.0


def test_missing_address_graceful_handling():
    """Verify that missing address in one or both does not raise exception or NaNs."""
    feats = extract_pair_features(
        "Shree Ganesh Traders", "", "India",
        "Shree Ganesh Traders Pvt Ltd", "Shop 4, MG Road, Mumbai 400001", "India"
    )
    assert len(feats) == len(FEATURE_NAMES)
    arr = np.array(feats, dtype=np.float32)
    assert not np.isnan(arr).any()

    feat_map = dict(zip(FEATURE_NAMES, feats))
    assert feat_map["is_s1_addr_missing"] == 1.0
    assert feat_map["is_cand_addr_missing"] == 0.0
    assert feat_map["name_norm_exact"] == 1.0
    # Missing address should NOT make it crash or produce NaNs
    assert feat_map["addr_levenshtein_sim"] == 0.0


def test_transliteration_indic_pair():
    """Verify Indic vs Latin pair feature extraction."""
    # Devanagari "होटल ताज" vs "Hotel Taj"
    feats = extract_pair_features(
        "होटल ताज", "गेटवे ऑफ इंडिया, मुंबई", "India",
        "Hotel Taj", "Gateway of India, Mumbai", "India"
    )
    arr = np.array(feats, dtype=np.float32)
    assert not np.isnan(arr).any()

    feat_map = dict(zip(FEATURE_NAMES, feats))
    assert feat_map["name_script_compatible"] == 0.0
    assert feat_map["name_translit_sim"] > 0.60


def test_completely_different_pair():
    """Verify features for completely unrelated entities."""
    feats = extract_pair_features(
        "Apex Medical Clinic", "404 Health Ave, Chicago, IL 60601", "US",
        "Zeta Cloud Analytics", "99 Tech Blvd, Seattle, WA 98101", "US"
    )
    feat_map = dict(zip(FEATURE_NAMES, feats))
    assert feat_map["name_norm_exact"] == 0.0
    assert feat_map["name_levenshtein_sim"] < 0.4
    assert feat_map["addr_postal_exact"] == 0.0
    assert feat_map["inter_both_weak"] == 1.0
