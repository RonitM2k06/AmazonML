"""
Pairwise Feature Extraction Engine for Business Entity Resolution.
Extracts fine-grained Name, Address, Phonetic/Transliteration, Postal/House,
and Cross-Field Interaction features for (Source 1, Candidate) record pairs.
"""

import math
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from rapidfuzz import distance, fuzz

from normalization import (
    normalize_unicode,
    normalize_whitespace,
    normalize_lowercase,
    strip_punctuation,
    normalize_alphanumeric,
    strip_legal_suffixes,
    normalize_business_name,
    expand_address_abbreviations,
    normalize_address,
    extract_postal_codes,
    extract_numbers,
    tokenize,
    token_jaccard,
    token_overlap,
    character_similarity,
    token_set_similarity,
    token_sort_similarity,
)
from blocking import (
    transliterate_indic,
    clean_business_name,
    get_consonant_skeleton,
    GENERIC_WORDS,
    ADDRESS_STOPWORDS,
)

# Indic Unicode detection regex
RE_INDIC = re.compile(r"[\u0900-\u0D7F]")
RE_DIGITS = re.compile(r"\d")


def _get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity."""
    ng1 = _get_char_ngrams(s1, n)
    ng2 = _get_char_ngrams(s2, n)
    if not ng1 and not ng2:
        return 1.0
    if not ng1 or not ng2:
        return 0.0
    return float(len(ng1.intersection(ng2)) / len(ng1.union(ng2)))


def _extract_acronym(tokens: List[str]) -> str:
    """Extract initials/acronym from list of tokens."""
    return "".join(t[0] for t in tokens if t and t[0].isalnum()).lower()


# Complete ordered list of feature names
FEATURE_NAMES: List[str] = [
    # --- NAME FEATURES (27) ---
    "name_raw_exact",
    "name_norm_exact",
    "name_lower_exact",
    "name_punct_strip_exact",
    "name_alnum_exact",
    "name_legal_stripped_exact",
    "name_levenshtein_sim",
    "name_edit_distance",
    "name_jaro_winkler",
    "name_ngram3_sim",
    "name_ngram4_sim",
    "name_token_jaccard",
    "name_token_overlap_ratio",
    "name_token_containment",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_common_token_count",
    "name_rare_token_overlap",
    "name_rare_token_weighted_sim",
    "name_len_ratio",
    "name_token_count_diff",
    "name_mean_token_len_diff",
    "name_word_order_sim",
    "name_acronym_match",
    "name_fused_exact",
    "name_script_compatible",
    "name_translit_sim",
    "name_skeleton_overlap",
    "name_skeleton_exact",
    # --- ADDRESS FEATURES (17) ---
    "addr_norm_exact",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_levenshtein_sim",
    "addr_token_set_ratio",
    "addr_len_ratio",
    "addr_token_count_diff",
    "addr_digit_overlap",
    "addr_postal_exact",
    "addr_postal_prefix2",
    "addr_postal_prefix3",
    "addr_house_num_exact",
    "addr_street_overlap",
    "addr_rare_token_overlap",
    "is_s1_addr_missing",
    "is_cand_addr_missing",
    "is_either_addr_short",
    # --- CROSS-FIELD INTERACTION FEATURES (10) ---
    "inter_name_x_addr",
    "inter_max_name_addr",
    "inter_min_name_addr",
    "inter_name_strong_addr_weak",
    "inter_name_weak_addr_strong",
    "inter_both_strong",
    "inter_both_weak",
    "inter_postal_x_name",
    "inter_house_x_name",
    "inter_translit_x_addr",
]

FEATURE_GROUPS = {
    "name_only": [f for f in FEATURE_NAMES if f.startswith("name_")],
    "addr_only": [f for f in FEATURE_NAMES if f.startswith("addr_") or f.startswith("is_")],
    "name_plus_addr": [
        f for f in FEATURE_NAMES if not f.startswith("inter_") and not f.startswith("addr_postal") and not f.startswith("addr_house") and not f.endswith("rare_token_overlap") and not f.endswith("translit_sim")
    ],
    "postal_house": [
        f for f in FEATURE_NAMES if "postal" in f or "house" in f
    ],
    "rare_tokens": [
        f for f in FEATURE_NAMES if "rare" in f
    ],
    "transliteration": [
        f for f in FEATURE_NAMES if "translit" in f or "script" in f
    ],
    "interaction": [
        f for f in FEATURE_NAMES if f.startswith("inter_")
    ],
    "full": FEATURE_NAMES,
}


def extract_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    cand_name: str,
    cand_addr: str,
    cand_country: str,
) -> List[float]:
    """
    Extract deterministic feature vector for an entity pair.
    Returns list of float values matching FEATURE_NAMES.
    Guaranteed zero NaNs, Infs, or exceptions.
    """
    s1_name_raw = s1_name or ""
    cand_name_raw = cand_name or ""
    s1_addr_raw = s1_addr or ""
    cand_addr_raw = cand_addr or ""

    # Normalizations
    s1_name_norm, s1_name_toks = clean_business_name(s1_name_raw)
    cand_name_norm, cand_name_toks = clean_business_name(cand_name_raw)

    s1_lower = normalize_lowercase(s1_name_raw)
    cand_lower = normalize_lowercase(cand_name_raw)

    s1_punct_strip = strip_punctuation(s1_name_raw)
    cand_punct_strip = strip_punctuation(cand_name_raw)

    s1_alnum = normalize_alphanumeric(s1_name_raw)
    cand_alnum = normalize_alphanumeric(cand_name_raw)

    s1_legal_stripped = strip_legal_suffixes(s1_lower)
    cand_legal_stripped = strip_legal_suffixes(cand_lower)

    # 1. Exact Equality Name Features
    name_raw_exact = float(s1_name_raw.strip() == cand_name_raw.strip() and len(s1_name_raw) > 0)
    name_norm_exact = float(s1_name_norm == cand_name_norm and len(s1_name_norm) > 0)
    name_lower_exact = float(s1_lower == cand_lower and len(s1_lower) > 0)
    name_punct_strip_exact = float(s1_punct_strip == cand_punct_strip and len(s1_punct_strip) > 0)
    name_alnum_exact = float(s1_alnum == cand_alnum and len(s1_alnum) > 0)
    name_legal_stripped_exact = float(s1_legal_stripped == cand_legal_stripped and len(s1_legal_stripped) > 0)

    # 2. String Similarities
    name_levenshtein_sim = character_similarity(s1_name_norm, cand_name_norm)
    name_edit_distance = float(distance.Levenshtein.distance(s1_name_norm, cand_name_norm))
    name_jaro_winkler = float(distance.JaroWinkler.similarity(s1_name_norm, cand_name_norm))
    name_ngram3_sim = _ngram_jaccard(s1_name_norm, cand_name_norm, 3)
    name_ngram4_sim = _ngram_jaccard(s1_name_norm, cand_name_norm, 4)

    # 3. Token Similarities
    name_token_jaccard = token_jaccard(s1_name_toks, cand_name_toks)
    name_token_overlap_ratio = token_overlap(s1_name_toks, cand_name_toks)

    s1_set = set(s1_name_toks)
    cand_set = set(cand_name_toks)
    common_toks = s1_set.intersection(cand_set)
    name_common_token_count = float(len(common_toks))
    name_token_containment = float(s1_set.issubset(cand_set) or cand_set.issubset(s1_set)) if s1_set and cand_set else 0.0

    name_token_sort_ratio = token_sort_similarity(s1_name_norm, cand_name_norm)
    name_token_set_ratio = token_set_similarity(s1_name_norm, cand_name_norm)

    # Rare token overlap (tokens with len >= 6 and not generic)
    s1_rare = {t for t in s1_name_toks if len(t) >= 6 and t not in GENERIC_WORDS}
    cand_rare = {t for t in cand_name_toks if len(t) >= 6 and t not in GENERIC_WORDS}
    rare_common = s1_rare.intersection(cand_rare)
    name_rare_token_overlap = float(len(rare_common))
    name_rare_token_weighted_sim = float(len(rare_common) / max(1, len(s1_rare.union(cand_rare)))) if s1_rare or cand_rare else 0.0

    # Length statistics
    l1 = len(s1_name_norm)
    l2 = len(cand_name_norm)
    name_len_ratio = float(min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 1.0
    name_token_count_diff = float(abs(len(s1_name_toks) - len(cand_name_toks)))

    m1 = np.mean([len(t) for t in s1_name_toks]) if s1_name_toks else 0.0
    m2 = np.mean([len(t) for t in cand_name_toks]) if cand_name_toks else 0.0
    name_mean_token_len_diff = float(abs(m1 - m2))

    # Word order & Acronym
    name_word_order_sim = float(s1_name_toks == cand_name_toks and len(s1_name_toks) > 0)
    acronym1 = _extract_acronym(s1_name_toks)
    acronym2 = _extract_acronym(cand_name_toks)
    name_acronym_match = float(
        (acronym1 == cand_name_norm and len(acronym1) >= 2)
        or (acronym2 == s1_name_norm and len(acronym2) >= 2)
        or (acronym1 == acronym2 and len(acronym1) >= 2)
    )

    fused1 = "".join(s1_name_toks)
    fused2 = "".join(cand_name_toks)
    name_fused_exact = float(fused1 == fused2 and len(fused1) >= 5)

    # Script & Transliteration
    s1_has_indic = bool(RE_INDIC.search(s1_name_raw))
    cand_has_indic = bool(RE_INDIC.search(cand_name_raw))
    name_script_compatible = float(s1_has_indic == cand_has_indic)

    if s1_has_indic or cand_has_indic:
        trans1 = transliterate_indic(s1_name_raw)
        trans2 = transliterate_indic(cand_name_raw)
        name_translit_sim = character_similarity(normalize_business_name(trans1), normalize_business_name(trans2))
    else:
        name_translit_sim = name_levenshtein_sim

    # Consonant phonetic skeleton matching (invariant across loanwords and scripts)
    s1_skels = {get_consonant_skeleton(t) for t in s1_name_toks if t not in GENERIC_WORDS and len(t) >= 3}
    cand_skels = {get_consonant_skeleton(t) for t in cand_name_toks if t not in GENERIC_WORDS and len(t) >= 3}
    if s1_has_indic or cand_has_indic:
        _, trans_s1_toks = clean_business_name(transliterate_indic(s1_name_raw))
        _, trans_c_toks = clean_business_name(transliterate_indic(cand_name_raw))
        s1_skels.update(get_consonant_skeleton(t) for t in trans_s1_toks if t not in GENERIC_WORDS and len(t) >= 3)
        cand_skels.update(get_consonant_skeleton(t) for t in trans_c_toks if t not in GENERIC_WORDS and len(t) >= 3)

    skel_common = s1_skels.intersection(cand_skels)
    name_skeleton_overlap = float(len(skel_common))
    name_skeleton_exact = float(len(skel_common) > 0 and s1_skels == cand_skels)

    # --- ADDRESS FEATURES ---
    s1_addr_norm = normalize_address(s1_addr_raw)
    cand_addr_norm = normalize_address(cand_addr_raw)

    is_s1_addr_missing = float(not s1_addr_norm)
    is_cand_addr_missing = float(not cand_addr_norm)
    is_either_addr_short = float(len(s1_addr_norm) < 8 or len(cand_addr_norm) < 8)

    if is_s1_addr_missing or is_cand_addr_missing:
        # Graceful defaults for missing address
        addr_norm_exact = 0.0
        addr_token_jaccard = 0.0
        addr_token_overlap = 0.0
        addr_levenshtein_sim = 0.0
        addr_token_set_ratio = 0.0
        addr_len_ratio = 0.0
        addr_token_count_diff = 0.0
        addr_digit_overlap = 0.0
        addr_postal_exact = 0.0
        addr_postal_prefix2 = 0.0
        addr_postal_prefix3 = 0.0
        addr_house_num_exact = 0.0
        addr_street_overlap = 0.0
        addr_rare_token_overlap = 0.0
    else:
        addr_norm_exact = float(s1_addr_norm == cand_addr_norm)
        s1_addr_toks = tokenize(s1_addr_norm)
        cand_addr_toks = tokenize(cand_addr_norm)

        addr_token_jaccard = token_jaccard(s1_addr_toks, cand_addr_toks)
        addr_token_overlap = token_overlap(s1_addr_toks, cand_addr_toks)
        addr_levenshtein_sim = character_similarity(s1_addr_norm, cand_addr_norm)
        addr_token_set_ratio = token_set_similarity(s1_addr_norm, cand_addr_norm)

        al1 = len(s1_addr_norm)
        al2 = len(cand_addr_norm)
        addr_len_ratio = float(min(al1, al2) / max(al1, al2)) if max(al1, al2) > 0 else 1.0
        addr_token_count_diff = float(abs(len(s1_addr_toks) - len(cand_addr_toks)))

        # Digit overlap
        d1 = set(RE_DIGITS.findall(s1_addr_norm))
        d2 = set(RE_DIGITS.findall(cand_addr_norm))
        addr_digit_overlap = float(len(d1.intersection(d2)) / max(1, len(d1.union(d2)))) if d1 or d2 else 1.0

        # Postal code features
        p1_list = extract_postal_codes(s1_addr_raw)
        p2_list = extract_postal_codes(cand_addr_raw)
        if p1_list and p2_list:
            p1 = p1_list[0]
            p2 = p2_list[0]
            addr_postal_exact = float(p1 == p2)
            addr_postal_prefix2 = float(p1[:2] == p2[:2])
            addr_postal_prefix3 = float(p1[:3] == p2[:3])
        else:
            addr_postal_exact = 0.0
            addr_postal_prefix2 = 0.0
            addr_postal_prefix3 = 0.0

        # House number features
        nums1 = extract_numbers(s1_addr_norm)
        nums2 = extract_numbers(cand_addr_norm)
        addr_house_num_exact = float(bool(nums1 and nums2 and nums1[0] == nums2[0]))

        # Street & Rare Address Tokens
        street_toks1 = {t for t in s1_addr_toks if t not in ADDRESS_STOPWORDS and len(t) >= 4}
        street_toks2 = {t for t in cand_addr_toks if t not in ADDRESS_STOPWORDS and len(t) >= 4}
        street_common = street_toks1.intersection(street_toks2)
        addr_street_overlap = float(len(street_common) / max(1, min(len(street_toks1), len(street_toks2)))) if street_toks1 and street_toks2 else 0.0

        rare_addr1 = {t for t in street_toks1 if len(t) >= 7}
        rare_addr2 = {t for t in street_toks2 if len(t) >= 7}
        addr_rare_token_overlap = float(len(rare_addr1.intersection(rare_addr2)))

    # --- CROSS-FIELD INTERACTIONS ---
    primary_name_sim = max(name_levenshtein_sim, name_token_set_ratio, name_translit_sim)
    primary_addr_sim = max(addr_levenshtein_sim, addr_token_set_ratio)

    inter_name_x_addr = primary_name_sim * primary_addr_sim
    inter_max_name_addr = max(primary_name_sim, primary_addr_sim)
    inter_min_name_addr = min(primary_name_sim, primary_addr_sim)

    inter_name_strong_addr_weak = float(primary_name_sim >= 0.85 and primary_addr_sim < 0.35)
    inter_name_weak_addr_strong = float(primary_name_sim < 0.50 and primary_addr_sim >= 0.85)
    inter_both_strong = float(primary_name_sim >= 0.80 and primary_addr_sim >= 0.80)
    inter_both_weak = float(primary_name_sim < 0.50 and primary_addr_sim < 0.40)

    inter_postal_x_name = addr_postal_exact * primary_name_sim
    inter_house_x_name = addr_house_num_exact * primary_name_sim
    inter_translit_x_addr = name_translit_sim * primary_addr_sim

    return [
        name_raw_exact,
        name_norm_exact,
        name_lower_exact,
        name_punct_strip_exact,
        name_alnum_exact,
        name_legal_stripped_exact,
        name_levenshtein_sim,
        name_edit_distance,
        name_jaro_winkler,
        name_ngram3_sim,
        name_ngram4_sim,
        name_token_jaccard,
        name_token_overlap_ratio,
        name_token_containment,
        name_token_sort_ratio,
        name_token_set_ratio,
        name_common_token_count,
        name_rare_token_overlap,
        name_rare_token_weighted_sim,
        name_len_ratio,
        name_token_count_diff,
        name_mean_token_len_diff,
        name_word_order_sim,
        name_acronym_match,
        name_fused_exact,
        name_script_compatible,
        name_translit_sim,
        name_skeleton_overlap,
        name_skeleton_exact,
        addr_norm_exact,
        addr_token_jaccard,
        addr_token_overlap,
        addr_levenshtein_sim,
        addr_token_set_ratio,
        addr_len_ratio,
        addr_token_count_diff,
        addr_digit_overlap,
        addr_postal_exact,
        addr_postal_prefix2,
        addr_postal_prefix3,
        addr_house_num_exact,
        addr_street_overlap,
        addr_rare_token_overlap,
        is_s1_addr_missing,
        is_cand_addr_missing,
        is_either_addr_short,
        inter_name_x_addr,
        inter_max_name_addr,
        inter_min_name_addr,
        inter_name_strong_addr_weak,
        inter_name_weak_addr_strong,
        inter_both_strong,
        inter_both_weak,
        inter_postal_x_name,
        inter_house_x_name,
        inter_translit_x_addr,
    ]
