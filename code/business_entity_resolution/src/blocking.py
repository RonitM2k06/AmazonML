"""
Candidate Generation and Blocking Module for Amazon ML Challenge 2026.
Implements high-recall, memory-efficient multi-pass blocking partitioned by country.
Includes unified Indic multi-script transliteration, domain name normalization,
two-token signatures, address signatures, and controlled n-gram blocking.
"""

import os
import sys
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np

# Import normalization functions

from normalization import (
    normalize_unicode,
    normalize_lowercase,
    normalize_whitespace,
    strip_punctuation,
    normalize_alphanumeric,
    strip_legal_suffixes,
    normalize_business_name,
    expand_address_abbreviations,
    normalize_address,
    extract_postal_codes,
    extract_numbers,
    tokenize,
)

# Address Stopwords to prevent bucket explosion
ADDRESS_STOPWORDS = {
    "street", "road", "rd", "st", "ave", "avenue", "lane", "ln", "dr", "drive",
    "blvd", "boulevard", "hwy", "highway", "ct", "court", "pl", "place", "sq",
    "square", "ste", "suite", "apt", "apartment", "fl", "floor", "bldg", "building",
    "dept", "department", "near", "opp", "opposite", "behind", "beside", "adjacent",
    "nagar", "colony", "cross", "main", "sector", "phase", "marg", "dist",
    "district", "city", "state", "post", "box", "po", "pob", "pobox", "block",
    "unit", "house", "no", "number", "kh", "plot", "area", "bazar", "chowk",
}

# Generic High-Frequency Words to avoid in single-token blocks
GENERIC_WORDS = {
    "inc", "llc", "corp", "ltd", "pvt", "limited", "company", "the", "and", "services",
    "solutions", "enterprises", "street", "road", "avenue", "north", "south", "east", "west",
    "center", "centre", "national", "global", "general", "medical", "dental", "health", "care",
    "hospital", "clinic", "pharma", "pharmacy", "hotel", "restaurant", "store", "shop", "market",
    "associates", "consulting", "group", "holdings", "international", "industries", "technologies",
    "financial", "insurance", "management", "logistics", "transport", "builders", "construction",
    "shri", "om", "sai", "sri", "jai", "maa", "balaji", "shree", "new", "dr", "mr", "ms",
    "india", "american", "bharat", "brothers", "sons", "agency", "agencies", "trading",
}

# Domain extension stripper
RE_DOMAIN = re.compile(r"\.(com|in|org|net|co|io|biz|info|us|fr|gov|edu)\b", re.IGNORECASE)

# Universal Indic Unicode phonetic mapping across all Indian scripts:
# Devanagari (0x0900), Bengali (0x0980), Gurmukhi (0x0A00), Gujarati (0x0A80),
# Tamil (0x0B80), Telugu (0x0C00), Kannada (0x0C80), Malayalam (0x0D00).
INDIC_OFFSETS = {
    0x02: 'n', 0x03: 'h', 0x04: 'a',
    0x05: 'a', 0x06: 'aa', 0x07: 'i', 0x08: 'ee', 0x09: 'u', 0x0A: 'oo',
    0x0B: 'ri', 0x0C: 'li', 0x0D: 'e',
    0x0E: 'e', 0x0F: 'e', 0x10: 'ai', 0x11: 'o', 0x12: 'o', 0x13: 'o', 0x14: 'au',
    0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'ng',
    0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j', 0x1D: 'jh', 0x1E: 'ny',
    0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n',
    0x24: 't', 0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n',
    0x2A: 'p', 0x2B: 'ph', 0x2C: 'b', 0x2D: 'bh', 0x2E: 'm',
    0x2F: 'y', 0x30: 'r', 0x32: 'l', 0x33: 'l', 0x35: 'v',
    0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h',
    0x3E: 'a', 0x3F: 'i', 0x40: 'ee', 0x41: 'u', 0x42: 'oo',
    0x43: 'ri', 0x44: 'ri', 0x45: 'e',
    0x46: 'e', 0x47: 'e', 0x48: 'ai', 0x49: 'o', 0x4A: 'o', 0x4B: 'o', 0x4C: 'au',
    0x4D: '',
}


def transliterate_indic(text: str) -> str:
    """Universal transliteration of any Indic script (Hindi, Gujarati, Kannada, etc.) to Latin."""
    if not text:
        return ""
    res = []
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            block_offset = cp % 0x80
            if block_offset in INDIC_OFFSETS:
                res.append(INDIC_OFFSETS[block_offset])
        elif cp < 128:
            res.append(ch)
    return "".join(res)


def get_consonant_skeleton(word: str) -> str:
    """
    Compute phonetic consonant skeleton invariant to cross-lingual transliterations and loanwords.
    Normalizes digraphs (ph->f, th->t, kh->k, ch->c), collapses vowels and duplicate consonants.
    """
    if not word:
        return ""
    w = word.lower()
    replacements = [
        ("sh", "s"), ("ph", "f"), ("th", "t"),
        ("kh", "k"), ("gh", "g"), ("dh", "d"), ("bh", "b"),
        ("jh", "j"), ("ck", "k"), ("qu", "k")
    ]
    for src, dst in replacements:
        w = w.replace(src, dst)
    w = w.replace("ch", "c")
    w = w.replace("c", "k").replace("q", "k").replace("x", "ks").replace("z", "s")
    first = w[0]
    rest = re.sub(r"[aeiouy]", "", w[1:])
    return re.sub(r"(.)\1+", r"\1", first + rest)


# Alias for backward compatibility
transliterate_hindi = transliterate_indic


def get_soundex(token: str) -> str:
    """Compute American Soundex code for a single word token."""
    if not token or not token.isalpha():
        return ""
    token = token.upper()
    soundex_mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    first_char = token[0]
    encoded = [first_char]
    for ch in token[1:]:
        code = soundex_mapping.get(ch, "")
        if code and code != encoded[-1]:
            encoded.append(code)
    digits = [c for c in encoded[1:] if c.isdigit()]
    result = first_char + "".join(digits)
    return (result + "000")[:4]


def clean_business_name(name: str) -> Tuple[str, List[str]]:
    """Clean business name, strip legal suffixes & domains, and return normalized string + tokens."""
    if not name:
        return "", []
    # Strip domain extensions like .com, .in
    text = RE_DOMAIN.sub(" ", name)
    norm = normalize_business_name(text, strip_legal=True)
    tokens = [t for t in tokenize(norm) if t not in GENERIC_WORDS and len(t) >= 2]
    if not tokens:
        tokens = tokenize(norm)
    return norm, tokens


def extract_blocking_keys(
    name: str,
    address: str,
    country: str,
    strategies: Set[str],
) -> List[Tuple]:
    """
    Extract discrete blocking keys for a record based on requested strategies.
    Every key is scoped by country: (country, strategy_name, key_components...).
    """
    keys = []
    if not country:
        country = "UNKNOWN"

    norm_name, name_tokens = clean_business_name(name)
    postals = extract_postal_codes(address) if address else []
    addr_numbers = extract_numbers(address) if address else []
    h_nums = []
    if addr_numbers:
        for hn in addr_numbers[:4]:
            h_nums.append(hn)
            hn_clean = hn.lstrip("0")
            if hn_clean and hn_clean != hn:
                h_nums.append(hn_clean)

    # Complex door/plot pattern (e.g. 8-2-412 or E-5/140)
    complex_codes = []
    if address:
        raw_codes = re.findall(r"[A-Za-z0-9]+[/\-\.][A-Za-z0-9]+(?:[/\-\.][A-Za-z0-9]+)?", address)
        for rc in raw_codes[:3]:
            clean_rc = re.sub(r"[^a-zA-Z0-9]", "", rc).lower()
            if len(clean_rc) >= 3:
                complex_codes.append(clean_rc)

    # Check if Indic script present (Hindi, Gujarati, Kannada, etc.)
    has_indic = bool(re.search(r"[\u0900-\u0D7F]", name))
    names_to_index = [norm_name]
    tokens_to_index = [name_tokens]

    # Domain stems (e.g. columbushungerallied.com -> columbushungerallied)
    domain_stems = []
    if name:
        parts = re.findall(r"([a-zA-Z0-9\-]+)\.(?:com|in|org|net|co\.in|gov|edu)", name.lower())
        for p in parts:
            domain_stems.append(p)
            for sub_p in re.split(r"[\-\.]", p):
                if len(sub_p) >= 4 and sub_p not in GENERIC_WORDS:
                    domain_stems.append(sub_p)
    domain_stems = list(set(domain_stems))
    if domain_stems:
        tokens_to_index.append(domain_stems)
        for ds in domain_stems:
            keys.append((country, "domain_stem", ds))

    if has_indic:
        trans_name = transliterate_indic(name)
        norm_trans, trans_tokens = clean_business_name(trans_name)
        if norm_trans:
            names_to_index.append(norm_trans)
            tokens_to_index.append(trans_tokens)

    # Address tokens
    norm_addr = normalize_address(address, expand_abbr=True) if address else ""
    addr_stopwords = ADDRESS_STOPWORDS.union({
        "flat", "door", "plot", "shop", "office", "room", "gala", "survey", "khasra",
        "hissa", "floor", "building", "bldg", "road", "street", "near", "opp", "opposite", "behind"
    })
    addr_toks = [t for t in tokenize(norm_addr) if t not in addr_stopwords and len(t) >= 4]

    # 1. STRATEGY A: Exact / Normalized Name
    if "exact_norm" in strategies:
        for n_str in names_to_index:
            if n_str:
                keys.append((country, "exact_norm", n_str))
                # Concatenated tokens (catches domain names like anilikaenergy)
                fused = "".join(tokenize(n_str))
                if len(fused) >= 6:
                    keys.append((country, "fused_name", fused))

    if "sorted_tokens" in strategies:
        for t_list in tokens_to_index:
            if len(t_list) >= 2:
                keys.append((country, "sorted_tokens", " ".join(sorted(t_list))))

    # 2. STRATEGY B: Postal + Name Tokens
    if "postal_name" in strategies and postals:
        for p in postals[:2]:
            for t_list in tokens_to_index:
                for tok in t_list[:4]:
                    if len(tok) >= 3:
                        keys.append((country, "post_name_tok", p, tok))

    # 3. STRATEGY C: Address + Name
    if "address_name" in strategies and address:
        for hn in h_nums[:3]:
            for t_list in tokens_to_index:
                for tok in t_list[:3]:
                    if len(tok) >= 3:
                        keys.append((country, "house_name_tok", hn, tok))

        if addr_toks:
            for a_tok in addr_toks[:4]:
                for t_list in tokens_to_index:
                    for n_tok in t_list[:3]:
                        if len(n_tok) >= 3:
                            keys.append((country, "addr_name_tok", a_tok, n_tok))

    # 4. STRATEGY: Two-Token Signatures (High Precision Multi-Token Matching)
    if "two_tokens" in strategies:
        for t_list in tokens_to_index:
            if len(t_list) >= 2:
                top2 = tuple(sorted(t_list[:3]))[:2]
                keys.append((country, "two_tokens", top2))

    # 5. STRATEGY: Address Signatures (Name-Independent Address Anchors)
    if "address_signature" in strategies and address:
        for hn in h_nums[:3]:
            if addr_toks:
                keys.append((country, "house_addr_tok", hn, addr_toks[0]))
            if postals:
                keys.append((country, "post_house_num", postals[0], hn))

        if postals and addr_toks:
            keys.append((country, "post_addr_tok", postals[0], addr_toks[0]))

        if complex_codes and addr_toks:
            for cc in complex_codes:
                for at in addr_toks[:3]:
                    if len(at) >= 4:
                        keys.append((country, "door_addr_tok", cc, at))

    # 6. STRATEGY D: Cleaned Character N-Gram
    if "char_ngram" in strategies:
        for n_str in names_to_index:
            clean_toks = [t for t in tokenize(n_str) if t not in GENERIC_WORDS and len(t) >= 4]
            core = " ".join(clean_toks) if clean_toks else n_str
            if len(core) >= 5:
                # 4-gram prefix + length bucket (length // 4)
                l_bucket = len(core) // 4
                keys.append((country, "pfx4_lbucket", core[:4], l_bucket))
                if postals:
                    keys.append((country, "post_pfx4", postals[0], core[:4]))

    # 7. STRATEGY: Distinctive Rare Name Token
    if "rare_token" in strategies:
        for t_list in tokens_to_index:
            for tok in t_list:
                if len(tok) >= 6 and tok not in GENERIC_WORDS:
                    keys.append((country, "rare_tok", tok))
                    break  # index only the first distinctive token

    # 8. STRATEGY: Phonetic Consonant Skeleton
    if "phonetic_skeleton" in strategies:
        for t_list in tokens_to_index:
            skels = [get_consonant_skeleton(tok) for tok in t_list if tok not in GENERIC_WORDS and len(tok) >= 3]
            # Skeleton 2-token signature
            if len(skels) >= 2:
                top2_skel = tuple(sorted(skels[:3]))[:2]
                keys.append((country, "skel_two_tok", top2_skel))
            # Distinctive skeleton token
            for sk in skels:
                if len(sk) >= 5:
                    keys.append((country, "skel_rare_tok", sk))
                    break

    # 9. STRATEGY: High-Precision Address Co-occurrence Pair (Expanded Window)
    if "address_pair" in strategies and address:
        if len(addr_toks) >= 2:
            sorted_by_len = sorted(addr_toks, key=lambda x: -len(x))
            top_distinctive = sorted_by_len[:8]
            for i in range(len(top_distinctive)):
                for j in range(i + 1, len(top_distinctive)):
                    pair = tuple(sorted([top_distinctive[i], top_distinctive[j]]))
                    keys.append((country, "addr_pair_sig", pair))

    return keys


class BlockIndex:
    """
    Inverted Index for Candidate Generation.
    Maps (country, strategy, ...) blocking keys to target entity IDs.
    Guards against bucket explosion via max_bucket_size.
    """

    def __init__(self, max_bucket_size: int = 1000):
        self.max_bucket_size = max_bucket_size
        self.index: Dict[Tuple, List[str]] = defaultdict(list)
        self.oversized_keys: Set[Tuple] = set()
        self.total_postings = 0

    def add(self, entity_id: str, keys: List[Tuple]):
        """Index a single target record with its extracted keys."""
        for k in keys:
            if k in self.oversized_keys:
                continue
            bucket = self.index[k]
            if len(bucket) >= self.max_bucket_size:
                self.oversized_keys.add(k)
                del self.index[k]
            else:
                bucket.append(entity_id)
                self.total_postings += 1

    def query(self, keys: List[Tuple], max_candidates: int = 200) -> Set[str]:
        """Retrieve candidate IDs matching any of the query keys."""
        candidates = set()
        for k in keys:
            if k in self.oversized_keys:
                continue
            bucket = self.index.get(k)
            if bucket:
                candidates.update(bucket)
                if len(candidates) >= max_candidates:
                    break
        return candidates

    def get_stats(self) -> Dict[str, Union[int, float]]:
        active_keys = len(self.index)
        sizes = [len(v) for v in self.index.values()]
        return {
            "total_keys": active_keys,
            "total_postings": self.total_postings,
            "oversized_keys_pruned": len(self.oversized_keys),
            "max_bucket_size": max(sizes) if sizes else 0,
            "median_bucket_size": float(np.median(sizes)) if sizes else 0.0,
            "mean_bucket_size": round(float(np.mean(sizes)), 2) if sizes else 0.0,
        }

    def get_top_buckets(self, top_k: int = 50) -> List[Tuple[Tuple, int]]:
        items = [(k, len(v)) for k, v in self.index.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]


def build_target_index(
    target_records: Iterable[Tuple[str, str, str, str]],
    strategies: Set[str],
    max_bucket_size: int = 1000,
) -> BlockIndex:
    index = BlockIndex(max_bucket_size=max_bucket_size)
    for eid, name, addr, country in target_records:
        keys = extract_blocking_keys(name, addr, country, strategies)
        index.add(eid, keys)
    return index


def generate_candidates(
    s1_records: Iterable[Tuple[str, str, str, str]],
    target_index: BlockIndex,
    strategies: Set[str],
    max_candidates_per_s1: int = 200,
) -> Dict[str, Set[str]]:
    candidates = {}
    for s1_id, name, addr, country in s1_records:
        keys = extract_blocking_keys(name, addr, country, strategies)
        cands = target_index.query(keys, max_candidates=max_candidates_per_s1)
        candidates[s1_id] = cands
    return candidates
