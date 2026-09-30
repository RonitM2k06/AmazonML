"""
Text Normalization and Similarity Module for Business Entity Resolution.
Provides Unicode normalization, case/punctuation cleaning, legal suffix extraction,
address tokenization, postal code extraction, and string similarity metrics.
"""

import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Set, Tuple
from rapidfuzz import fuzz, distance


# Common Business Legal Suffixes across US, India, and France
LEGAL_SUFFIXES = [
    # US / General
    r"\bincorporated\b", r"\binc\b",
    r"\bcorporation\b", r"\bcorp\b",
    r"\blimited liability company\b", r"\bllc\b", r"\bl\.l\.c\b",
    r"\blimited liability partnership\b", r"\bllp\b", r"\bl\.l\.p\b",
    r"\bcompany\b", r"\bco\b",
    r"\blimited\b", r"\bltd\b",
    r"\bholdings?\b",
    r"\bgroup\b",
    r"\benterprises?\b",
    r"\bservices?\b",
    r"\bsolutions?\b",
    r"\btechnologies\b", r"\btech\b",
    # India
    r"\bprivate limited\b", r"\bpvt ltd\b", r"\bpvt\. ltd\b", r"\bprivate ltd\b",
    r"\bpvt\b", r"\bplc\b",
    # France / Europe
    r"\bsarl\b", r"\bs\.a\.r\.l\b",
    r"\bsas\b", r"\bs\.a\.s\b",
    r"\bsasu\b",
    r"\bsa\b", r"\bs\.a\b",
    r"\beurl\b",
    r"\bsci\b",
]

# Precompile regex for legal suffix stripping
RE_LEGAL_SUFFIXES = re.compile(
    r"\s+(" + "|".join(LEGAL_SUFFIXES) + r")\s*$", re.IGNORECASE
)

# Address abbreviations dictionary
ADDRESS_ABBREVIATIONS = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\bdept\b": "department",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
}

# Precompile address abbreviation regexes
COMPILED_ADDR_ABBR = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in ADDRESS_ABBREVIATIONS.items()
]

# Postal / PIN code regex
# Matches 6-digit (India) or 5-digit (US/France)
RE_POSTAL_CODE = re.compile(r"\b(\d{5,6})\b")
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")
RE_NON_ALPHANUMERIC = re.compile(r"[^a-zA-Z0-9\s]")


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFKD and strip Latin combining diacritical marks while preserving Indic scripts."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    # Strip only Latin combining diacritical marks (U+0300 - U+036F)
    stripped = "".join(c for c in decomposed if not (0x0300 <= ord(c) <= 0x036F))
    return unicodedata.normalize("NFKC", stripped)


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace and strip."""
    if not text:
        return ""
    return RE_WHITESPACE.sub(" ", text).strip()


def normalize_lowercase(text: str) -> str:
    """Convert to lowercase and normalize whitespace."""
    if not text:
        return ""
    return normalize_whitespace(text.lower())


def strip_punctuation(text: str) -> str:
    """Remove punctuation marks and collapse whitespace, preserving Indic scripts and accents."""
    if not text:
        return ""
    try:
        text.encode("ascii")
        return normalize_whitespace(RE_PUNCTUATION.sub(" ", text))
    except UnicodeEncodeError:
        # Non-ASCII: preserve letters (L), marks (M), numbers (N), and separators (Z)
        chars = [c if (unicodedata.category(c)[0] in "LMNZ" or c == " ") else " " for c in text]
        return normalize_whitespace("".join(chars))


def normalize_alphanumeric(text: str) -> str:
    """Keep only ASCII alphanumeric characters and whitespace."""
    if not text:
        return ""
    norm = normalize_unicode(text).lower()
    return normalize_whitespace(RE_NON_ALPHANUMERIC.sub(" ", norm))


def strip_legal_suffixes(name: str) -> str:
    """Strip common legal suffixes (Inc, LLC, Pvt Ltd, Sarl, etc.)."""
    if not name:
        return ""
    cleaned = name.strip()
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = RE_LEGAL_SUFFIXES.sub("", cleaned).strip()
    return cleaned


def normalize_business_name(name: str, strip_legal: bool = True) -> str:
    """Standard normalization for business names."""
    if not name:
        return ""
    text = normalize_unicode(name).lower()
    text = strip_punctuation(text)
    if strip_legal:
        text = strip_legal_suffixes(text)
    return normalize_whitespace(text)


def expand_address_abbreviations(address: str) -> str:
    """Expand common street and landmark abbreviations."""
    if not address:
        return ""
    norm = address
    for pattern, replacement in COMPILED_ADDR_ABBR:
        norm = pattern.sub(replacement, norm)
    return norm


def normalize_address(address: str, expand_abbr: bool = True) -> str:
    """Standard normalization for business addresses."""
    if not address:
        return ""
    text = normalize_unicode(address).lower()
    text = strip_punctuation(text)
    if expand_abbr:
        text = expand_address_abbreviations(text)
    return normalize_whitespace(text)


def extract_postal_codes(address: str) -> List[str]:
    """Extract 5 or 6 digit postal/PIN codes from address."""
    if not address:
        return []
    return RE_POSTAL_CODE.findall(address)


def extract_numbers(text: str) -> List[str]:
    """Extract all numeric tokens from text."""
    if not text:
        return []
    return RE_NUMBERS.findall(text)


def tokenize(text: str) -> List[str]:
    """Tokenize normalized text into whitespace-delimited tokens."""
    if not text:
        return []
    return text.split()


def token_jaccard(tokens1: Iterable[str], tokens2: Iterable[str]) -> float:
    """Compute Jaccard similarity between two token collections."""
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return float(intersection / union) if union > 0 else 0.0


def token_overlap(tokens1: Iterable[str], tokens2: Iterable[str]) -> float:
    """Compute token overlap (containment) ratio: |A ∩ B| / min(|A|, |B|)."""
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    min_len = min(len(s1), len(s2))
    intersection = len(s1.intersection(s2))
    return float(intersection / min_len) if min_len > 0 else 0.0


def character_similarity(s1: str, s2: str) -> float:
    """Compute normalized character Levenshtein similarity in [0, 1]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return float(distance.Levenshtein.normalized_similarity(s1, s2))


def token_set_similarity(s1: str, s2: str) -> float:
    """Compute token set ratio similarity in [0, 1] using rapidfuzz."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return float(fuzz.token_set_ratio(s1, s2) / 100.0)


def token_sort_similarity(s1: str, s2: str) -> float:
    """Compute token sort ratio similarity in [0, 1] using rapidfuzz."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return float(fuzz.token_sort_ratio(s1, s2) / 100.0)
