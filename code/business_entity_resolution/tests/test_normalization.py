"""
Unit tests for text normalization and similarity utilities.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from normalization import (
    normalize_unicode,
    normalize_whitespace,
    normalize_lowercase,
    strip_punctuation,
    strip_legal_suffixes,
    normalize_business_name,
    expand_address_abbreviations,
    normalize_address,
    extract_postal_codes,
    extract_numbers,
    token_jaccard,
    token_overlap,
    character_similarity,
    token_set_similarity,
)


def test_unicode_and_accents():
    # French accented
    assert normalize_unicode("Café Société") == "Cafe Societe"
    # Hindi preservation (Devanagari characters should remain intact when not combining diacritics)
    hindi = "राम मार्केटिंग"
    assert normalize_unicode(hindi) == hindi


def test_legal_suffix_stripping():
    assert strip_legal_suffixes("Amazon Inc") == "Amazon"
    assert strip_legal_suffixes("Tata Motors Pvt Ltd") == "Tata Motors"
    assert strip_legal_suffixes("Google LLC") == "Google"
    assert strip_legal_suffixes("Societe Generale SARL") == "Societe Generale"


def test_address_abbreviation_expansion():
    addr = "123 Main St, Near Opp State Bank, 5th Fl"
    expanded = expand_address_abbreviations(addr)
    assert "street" in expanded.lower()
    assert "opposite" in expanded.lower()
    assert "floor" in expanded.lower()


def test_postal_code_extraction():
    assert extract_postal_codes("1795 Westchester Dr, High Point, NC 27262") == ["27262"]
    assert extract_postal_codes("New Delhi, Delhi 110001, India") == ["110001"]


def test_token_jaccard_and_overlap():
    t1 = ["apple", "banana"]
    t2 = ["banana", "cherry"]
    assert token_jaccard(t1, t2) == 1.0 / 3.0
    assert token_overlap(t1, t2) == 1.0 / 2.0


def test_character_similarity():
    assert character_similarity("abc", "abc") == 1.0
    assert character_similarity("abc", "xyz") == 0.0
    assert 0.6 < character_similarity("Walmart", "Wal-mart") < 1.0
