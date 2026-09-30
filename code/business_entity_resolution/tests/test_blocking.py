"""
Unit tests for blocking and candidate generation module.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from blocking import (
    extract_blocking_keys,
    transliterate_hindi,
    get_soundex,
    BlockIndex,
    build_target_index,
    generate_candidates,
)


def test_transliterate_hindi():
    hindi = "राम मार्केटिंग प्राइवेट लिमिटेड"
    trans = transliterate_hindi(hindi)
    assert "ram" in trans.lower()
    assert "marketing" in trans.lower()


def test_soundex():
    assert get_soundex("Robert") == "R163"
    assert get_soundex("Rupert") == "R163"
    assert get_soundex("Ashcraft") == "A261"


def test_extract_blocking_keys():
    name = "Walmart Supercenter Inc."
    addr = "702 SW 8th St, Bentonville, AR 72716"
    country = "US"

    strategies = {"exact_norm", "sorted_tokens", "postal_name", "address_name"}
    keys = extract_blocking_keys(name, addr, country, strategies)

    key_types = {k[1] for k in keys}
    assert "exact_norm" in key_types
    assert "post_name_tok" in key_types
    # All keys should have country as first element
    for k in keys:
        assert k[0] == "US"


def test_inverted_index_and_candidate_generation():
    target_data = [
        ("S2-101", "Walmart Supercenter", "702 SW 8th St, Bentonville, AR 72716", "US"),
        ("S3-202", "Target Store", "1000 Nicollet Mall, Minneapolis, MN 55403", "US"),
    ]
    strategies = {"exact_norm", "postal_name"}
    index = build_target_index(target_data, strategies, max_bucket_size=10)

    s1_data = [
        ("S1-1", "Walmart Inc", "702 SW 8th St, Bentonville, AR 72716", "US"),
        ("S1-2", "Target Corporation", "1000 Nicollet Mall, Minneapolis, MN 55403", "US"),
        ("S1-3", "Unknown Bakery", "123 Random St", "US"),
    ]

    candidates = generate_candidates(s1_data, index, strategies)
    assert "S2-101" in candidates["S1-1"]
    assert "S3-202" in candidates["S1-2"]
    assert len(candidates["S1-3"]) == 0  # no candidates for unknown


def test_oversized_bucket_pruning():
    index = BlockIndex(max_bucket_size=3)
    key = ("US", "dummy", "common_val")
    index.add("S2-1", [key])
    index.add("S2-2", [key])
    index.add("S2-3", [key])
    assert key in index.index
    # 4th addition triggers pruning
    index.add("S2-4", [key])
    assert key not in index.index
    assert key in index.oversized_keys
    # Query should return nothing from pruned key
    assert len(index.query([key])) == 0
