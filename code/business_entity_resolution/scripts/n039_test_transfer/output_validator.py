#!/usr/bin/env python3
"""Requirement E: validates a matching_results.tsv-style output."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

_ID_RE = re.compile(r"^(S1|S2|S3)-\d+$")


def validate_output(path: Path) -> Dict[str, object]:
    checks = {}
    seen_pairs = set()
    invented_ids = []
    mixed_namespace_rows = []
    bad_id_format = []
    total_rows = 0
    total_pairs = 0

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        checks["header_valid"] = (header == "source1_entity_id\tmatched_entity_ids")
        for line in f:
            total_rows += 1
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) != 2:
                bad_id_format.append(line)
                continue
            s1_id, matched_str = parts
            if not _ID_RE.match(s1_id) or not s1_id.startswith("S1-"):
                bad_id_format.append(s1_id)
            if not matched_str:
                continue
            matched_ids = matched_str.split(",")
            s2_present = any(m.startswith("S2-") for m in matched_ids)
            s3_present = any(m.startswith("S3-") for m in matched_ids)
            # Mixing S2 and S3 within one row is EXPECTED (a match set can span
            # both target sources) -- what must never happen is a malformed ID
            # that belongs to neither namespace, or a namespace prefix that
            # doesn't match its own ID pattern.
            for m in matched_ids:
                if not _ID_RE.match(m):
                    bad_id_format.append(m)
                pair = (s1_id, m)
                if pair in seen_pairs:
                    invented_ids.append(f"DUPLICATE:{pair}")
                seen_pairs.add(pair)
                total_pairs += 1

    checks["total_rows"] = total_rows
    checks["total_pairs"] = total_pairs
    checks["duplicate_pairs"] = len([x for x in invented_ids if x.startswith("DUPLICATE:")])
    checks["malformed_ids"] = bad_id_format[:50]  # cap for report size
    checks["malformed_id_count"] = len(bad_id_format)
    checks["ALL_PASS"] = (
        checks["header_valid"] and checks["duplicate_pairs"] == 0 and checks["malformed_id_count"] == 0
    )
    return checks


if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1])
    result = validate_output(p)
    for k, v in result.items():
        print(f"{k}: {v}")
