#!/usr/bin/env python3
"""
Runtime guard (requirement F): asserts that no ground-truth/label file is
ever opened during a test-transfer run. This is a REAL runtime check, not a
comment -- it monkey-patches the builtin `open()` for the duration of a
`with ProvenanceGuard():` block and raises immediately if any forbidden path
is opened (by exact path, by resolved absolute path, or by substring match
on the filename "train_ground_truth.tsv" as a defense-in-depth backstop in
case a relative/symlinked path slips past the exact-match check).

Usage:
    from provenance_guard import ProvenanceGuard
    with ProvenanceGuard():
        run_pipeline(...)
"""

from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from config import FORBIDDEN_LABEL_FILES  # noqa: E402

_FORBIDDEN_BASENAMES = {"train_ground_truth.tsv"}
_FORBIDDEN_ABS = {os.path.abspath(p) for p in FORBIDDEN_LABEL_FILES}


class GroundTruthAccessViolation(RuntimeError):
    pass


class ProvenanceGuard:
    """Context manager that blocks opens of any forbidden ground-truth file."""

    def __init__(self, extra_forbidden_paths=None):
        self._orig_open = None
        self.opened_paths_log = []
        self.extra_forbidden_abs = {os.path.abspath(p) for p in (extra_forbidden_paths or [])}

    def _guarded_open(self, file, *args, **kwargs):
        try:
            path_str = os.fspath(file)
        except TypeError:
            path_str = str(file)

        basename = os.path.basename(path_str)
        abs_path = os.path.abspath(path_str)

        if basename in _FORBIDDEN_BASENAMES or abs_path in (_FORBIDDEN_ABS | self.extra_forbidden_abs):
            raise GroundTruthAccessViolation(
                f"BLOCKED: attempted to open forbidden ground-truth/label file: {path_str} "
                f"(resolved: {abs_path}). Test-transfer runs must never access labels."
            )

        self.opened_paths_log.append(path_str)
        return self._orig_open(file, *args, **kwargs)

    def __enter__(self):
        self._orig_open = builtins.open
        builtins.open = self._guarded_open
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.open = self._orig_open
        return False

    def audit_summary(self) -> dict:
        return {
            "total_files_opened": len(self.opened_paths_log),
            "unique_files_opened": sorted(set(self.opened_paths_log)),
            "forbidden_files_blocked": 0,  # if any were blocked, an exception would have propagated
        }


if __name__ == "__main__":
    print("[self-test] Verifying ProvenanceGuard blocks train_ground_truth.tsv ...")
    from config import TRAIN_DATA_DIR
    try:
        with ProvenanceGuard():
            with open(TRAIN_DATA_DIR / "train_ground_truth.tsv") as f:
                f.readline()
        print("[self-test] FAILED -- guard did not block the forbidden file!")
        raise SystemExit(1)
    except GroundTruthAccessViolation as e:
        print(f"[self-test] PASSED -- guard correctly blocked: {e}")

    print("[self-test] Verifying ProvenanceGuard allows a normal file ...")
    with ProvenanceGuard() as g:
        with open(__file__) as f:
            f.readline()
    print(f"[self-test] PASSED -- allowed normal open. Audit: {g.audit_summary()}")
