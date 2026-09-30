#!/usr/bin/env python3
"""
Device wrapper for src/two_tower_encoder.py that prefers MPS, then CUDA,
then CPU -- WITHOUT modifying two_tower_encoder.py's own hardcoded "cuda"
default parameters (src/two_tower_encoder.py lines 213, 267) or its
autocast call (line 242, which is already conditionally guarded by
`enabled=(device == "cuda")` and therefore does not crash on non-CUDA
devices -- it simply runs without autocast there).

This module only decides WHICH device string to pass in; it does not alter
the encoder's internals.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_CODE_DIR = Path(__file__).resolve().parents[2]
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_DIR / "src"))
sys.path.insert(0, str(_THIS_DIR))

from config import select_device  # noqa: E402


def encode_records_fused_auto_device(
    model,
    tokenizer,
    records: Dict[str, Tuple],
    batch_size: int = 256,
    progress_every: int = 25000,
) -> Tuple[List[str], np.ndarray]:
    """
    Calls src/two_tower_encoder.py::encode_records_fused with an
    MPS-then-CUDA-then-CPU device choice, moving the model to that device
    first. Falls back to CPU if MPS raises (some ops are not yet
    MPS-implemented for every transformer op in every torch version).
    """
    from two_tower_encoder import encode_records_fused

    device = select_device()
    try:
        model.to(device)
        return encode_records_fused(
            model, tokenizer, records, batch_size=batch_size, device=device,
            progress_every=progress_every,
        )
    except (RuntimeError, NotImplementedError) as exc:
        if device != "cpu":
            print(f"[device_utils] {device} failed ({exc}); falling back to cpu.", flush=True)
            model.to("cpu")
            return encode_records_fused(
                model, tokenizer, records, batch_size=batch_size, device="cpu",
                progress_every=progress_every,
            )
        raise

