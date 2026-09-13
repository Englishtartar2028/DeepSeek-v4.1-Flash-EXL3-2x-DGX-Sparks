#!/usr/bin/env python3
"""Host-side K-map contract for the mixed-K EXL3 2.9 bpw checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KMAP = json.loads((ROOT / "files" / "exl3_k_map.json").read_text())


def test_k_map() -> None:
    assert KMAP["codebook"] == "mul1"
    assert KMAP["mul1_marker_i32"] == -2082680531
    routed = {int(k): int(v) for k, v in KMAP["routed"].items()}
    assert routed[0] == 3
    assert routed[17] == 3
    assert routed[18] == 2
    assert routed[22] == 2
    assert routed[23] == 3
    assert routed[39] == 3
    assert set(routed) == set(range(40))
    shared = {int(k): int(v) for k, v in KMAP["shared"].items()}
    assert shared[0] == 5
    assert shared[29] == 4  # mixed 4/5; infer the rest from trellis at load
    assert KMAP["engram"]["1"] == 5
    assert KMAP["engram"]["14"] == 4
    assert KMAP["mtp_bits"] == 4
    assert KMAP["attn_default"] == 5
    assert KMAP["indexer_wk_bits"] == 8
    print("k_map contract OK")


if __name__ == "__main__":
    test_k_map()
