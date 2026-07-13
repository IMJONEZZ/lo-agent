"""Phase-6 tests: LD_PRELOAD steer-file format + linkage detection."""

from __future__ import annotations

import struct

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from local_harness.jlens import manager  # noqa: E402


def test_steer_file_roundtrip(tmp_path):
    path = tmp_path / "steer.bin"
    edits = [
        {"layer": 3, "pos_start": 0, "pos_end": -1, "vector": [1.0, 2.0, 3.0, 4.0]},
        {"layer": 7, "pos_start": 2, "pos_end": 5, "vector": [0.5, -0.5]},
    ]
    manager.write_steer_file(edits, str(path))
    raw = path.read_bytes()
    assert raw[:4] == b"LJS1"
    (n,) = struct.unpack_from("<I", raw, 4)
    assert n == 2
    # parse the first edit exactly as the C++ shim does
    off = 8
    layer, ps, pe, d = struct.unpack_from("<iiii", raw, off)
    off += 16
    vec = np.frombuffer(raw, "<f4", count=d, offset=off)
    assert (layer, ps, pe, d) == (3, 0, -1, 4)
    np.testing.assert_allclose(vec, [1.0, 2.0, 3.0, 4.0])


def test_detect_linkage_missing():
    ok, why = manager.detect_llama_linkage("/no/such/binary")
    assert not ok and "not found" in why


def test_shim_source_present():
    # the shim C++ ships with the package (built on demand by `lo lens shim`)
    assert manager._SHIM_SRC.is_file()
    assert "llama_init_from_model" in manager._SHIM_SRC.read_text()
