"""Portable bundle integrity and metadata binding checks (CPU only)."""
import hashlib
import json

import pytest

from vla_edge.backends.abcvla.bundle import AbcVlaError, engine_declaration, verify_file
from vla_edge.scripts.verify_release import verify


def declaration(root, name, payload):
    (root / name).write_bytes(payload)
    return {"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_engine_metadata_is_bound_to_released_plan(tmp_path):
    plan = declaration(tmp_path, "model.plan", b"test engine bytes")
    meta = {"schema_version": 1, "kind": "vla-edge-engine", "file": "model.plan",
            "sha256": plan["sha256"], "bytes": plan["bytes"]}
    plan["metadata"] = declaration(tmp_path, "metadata.json", json.dumps(meta).encode())
    entry = engine_declaration({"root": tmp_path}, plan, "vision")
    assert entry["path"] == str(tmp_path / "model.plan")
    meta["sha256"] = "0" * 64
    plan["metadata"] = declaration(tmp_path, "metadata.json", json.dumps(meta).encode())
    with pytest.raises(AbcVlaError, match="does not match"):
        engine_declaration({"root": tmp_path}, plan, "vision")


def test_release_detects_corruption_and_path_escape(tmp_path):
    item = declaration(tmp_path, "weights.bin", b"original")
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"schema_version": 1, "files": {"weights.bin": item}}))
    assert verify(tmp_path) == 1
    (tmp_path / "weights.bin").write_bytes(b"modified")
    with pytest.raises(AbcVlaError, match="mismatch"):
        verify(tmp_path)
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"value")
    with pytest.raises(AbcVlaError, match="escapes"):
        verify_file(tmp_path, {"path": "../outside.bin"}, "test")
