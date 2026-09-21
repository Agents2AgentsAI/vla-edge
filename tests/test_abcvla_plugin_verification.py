"""The device verifier must actually initialize CUDA plugins, not just hash them."""

import json
import sys
from types import SimpleNamespace

import pytest

from vla_edge.scripts import verify_abcvla_device as verifier


@pytest.mark.parametrize("fail", [False, True])
def test_device_verification_checks_plugin_initialization(
    monkeypatch, tmp_path, capsys, fail
):
    descriptor = {"engines": {"vision": {}, "flow": {}, "prefill": {"213": {}}}}
    (tmp_path / "abcvla-serving.json").write_text(json.dumps(descriptor))
    (tmp_path / "engine.json").write_text(json.dumps({"requires": {}}))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace())
    monkeypatch.setattr(sys, "argv", ["verify", "--bundle", str(tmp_path)])
    monkeypatch.setattr(
        verifier, "detect_device", lambda *args: {"name": "test", "tensorrt": "test"}
    )
    monkeypatch.setattr(verifier, "verify_device_requirements", lambda *args: None)
    monkeypatch.setattr(
        verifier,
        "engine_declaration",
        lambda desc, raw, label: {
            "metadata": str(tmp_path / "engine.json"),
            "stage": label,
        },
    )
    initialized = []

    def load(trt, entry):
        initialized.append(entry["stage"])
        if fail:
            raise RuntimeError("CUDA symbol not found")

    monkeypatch.setattr(verifier, "_load_plugins", load)
    if fail:
        with pytest.raises(RuntimeError, match="CUDA symbol"):
            verifier.main()
        assert "initialization pass" not in capsys.readouterr().out
    else:
        verifier.main()
        assert initialized == ["vision", "flow", "prefill-213"]
        assert capsys.readouterr().out.count("CUDA plugin initialization pass") == 3
