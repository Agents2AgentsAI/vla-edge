"""Compact TensorRT host-runtime resolution tests."""

from __future__ import annotations

import json

import pytest

from vla_edge.backends.tensorrt.host import (
    ACTION_EXPERT_PREFIX,
    EMBEDDING_PREFIX,
    HOST_FORMAT_VERSION,
    HostRuntimeError,
    required_prefixes,
    resolve_local_host,
)


def _write_host(path, repo_id="allenai/test"):
    path.mkdir(parents=True)
    (path / "host.json").write_text(
        json.dumps(
            {
                "format_version": HOST_FORMAT_VERSION,
                "repo_id": repo_id,
                "dtype": "bfloat16",
            }
        )
    )
    (path / "host.safetensors").write_bytes(b"weights")


def test_serving_config_resolves_shared_host_directory(tmp_path):
    engine = tmp_path / "yam-champion"
    host = tmp_path / "host" / "yam"
    engine.mkdir()
    _write_host(host)
    (engine / "serving.json").write_text(json.dumps({"host_dir": "../host/yam"}))

    assert resolve_local_host(engine, "allenai/test") == host.resolve()


def test_declared_missing_host_is_not_silently_downloaded(tmp_path):
    engine = tmp_path / "yam-champion"
    engine.mkdir()
    (engine / "serving.json").write_text(json.dumps({"host_dir": "../host/yam"}))

    with pytest.raises(HostRuntimeError, match="not a directory"):
        resolve_local_host(engine, "allenai/test")


def test_host_for_wrong_checkpoint_is_rejected(tmp_path):
    engine = tmp_path / "engine"
    host = engine / "host"
    engine.mkdir()
    _write_host(host, repo_id="allenai/other")

    with pytest.raises(HostRuntimeError, match="this embodiment uses"):
        resolve_local_host(engine, "allenai/test")


def test_action_expert_is_required_only_for_compiled_flow():
    assert required_prefixes(action_expert=False) == (EMBEDDING_PREFIX,)
    assert required_prefixes(action_expert=True) == (
        EMBEDDING_PREFIX,
        ACTION_EXPERT_PREFIX,
    )
