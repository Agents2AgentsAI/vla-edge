"""Inference server startup tests. No checkpoint or GPU is required."""

from __future__ import annotations

import errno
import json
import logging
import sys
from types import SimpleNamespace

import pytest

from vla_edge.backends import molmoact2
from vla_edge.serving import server


def _write_engine_set(root, name, repo_id, *, fast_vision=False):
    engine = root / name
    host = root / "host" / name
    engine.mkdir(parents=True)
    host.mkdir(parents=True, exist_ok=True)
    (engine / "serving.json").write_text(
        json.dumps({"host_dir": f"../host/{name}"})
    )
    (host / "host.json").write_text(json.dumps({"repo_id": repo_id}))
    for filename in molmoact2._REQUIRED_ENGINE_FILES:
        (engine / filename).touch()
    if fast_vision:
        (engine / "vision_fp8.plan").touch()
    return engine


def test_bundle_root_selects_its_only_matching_engine_set(tmp_path):
    (tmp_path / "MANIFEST.json").write_text("{}")
    expected = _write_engine_set(
        tmp_path,
        "yam-champion",
        "allenai/MolmoAct2-BimanualYAM",
        fast_vision=True,
    )

    selected = server._resolve_engine_dir(
        tmp_path,
        repo_id="allenai/MolmoAct2-BimanualYAM",
        fast_vision=True,
    )

    assert selected == expected


def test_bundle_root_fast_vision_selects_accelerated_set(tmp_path):
    (tmp_path / "MANIFEST.json").write_text("{}")
    _write_engine_set(tmp_path, "yam", "allenai/MolmoAct2-BimanualYAM")
    expected = _write_engine_set(
        tmp_path,
        "yam-champion",
        "allenai/MolmoAct2-BimanualYAM",
        fast_vision=True,
    )

    selected = server._resolve_engine_dir(
        tmp_path,
        repo_id="allenai/MolmoAct2-BimanualYAM",
        fast_vision=True,
    )

    assert selected == expected


def test_bundle_root_lists_choices_when_selection_is_ambiguous(tmp_path):
    (tmp_path / "MANIFEST.json").write_text("{}")
    _write_engine_set(tmp_path, "yam", "allenai/MolmoAct2-BimanualYAM")
    _write_engine_set(
        tmp_path,
        "yam-champion",
        "allenai/MolmoAct2-BimanualYAM",
        fast_vision=True,
    )

    with pytest.raises(ValueError, match="multiple engine sets") as exc_info:
        server._resolve_engine_dir(
            tmp_path,
            repo_id="allenai/MolmoAct2-BimanualYAM",
            fast_vision=False,
        )

    assert "yam-champion" in str(exc_info.value)
    assert "yam" in str(exc_info.value)


def test_listener_reserves_port_until_closed():
    first = server._reserve_listener("127.0.0.1", 0)
    port = int(first.getsockname()[1])
    try:
        with pytest.raises(OSError) as exc_info:
            server._reserve_listener("127.0.0.1", port)
        assert exc_info.value.errno == errno.EADDRINUSE
    finally:
        first.close()

    replacement = server._reserve_listener("127.0.0.1", port)
    replacement.close()


@pytest.mark.parametrize(
    ("policy_name", "horizon"),
    [
        ("abcvla-bimanual-yam", 30),
        ("molmoact2-bimanual-yam", 30),
    ],
)
def test_health_reports_selected_policy_and_action_horizon(policy_name, horizon):
    import asyncio

    pytest.importorskip("fastapi")
    from vla_edge.config import get_embodiment, get_policy

    policy = get_policy(policy_name)
    pipeline = SimpleNamespace(
        policy=policy,
        embodiment=get_embodiment(policy.embodiment),
        backend=SimpleNamespace(rtc_available=False, action_horizon=horizon),
    )
    app = server.build_app(pipeline, "tensorrt")
    route = next(route for route in app.routes if route.path == "/act" and "GET" in route.methods)
    response = asyncio.run(route.endpoint())
    payload = json.loads(response.body)

    assert payload["policy"] == policy_name
    assert payload["model_family"] == policy.model_family
    assert payload["action_horizon"] == horizon
    assert payload["embodiment"] == "bimanual-yam"
    assert payload["cameras"] == ["top_cam", "left_cam", "right_cam"]
    assert payload["gripper"] == {
        "indices": [6, 13],
        "checkpoint_convention": "closed_0_open_1",
        "wire_convention": "closed_0_open_1",
        "state_source": policy.gripper_state,
    }


def test_port_conflict_is_reported_before_checkpoint_load(monkeypatch, caplog):
    def occupied(_host, _port):
        raise OSError(errno.EADDRINUSE, "Address already in use")

    monkeypatch.setattr(server, "_reserve_listener", occupied)
    monkeypatch.setattr(
        server,
        "_probe_existing_server",
        lambda _host, _port: {
            "status": "ok",
            "repo_id": "allenai/MolmoAct2-BimanualYAM",
            "backend": "tensorrt",
            "rtc_available": True,
        },
    )
    monkeypatch.setattr(
        server.Pipeline,
        "load",
        lambda *args, **kwargs: pytest.fail("checkpoint must not load"),
    )
    caplog.set_level(logging.ERROR)

    result = server.main(["--embodiment", "bimanual-yam", "--backend", "torch"])

    assert result == 1
    assert "no model was loaded" in caplog.text
    assert "MolmoAct2-BimanualYAM" in caplog.text
    assert "--port 8203" in caplog.text


def test_ctrl_c_after_uvicorn_shutdown_exits_cleanly_and_closes_listener(
    monkeypatch, caplog
):
    class FakeListener:
        closed = False

        def getsockname(self):
            return ("127.0.0.1", 8202)

        def close(self):
            self.closed = True

    class FakeUvicornServer:
        def __init__(self, _config):
            pass

        def run(self, *, sockets):
            assert sockets == [listener]
            raise KeyboardInterrupt

    listener = FakeListener()
    closed = []
    pipeline = SimpleNamespace(close=lambda: closed.append(True))
    uvicorn = SimpleNamespace(
        Config=lambda *args, **kwargs: (args, kwargs),
        Server=FakeUvicornServer,
    )
    monkeypatch.setattr(server, "_reserve_listener", lambda _host, _port: listener)
    monkeypatch.setattr(server.Pipeline, "load", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(server, "build_app", lambda *_args: object())
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    caplog.set_level(logging.INFO)

    result = server.main(
        ["--embodiment", "bimanual-yam", "--backend", "torch", "--no-warmup"]
    )

    assert result == 0
    assert listener.closed
    assert closed == [True]
    assert "server stopped by user" in caplog.text


def test_checkpoint_load_failure_still_closes_listener(monkeypatch):
    class FakeListener:
        closed = False

        def getsockname(self):
            return ("127.0.0.1", 8202)

        def close(self):
            self.closed = True

    listener = FakeListener()
    monkeypatch.setattr(server, "_reserve_listener", lambda _host, _port: listener)

    def fail_load(*_args, **_kwargs):
        raise RuntimeError("checkpoint failed")

    monkeypatch.setattr(server.Pipeline, "load", fail_load)

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        server.main(
            ["--embodiment", "bimanual-yam", "--backend", "torch", "--no-warmup"]
        )

    assert listener.closed


@pytest.mark.parametrize(
    ("bind_host", "probe_host"),
    [
        ("0.0.0.0", "127.0.0.1"),
        ("::", "::1"),
        ("127.0.0.1", "127.0.0.1"),
    ],
)
def test_probe_host_uses_loopback_for_wildcard_binds(bind_host, probe_host):
    assert server._probe_host(bind_host) == probe_host
