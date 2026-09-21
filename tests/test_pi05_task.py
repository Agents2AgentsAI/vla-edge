"""Pi0.5 task dispatch, camera order and RTC rejection without hardware."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parents[1] / "examples/bimanual-yam"


def load(name):
    spec = importlib.util.spec_from_file_location(
        "test_pi05_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stub_rollout(monkeypatch):
    calls = []
    module = SimpleNamespace(
        control_settings=lambda: None,
        preflight=lambda here: ["rig"],
        run=lambda *args: calls.append(args) or 0,
    )
    monkeypatch.setitem(sys.modules, "pi05_rollout", module)
    return calls


def fake_client():
    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def bind(self, **kwargs):
            return SimpleNamespace(
                policy="pi05-bimanual-yam",
                model_family="pi05",
                action_horizon=16,
                gripper_state="commanded",
            )

        def health(self):
            return {}

        def close(self):
            pass

    return Client


def test_task_check_does_not_launch_controller(monkeypatch):
    task = load("run_task")
    calls = stub_rollout(monkeypatch)
    monkeypatch.setattr(task, "ActClient", fake_client())
    monkeypatch.setattr(task.os, "chdir", lambda p: None)
    monkeypatch.setattr(
        task.os, "execvpe", lambda *args: pytest.fail("controller launched")
    )
    monkeypatch.delenv("YAM_RTC", raising=False)
    assert task.main(["fold", "--check"]) == 0
    assert calls == []


@pytest.mark.parametrize("option", ["rtc", "prefix", "horizon"])
def test_invalid_pi_control_settings_fail_before_launch(monkeypatch, option):
    task = load("run_task")
    monkeypatch.setattr(task, "ActClient", fake_client())
    monkeypatch.setattr(task.os, "chdir", lambda p: None)
    monkeypatch.setattr(
        task.os, "execvpe", lambda *args: pytest.fail("controller launched")
    )
    args = ["fold"]
    monkeypatch.delenv("YAM_RTC", raising=False)
    monkeypatch.delenv("YAM_ACTION_HORIZON", raising=False)
    if option == "rtc":
        monkeypatch.setenv("YAM_RTC", "1")
    if option == "prefix":
        args += ["--rtc-prefix-length", "5"]
    if option == "horizon":
        monkeypatch.setenv("YAM_ACTION_HORIZON", "30")
    with pytest.raises(SystemExit):
        task.main(args)


def test_pi_dispatch_uses_independent_motion_controller(monkeypatch):
    task = load("run_task")
    monkeypatch.setattr(task, "ActClient", fake_client())
    monkeypatch.setattr(task.os, "chdir", lambda p: None)
    monkeypatch.delenv("YAM_RTC", raising=False)
    monkeypatch.delenv("YAM_ACTION_HORIZON", raising=False)
    calls = stub_rollout(monkeypatch)
    monkeypatch.setattr(
        task.os, "execvpe", lambda *args: pytest.fail("legacy controller launched")
    )
    assert task.main(["fold"]) == 0
    assert calls[0][0].instruction == "fold"
    assert calls[0][1] == ["rig"]


def test_client_preserves_camera_rgb_order(monkeypatch):
    mod = load("pi05_client")
    calls = []

    class Client(fake_client()):
        def act(self, **kwargs):
            calls.append(kwargs)
            return np.zeros((16, 14), np.float32), 63.0

    monkeypatch.setattr(mod, "ActClient", Client)
    monkeypatch.delenv("YAM_RTC", raising=False)
    monkeypatch.delenv("YAM_ACTION_HORIZON", raising=False)
    client = mod.Pi05Client()
    obs = {
        "front_camera_rgb": np.array([255, 0, 0]),
        "left_camera_rgb": np.array([0, 255, 0]),
        "right_camera_rgb": np.array([0, 0, 255]),
        "joint_positions": np.arange(14),
    }
    result = client.inference(client.prepare_input(obs, "fold"))
    assert result["actions"].shape == (16, 14)
    np.testing.assert_array_equal(
        calls[0]["cameras"]["top_cam"], obs["front_camera_rgb"]
    )
    np.testing.assert_array_equal(
        calls[0]["cameras"]["right_cam"], obs["right_camera_rgb"]
    )
    with pytest.raises(ValueError, match="RTC"):
        client.inference({}, rtc={})
