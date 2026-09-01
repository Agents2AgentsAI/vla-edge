"""Headless camera viewer tests. No camera or OpenCV installation is required."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

YAM_DIR = Path(__file__).parents[1] / "examples" / "bimanual-yam"


def _load_module(monkeypatch, filename, module_name):
    fake_zmq = ModuleType("zmq")
    fake_zmq.Again = type("Again", (Exception,), {})
    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)
    spec = importlib.util.spec_from_file_location(module_name, YAM_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_headless_auto_view_delegates_before_importing_opencv(monkeypatch, capsys):
    client = _load_module(monkeypatch, "camera_client.py", "camera_client_cli_test")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setitem(sys.modules, "cv2", None)
    called = []
    monkeypatch.setattr(
        client,
        "_run_browser_viewer",
        lambda endpoint: called.append(endpoint) or 0,
    )

    assert client.main(["--mode", "sub"]) == 0

    assert called == ["tcp://127.0.0.1:5556"]
    assert "No graphical display detected" in capsys.readouterr().out


def test_explicit_window_without_display_exits_cleanly(monkeypatch, capsys):
    client = _load_module(monkeypatch, "camera_client.py", "camera_client_window_test")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setitem(sys.modules, "cv2", None)

    with pytest.raises(SystemExit) as exc_info:
        client.main(["--view", "window"])

    assert exc_info.value.code == 2
    assert "needs a graphical display" in capsys.readouterr().err


def test_unreachable_display_is_not_passed_to_qt(monkeypatch):
    client = _load_module(monkeypatch, "camera_client.py", "camera_client_x_test")
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(client.shutil, "which", lambda _name: "/usr/bin/xdpyinfo")
    monkeypatch.setattr(
        client.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1),
    )

    assert client._display_available() is False


def test_viewer_uses_model_camera_order(monkeypatch):
    client = _load_module(monkeypatch, "camera_client.py", "camera_client_order_test")
    frames = {
        "right_camera": np.zeros((2, 2, 3), dtype=np.uint8),
        "left_camera": np.zeros((2, 2, 3), dtype=np.uint8),
        "front_camera": np.zeros((2, 2, 3), dtype=np.uint8),
    }

    assert client._ordered_camera_names(frames) == [
        "front_camera",
        "left_camera",
        "right_camera",
    ]


def test_browser_encoder_uses_model_camera_order(monkeypatch):
    viewer = _load_module(monkeypatch, "mjpeg_viewer.py", "mjpeg_viewer_test")
    labels = []

    class FakeCV2:
        FONT_HERSHEY_SIMPLEX = 1
        LINE_AA = 2
        COLOR_RGB2BGR = 3
        IMWRITE_JPEG_QUALITY = 4

        @staticmethod
        def putText(image, label, *_args):
            labels.append(label)
            return image

        @staticmethod
        def resize(image, _size):
            return image

        @staticmethod
        def cvtColor(image, _conversion):
            return image

        @staticmethod
        def imencode(_extension, _image, _params):
            return True, np.asarray([1, 2, 3], dtype=np.uint8)

    frames = {
        "left_camera": np.zeros((2, 2, 3), dtype=np.uint8),
        "front_camera": np.zeros((2, 2, 3), dtype=np.uint8),
        "right_camera": np.zeros((2, 2, 3), dtype=np.uint8),
    }

    assert viewer._encode_grid(frames, FakeCV2, np) == b"\x01\x02\x03"
    assert labels == ["front_camera", "left_camera", "right_camera"]
