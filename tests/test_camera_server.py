import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

YAM_DIR = Path(__file__).parents[1] / "examples" / "bimanual-yam"
sys.path.insert(0, str(YAM_DIR))

fake_zmq = ModuleType("zmq")
fake_zmq.REP = 1
fake_zmq.PUB = 2
fake_zmq.POLLIN = 3
fake_zmq.LINGER = 4
fake_zmq.RCVTIMEO = 5
fake_zmq.SNDTIMEO = 6
fake_zmq.Again = type("Again", (Exception,), {})
fake_zmq.ZMQError = type("ZMQError", (Exception,), {})
fake_omegaconf = ModuleType("omegaconf")


class FakeOmegaConf:
    config = None

    @classmethod
    def load(cls, _path):
        return cls.config

    @staticmethod
    def to_container(config, resolve=True):
        assert resolve
        return config


fake_omegaconf.OmegaConf = FakeOmegaConf
previous_zmq = sys.modules.get("zmq")
previous_omegaconf = sys.modules.get("omegaconf")
sys.modules["zmq"] = fake_zmq
sys.modules["omegaconf"] = fake_omegaconf
MODULE_PATH = YAM_DIR / "camera_server.py"
SPEC = importlib.util.spec_from_file_location("camera_server", MODULE_PATH)
assert SPEC and SPEC.loader
camera_server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = camera_server
try:
    SPEC.loader.exec_module(camera_server)
finally:
    if previous_zmq is None:
        sys.modules.pop("zmq", None)
    else:
        sys.modules["zmq"] = previous_zmq
    if previous_omegaconf is None:
        sys.modules.pop("omegaconf", None)
    else:
        sys.modules["omegaconf"] = previous_omegaconf


def _config():
    return {
        "sensors": {
            "cameras": {
                "left_camera": {"device_id": "left"},
                "front_camera": {"device_id": "front"},
                "right_camera": {"device_id": "right"},
            }
        }
    }


def test_partial_camera_startup_closes_previously_opened_camera(monkeypatch):
    opened = []

    class Camera:
        def __init__(self, device_id):
            if device_id == "front":
                raise RuntimeError("Device or resource busy")
            self.device_id = device_id
            self.closed = False
            opened.append(self)

        def close(self):
            self.closed = True

    FakeOmegaConf.config = _config()
    monkeypatch.setattr(
        camera_server,
        "get_device_ids",
        lambda: ["left", "front", "right"],
    )
    monkeypatch.setattr(camera_server, "RealSenseCamera", Camera)

    with pytest.raises(RuntimeError, match="front_camera.*resource busy"):
        camera_server._build_cameras_from_config(Path("rig.yaml"))

    assert len(opened) == 1
    assert opened[0].device_id == "left"
    assert opened[0].closed


def test_missing_camera_is_rejected_before_any_pipeline_opens(monkeypatch):
    FakeOmegaConf.config = _config()
    monkeypatch.setattr(camera_server, "get_device_ids", lambda: ["left", "front"])
    monkeypatch.setattr(
        camera_server,
        "RealSenseCamera",
        lambda _device_id: pytest.fail("pipeline should not open"),
    )

    with pytest.raises(RuntimeError, match="right_camera"):
        camera_server._build_cameras_from_config(Path("rig.yaml"))


def test_existing_healthy_server_prevents_camera_discovery(monkeypatch):
    monkeypatch.setattr(
        camera_server,
        "_probe_existing_server",
        lambda _endpoint: (True, None),
    )
    monkeypatch.setattr(
        camera_server,
        "_build_cameras_from_config",
        lambda _path: pytest.fail("existing server must retain camera ownership"),
    )

    assert camera_server.main(["--config", "rig.yaml"]) == 0


def test_existing_unhealthy_server_is_reported_without_touching_cameras(monkeypatch):
    monkeypatch.setattr(
        camera_server,
        "_probe_existing_server",
        lambda _endpoint: (True, "stale frame"),
    )
    monkeypatch.setattr(
        camera_server,
        "_build_cameras_from_config",
        lambda _path: pytest.fail("existing server must retain camera ownership"),
    )

    assert camera_server.main(["--config", "rig.yaml"]) == 1
