import importlib
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

YAM_DIR = Path(__file__).parents[1] / "examples" / "bimanual-yam"
sys.path.insert(0, str(YAM_DIR))
realsense = importlib.import_module("gello_min.realsense_camera")


def test_device_discovery_does_not_reset_cameras(monkeypatch):
    class Device:
        def __init__(self, serial):
            self.serial = serial

        def get_info(self, _field):
            return self.serial

        def hardware_reset(self):
            raise AssertionError("discovery must not reset active cameras")

    rs = ModuleType("pyrealsense2")
    rs.camera_info = SimpleNamespace(serial_number=object())
    rs.context = lambda: SimpleNamespace(
        query_devices=lambda: [Device("front"), Device("left"), Device("right")]
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)

    assert realsense.get_device_ids() == ["front", "left", "right"]


def test_constructor_releases_pipeline_when_start_fails(monkeypatch):
    pipelines = []

    class Pipeline:
        def __init__(self):
            self.stopped = False
            pipelines.append(self)

        def start(self, _config):
            raise RuntimeError("Device or resource busy")

        def stop(self):
            self.stopped = True

    class Config:
        def enable_device(self, _device_id):
            pass

        def enable_stream(self, *_args):
            pass

    rs = ModuleType("pyrealsense2")
    rs.align = lambda _stream: object()
    rs.pipeline = Pipeline
    rs.config = Config
    rs.stream = SimpleNamespace(color=1, depth=2)
    rs.format = SimpleNamespace(z16=1, bgr8=2)
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)

    with pytest.raises(RuntimeError, match="resource busy"):
        realsense.RealSenseCamera("front")

    assert len(pipelines) == 1
    assert pipelines[0].stopped


def test_close_is_idempotent_and_stops_pipeline():
    class Pipeline:
        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    camera = realsense.RealSenseCamera.__new__(realsense.RealSenseCamera)
    camera._closed = False
    camera._stop_event = threading.Event()
    camera._lock = threading.Lock()
    camera._pipeline = Pipeline()
    camera._capture_thread = None
    camera._device_id = "front"
    pipeline = camera._pipeline

    camera.close()
    camera.close()

    assert camera._stop_event.is_set()
    assert pipeline.stop_calls == 1
    assert camera._pipeline is None
