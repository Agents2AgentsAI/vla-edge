import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

YAM_DIR = Path(__file__).parents[1] / "examples" / "bimanual-yam"
sys.path.insert(0, str(YAM_DIR))
yam = importlib.import_module("gello_min.yam")
env_module = importlib.import_module("gello_min.env")
robot_module = importlib.import_module("gello_min.robot")


class FakeI2RTRobot:
    def __init__(self):
        self.closed = False

    def get_joint_pos(self):
        return np.zeros(7)

    def command_joint_pos(self, _target):
        pass

    def close(self):
        self.closed = True


def test_configured_gripper_limits_are_forwarded_to_i2rt(monkeypatch):
    calls = []

    def get_yam_robot(**kwargs):
        calls.append(kwargs)
        return FakeI2RTRobot()

    i2rt = ModuleType("i2rt")
    i2rt.__path__ = []
    robots = ModuleType("i2rt.robots")
    robots.__path__ = []
    get_robot = ModuleType("i2rt.robots.get_robot")
    get_robot.get_yam_robot = get_yam_robot
    utils = ModuleType("i2rt.robots.utils")
    utils.GripperType = SimpleNamespace(LINEAR_4310="linear_4310")
    monkeypatch.setitem(sys.modules, "i2rt", i2rt)
    monkeypatch.setitem(sys.modules, "i2rt.robots", robots)
    monkeypatch.setitem(sys.modules, "i2rt.robots.get_robot", get_robot)
    monkeypatch.setitem(sys.modules, "i2rt.robots.utils", utils)

    yam.YAMRobot(channel="can_left", gripper_limits=[6.4, 1.23])

    assert calls[0]["channel"] == "can_left"
    np.testing.assert_allclose(
        calls[0]["gripper_limits_override"],
        [6.4, 1.23],
    )


def test_close_uses_clean_i2rt_shutdown_once(monkeypatch):
    robot = FakeI2RTRobot()
    clean_closes = []
    home_arms = ModuleType("home_arms")
    home_arms.close_robot_cleanly = lambda value: clean_closes.append(value)
    monkeypatch.setitem(sys.modules, "home_arms", home_arms)

    adapter = yam.YAMRobot.__new__(yam.YAMRobot)
    adapter.robot = robot
    adapter._channel = "can_left"
    adapter._closed = False
    disable_probes = []
    home_arms.disable_and_probe = (
        lambda channels: disable_probes.append(tuple(channels)) or True
    )
    adapter.close()
    adapter.close()

    assert clean_closes == [robot]
    assert disable_probes == [("can_left",)]


def test_close_reports_a_motor_that_does_not_disable(monkeypatch):
    robot = FakeI2RTRobot()
    home_arms = ModuleType("home_arms")
    home_arms.close_robot_cleanly = lambda _value: None
    home_arms.disable_and_probe = lambda _channels: False
    monkeypatch.setitem(sys.modules, "home_arms", home_arms)

    adapter = yam.YAMRobot.__new__(yam.YAMRobot)
    adapter.robot = robot
    adapter._channel = "can_right"
    adapter._closed = False

    with pytest.raises(RuntimeError, match="motors stayed enabled"):
        adapter.close()

    assert adapter._closed


def test_bimanual_close_releases_second_arm_when_first_close_fails():
    class Closable:
        def __init__(self, error=None):
            self.error = error
            self.closed = False

        def close(self):
            self.closed = True
            if self.error is not None:
                raise self.error

    left = Closable(RuntimeError("left close failed"))
    right = Closable()
    robot = robot_module.BimanualRobot(left, right)

    with pytest.raises(RuntimeError, match="failed to close 1 arm"):
        robot.close()

    assert left.closed
    assert right.closed


def test_environment_closes_robot_before_camera_resources():
    events = []

    class Closable:
        def __init__(self, name):
            self.name = name

        def num_dofs(self):
            return 0

        def close(self):
            events.append(self.name)

    env = env_module.RobotEnv(
        Closable("robot"),
        camera_dict={"wrist": Closable("camera")},
        camera_client=Closable("client"),
    )

    env.close()

    assert events == ["robot", "client", "camera"]
