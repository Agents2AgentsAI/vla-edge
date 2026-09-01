import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "examples" / "bimanual-yam" / "home_arms.py"
SPEC = importlib.util.spec_from_file_location("home_arms", MODULE_PATH)
assert SPEC and SPEC.loader
home_arms = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = home_arms
SPEC.loader.exec_module(home_arms)


class FakeRobot:
    def __init__(self, state):
        self.state = np.asarray(state, dtype=float)
        self.commands = []
        self.closed = False

    def get_observations(self):
        return {"joint_pos": self.state.copy()}

    def command_joint_pos(self, command):
        self.state = np.asarray(command, dtype=float)
        self.commands.append(self.state.copy())

    def close(self):
        self.closed = True


def test_home_moves_both_arms_together_and_closes_them():
    robots = {
        "can_left": FakeRobot([0.5, -0.3, 0.2, 0.1, 0.0, -0.1]),
        "can_right": FakeRobot([-0.4, 0.2, 0.1, -0.2, 0.3, 0.0]),
    }

    result = home_arms.home_arms_together(
        tuple(robots),
        make_robot=robots.__getitem__,
        sleep=lambda _seconds: None,
    )

    assert result == home_arms.HomeResult(frozenset(robots), True)
    assert all(robot.closed for robot in robots.values())
    assert all(np.allclose(robot.state, 0.0) for robot in robots.values())
    assert len({len(robot.commands) for robot in robots.values()}) == 1


def test_home_isolates_a_channel_that_cannot_open():
    right = FakeRobot([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])

    def make_robot(channel):
        if channel == "can_left":
            raise RuntimeError("bus unavailable")
        return right

    result = home_arms.home_arms_together(
        ("can_left", "can_right"),
        make_robot=make_robot,
        sleep=lambda _seconds: None,
    )

    assert result == home_arms.HomeResult(frozenset({"can_right"}), False)
    assert right.closed
    assert np.allclose(right.state, 0.0)


class FakeBus:
    def recv(self, timeout):
        return None


class FakeGripperInterface:
    def __init__(self):
        self.bus = FakeBus()
        self.off = False
        self.closed = False

    def motor_on(self, _motor_id, _motor_type):
        return SimpleNamespace(position=0.0)

    def set_control(self, _motor_id, _motor_type, target, *_args):
        if target <= -0.4:
            return SimpleNamespace(position=-0.1, torque=1.0)
        return SimpleNamespace(position=target, torque=0.0)

    def motor_off(self, _motor_id):
        self.off = True

    def close(self):
        self.closed = True


def test_open_gripper_stops_on_contact_and_disables_motor():
    interface = FakeGripperInterface()

    ok = home_arms.open_gripper(
        "can_left",
        make_interface=lambda _channel: interface,
        sleep=lambda _seconds: None,
        motor_type=object(),
    )

    assert ok
    assert interface.off
    assert interface.closed


class FakeProbeInterface:
    def __init__(self):
        self.bus = FakeBus()
        self.closed = False

    def _get_frame_id(self, motor_id):
        return motor_id

    def _send_message_get_response(self, *_args):
        return object()

    def parse_recv_message(self, *_args, **_kwargs):
        return SimpleNamespace(error_code="0x0")

    def close(self):
        self.closed = True


def test_disable_probe_requires_all_motor_responses():
    interfaces = []

    def make_interface(_channel):
        interface = FakeProbeInterface()
        interfaces.append(interface)
        return interface

    assert home_arms.disable_and_probe(
        ("can_left", "can_right"),
        make_interface=make_interface,
        motor_type=object(),
    )
    assert len(interfaces) == 2
    assert all(interface.closed for interface in interfaces)


def test_legacy_close_joins_control_thread_before_socket_close():
    stopped = threading.Event()

    class LegacyChain:
        def __init__(self):
            self.running = True
            self.socket_closed = False

        def _set_torques_and_update_state(self):
            while self.running:
                time.sleep(0.001)
            stopped.set()

        def close(self):
            assert stopped.is_set()
            self.socket_closed = True

    class LegacyRobot:
        def __init__(self):
            self.motor_chain = LegacyChain()
            self._stop_event = threading.Event()
            self._server_thread = threading.Thread(
                target=self._stop_event.wait,
            )
            self._server_thread.start()
            self.control_thread = threading.Thread(
                target=self.motor_chain._set_torques_and_update_state,
            )
            self.control_thread.start()

        def close(self):
            self._stop_event.set()
            self._server_thread.join()
            self.motor_chain.close()

    robot = LegacyRobot()

    home_arms.close_robot_cleanly(robot)

    assert not robot.control_thread.is_alive()
    assert robot.motor_chain.socket_closed
