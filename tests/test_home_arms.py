import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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


class ActiveRobot:
    def __init__(self, n_dofs=14):
        self.state = np.tile([0.4, -0.3, 0.2, 0.1, -0.1, 0.2, 0.3], n_dofs // 7)
        self.commands = []
        self.enabled = True

    def get_joint_state(self):
        return self.state.copy()

    def command_joint_state(self, command):
        assert self.enabled
        self.commands.append(np.asarray(command).copy())
        self.state = np.asarray(command).copy()

    def close(self):
        pytest.fail("active homing must leave closing to its owner")


@pytest.mark.parametrize("n_dofs", [7, 14])
def test_active_homing_keeps_torque_through_home_and_gripper_open(monkeypatch, n_dofs):
    monkeypatch.setattr(
        home_arms,
        "home_arms_together",
        lambda *args, **kwargs: pytest.fail("new arm driver"),
    )
    monkeypatch.setattr(
        home_arms,
        "open_gripper",
        lambda *args, **kwargs: pytest.fail("new gripper driver"),
    )
    monkeypatch.setattr(
        home_arms,
        "disable_and_probe",
        lambda *args, **kwargs: pytest.fail("premature disable"),
    )
    robot = ActiveRobot(n_dofs)
    initial = robot.state.copy()
    delays = []
    home_arms.home_active_robot(robot, sleep=delays.append)
    commands = np.array(robot.commands)
    arms = [i for i in range(n_dofs) if i % 7 != 6]
    grips = list(range(6, n_dofs, 7))
    arm_home = next(
        i for i, command in enumerate(commands) if np.all(command[arms] == 0)
    )
    np.testing.assert_array_equal(
        commands[: arm_home + 1, grips], np.tile(initial[grips], (arm_home + 1, 1))
    )
    np.testing.assert_array_equal(commands[arm_home:, arms], 0)
    np.testing.assert_array_equal(commands[-1, grips], 1)
    deltas = np.diff(np.vstack([initial, commands]), axis=0)
    assert (
        np.max(np.abs(deltas[:, arms]))
        <= home_arms.DEFAULT_HOME_MAX_VEL / home_arms.CONTROL_HZ + 1e-9
    )
    assert np.max(np.abs(deltas[:, grips])) <= 0.15 / home_arms.CONTROL_HZ + 1e-9
    assert robot.enabled and sum(delays) >= home_arms.MIN_HOME_DURATION_S


def test_active_home_starts_from_last_command_without_a_handover_jump():
    robot = ActiveRobot()
    previous = robot.state + 0.2
    home_arms.home_active_robot(robot, previous_command=previous, sleep=lambda _: None)
    arms = [i for i in range(14) if i % 7 != 6]
    assert (
        np.max(np.abs(robot.commands[0][arms] - previous[arms]))
        <= home_arms.DEFAULT_HOME_MAX_VEL / home_arms.CONTROL_HZ
    )


def test_failed_active_arm_home_does_not_open_grippers():
    robot = ActiveRobot()
    initial = robot.state.copy()

    def stuck(command):
        robot.commands.append(np.asarray(command).copy())
        # Live feedback remains away from q=0.

    robot.command_joint_state = stuck
    with pytest.raises(RuntimeError, match="homing error"):
        home_arms.home_active_robot(robot, sleep=lambda _: None)
    assert all(
        np.array_equal(command[[6, 13]], initial[[6, 13]]) for command in robot.commands
    )


def test_active_homing_aborts_immediately_when_live_feedback_fails():
    robot = ActiveRobot()

    def state():
        if robot.commands:
            raise RuntimeError("motor chain stopped")
        return robot.state.copy()

    robot.get_joint_state = state
    with pytest.raises(RuntimeError, match="motor chain stopped"):
        home_arms.home_active_robot(robot, sleep=lambda _: None)
    assert len(robot.commands) == 1


@pytest.mark.parametrize("argv,cwd,expected", [
    ([b"python3", b"/checkout/examples/bimanual-yam/run_task.py", b"fold"], None, True),
    ([b"python3", b"-B", b"examples/bimanual-yam/run_task.py", b"fold"], None, True),
    ([b"python3", b"run_task.py", b"fold"], "/checkout/examples/bimanual-yam", True),
    ([b"python3", b"run_task.py"], "/checkout/other", False),
    ([b"python3", b"/checkout/other/run_task.py"], None, False),
    ([b"python3", b"-m", b"pytest", b"/checkout/examples/bimanual-yam/run_task.py"], None, False),
    ([b"bash", b"-c", b"cat /checkout/examples/bimanual-yam/run_task.py"], None, False),
])
def test_shared_task_detection_is_scoped_to_the_yam_controller(argv, cwd, expected):
    assert home_arms._is_shared_yam_task(argv, cwd) is expected
