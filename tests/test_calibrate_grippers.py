import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

YAM_DIR = Path(__file__).parents[1] / "examples" / "bimanual-yam"
sys.path.insert(0, str(YAM_DIR))
MODULE_PATH = YAM_DIR / "calibrate_grippers.py"
SPEC = importlib.util.spec_from_file_location("calibrate_grippers", MODULE_PATH)
assert SPEC and SPEC.loader
calibrate_grippers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibrate_grippers
SPEC.loader.exec_module(calibrate_grippers)


class FakeBus:
    def recv(self, timeout):
        return None


class FakeGripperInterface:
    def __init__(
        self,
        *,
        opened=1.23,
        closed=6.4,
        rotor_temp=30.0,
        enable_error="0x1",
    ):
        self.open_stop = opened
        self.closed_stop = closed
        self.position = (opened + closed) / 2.0
        self.rotor_temp = rotor_temp
        self.enable_error = enable_error
        self.enabled = False
        self.off = False
        self.interface_closed = False
        self.command_leads = []
        self.bus = FakeBus()

    def _feedback(self, *, error_code):
        return SimpleNamespace(
            error_code=error_code,
            position=self.position,
            torque=0.0,
            temperature_rotor=self.rotor_temp,
            temperature_mos=32.0,
        )

    def _get_frame_id(self, motor_id):
        return motor_id

    def _send_message_get_response(self, *_args):
        command = _args[-1][-1]
        if command == 0xFC:
            self.enabled = self.enable_error == "0x1"
            return self._feedback(error_code=self.enable_error)
        assert command == 0xFD
        self.enabled = False
        return self._feedback(error_code="0x0")

    def parse_recv_message(self, message, *_args, **_kwargs):
        return message

    def set_control(self, _motor_id, _motor_type, target, _vel, kp, _kd, _torque):
        assert self.enabled
        previous = self.position
        self.command_leads.append(abs(target - previous))
        self.position = float(np.clip(target, self.open_stop, self.closed_stop))
        feedback = self._feedback(error_code="0x1")
        feedback.torque = kp * (target - self.position)
        feedback.velocity = self.position - previous
        return feedback

    def motor_off(self, _motor_id):
        self.enabled = False
        self.off = True

    def close(self):
        self.interface_closed = True


def test_measure_limits_accepts_shifted_motor_zero_and_closes_interface():
    interface = FakeGripperInterface()

    measured = calibrate_grippers.measure_limits(
        "can_left",
        make_interface=lambda _channel: interface,
        sleep=lambda _seconds: None,
        motor_type=object(),
    )

    np.testing.assert_allclose(measured, [6.4, 1.23])
    assert max(interface.command_leads) <= (
        calibrate_grippers.CALIBRATION_MAX_FOLLOWING_ERROR_RAD + 1e-9
    )
    assert interface.off
    assert interface.interface_closed


def test_calibration_sweep_runs_at_about_point_eight_radians_per_second():
    speed = calibrate_grippers.CALIBRATION_STEP_RAD / (
        calibrate_grippers.CALIBRATION_SAMPLES_PER_STEP
        * calibrate_grippers.CALIBRATION_SAMPLE_PERIOD_S
    )

    assert speed == pytest.approx(5 / 6)
    assert calibrate_grippers.CALIBRATION_SAMPLES_PER_STEP == 3


def test_measurement_failure_still_disables_and_closes_interface():
    interface = FakeGripperInterface()

    def fail_control(*_args):
        raise RuntimeError("CAN write failed")

    interface.set_control = fail_control

    with pytest.raises(RuntimeError, match="CAN write failed"):
        calibrate_grippers.measure_limits(
            "can_left",
            make_interface=lambda _channel: interface,
            sleep=lambda _seconds: None,
            motor_type=object(),
        )

    assert interface.off
    assert interface.interface_closed


def test_hot_gripper_is_refused_before_it_is_enabled():
    interface = FakeGripperInterface(rotor_temp=50.0)

    with pytest.raises(RuntimeError, match="wait until"):
        calibrate_grippers.measure_limits(
            "can_left",
            make_interface=lambda _channel: interface,
            sleep=lambda _seconds: None,
            motor_type=object(),
        )

    assert not interface.enabled
    assert interface.off
    assert interface.interface_closed


def test_motor_fault_is_not_cleared_automatically():
    interface = FakeGripperInterface(enable_error="0xc")

    with pytest.raises(RuntimeError, match="will not clear a motor fault"):
        calibrate_grippers.measure_limits(
            "can_left",
            make_interface=lambda _channel: interface,
            sleep=lambda _seconds: None,
            motor_type=object(),
        )

    assert not interface.enabled
    assert interface.off
    assert interface.interface_closed


@pytest.mark.parametrize(
    "limits",
    (None, [0.0, 0.2], [0.0, 8.0], [0.0, np.nan]),
)
def test_validate_limits_rejects_implausible_measurements(limits):
    with pytest.raises(ValueError):
        calibrate_grippers.validate_limits(limits)


def test_render_and_write_preserve_configs_and_first_backups(tmp_path):
    left = tmp_path / "left.yaml"
    right = tmp_path / "right.yaml"
    original = """\
robot:
  channel: can_left
  gripper_limits: null  # measured once
hz: 30
"""
    left.write_text(original)
    right.write_text(original.replace("can_left", "can_right"))

    left_text = calibrate_grippers.render_gripper_limits(original, [0.0, -5.1])
    right_text = calibrate_grippers.render_gripper_limits(
        original.replace("can_left", "can_right"), [0.1, -5.0]
    )
    calibrate_grippers.write_calibration(left, right, left_text, right_text)
    calibrate_grippers.write_calibration(left, right, left_text, right_text)

    assert "gripper_limits: [0.0, -5.1]  # measured once" in left.read_text()
    assert "hz: 30" in left.read_text()
    assert (tmp_path / "left.yaml.calibration.bak").read_text() == original


def test_keyboard_interrupt_still_disables_both_buses(monkeypatch, tmp_path):
    disabled = []
    monkeypatch.setattr(
        calibrate_grippers,
        "parse_args",
        lambda: SimpleNamespace(
            yes=True,
            left_config=tmp_path / "left.yaml",
            right_config=tmp_path / "right.yaml",
        ),
    )
    monkeypatch.setattr(
        calibrate_grippers,
        "_configured_channel",
        lambda _path, env_name: {
            "YAM_CAN_LEFT": "can_left",
            "YAM_CAN_RIGHT": "can_right",
        }[env_name],
    )
    monkeypatch.setattr(calibrate_grippers, "_confirm", lambda _yes: True)
    monkeypatch.setattr(
        calibrate_grippers.home_arms, "acquire_home_lock", lambda: object()
    )
    monkeypatch.setattr(
        calibrate_grippers.home_arms, "stop_robot_drivers", lambda: True
    )
    monkeypatch.setattr(
        calibrate_grippers,
        "measure_limits",
        lambda _channel: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    monkeypatch.setattr(
        calibrate_grippers.home_arms,
        "disable_and_probe",
        lambda channels: disabled.append(tuple(channels)) or True,
    )

    with pytest.raises(KeyboardInterrupt):
        calibrate_grippers.main()

    assert disabled == [("can_left", "can_right")]
