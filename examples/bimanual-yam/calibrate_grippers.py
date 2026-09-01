#!/usr/bin/env python3
"""Measure and save the physical closed/open limits of both YAM grippers."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import home_arms
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_LEFT_CONFIG = HERE / "configs" / "yam_left.yaml"
DEFAULT_RIGHT_CONFIG = HERE / "configs" / "yam_right.yaml"
MIN_GRIPPER_TRAVEL_RAD = 4.0
MAX_GRIPPER_TRAVEL_RAD = 6.8
GRIPPER_MOTOR_ID = 7
CALIBRATION_STEP_RAD = 0.05
CALIBRATION_MAX_SEARCH_RAD = MAX_GRIPPER_TRAVEL_RAD + 0.5
CALIBRATION_KP = 4.0
CALIBRATION_KD = 0.5
CALIBRATION_SAMPLES_PER_STEP = 3
CALIBRATION_SAMPLE_PERIOD_S = 0.02
CALIBRATION_CONTACT_TORQUE_NM = 0.55
CALIBRATION_CONTACT_ERROR_RAD = 0.12
CALIBRATION_CONTACT_MOTION_RAD = 0.015
CALIBRATION_CONTACT_STEPS = 2
CALIBRATION_MAX_FOLLOWING_ERROR_RAD = 0.25
CALIBRATION_MAX_ROTOR_TEMP_C = 45.0
CALIBRATION_MAX_MOS_TEMP_C = 60.0


def _gripper_interface(channel: str) -> Any:
    return home_arms._gripper_interface(channel)


def _check_feedback(feedback: Any, channel: str) -> tuple[float, float]:
    position = float(feedback.position)
    torque = float(feedback.torque)
    rotor_temp = float(feedback.temperature_rotor)
    mos_temp = float(feedback.temperature_mos)
    values = np.asarray([position, torque, rotor_temp, mos_temp], dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{channel}: non-finite gripper feedback: {values}")
    if rotor_temp > CALIBRATION_MAX_ROTOR_TEMP_C:
        raise RuntimeError(
            f"{channel}: gripper rotor is {rotor_temp:.0f} C; wait until it is "
            f"at or below {CALIBRATION_MAX_ROTOR_TEMP_C:.0f} C"
        )
    if mos_temp > CALIBRATION_MAX_MOS_TEMP_C:
        raise RuntimeError(
            f"{channel}: gripper MOS temperature is {mos_temp:.0f} C; wait until "
            f"it is at or below {CALIBRATION_MAX_MOS_TEMP_C:.0f} C"
        )
    return position, torque


def _disable_and_read(interface: Any, motor_type: Any) -> Any:
    frame_id = interface._get_frame_id(GRIPPER_MOTOR_ID)
    message = interface._send_message_get_response(
        frame_id,
        GRIPPER_MOTOR_ID,
        [0xFF] * 7 + [0xFD],
    )
    return interface.parse_recv_message(
        message,
        motor_type,
        ignore_error=True,
    )


def _enable_and_read(interface: Any, motor_type: Any) -> Any:
    frame_id = interface._get_frame_id(GRIPPER_MOTOR_ID)
    message = interface._send_message_get_response(
        frame_id,
        GRIPPER_MOTOR_ID,
        [0xFF] * 7 + [0xFC],
    )
    return interface.parse_recv_message(
        message,
        motor_type,
        ignore_error=True,
    )


def _command_position(
    interface: Any,
    motor_type: Any,
    channel: str,
    target: float,
    sleep: Callable[[float], None],
) -> Any:
    feedback = interface.set_control(
        GRIPPER_MOTOR_ID,
        motor_type,
        target,
        0.0,
        CALIBRATION_KP,
        CALIBRATION_KD,
        0.0,
    )
    _check_feedback(feedback, channel)
    sleep(CALIBRATION_SAMPLE_PERIOD_S)
    return feedback


def _find_stop(
    interface: Any,
    motor_type: Any,
    channel: str,
    start_feedback: Any,
    *,
    direction: int,
    label: str,
    sleep: Callable[[float], None],
) -> Any:
    if direction not in {-1, 1}:
        raise ValueError("calibration direction must be -1 or +1")

    start_position, _ = _check_feedback(start_feedback, channel)
    target = start_position
    feedback = start_feedback
    contact_steps = 0
    steps = math.ceil(CALIBRATION_MAX_SEARCH_RAD / CALIBRATION_STEP_RAD)
    print(f"{channel}: finding {label} stop", flush=True)

    for _ in range(steps):
        step_start, _ = _check_feedback(feedback, channel)
        target += direction * CALIBRATION_STEP_RAD
        target_lead = direction * (target - step_start)
        # CALIBRATION_KP * this cap bounds the PD contact command at 1 Nm.
        if target_lead > CALIBRATION_MAX_FOLLOWING_ERROR_RAD:
            target = step_start + direction * CALIBRATION_MAX_FOLLOWING_ERROR_RAD
        for _ in range(CALIBRATION_SAMPLES_PER_STEP):
            feedback = _command_position(
                interface,
                motor_type,
                channel,
                target,
                sleep,
            )

        position, torque = _check_feedback(feedback, channel)
        following_error = abs(target - position)
        motion = abs(position - step_start)
        contact = (
            abs(torque) >= CALIBRATION_CONTACT_TORQUE_NM
            and following_error >= CALIBRATION_CONTACT_ERROR_RAD
            and motion <= CALIBRATION_CONTACT_MOTION_RAD
        )
        contact_steps = contact_steps + 1 if contact else 0
        if contact_steps < CALIBRATION_CONTACT_STEPS:
            continue

        endpoint = position
        for _ in range(CALIBRATION_SAMPLES_PER_STEP):
            feedback = _command_position(
                interface,
                motor_type,
                channel,
                endpoint,
                sleep,
            )
        print(f"{channel}: {label} stop at {endpoint:+.3f} rad", flush=True)
        return feedback

    raise RuntimeError(
        f"{channel}: no {label} stop found within "
        f"{CALIBRATION_MAX_SEARCH_RAD:.1f} rad"
    )


def validate_limits(limits: Sequence[float]) -> np.ndarray:
    if limits is None:
        raise ValueError("gripper_limits is null")
    try:
        measured = np.asarray(limits, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected two numeric gripper limits, got {limits!r}") from exc
    if measured.shape != (2,) or not np.all(np.isfinite(measured)):
        raise ValueError(f"expected two finite gripper limits, got {measured}")
    if measured[0] <= measured[1]:
        raise ValueError(
            "expected the closed limit to be greater than the open limit for "
            f"this YAM gripper, got {measured}"
        )
    travel = abs(float(measured[1] - measured[0]))
    if not MIN_GRIPPER_TRAVEL_RAD <= travel <= MAX_GRIPPER_TRAVEL_RAD:
        raise ValueError(
            f"measured travel {travel:.3f} rad is outside the expected "
            f"{MIN_GRIPPER_TRAVEL_RAD:.1f}-{MAX_GRIPPER_TRAVEL_RAD:.1f} rad range"
        )
    return measured


def measure_limits(
    channel: str,
    *,
    make_interface: Callable[[str], Any] = _gripper_interface,
    sleep: Callable[[float], None] = time.sleep,
    motor_type: Any = None,
) -> np.ndarray:
    """Measure one gripper without starting I2RT's motor-chain threads."""
    if motor_type is None:
        from i2rt.motor_drivers.dm_driver import MotorType

        motor_type = MotorType.DM4310

    print(f"{channel}: measuring gripper limits", flush=True)
    interface = None
    try:
        interface = make_interface(channel)
        for _ in range(40):
            if interface.bus.recv(timeout=0.05) is None:
                break

        disabled_feedback = _disable_and_read(interface, motor_type)
        _check_feedback(disabled_feedback, channel)
        if str(disabled_feedback.error_code) != "0x0":
            raise RuntimeError(
                f"{channel}: motor 7 did not enter the disabled state "
                f"(reported {disabled_feedback.error_code})"
            )
        print(
            f"{channel}: motor temperatures: "
            f"rotor={float(disabled_feedback.temperature_rotor):.0f} C, "
            f"MOS={float(disabled_feedback.temperature_mos):.0f} C",
            flush=True,
        )

        feedback = _enable_and_read(interface, motor_type)
        if str(feedback.error_code) != "0x1":
            raise RuntimeError(
                f"{channel}: motor 7 refused to enable "
                f"(reported {feedback.error_code}); calibration will not clear "
                "a motor fault automatically"
            )
        hold_position, _ = _check_feedback(feedback, channel)
        for _ in range(CALIBRATION_SAMPLES_PER_STEP):
            feedback = _command_position(
                interface,
                motor_type,
                channel,
                hold_position,
                sleep,
            )

        feedback = _find_stop(
            interface,
            motor_type,
            channel,
            feedback,
            direction=1,
            label="closed",
            sleep=sleep,
        )
        closed, _ = _check_feedback(feedback, channel)
        feedback = _find_stop(
            interface,
            motor_type,
            channel,
            feedback,
            direction=-1,
            label="open",
            sleep=sleep,
        )
        opened, _ = _check_feedback(feedback, channel)
        measured = validate_limits([closed, opened])
        print(
            f"{channel}: closed={measured[0]:+.3f}, "
            f"open={measured[1]:+.3f}, "
            f"travel={abs(measured[1] - measured[0]):.3f} rad",
            flush=True,
        )
        return measured
    finally:
        if interface is not None:
            try:
                interface.motor_off(GRIPPER_MOTOR_ID)
            except Exception as exc:  # noqa: BLE001 - final probe verifies shutdown
                print(f"{channel}: gripper disable failed: {exc}", file=sys.stderr)
            try:
                interface.close()
            except Exception as exc:  # noqa: BLE001 - final probe verifies shutdown
                print(f"{channel}: CAN close failed: {exc}", file=sys.stderr)


def render_gripper_limits(text: str, limits: Sequence[float]) -> str:
    from configure_rig import _replace_nested_scalar

    measured = validate_limits(limits)
    rendered = json.dumps([float(value) for value in measured])
    return _replace_nested_scalar(text, "robot", "gripper_limits", rendered)


def write_calibration(
    left_path: Path,
    right_path: Path,
    left_text: str,
    right_text: str,
) -> None:
    for path in (left_path, right_path):
        backup = path.with_name(f"{path.name}.calibration.bak")
        if not backup.exists():
            shutil.copy2(path, backup)

    pending = []
    for path, text in ((left_path, left_text), (right_path, right_text)):
        temporary = path.with_name(f".{path.name}.calibration.tmp")
        temporary.write_text(text, encoding="utf-8")
        pending.append((temporary, path))
    for temporary, path in pending:
        os.replace(temporary, path)


def _configured_channel(config_path: Path, env_name: str) -> str:
    import yaml

    if override := os.environ.get(env_name):
        return override
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return str(config["robot"]["channel"])


def _confirm(assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(
            "Refusing to move grippers without an interactive terminal or --yes.",
            file=sys.stderr,
        )
        return False
    print("This moves each gripper through its full physical travel.")
    print("Remove all objects, home the arms, and keep the e-stop within reach.")
    answer = input("Measure and save both grippers [y/N]: ")
    return answer.strip().lower() in {"y", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="skip the motion prompt")
    parser.add_argument("--left-config", type=Path, default=DEFAULT_LEFT_CONFIG)
    parser.add_argument("--right-config", type=Path, default=DEFAULT_RIGHT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left_path = args.left_config.resolve()
    right_path = args.right_config.resolve()
    channels = (
        _configured_channel(left_path, "YAM_CAN_LEFT"),
        _configured_channel(right_path, "YAM_CAN_RIGHT"),
    )
    if channels[0] == channels[1]:
        print(f"Both arms resolve to {channels[0]!r}; refusing calibration.")
        return 2
    if not _confirm(args.yes):
        print("No motion commanded.")
        return 2

    home_lock = home_arms.acquire_home_lock()
    if home_lock is None:
        print("Another homing or calibration process is already running.")
        return 1
    if not home_arms.stop_robot_drivers():
        print("Could not stop all robot controllers.", file=sys.stderr)
        return 1

    limits: list[np.ndarray] = []
    calibration_ok = True
    try:
        for side, channel in zip(("left", "right"), channels, strict=True):
            try:
                print(f"--- {side} gripper on {channel} ---", flush=True)
                limits.append(measure_limits(channel))
            except Exception as exc:  # noqa: BLE001 - disable both buses below
                print(f"{side} gripper calibration failed: {exc}", file=sys.stderr)
                calibration_ok = False
                break
    finally:
        # This also runs on Ctrl-C, so a partial sweep never leaves a gripper
        # or an arm energized.
        print("--- final motor state ---", flush=True)
        disabled = home_arms.disable_and_probe(channels)

    if not calibration_ok or len(limits) != 2 or not disabled:
        print("Calibration was not saved.", file=sys.stderr)
        return 1

    left_text = render_gripper_limits(left_path.read_text(encoding="utf-8"), limits[0])
    right_text = render_gripper_limits(
        right_path.read_text(encoding="utf-8"), limits[1]
    )
    write_calibration(left_path, right_path, left_text, right_text)
    print(f"Saved left limits to {left_path}")
    print(f"Saved right limits to {right_path}")
    print("Gripper calibration complete. Normal launches will skip the limit sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
