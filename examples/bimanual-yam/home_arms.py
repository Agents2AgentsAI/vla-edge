#!/usr/bin/env python3
"""Home both YAM arms, open their grippers, and disable every motor.

The default command moves hardware. It stops any other YAM controller, holds
both arms at their measured poses, ramps them to encoder q=0 together, opens
the grippers, and finishes by sending a disable frame to all 14 motors.

``--status`` sends only disable frames and reports every motor response. It
does not command a position, but it will stop an active controller first.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_LEFT_CONFIG = HERE / "configs" / "yam_left.yaml"
DEFAULT_RIGHT_CONFIG = HERE / "configs" / "yam_right.yaml"

CONTROL_HZ = 30.0
MIN_HOME_DURATION_S = 4.0
DEFAULT_HOME_MAX_VEL = 0.5
HOME_TOLERANCE_RAD = 0.08
DRIVER_STOP_GRACE_S = 25.0
GRIPPER_MOTOR_ID = 7
GRIPPER_STEP_RAD = 0.1
GRIPPER_MAX_TRAVEL_RAD = 6.5
GRIPPER_STALL_TORQUE = 0.8
GRIPPER_STALL_ERROR_RAD = 0.2
HOME_LOCK_PATH = Path(os.environ.get("YAM_HOME_LOCK", "/tmp/vla-edge-yam-home.lock"))

ROBOT_DRIVER_BASENAMES = (
    b"launch_yaml_eval_molmoact.py",
    b"lerobot_runner.py",
)


@dataclass(frozen=True)
class HomeResult:
    homed_channels: frozenset[str]
    ok: bool


def _configured_channel(env_name: str, config_path: Path, fallback: str) -> str:
    import yaml

    if override := os.environ.get(env_name):
        return override
    try:
        with config_path.open(encoding="utf-8") as stream:
            return str(yaml.safe_load(stream)["robot"]["channel"])
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return fallback


def configured_channels() -> tuple[str, str]:
    left_config = Path(os.environ.get("YAM_LEFT_CONFIG", DEFAULT_LEFT_CONFIG))
    right_config = Path(os.environ.get("YAM_RIGHT_CONFIG", DEFAULT_RIGHT_CONFIG))
    return (
        _configured_channel("YAM_CAN_LEFT", left_config, "can_left"),
        _configured_channel("YAM_CAN_RIGHT", right_config, "can_right"),
    )


def acquire_home_lock() -> Any | None:
    handle = HOME_LOCK_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def _ancestor_pids(pid: int | None = None) -> set[int]:
    current = os.getpid() if pid is None else pid
    ancestors = {current}
    while current > 1:
        try:
            fields = Path(f"/proc/{current}/stat").read_text(encoding="utf-8").split()
            current = int(fields[3])
        except (OSError, IndexError, ValueError):
            break
        if current in ancestors:
            break
        ancestors.add(current)
    return ancestors


def find_robot_drivers() -> list[int]:
    excluded = _ancestor_pids()
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(arg.endswith(name) for arg in argv for name in ROBOT_DRIVER_BASENAMES):
            matches.append(pid)
    return sorted(matches)


def stop_robot_drivers(grace_s: float = DRIVER_STOP_GRACE_S) -> bool:
    victims = find_robot_drivers()
    for pid in victims:
        print(f"stopping robot controller pid {pid}", flush=True)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    if victims:
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline and find_robot_drivers():
            time.sleep(0.25)
        for pid in find_robot_drivers():
            print(
                f"controller pid {pid} did not stop cleanly; sending SIGKILL",
                flush=True,
            )
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(1.0)
    return not find_robot_drivers()


def _robot_factory(channel: str) -> Any:
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    return get_yam_robot(
        channel=channel,
        gripper_type=GripperType.NO_GRIPPER,
        zero_gravity_mode=False,
    )


def _joint_positions(robot: Any) -> np.ndarray:
    positions = np.asarray(robot.get_observations()["joint_pos"], dtype=float)
    if positions.shape != (6,) or not np.all(np.isfinite(positions)):
        raise RuntimeError(f"invalid six-joint state: {positions}")
    return positions


def _legacy_control_thread(chain: Any) -> threading.Thread | None:
    """Find the unexposed CAN control thread in I2RT 1.3.3."""
    for thread in threading.enumerate():
        target = getattr(thread, "_target", None)
        if (
            getattr(target, "__self__", None) is chain
            and getattr(target, "__name__", "")
            == "_set_torques_and_update_state"
        ):
            return thread
    return None


def close_robot_cleanly(robot: Any) -> None:
    """Close a robot without racing I2RT 1.3.3's CAN control thread."""
    chain = getattr(robot, "motor_chain", None)
    if chain is None or hasattr(chain, "_control_thread"):
        robot.close()
        return

    control_thread = _legacy_control_thread(chain)
    if control_thread is None:
        robot.close()
        return

    # MotorChainRobot.close() already stops this server first. Repeat that
    # ordering here, then stop and join the lower-level thread that 1.3.3 did
    # not retain. Calling robot.close() afterward is safe and keeps I2RT's
    # normal recording cleanup and user-visible status message.
    stop_event = getattr(robot, "_stop_event", None)
    server_thread = getattr(robot, "_server_thread", None)
    if stop_event is not None:
        stop_event.set()
    if server_thread is not None and server_thread is not threading.current_thread():
        server_thread.join()
    chain.running = False
    if control_thread is not threading.current_thread():
        control_thread.join()
    robot.close()


def home_arms_together(
    channels: Sequence[str],
    *,
    max_joint_vel: float = DEFAULT_HOME_MAX_VEL,
    make_robot: Callable[[str], Any] = _robot_factory,
    sleep: Callable[[float], None] = time.sleep,
) -> HomeResult:
    if not math.isfinite(max_joint_vel) or max_joint_vel <= 0.0:
        raise ValueError("home max joint velocity must be finite and positive")

    robots: dict[str, Any] = {}
    starts: dict[str, np.ndarray] = {}
    ok = True
    for channel in channels:
        print(f"opening {channel} for homing", flush=True)
        robot = None
        try:
            robot = make_robot(channel)
            starts[channel] = _joint_positions(robot)
            robots[channel] = robot
        except Exception as exc:  # noqa: BLE001 - isolate a failed arm
            print(f"{channel}: could not start homing: {exc}", flush=True)
            if robot is not None:
                try:
                    close_robot_cleanly(robot)
                except Exception as close_exc:  # noqa: BLE001
                    print(f"{channel}: cleanup after open failure: {close_exc}")
            ok = False

    if not robots:
        return HomeResult(frozenset(), False)

    active = set(robots)
    max_delta = max(float(np.max(np.abs(starts[channel]))) for channel in active)
    duration = max(MIN_HOME_DURATION_S, max_delta / max_joint_vel)
    steps = max(1, math.ceil(duration * CONTROL_HZ))
    print(
        f"homing {', '.join(sorted(active))} together over {duration:.1f} s", flush=True
    )

    homed: set[str] = set()
    try:
        for index in range(steps + 1):
            alpha = index / steps
            for channel in tuple(active):
                try:
                    robots[channel].command_joint_pos((1.0 - alpha) * starts[channel])
                except Exception as exc:  # noqa: BLE001 - finish the other arm
                    print(f"{channel}: homing command failed: {exc}", flush=True)
                    active.remove(channel)
                    ok = False
            if not active:
                break
            sleep(1.0 / CONTROL_HZ)
        sleep(0.5)

        for channel in active:
            try:
                final = _joint_positions(robots[channel])
                print(
                    f"{channel}: {np.round(starts[channel], 3)} -> {np.round(final, 3)}",
                    flush=True,
                )
                if float(np.max(np.abs(final))) > HOME_TOLERANCE_RAD:
                    raise RuntimeError(
                        f"final error exceeds {HOME_TOLERANCE_RAD:.2f} rad"
                    )
                homed.add(channel)
            except Exception as exc:  # noqa: BLE001 - report all channels
                print(f"{channel}: homing verification failed: {exc}", flush=True)
                ok = False
    finally:
        for channel, robot in robots.items():
            try:
                close_robot_cleanly(robot)
            except Exception as exc:  # noqa: BLE001 - final probe is the authority
                print(f"{channel}: close failed: {exc}", flush=True)
                ok = False

    return HomeResult(frozenset(homed), ok and homed == set(channels))


def _gripper_interface(channel: str) -> Any:
    from i2rt.motor_drivers.dm_driver import (
        ControlMode,
        DMSingleMotorCanInterface,
    )

    return DMSingleMotorCanInterface(
        channel=channel,
        bustype="socketcan",
        control_mode=ControlMode.MIT,
    )


def open_gripper(
    channel: str,
    *,
    make_interface: Callable[[str], Any] = _gripper_interface,
    sleep: Callable[[float], None] = time.sleep,
    motor_type: Any = None,
) -> bool:
    if motor_type is None:
        from i2rt.motor_drivers.dm_driver import MotorType

        motor_type = MotorType.DM4310

    interface = None
    ok = True
    reached_stop = False
    try:
        interface = make_interface(channel)
        for _ in range(40):
            if interface.bus.recv(timeout=0.05) is None:
                break
        position = interface.motor_on(GRIPPER_MOTOR_ID, motor_type).position

        def hold(target: float) -> Any:
            return interface.set_control(
                GRIPPER_MOTOR_ID,
                motor_type,
                target,
                0.0,
                8.0,
                0.5,
                0.0,
            )

        feedback = hold(position)
        for _ in range(19):
            sleep(0.02)
            feedback = hold(position)

        target = position
        steps = math.ceil(GRIPPER_MAX_TRAVEL_RAD / GRIPPER_STEP_RAD)
        for _ in range(steps):
            target -= GRIPPER_STEP_RAD
            for _ in range(4):
                feedback = hold(target)
                sleep(0.02)
            error = abs(feedback.position - target)
            if (
                abs(feedback.torque) > GRIPPER_STALL_TORQUE
                and error > GRIPPER_STALL_ERROR_RAD
            ):
                reached_stop = True
                break
        if not reached_stop:
            raise RuntimeError(
                f"no open stop detected within {GRIPPER_MAX_TRAVEL_RAD:.1f} rad"
            )
        print(f"{channel}: gripper open at {feedback.position:+.3f} rad", flush=True)
    except Exception as exc:  # noqa: BLE001 - cleanup still must run
        print(f"{channel}: gripper open failed: {exc}", flush=True)
        ok = False
    finally:
        if interface is not None:
            try:
                interface.motor_off(GRIPPER_MOTOR_ID)
                interface.close()
            except Exception as exc:  # noqa: BLE001 - report cleanup failure
                print(f"{channel}: gripper cleanup failed: {exc}", flush=True)
                ok = False
    return ok and reached_stop


def disable_and_probe(
    channels: Sequence[str],
    *,
    make_interface: Callable[[str], Any] = _gripper_interface,
    motor_type: Any = None,
) -> bool:
    if motor_type is None:
        from i2rt.motor_drivers.dm_driver import MotorType

        motor_type = MotorType.DM4310

    all_ok = True
    for channel in channels:
        interface = None
        states = []
        try:
            interface = make_interface(channel)
            for _ in range(40):
                if interface.bus.recv(timeout=0.05) is None:
                    break
            for motor_id in range(1, 8):
                try:
                    frame_id = interface._get_frame_id(motor_id)
                    message = interface._send_message_get_response(
                        frame_id,
                        motor_id,
                        [0xFF] * 7 + [0xFD],
                    )
                    state = interface.parse_recv_message(
                        message,
                        motor_type,
                        ignore_error=True,
                    ).error_code
                except Exception:  # noqa: BLE001 - retain one status per motor
                    state = "?"
                states.append(state)
        except Exception as exc:  # noqa: BLE001 - continue with the other bus
            print(f"{channel}: disable probe failed: {exc}", flush=True)
            states = ["?"] * 7
        finally:
            if interface is not None:
                try:
                    interface.close()
                except Exception as exc:  # noqa: BLE001
                    print(f"{channel}: probe close failed: {exc}", flush=True)
                    all_ok = False

        channel_ok = len(states) == 7 and all(state == "0x0" for state in states)
        all_ok &= channel_ok
        label = "OK (all disabled)" if channel_ok else "ATTENTION"
        print(f"{channel}: {' '.join(states)}  {label}", flush=True)
    return all_ok


def _confirm_motion(assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(
            "Refusing to move without an interactive terminal or --yes.",
            file=sys.stderr,
        )
        return False
    answer = input(
        "This will move both arms to encoder q=0 and open the grippers. Continue [y/N]: "
    )
    return answer.strip().lower() in {"y", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        action="store_true",
        help="stop controllers, disable every motor, and report responses; no position commands",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the motion confirmation"
    )
    parser.add_argument(
        "--max-joint-vel",
        type=float,
        default=float(os.environ.get("YAM_HOME_MAX_JOINT_VEL", DEFAULT_HOME_MAX_VEL)),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    home_lock = acquire_home_lock()
    if home_lock is None:
        print("Another homing process is already running; leaving it in control.")
        return 0
    channels = configured_channels()
    if len(set(channels)) != 2:
        print(
            f"Left and right arms resolve to the same CAN interface: {channels}",
            file=sys.stderr,
        )
        return 2
    if not args.status and not _confirm_motion(args.yes):
        print("No motion commanded.")
        return 2
    if not stop_robot_drivers():
        print("Could not stop all robot controllers.", file=sys.stderr)
        return 1
    if args.status:
        return 0 if disable_and_probe(channels) else 1

    result = home_arms_together(channels, max_joint_vel=args.max_joint_vel)
    grippers_ok = True
    for channel in channels:
        if channel in result.homed_channels:
            grippers_ok &= open_gripper(channel)
        else:
            print(
                f"{channel}: skipping gripper because the arm did not home", flush=True
            )
            grippers_ok = False

    print("--- final motor state ---", flush=True)
    disabled = disable_and_probe(channels)
    if result.ok and grippers_ok and disabled:
        print("arms homed; grippers open; all motors disabled", flush=True)
        return 0
    print("homing incomplete; all reachable motors were sent disable", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
