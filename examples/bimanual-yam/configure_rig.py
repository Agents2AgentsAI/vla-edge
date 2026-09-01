"""Discover and configure the cameras, CAN links, and start poses for a YAM rig."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LEFT_CONFIG = HERE / "configs" / "yam_left.yaml"
DEFAULT_RIGHT_CONFIG = HERE / "configs" / "yam_right.yaml"
CAN_BITRATE = 1_000_000
CAN_MOVE_MIN_DEGREES = 5.0
CAN_MOVE_MARGIN_DEGREES = 3.0


@dataclass(frozen=True)
class Camera:
    serial: str
    name: str


@dataclass(frozen=True)
class CanLink:
    name: str
    up: bool
    bitrate: int | None
    parent: str


def discover_cameras() -> list[Camera]:
    import pyrealsense2 as rs

    devices = rs.context().query_devices()
    return [
        Camera(
            serial=device.get_info(rs.camera_info.serial_number),
            name=device.get_info(rs.camera_info.name),
        )
        for device in devices
    ]


def _can_links_from_json(items: list[dict[str, object]]) -> list[CanLink]:
    links = []
    for item in items:
        linkinfo = item.get("linkinfo") or {}
        info_data = linkinfo.get("info_data") or {}
        bittiming = info_data.get("bittiming") or {}
        bitrate = bittiming.get("bitrate")
        parent_parts = [item.get("parentbus"), item.get("parentdev")]
        links.append(
            CanLink(
                name=str(item["ifname"]),
                up="UP" in (item.get("flags") or []),
                bitrate=int(bitrate) if bitrate is not None else None,
                parent="/".join(str(part) for part in parent_parts if part)
                or "unknown",
            )
        )
    return sorted(links, key=lambda link: (not link.name.startswith("can_"), link.name))


def discover_can_links() -> list[CanLink]:
    if shutil.which("ip") is None:
        raise RuntimeError("The `ip` command is required to configure CAN links.")
    result = subprocess.run(
        ["ip", "-details", "-json", "link", "show", "type", "can"],
        capture_output=True,
        text=True,
        check=True,
    )
    return _can_links_from_json(json.loads(result.stdout))


def _read_can_positions(channel: str) -> dict[int, float]:
    """Read output positions without enabling or commanding the motors."""
    from i2rt.motor_config_tool.dm_motor_registers import DMRegAddr, read_register
    from i2rt.motor_config_tool.utils import RawCanInterface

    interface = RawCanInterface(
        channel=channel,
        bustype="socketcan",
        bitrate=CAN_BITRATE,
    )
    positions = {}
    try:
        for motor_id in range(1, 8):
            try:
                position = float(read_register(interface, motor_id, DMRegAddr.XOUT))
                if math.isfinite(position):
                    positions[motor_id] = position
            except RuntimeError:
                continue
    finally:
        interface.close()
    if not positions:
        raise RuntimeError(f"No motors answered on {channel}")
    return positions


def _can_motion_degrees(
    before: dict[str, dict[int, float]],
    after: dict[str, dict[int, float]],
) -> dict[str, float]:
    motion = {}
    for channel, first_positions in before.items():
        second_positions = after.get(channel, {})
        common_ids = first_positions.keys() & second_positions.keys()
        if not common_ids:
            continue
        motion[channel] = max(
            abs(
                math.degrees(
                    math.remainder(
                        second_positions[motor_id] - first_positions[motor_id],
                        2 * math.pi,
                    )
                )
            )
            for motor_id in common_ids
        )
    return motion


def _identify_moved_can(motion: dict[str, float]) -> str | None:
    ranked = sorted(motion.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return None
    winner, winner_degrees = ranked[0]
    runner_up_degrees = ranked[1][1] if len(ranked) > 1 else 0.0
    if winner_degrees < CAN_MOVE_MIN_DEGREES:
        return None
    if winner_degrees < runner_up_degrees + CAN_MOVE_MARGIN_DEGREES:
        return None
    return winner


def _active_usb_can_names(links: list[CanLink]) -> list[str]:
    return [
        link.name
        for link in links
        if link.parent.startswith("usb/") and link.up and link.bitrate == CAN_BITRATE
    ]


def _verify_left_can(links: list[CanLink]) -> str | None:
    candidates = _active_usb_can_names(links)
    if len(candidates) != 2:
        print(
            "Automatic CAN identification needs exactly two USB CAN interfaces "
            "UP at 1 Mbit/s."
        )
        return None

    try:
        answer = input(
            "Identify the LEFT arm by hand-moving one joint? [Y/n]: "
        ).strip()
    except EOFError as exc:
        raise RuntimeError(
            "Interactive input ended before setup was complete."
        ) from exc
    if answer.lower() in {"n", "no"}:
        print("Trace each adapter's CAN cable, then choose its interface manually.")
        return None

    print(
        "Stop any robot controller. Keep both arms powered with their motors disabled."
    )
    print("This reads joint positions only. It does not enable or command the motors.")
    try:
        input("Keep both arms still, then press Enter to take a baseline: ")
    except EOFError as exc:
        raise RuntimeError(
            "Interactive input ended before setup was complete."
        ) from exc
    try:
        before = {channel: _read_can_positions(channel) for channel in candidates}
    except (ImportError, RuntimeError) as exc:
        print(f"Could not read the CAN buses: {exc}")
        return None

    try:
        input(
            "Hand-move one joint on the LEFT arm by at least 10 degrees, "
            "hold it there, then press Enter: "
        )
    except EOFError as exc:
        raise RuntimeError(
            "Interactive input ended before setup was complete."
        ) from exc
    try:
        after = {channel: _read_can_positions(channel) for channel in candidates}
    except RuntimeError as exc:
        print(f"Could not read the CAN buses: {exc}")
        return None

    motion = _can_motion_degrees(before, after)
    for channel in candidates:
        print(
            f"  {channel}: largest joint change {motion.get(channel, 0.0):.1f} degrees"
        )
    detected = _identify_moved_can(motion)
    if detected is None:
        print("The result was inconclusive. Confirm the cables and choose manually.")
        return None
    print(f"Detected the LEFT arm on {detected}.")
    return detected


def _load_config(path: Path) -> dict[str, object]:
    import yaml

    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _camera_serial(config: dict[str, object], role: str) -> str:
    return str(config["sensors"]["cameras"][role]["device_id"])


def _start_joints(config: dict[str, object]) -> list[float]:
    return [float(value) for value in config["agent"]["start_joints"]]


def _choose(
    label: str,
    options: list[str],
    default: str | None,
    requested: str | None,
    used: set[str],
) -> str:
    if requested is not None:
        if requested not in options:
            raise ValueError(f"{label}: {requested!r} was not detected")
        if requested in used:
            raise ValueError(f"{label}: {requested!r} is already assigned")
        return requested

    while True:
        default_hint = (
            f" [{default}]" if default in options and default not in used else ""
        )
        try:
            answer = input(f"{label}{default_hint}: ").strip()
        except EOFError as exc:
            raise RuntimeError(
                "Interactive input ended before setup was complete."
            ) from exc
        if not answer and default_hint:
            answer = str(default)
        elif answer.isdigit() and 1 <= int(answer) <= len(options):
            answer = options[int(answer) - 1]
        if answer not in options:
            print(f"Choose 1-{len(options)} or enter an exact device name.")
            continue
        if answer in used:
            print(f"{answer} is already assigned.")
            continue
        return answer


def _choose_start_pose(
    label: str, current: list[float], supplied: list[float] | None
) -> list[float]:
    if supplied is not None:
        return supplied
    print(f"{label} start_joints target: {current}")
    print("  first six values: joint angles in radians")
    print("  seventh value: normalized gripper position (0=closed, 1=open)")
    while True:
        try:
            answer = input(
                "Type KEEP after checking this target is collision-free, "
                "or enter seven joint values: "
            ).strip()
        except EOFError as exc:
            raise RuntimeError(
                "Interactive input ended before setup was complete."
            ) from exc
        if answer.lower() == "keep":
            return current
        try:
            values = [float(value) for value in answer.replace(",", " ").split()]
        except ValueError:
            values = []
        if len(values) == 7:
            return values
        print("Enter exactly seven numbers, or type KEEP.")


def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        return input(f"{prompt} [y/N]: ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def capture_camera_sheet(cameras: list[Camera], output: Path) -> Path | None:
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    labeled_frames = []
    for index, camera in enumerate(cameras, start=1):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(camera.serial)
        config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
        try:
            pipeline.start(config)
            frame = None
            for _ in range(15):
                frame = pipeline.wait_for_frames(timeout_ms=3000).get_color_frame()
            if not frame:
                raise RuntimeError("no color frame")
            image = np.asanyarray(frame.get_data()).copy()
            label = f"{index}: {camera.name}  {camera.serial}"
            cv2.rectangle(image, (0, 0), (640, 34), (0, 0, 0), -1)
            cv2.putText(
                image,
                label,
                (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            labeled_frames.append(image)
        except RuntimeError as exc:
            print(f"warning: could not capture {camera.serial}: {exc}", file=sys.stderr)
        finally:
            try:
                pipeline.stop()
            except RuntimeError:
                pass

    if not labeled_frames:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.hstack(labeled_frames))
    return output


def capture_camera_server_sheet(
    configured_roles: dict[str, str], output: Path
) -> Path | None:
    import cv2
    import numpy as np
    from camera_client import CameraClient

    client = CameraClient(
        "tcp://127.0.0.1:5555",
        request_timeout_ms=500,
        max_frame_age_sec=5.0,
    )
    try:
        observations = client.get_obs()
    except (RuntimeError, TimeoutError):
        return None
    finally:
        client.close()

    labeled_frames = []
    for index, role in enumerate(
        ("front_camera", "left_camera", "right_camera"), start=1
    ):
        if role not in observations:
            return None
        image = cv2.cvtColor(observations[role], cv2.COLOR_RGB2BGR)
        label = f"{index}: {role}  {configured_roles[role]}"
        cv2.rectangle(image, (0, 0), (640, 34), (0, 0, 0), -1)
        cv2.putText(
            image,
            label,
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        labeled_frames.append(image)

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.hstack(labeled_frames))
    return output


def _replace_nested_scalar(text: str, parent: str, key: str, rendered: str) -> str:
    lines = text.splitlines(keepends=True)
    parent_pattern = re.compile(rf"^(?P<indent>\s*){re.escape(parent)}:\s*(?:#.*)?$")
    key_pattern = re.compile(
        rf"^(?P<prefix>\s*{re.escape(key)}\s*:)[^#]*?(?P<comment>\s+#.*)?$"
    )

    for parent_index, line in enumerate(lines):
        parent_body = line.rstrip("\r\n")
        parent_match = parent_pattern.match(parent_body)
        if not parent_match:
            continue
        parent_indent = len(parent_match.group("indent"))
        for child_index in range(parent_index + 1, len(lines)):
            child = lines[child_index]
            body = child.rstrip("\r\n")
            if body.strip() and len(body) - len(body.lstrip()) <= parent_indent:
                break
            key_match = key_pattern.match(body)
            if not key_match:
                continue
            newline = child[len(body) :]
            comment = key_match.group("comment") or ""
            lines[child_index] = (
                f"{key_match.group('prefix')} {rendered}{comment}{newline}"
            )
            return "".join(lines)
        break
    raise ValueError(f"Could not find {parent}.{key} in config")


def render_configs(
    left_text: str,
    right_text: str,
    camera_roles: dict[str, str],
    left_can: str,
    right_can: str,
    left_start: list[float],
    right_start: list[float],
) -> tuple[str, str]:
    for role, serial in camera_roles.items():
        left_text = _replace_nested_scalar(
            left_text, role, "device_id", json.dumps(serial)
        )
    left_text = _replace_nested_scalar(
        left_text, "robot", "channel", json.dumps(left_can)
    )
    left_text = _replace_nested_scalar(
        left_text, "agent", "start_joints", json.dumps(left_start)
    )
    right_text = _replace_nested_scalar(
        right_text, "robot", "channel", json.dumps(right_can)
    )
    right_text = _replace_nested_scalar(
        right_text, "agent", "start_joints", json.dumps(right_start)
    )
    return left_text, right_text


def _write_configs(
    left_path: Path, right_path: Path, left_text: str, right_text: str
) -> None:
    for path in (left_path, right_path):
        backup = path.with_name(f"{path.name}.setup.bak")
        if not backup.exists():
            shutil.copy2(path, backup)

    pending = []
    for path, text in ((left_path, left_text), (right_path, right_text)):
        temporary = path.with_name(f".{path.name}.setup.tmp")
        temporary.write_text(text, encoding="utf-8")
        pending.append((temporary, path))
    for temporary, path in pending:
        os.replace(temporary, path)


def _sudo_prefix() -> list[str]:
    if os.geteuid() == 0:
        return []
    if shutil.which("sudo") is None:
        raise RuntimeError("sudo is required to configure CAN links")
    subprocess.run(["sudo", "-v"], check=True)
    return ["sudo"]


def configure_can_links(names: list[str], dry_run: bool) -> None:
    current = {link.name: link for link in discover_can_links()}
    needs_change = [
        name
        for name in names
        if not current[name].up or current[name].bitrate != CAN_BITRATE
    ]
    if not needs_change:
        print("CAN links are already UP at 1 Mbit/s.")
        return

    prefix = [] if dry_run else _sudo_prefix()
    for name in needs_change:
        commands = (
            ["ip", "link", "set", "dev", name, "down"],
            [
                "ip",
                "link",
                "set",
                "dev",
                name,
                "type",
                "can",
                "bitrate",
                str(CAN_BITRATE),
            ],
            ["ip", "link", "set", "dev", name, "up"],
        )
        for command in commands:
            if dry_run:
                print("would run:", " ".join([*prefix, *command]))
            else:
                subprocess.run([*prefix, *command], check=True)

    if dry_run:
        return
    verified = {link.name: link for link in discover_can_links()}
    for name in names:
        link = verified[name]
        if not link.up or link.bitrate != CAN_BITRATE:
            raise RuntimeError(f"{name} did not come UP at 1 Mbit/s")


def _print_camera_options(cameras: list[Camera]) -> None:
    print("\nDetected RealSense cameras:")
    for index, camera in enumerate(cameras, start=1):
        print(f"  {index}. {camera.serial}  {camera.name}")


def _print_can_options(links: list[CanLink]) -> None:
    print("\nDetected CAN interfaces:")
    for index, link in enumerate(links, start=1):
        rate = (
            f"{link.bitrate / 1_000_000:g} Mbit/s" if link.bitrate else "unconfigured"
        )
        state = "UP" if link.up else "DOWN"
        print(f"  {index}. {link.name}  {state}, {rate}, {link.parent}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-camera")
    parser.add_argument("--left-camera")
    parser.add_argument("--right-camera")
    parser.add_argument("--left-can")
    parser.add_argument("--right-can")
    parser.add_argument("--left-start-joints", nargs=7, type=float)
    parser.add_argument("--right-start-joints", nargs=7, type=float)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--skip-can-up", action="store_true")
    parser.add_argument("--skip-can-verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="accept the final summary")
    parser.add_argument("--left-config", type=Path, default=DEFAULT_LEFT_CONFIG)
    parser.add_argument("--right-config", type=Path, default=DEFAULT_RIGHT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left_config = args.left_config.resolve()
    right_config = args.right_config.resolve()
    left_data = _load_config(left_config)
    right_data = _load_config(right_config)

    cameras = discover_cameras()
    if len(cameras) != 3:
        raise RuntimeError(
            f"Expected exactly three RealSense cameras, found {len(cameras)}"
        )
    _print_camera_options(cameras)
    print("Press Enter to keep any value shown in brackets.")
    configured_camera_roles = {
        role: _camera_serial(left_data, role)
        for role in ("front_camera", "left_camera", "right_camera")
    }
    if not args.no_preview:
        sheet_path = HERE / "yam_eval_runs" / "rig_setup" / "cameras.jpg"
        sheet = capture_camera_server_sheet(configured_camera_roles, sheet_path)
        if sheet is None:
            sheet = capture_camera_sheet(cameras, sheet_path)
        if sheet:
            print(f"Camera snapshots: {sheet}")

    serials = [camera.serial for camera in cameras]
    camera_requests = {
        "front_camera": args.front_camera,
        "left_camera": args.left_camera,
        "right_camera": args.right_camera,
    }
    camera_roles = {}
    used_cameras: set[str] = set()
    for role in ("front_camera", "left_camera", "right_camera"):
        selected = _choose(
            f"Serial for {role}",
            serials,
            configured_camera_roles[role],
            camera_requests[role],
            used_cameras,
        )
        camera_roles[role] = selected
        used_cameras.add(selected)

    can_links = discover_can_links()
    if len(can_links) < 2:
        raise RuntimeError(
            f"Expected at least two CAN interfaces, found {len(can_links)}"
        )
    _print_can_options(can_links)
    can_names = [link.name for link in can_links]
    detected_left_can = None
    if (
        args.left_can is None
        and args.right_can is None
        and not args.skip_can_verify
        and not args.dry_run
    ):
        detected_left_can = _verify_left_can(can_links)

    configured_left_can = str(left_data["robot"]["channel"])
    configured_right_can = str(right_data["robot"]["channel"])
    left_default = detected_left_can or configured_left_can
    right_default = configured_right_can
    if detected_left_can is not None:
        other_candidates = [
            name
            for name in _active_usb_can_names(can_links)
            if name != detected_left_can
        ]
        if len(other_candidates) == 1:
            right_default = other_candidates[0]

    used_can: set[str] = set()
    left_can = _choose(
        "CAN interface connected to the LEFT arm",
        can_names,
        left_default,
        args.left_can,
        used_can,
    )
    used_can.add(left_can)
    right_can = _choose(
        "CAN interface connected to the RIGHT arm",
        can_names,
        right_default,
        args.right_can,
        used_can,
    )

    left_start = _choose_start_pose(
        "Left arm", _start_joints(left_data), args.left_start_joints
    )
    right_start = _choose_start_pose(
        "Right arm", _start_joints(right_data), args.right_start_joints
    )

    print("\nRig configuration:")
    print(
        f"  cameras: front={camera_roles['front_camera']}, left={camera_roles['left_camera']}, right={camera_roles['right_camera']}"
    )
    print(f"  CAN: left={left_can}, right={right_can}")
    print(f"  left start_joints:  {left_start}")
    print(f"  right start_joints: {right_start}")
    if not _confirm("Apply this configuration", args.yes):
        print("No changes made.")
        return 1

    if not args.skip_can_up:
        configure_can_links([left_can, right_can], args.dry_run)

    left_text, right_text = render_configs(
        left_config.read_text(encoding="utf-8"),
        right_config.read_text(encoding="utf-8"),
        camera_roles,
        left_can,
        right_can,
        left_start,
        right_start,
    )
    if args.dry_run:
        print("Dry run complete. No files or CAN links were changed.")
        return 0

    _write_configs(left_config, right_config, left_text, right_text)
    print(f"Wrote {left_config}")
    print(f"Wrote {right_config}")
    print("Rig setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
