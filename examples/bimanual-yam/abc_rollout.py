"""ABC-VLA hard-prefix rollout on the example's calibrated YAM rig.

Scheduling is adapted from amazon-far/abc (Apache-2.0).
The action schedule follows ABC's one-request-ahead RTC: bootstrap without
an invented prefix, request with exactly the remaining committed commands,
then discard the returned prefix rows. Each command gets a full control tick.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from vla_edge.protocol.client import ActClient
from vla_edge.scripts.check_abc_rollout import validate_abc_rtc

logger = logging.getLogger(__name__)

CAMERAS = {
    "top_cam": "front_camera_rgb",
    "left_cam": "left_camera_rgb",
    "right_cam": "right_camera_rgb",
}
ARM_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def validate_contract(metadata, prefix, chunk):
    validate_abc_rtc(metadata, prefix, chunk)
    if (
        metadata.get("cameras") != list(CAMERAS)
        or metadata.get("embodiment") != "bimanual-yam"
    ):
        raise ValueError("ABC requires the top/left/right bimanual YAM camera contract")
    gripper = metadata.get("gripper", {})
    if (
        gripper.get("state_source") != "measured"
        or gripper.get("wire_convention") != "closed_0_open_1"
        or gripper.get("indices") != [6, 13]
        or not metadata.get("rtc")
    ):
        raise ValueError(
            "ABC requires measured closed=0/open=1 grippers and hard-prefix RTC"
        )


def preflight(here, metadata, prefix, chunk):
    """Read configuration, CAN link state and camera frames; never construct a robot."""
    import yaml
    from calibrate_grippers import validate_limits
    from camera_client import CameraClient

    validate_contract(metadata, prefix, chunk)
    configs = []
    for side in ("left", "right"):
        path = here / "configs" / f"yam_{side}.yaml"
        cfg = yaml.safe_load(path.read_text())
        cfg["robot"]["channel"] = os.getenv(
            f"YAM_CAN_{side.upper()}", cfg["robot"]["channel"]
        )
        validate_limits(cfg["robot"].get("gripper_limits"))
        start = np.asarray(cfg["agent"]["start_joints"], dtype=float)
        if (
            start.shape != (7,)
            or not np.isfinite(start).all()
            or not 0 <= start[6] <= 1
        ):
            raise ValueError(f"invalid {side} start_joints")
        configs.append(cfg)
    channels = [cfg["robot"]["channel"] for cfg in configs]
    if len(set(channels)) != 2:
        raise ValueError("left and right CAN interfaces must be different")
    if float(configs[0].get("hz", 30)) != 30:
        raise ValueError("ABC-VLA uses a 30 Hz action clock; set hz: 30")
    if int(configs[0].get("max_steps", 2500)) <= 0:
        raise ValueError("max_steps must be positive")
    velocity = float(os.getenv("YAM_MAX_JOINT_VEL", "2.2"))
    if not math.isfinite(velocity) or velocity <= 0:
        raise ValueError("YAM_MAX_JOINT_VEL must be finite and positive")
    for channel in channels:
        result = subprocess.run(
            ["ip", "-j", "link", "show", "dev", channel],
            check=True,
            capture_output=True,
            text=True,
        )
        if "UP" not in json.loads(result.stdout)[0]["flags"]:
            raise ValueError(f"{channel} is not UP; run configure_rig.py")
    camera = configs[0].get("eval", {}).get("camera_server", {})
    if not camera.get("enabled"):
        raise ValueError("enable eval.camera_server and start start_camera_server.sh")
    client = CameraClient(
        camera["endpoint"],
        request_timeout_ms=2000,
        max_frame_age_sec=float(camera.get("max_frame_age_sec", 0.5)),
    )
    try:
        frames = client.get_obs()
        for key in ("front_camera", "left_camera", "right_camera"):
            value = np.asarray(frames[key])
            if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
                raise ValueError(f"{key} must be HWC RGB uint8")
    finally:
        client.close()
    return configs


def _state(observation):
    state = np.asarray(observation["joint_positions"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("ABC needs 14 finite measured joint/gripper positions")
    return state.copy()


def limit_chunk(actions, previous_command, period, max_joint_vel):
    """Plan bounded setpoints before any of them become an RTC commitment.

    Arm setpoints are limited relative to the preceding command, not measured
    position (which can lag). Gripper commands retain the native ABC semantics.
    """
    if not math.isfinite(period) or period <= 0:
        raise ValueError("control period must be finite and positive")
    if not math.isfinite(max_joint_vel) or max_joint_vel <= 0:
        raise ValueError("max_joint_vel must be finite and positive")
    planned = np.asarray(actions, dtype=np.float32).copy()
    previous = np.asarray(previous_command, dtype=np.float32).copy()
    if planned.ndim != 2 or planned.shape[1] != 14 or not np.isfinite(planned).all():
        raise ValueError("invalid ABC action chunk")
    if previous.shape != (14,) or not np.isfinite(previous).all():
        raise ValueError("invalid preceding command")
    max_step = max_joint_vel * period
    for command in planned:
        lower = previous[ARM_INDICES].astype(np.float64) - max_step
        upper = previous[ARM_INDICES].astype(np.float64) + max_step
        command[ARM_INDICES] = np.clip(command[ARM_INDICES], lower, upper)
        previous = command.copy()
    return planned


def check_command(command, previous_command, period, max_joint_vel):
    """Check the committed plan without changing a command after prefix capture."""
    if command.shape != (14,) or not np.isfinite(command).all():
        raise ValueError("invalid ABC action")
    # The checkpoint may extrapolate beyond nominal gripper endpoints.
    # Preserve native ABC/I2RT gripper commands; do not clip them here.
    delta = np.max(np.abs(command[ARM_INDICES] - previous_command[ARM_INDICES]))
    if delta > max_joint_vel * period + 1e-6:
        raise ValueError(
            f"committed ABC command violates the setpoint-rate limit: {delta:.4f} rad/tick"
        )


def run_episode(
    env,
    client,
    instruction,
    *,
    prefix=5,
    chunk=6,
    max_steps=2500,
    max_joint_vel=2.2,
    initial_command=None,
    stop=lambda: False,
    record=lambda *args: None,
    on_command=lambda command: None,
    record_request=lambda *args: None,
    save_frames=False,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Run one episode. Dependencies can be faked to verify scheduling without hardware."""
    if not 1 <= prefix <= 7 or not prefix < chunk <= 30 - prefix:
        raise ValueError("require 1 <= prefix <= 7 and prefix < chunk <= 30-prefix")
    period = env.control_period_s
    random = np.random.default_rng()

    def request(obs, committed=None, anchor=None):
        cameras = {wire: np.asarray(obs[key]).copy() for wire, key in CAMERAS.items()}
        state = _state(obs)
        noise = random.standard_normal((1, 30, 14)).astype(np.float32)
        begin = clock()
        actions, elapsed = client.act(
            cameras=cameras,
            instruction=instruction,
            state=state,
            noise=noise,
            prefix_actions=committed,
            prefix_length=None if committed is None else len(committed),
        )
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (30, 14) or not np.isfinite(actions).all():
            raise ValueError("ABC server returned invalid actions")
        start = 0 if committed is None else len(committed)
        raw_chunk = actions[start : start + chunk]
        # The new suffix starts after the final command already committed to
        # execution. Never alter the remaining prefix or clamp it at send time.
        anchor = committed[-1] if committed is not None else anchor
        command_chunk = limit_chunk(raw_chunk, anchor, period, max_joint_vel)
        record_request(
            {
                "observation_time": begin,
                "state": state.tolist(),
                "noise": noise.tolist(),
                "prefix": None if committed is None else committed.tolist(),
                "actions": actions.tolist(),
                "command_anchor": np.asarray(anchor).tolist(),
                "command_chunk": command_chunk.tolist(),
                "limited_arm_values": int(
                    np.count_nonzero(
                        command_chunk[:, ARM_INDICES] != raw_chunk[:, ARM_INDICES]
                    )
                ),
                "server_ms": elapsed,
            }
        )
        return command_chunk

    if stop():
        return 0
    first_observation = env.get_obs()
    last_command = (
        _state(first_observation)
        if initial_command is None
        else np.asarray(initial_command, dtype=np.float32).copy()
    )
    actions = request(first_observation, anchor=last_command)
    steps = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        while steps < max_steps and not stop():
            pending = None
            for row, command in enumerate(actions):
                if stop() or steps >= max_steps:
                    return steps
                if row == chunk - prefix and steps + (chunk - row) < max_steps:
                    # A fresh observation precedes the first committed prefix command.
                    obs = env.get_obs()
                    pending = executor.submit(request, obs, actions[row:].copy())
                before = env.get_obs() if save_frames else env.get_robot_state()
                _state(before)  # Retain finite measured-feedback validation.
                check_command(command, last_command, period, max_joint_vel)
                env.robot().command_joint_state(command.copy())
                last_command = command.copy()
                on_command(last_command.copy())
                # Anchor to actual command submission: no catch-up bursts, including
                # the final row before waiting for the next inference.
                deadline = clock() + period
                while (remaining := deadline - clock()) > 0:
                    if stop():
                        return steps
                    sleep(min(remaining, 0.02))
                after = env.get_robot_state()
                record(before, after, command.copy(), clock())
                steps += 1
            if steps >= max_steps or stop():
                return steps
            if pending is None:
                raise RuntimeError("RTC request was not scheduled")
            # Slow inference holds the last target; never consume stale actions.
            actions = pending.result(timeout=30)
    return steps


def build_env(configs):
    from camera_client import CameraClient
    from gello_min.env import RobotEnv
    from gello_min.launch_utils import instantiate_from_dict
    from gello_min.robot import BimanualRobot

    camera = configs[0]["eval"]["camera_server"]
    resources = []
    try:
        cam = CameraClient(
            camera["endpoint"],
            request_timeout_ms=int(camera.get("request_timeout_ms", 500)),
            max_frame_age_sec=float(camera.get("max_frame_age_sec", 0.5)),
        )
        resources.append(cam)
        robots = []
        for cfg in configs:
            robot = instantiate_from_dict(cfg["robot"])
            resources.append(robot)
            robots.append(robot)
        return RobotEnv(BimanualRobot(*robots), control_rate_hz=30, camera_client=cam)
    except BaseException:
        for resource in reversed(resources):
            try:
                resource.close()
            except Exception:
                logger.exception("Resource cleanup failed during ABC startup")
        # A failing constructor can have enabled only part of an arm before
        # returning a robot object. Disable both configured buses as well.
        from home_arms import disable_and_probe

        disable_and_probe([cfg["robot"]["channel"] for cfg in configs])
        raise


def move_to_start(env, configs, velocity, stop, on_command=lambda command: None):
    target = np.concatenate([cfg["agent"]["start_joints"] for cfg in configs])
    current = _state(env.get_robot_state())
    steps = max(
        1,
        math.ceil(
            np.max(np.abs(target[ARM_INDICES] - current[ARM_INDICES]))
            / (velocity * env.control_period_s)
        ),
        math.ceil(np.max(np.abs(target[[6, 13]] - current[[6, 13]])) / 0.15),
    )
    for command in np.linspace(current, target, steps + 1)[1:]:
        if stop():
            raise KeyboardInterrupt
        env.robot().command_joint_state(command)
        on_command(command.copy())
        time.sleep(env.control_period_s)
    return target.copy()


def shutdown(env, channels, *, home, previous_command=None, max_joint_vel=0.5):
    from home_arms import disable_and_probe, home_active_robot

    if env is None:
        return
    try:
        if home:
            home_active_robot(
                env.robot(),
                previous_command=previous_command,
                max_joint_vel=max_joint_vel,
            )
    finally:
        # Disable only after normal homing; faults/interruptions still release
        # every motor, and never construct another controller to try homing again.
        try:
            env.close()
        finally:
            if not disable_and_probe(channels):
                raise RuntimeError(
                    "ABC shutdown incomplete; check the reported motor state"
                )


def run(args, configs, here):
    from eval_utils import EvalRolloutSaver

    storage = configs[0]["storage"]
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(storage["base_dir"]) / "data" / "abc-vla" / timestamp
    saver = EvalRolloutSaver(
        root, args.instruction, save_frames=bool(storage.get("save_frames", False))
    )
    stop_file = Path(os.getenv("YAM_STAGE_FILE", "/tmp/yam_done"))
    stop_file.unlink(missing_ok=True)
    velocity = float(os.getenv("YAM_MAX_JOINT_VEL", "2.2"))
    channels = [cfg["robot"]["channel"] for cfg in configs]
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "policy": "abcvla-bimanual-yam",
                "instruction": args.instruction,
                "server": args.server,
                "prefix_length": args.rtc_prefix_length,
                "execute_chunk_dim": args.execute_chunk_dim,
                "hz": 30,
                "max_joint_vel": velocity,
                "arm_limiter": "per_tick",
                "prefix_source": "committed_commands",
                "gripper_state": "measured",
                "bootstrap": "unprefixed",
                "stop_file": str(stop_file),
            },
            indent=2,
        )
    )
    env = None
    last_command = None
    normal_stop = False

    def remember_command(command):
        nonlocal last_command
        last_command = np.asarray(command).copy()

    client = ActClient(args.server, timeout_s=5)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    print(f"[run_task] recordings: {root.resolve()}")
    print(f"[run_task] stop: touch {stop_file}")
    try:
        contract = client.bind(policy="abcvla-bimanual-yam")
        validate_contract(
            client.health(), args.rtc_prefix_length, args.execute_chunk_dim
        )
        if contract.gripper_state != "measured":
            raise ValueError("ABC requires measured gripper feedback")
        env = build_env(configs)
        initial_command = move_to_start(
            env, configs, velocity, stop_file.exists, on_command=remember_command
        )
        with (
            (root / "commands.jsonl").open("w") as commands,
            (root / "requests.jsonl").open("w") as requests,
        ):

            def record(before, after, command, at):
                saver.add_step(before, after)
                commands.write(
                    json.dumps(
                        {
                            "time": at,
                            "target": command.tolist(),
                            "measured": _state(after).tolist(),
                        }
                    )
                    + "\n"
                )

            def record_request(value):
                requests.write(json.dumps(value) + "\n")
                requests.flush()

            run_episode(
                env,
                client,
                args.instruction,
                prefix=args.rtc_prefix_length,
                chunk=args.execute_chunk_dim,
                max_steps=int(configs[0].get("max_steps", 2500)),
                max_joint_vel=velocity,
                initial_command=initial_command,
                stop=stop_file.exists,
                record=record,
                on_command=remember_command,
                record_request=record_request,
                save_frames=saver.save_frames,
            )
        normal_stop = True
    except KeyboardInterrupt:
        normal_stop = True
        print("[run_task] stopping ABC rollout")
    except Exception as exc:
        saver.write_err(str(exc), saver.num_steps)
        raise
    finally:
        # Release both arms before flushing recordings, even after a controller fault.
        try:
            shutdown(
                env,
                channels,
                home=normal_stop,
                previous_command=last_command,
                max_joint_vel=velocity,
            )
        finally:
            try:
                client.close()
                saver.flush()
            finally:
                signal.signal(signal.SIGTERM, previous)
    return 0
