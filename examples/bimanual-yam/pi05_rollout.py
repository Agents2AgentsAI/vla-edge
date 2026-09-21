"""Synchronous Pi0.5 chunks with an independent bounded YAM motion owner.

Importing this module never opens hardware. Only run() constructs the motion
process, after server/camera/configuration checks have completed.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CAMERAS = {
    "top_cam": "front_camera",
    "left_cam": "left_camera",
    "right_cam": "right_camera",
}
GRIP = [6, 13]


def control_settings():
    from pi05_motion import MotionLimits

    if os.getenv("YAM_ASYNC_PLAN", "0").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
        "",
    }:
        raise ValueError(
            "Pi0.5 uses synchronous chunks; unset YAM_ASYNC_PLAN or set it to 0"
        )
    if os.getenv("YAM_RTC", "0").strip().lower() not in {"0", "false", "off", "no", ""}:
        raise ValueError("this Pi0.5 release does not support RTC")
    horizon = int(os.getenv("YAM_ACTION_HORIZON", "16"))
    if not 1 <= horizon <= 16:
        raise ValueError("YAM_ACTION_HORIZON must be within 1..16")
    velocity = float(os.getenv("YAM_MAX_JOINT_VEL", "2.2"))
    if not math.isfinite(velocity) or velocity <= 0:
        raise ValueError("YAM_MAX_JOINT_VEL must be finite and positive")
    return horizon, MotionLimits(velocity=velocity)


def preflight(here):
    """Read rig/camera/CAN configuration without constructing a motor controller."""
    import yaml
    from calibrate_grippers import validate_limits
    from camera_client import CameraClient

    control_settings()  # Includes dependency/option validation before hardware.
    configs = []
    for side in ("left", "right"):
        cfg = yaml.safe_load((here / "configs" / f"yam_{side}.yaml").read_text())
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
        if float(cfg.get("hz", 30)) != 30:
            raise ValueError("Pi0.5 uses 30 Hz policy actions; configure hz: 30")
        configs.append(cfg)
    channels = [cfg["robot"]["channel"] for cfg in configs]
    if len(set(channels)) != 2:
        raise ValueError("left and right CAN channels must be different")
    for channel in channels:
        output = subprocess.run(
            ["ip", "-j", "link", "show", "dev", channel],
            check=True,
            capture_output=True,
            text=True,
        )
        if "UP" not in json.loads(output.stdout)[0]["flags"]:
            raise ValueError(f"{channel} is not UP; run configure_rig.py")
    max_steps = int(configs[0].get("max_steps", 2500))
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    options = configs[0].get("eval", {}).get("camera_server", {})
    if not options.get("enabled"):
        raise ValueError("enable eval.camera_server and start start_camera_server.sh")
    camera = CameraClient(
        options["endpoint"],
        request_timeout_ms=1000,
        max_frame_age_sec=float(options.get("max_frame_age_sec", 0.5)),
    )
    try:
        validate_frames(camera.get_obs())
    finally:
        camera.close()
    return configs


def validate_frames(frames):
    for name in CAMERAS.values():
        value = np.asarray(frames[name])
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(f"{name} must be HWC RGB uint8")


def policy_state(snapshot):
    state = np.asarray(snapshot["measured"], dtype=np.float32).copy()
    reference = np.asarray(snapshot["reference"], dtype=np.float32)
    if (
        state.shape != (14,)
        or reference.shape != (14,)
        or not np.isfinite(state).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("invalid Pi0.5 robot state")
    state[GRIP] = reference[GRIP]  # Training uses commanded gripper opening.
    return state


def run_episode(
    motion,
    camera,
    client,
    instruction,
    *,
    horizon=16,
    max_steps=2500,
    stop=lambda: False,
    record=lambda value: None,
    observe=lambda before, after: None,
    save_frames=False,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Consume each chunk in order. Network waits never own the motor clock."""
    if not 1 <= horizon <= 16:
        raise ValueError("execution horizon must be within 1..16")
    steps = 0
    while steps < max_steps and not stop():
        frames = camera.get_obs()
        validate_frames(frames)
        snapshot = motion.check()
        state = policy_state(snapshot)
        requested = clock()
        actions, dt_ms = client.act(
            cameras={name: frames[key] for name, key in CAMERAS.items()},
            instruction=instruction,
            state=state,
            num_steps=10,
        )
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (16, 14) or not np.isfinite(actions).all():
            raise ValueError("Pi0.5 must return a finite 16x14 action chunk")
        record(
            {
                "kind": "chunk",
                "t": clock(),
                "step": steps,
                "elapsed_s": clock() - requested,
                "server_ms": dt_ms,
                "state": state.tolist(),
                "actions": actions.tolist(),
            }
        )
        # Recheck stop/owner health after every blocking policy request.
        if stop():
            break
        motion.check()
        for goal in actions[: min(horizon, max_steps - steps)]:
            if stop():
                return steps
            before = motion.check()
            started = clock()
            after = motion.set_goal(goal)
            record(
                {
                    "kind": "goal",
                    "t": started,
                    "step": steps,
                    "goal_seq": motion.seq,
                    "raw_target": goal.tolist(),
                    "reference": after["reference"].tolist(),
                    "measured": before["measured"].tolist(),
                }
            )
            # Preserve the action period; never burst goals to catch up after I/O.
            deadline = started + 1 / 30
            while clock() < deadline:
                if stop():
                    return steps + 1
                motion.check()
                sleep(min(0.01, max(0.0, deadline - clock())))
            after = motion.check()
            recorded = {"joint_positions": before["measured"]}
            if save_frames:
                # Capture fresh recording frames; motion pacing stays in its own process.
                recorded.update(
                    {key + "_rgb": value for key, value in camera.get_obs().items()}
                )
            observe(recorded, {"joint_positions": after["measured"]})
            steps += 1
    return steps


def run(args, configs, here):
    from camera_client import CameraClient
    from eval_utils import EvalRolloutSaver
    from home_arms import disable_and_probe
    from pi05_motion import MotionProcess

    from vla_edge.protocol.client import ActClient

    horizon, limits = control_settings()
    stop_file = Path(os.getenv("YAM_STAGE_FILE", "/tmp/yam_done"))
    stop_file.unlink(missing_ok=True)
    storage = configs[0]["storage"]
    root = (
        Path(storage["base_dir"])
        / "data"
        / "pi05"
        / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    saver = EvalRolloutSaver(
        root, args.instruction, save_frames=bool(storage.get("save_frames", False))
    )
    max_steps = int(configs[0].get("max_steps", 2500))
    manifest = {
        "policy": "pi05-bimanual-yam",
        "instruction": args.instruction,
        "server": args.server,
        "horizon": horizon,
        "model_horizon": 16,
        "policy_hz": 30,
        "reference_limits": asdict(limits),
        "async_plan": False,
        "rtc": False,
        "continuation": "synchronous chunks with continuous bounded references",
        "gripper_state": "commanded",
        "stop_file": str(stop_file),
        "max_steps": max_steps,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    result = {"status": "setup", "policy_actions": 0}
    motion = camera = client = None
    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        if motion is not None:
            motion.closing.set()
        raise KeyboardInterrupt

    old_handlers = {
        sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    print(
        f"[run_task] Pi0.5: synchronous {horizon}-row chunks, 100 Hz bounded motion",
        flush=True,
    )
    print(f"[run_task] recordings: {root.resolve()}", flush=True)
    print(f"[run_task] stop: touch {stop_file}", flush=True)
    try:
        client = ActClient(args.server, timeout_s=4)
        contract = client.bind(
            policy="pi05-bimanual-yam",
            model_family="pi05",
            embodiment="bimanual-yam",
            state_dim=14,
            action_dim=14,
            cameras=tuple(CAMERAS),
        )
        if contract.action_horizon != 16 or contract.gripper_state != "commanded":
            raise ValueError("Pi0.5 requires horizon 16 and commanded gripper state")
        options = configs[0]["eval"]["camera_server"]
        camera = CameraClient(
            options["endpoint"],
            request_timeout_ms=1000,
            max_frame_age_sec=float(options.get("max_frame_age_sec", 0.5)),
        )
        frames = camera.get_obs()
        validate_frames(frames)
        start = (
            configs[0]["agent"]["start_joints"] + configs[1]["agent"]["start_joints"]
        )
        # Exercise server output before enabling motors; discard the warmup targets.
        warm, _ = client.act(
            cameras={n: frames[k] for n, k in CAMERAS.items()},
            instruction=args.instruction,
            state=np.asarray(start, np.float32),
            num_steps=10,
        )
        if np.shape(warm) != (16, 14) or not np.isfinite(warm).all():
            raise ValueError("Pi0.5 preflight did not return finite 16x14 actions")
        if stop_file.exists():
            raise InterruptedError("Operator stop before enabling hardware")
        motion = MotionProcess(configs, start, root, stop_file, limits)
        with (root / "commands.jsonl").open("w") as log:
            log.write(
                json.dumps(
                    dict(
                        kind="header", t=time.monotonic(), wall=time.time(), **manifest
                    )
                )
                + "\n"
            )

            def record(value):
                log.write(json.dumps(value) + "\n")
                log.flush()

            result["policy_actions"] = run_episode(
                motion,
                camera,
                client,
                args.instruction,
                horizon=horizon,
                max_steps=max_steps,
                stop=lambda: interrupted or stop_file.exists(),
                record=record,
                observe=saver.add_step,
                save_frames=saver.save_frames,
            )
        result["status"] = "operator_stop" if stop_file.exists() else "step_limit"
    except (KeyboardInterrupt, InterruptedError):
        result["status"] = "operator_stop"
    except Exception as exc:
        logger.exception("Pi0.5 rollout failed; aborting motion")
        result.update(status="fault", reason=f"{type(exc).__name__}: {exc}")
        if motion is not None:
            motion.abort()
        saver.write_err(str(exc), saver.num_steps)
    finally:
        # Further Ctrl-C requests must not interrupt active-controller homing/close.
        for sig in old_handlers:
            signal.signal(sig, lambda *_: None)
        try:
            if motion is not None:
                try:
                    motion.close()
                except Exception as exc:
                    logger.exception("Pi0.5 motion shutdown failed")
                    result.update(status="fault", reason=str(exc))
                result["motion_outcome"] = motion.outcome
                if not motion.process.is_alive() and not disable_and_probe(
                    [cfg["robot"]["channel"] for cfg in configs]
                ):
                    result.update(status="fault", reason="Final motor disable check failed")
            if client is not None:
                client.close()
            if camera is not None:
                camera.close()
            saver.flush()
        finally:
            (root / "outcome.json").write_text(json.dumps(result, indent=2))
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    if result["status"] == "fault":
        raise RuntimeError(result["reason"])
    print("[run_task] Pi0.5 complete", flush=True)
    return 0
