"""MolmoAct eval launcher.

Runs N rollouts, prompting for an instruction each time. Saves all three
cameras frame-by-frame (PNG) plus the joint trajectory (``episode.h5``) per
rollout, classifies rollouts via cv2 keypress (y/n/q) or a post-timeout
stdin prompt, and converts the session's labeled rollouts to a LeRobot v3.0
dataset on the way out.

CLI (run from this example directory)::

    python launch_yaml_eval_molmoact.py \
        --left_config_path configs/yam_left.yaml \
        --right_config_path configs/yam_right.yaml \
        -n 10

Motion-control environment flags::

    YAM_ACTION_EMA=1.0       # arm EMA alpha (default 1.0 = raw, training-rig
                             # semantics; <1.0 enables optional smoothing)
    YAM_GRIPPER_RATE=0.15    # normalized gripper travel per control tick
    YAM_LIMITER=clamp        # clamp (default) = reference contract: ONE
                             #         command per tick, arm delta clamped to
                             #         +/- tick*V_MAX vs the MEASURED pose,
                             #         never blocks, never rewinds
                             # ramp  = legacy blocking catch-up)
    YAM_ASYNC_PLAN=1         # background planning + queue merge (default on;
                             # 0 = block per chunk)
    YAM_RTC=0                # RTC-guided queue replacement (default off;
                             # requires a MolmoAct2 server when enabled)
    YAM_RTC_HORIZON=10       # queued raw-action prefix sent to RTC
    YAM_RTC_TRIGGER=auto     # auto | continuous | remaining queue rows
    YAM_RTC_SCHEDULE=linear  # zeros | ones | linear | exp
    YAM_RTC_MAX_GUIDANCE=10  # maximum in-loop prefix guidance strength
"""

from __future__ import annotations

import atexit
import logging
import math
import os
import signal
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import tyro
from camera_client import CameraClient
from eval_utils import (
    EvalRolloutSaver,
    LiveCameraView,
    RolloutOutcome,
    convert_session_to_lerobot,
    move_rollout,
    prompt_instruction,
    resolve_label,
)
from gello_min.env import RobotEnv
from gello_min.launch_utils import instantiate_from_dict, move_to_start_position
from gello_min.logging_utils import log_collect_demos
from gello_min.realsense_camera import RealSenseCamera, get_device_ids
from gello_min.robot import BimanualRobot
from molmoact_client import MolmoAct, MolmoActLocal
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Give the root logger a handler so i2rt control-loop health is included in
# rollout logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"

# Maximum arm-joint velocity in rad/s. Keep this in sync with run_task.sh.
# Grippers use normalized positions and have a separate rate limit.
V_MAX = float(os.environ.get("YAM_MAX_JOINT_VEL", "2.2"))

# Setpoint limiter contract. Clamp applies one bounded command per control
# tick. Ramp retains the older blocking interpolation path for comparison.
LIMITER = os.environ.get("YAM_LIMITER", "clamp").strip().lower()
if LIMITER not in ("ramp", "clamp"):
    raise SystemExit(f"YAM_LIMITER must be 'ramp' or 'clamp', got {LIMITER!r}")


# ---------------------------------------------------------------------------
# Robot shutdown
# ---------------------------------------------------------------------------

_env: RobotEnv | None = None
_bimanual: bool = False
_left_cfg: dict[str, Any] | None = None
_right_cfg: dict[str, Any] | None = None
_park_done: bool = False
_env_close_done: bool = False
_termination_requested: bool = False


def _handle_sigterm(_signum: int, _frame: Any) -> None:
    """Turn SIGTERM into the launcher's normal flush-and-park path."""
    global _termination_requested
    _termination_requested = True
    raise KeyboardInterrupt


def _park_robot() -> None:
    """Park the arms at start_joints once, regardless of the exit path."""
    global _park_done
    if _park_done or _env is None:
        return
    _park_done = True
    print("Parking robot at start position...")
    try:
        if _bimanual:
            move_to_start_position(_env, True, _left_cfg, _right_cfg)
        else:
            move_to_start_position(_env, False, _left_cfg)
    except Exception as exc:  # noqa: BLE001  # best-effort cleanup
        logger.warning("Parking failed: %s", exc)


def _close_env() -> None:
    """Stop the motor control threads and de-energize both arms once."""
    global _env_close_done
    if _env_close_done or _env is None:
        return
    _env_close_done = True
    try:
        _env.close()
    except Exception as exc:  # noqa: BLE001 - wrapper performs final hardware check
        logger.warning("Robot shutdown failed: %s", exc)


def _shutdown_robot() -> None:
    """Park first, then release CAN ownership and motor torque."""
    try:
        _park_robot()
    finally:
        _close_env()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: str | None = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    num_rollouts: Annotated[int, tyro.conf.arg(aliases=("-n",))] = 1
    """How many rollouts to run in this session."""


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _build_env(
    args: Args,
) -> tuple[RobotEnv, dict[str, Any], dict[str, Any] | None, bool]:
    """Build cameras + robot(s) + RobotEnv from the launch configs.

    Camera source is decided by the ``eval.camera_server.enabled`` flag in the
    left config:

    * ``true``  -> connect to the long-lived camera server over ZMQ. RealSense
      devices are owned by that server; this process never opens them.
    * ``false`` -> open ``RealSenseCamera`` objects in-process (legacy path).
    """
    left_cfg = OmegaConf.to_container(OmegaConf.load(args.left_config_path), resolve=True)
    bimanual = args.right_config_path is not None
    right_cfg = (
        OmegaConf.to_container(OmegaConf.load(args.right_config_path), resolve=True)
        if bimanual else None
    )

    cam_server_cfg = ((left_cfg.get("eval") or {}).get("camera_server") or {})
    use_server = bool(cam_server_cfg.get("enabled", False))

    camera_dict = None
    camera_client = None
    if use_server:
        endpoint = str(cam_server_cfg.get("endpoint", "tcp://127.0.0.1:5555"))
        timeout_ms = int(cam_server_cfg.get("request_timeout_ms", 500))
        max_age = cam_server_cfg.get("max_frame_age_sec", 0.5)
        max_age = float(max_age) if max_age is not None else None
        print(f"[eval] Using camera server at {endpoint} (timeout={timeout_ms} ms)")
        camera_client = CameraClient(
            endpoint=endpoint,
            request_timeout_ms=timeout_ms,
            max_frame_age_sec=max_age,
        )
        if not camera_client.ping():
            raise RuntimeError(
                f"Camera server at {endpoint} did not respond to ping. "
                "Start it with start_camera_server.sh."
                "Start it with scripts/start_camera_server.sh."
            )
    else:
        ids = get_device_ids()
        print(f"Found {len(ids)} camera devices: {ids}")
        camera_cfg = left_cfg["sensors"]["cameras"]
        camera_dict = {
            "left_camera": RealSenseCamera(camera_cfg["left_camera"]["device_id"]),
            "front_camera": RealSenseCamera(camera_cfg["front_camera"]["device_id"]),
            "right_camera": RealSenseCamera(camera_cfg["right_camera"]["device_id"]),
        }

    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )
    left_robot = instantiate_from_dict(left_robot_cfg)

    if bimanual:
        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )
        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)
    else:
        robot = left_robot

    env = RobotEnv(
        robot,
        control_rate_hz=left_cfg.get("hz", 30),
        camera_dict=camera_dict,
        camera_client=camera_client,
    )
    return env, left_cfg, right_cfg, bimanual


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _InferenceResult:
    actions: Any
    elapsed_s: float
    requested_at_s: float
    planned_step: int
    predicted_delay: int | None = None
    prefix_len: int = 0


@dataclass(frozen=True)
class _PlanOptions:
    async_plan: bool
    inference_budget_s: float
    replan_threshold: float
    rtc: bool
    rtc_horizon: int
    rtc_trigger: str
    rtc_schedule: str
    rtc_max_guidance: float


class _AdaptivePlanLead:
    """Track inference latency and convert it to an async trigger lead."""

    def __init__(self, tick_s: float, seed_ticks: int = 8, alpha: float = 0.25):
        if tick_s <= 0.0:
            raise ValueError(f"control tick must be positive, got {tick_s}")
        self.tick_s = float(tick_s)
        self.alpha = float(alpha)
        self.ema_s = max(0.0, (int(seed_ticks) - 2) * self.tick_s)
        self._observations = 0

    def observe(self, elapsed_s: float) -> None:
        sample = max(0.0, float(elapsed_s))
        if self._observations == 0:
            self.ema_s = sample
        else:
            self.ema_s = self.alpha * sample + (1.0 - self.alpha) * self.ema_s
        self._observations += 1

    @property
    def ticks(self) -> int:
        return max(1, math.ceil(self.ema_s / self.tick_s) + 2)


class _MaxLatencyDelay:
    """Reference-style RTC delay estimate: max of the last 100 latencies."""

    def __init__(self, tick_s: float, window: int = 100):
        if tick_s <= 0.0:
            raise ValueError(f"control tick must be positive, got {tick_s}")
        if window < 1:
            raise ValueError(f"latency window must be positive, got {window}")
        self.tick_s = float(tick_s)
        self.samples = deque(maxlen=int(window))

    def observe(self, elapsed_s: float) -> None:
        sample = float(elapsed_s)
        if not math.isfinite(sample):
            raise ValueError(f"inference latency must be finite, got {sample}")
        self.samples.append(max(0.0, sample))

    @property
    def ticks(self) -> int:
        if not self.samples:
            return 0
        return math.ceil(max(self.samples) / self.tick_s)


class _RTCActionDelay:
    """Predict RTC delay in consumed actions, not nominal wall-clock ticks.

    The first guided request has no overlapping control-loop observation, so
    it falls back to the reference latency/fps estimate.  Once a request has
    completed, the number of actions actually consumed while it was in flight
    is the quantity that aligns both the prefix mask and queue replacement.
    """

    def __init__(self, tick_s: float, window: int = 100):
        self._latency_fallback = _MaxLatencyDelay(tick_s, window=window)
        self._consumed_samples = deque(maxlen=int(window))

    def observe(self, elapsed_s: float) -> None:
        self._latency_fallback.observe(elapsed_s)

    def observe_consumed(self, ticks: int) -> None:
        sample = int(ticks)
        if sample < 0:
            raise ValueError(f"consumed action ticks must be >= 0, got {sample}")
        self._consumed_samples.append(sample)

    @property
    def ticks(self) -> int:
        if self._consumed_samples:
            return max(self._consumed_samples)
        return self._latency_fallback.ticks

    @property
    def source(self) -> str:
        return "observed-actions" if self._consumed_samples else "wall-clock"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(
        f"{name} must be a boolean flag (0/1, false/true, no/yes, off/on), "
        f"got {raw!r}"
    )


def _planning_options() -> _PlanOptions:
    # Async planning keeps inference off the control-loop thread.
    async_plan = _env_flag("YAM_ASYNC_PLAN", default=True)
    inference_budget_s = float(os.environ.get("YAM_INFER_BUDGET_S", "0") or 0)
    # Replan near the end of a chunk. The observed lead-time floor below
    # prevents queue starvation when inference takes longer than this fraction.
    replan_threshold = float(os.environ.get("YAM_REPLAN_THRESHOLD", "0.2"))
    # RTC is opt-in. Validate it with the latency and task mix of the target
    # platform before enabling it by default.
    rtc = _env_flag("YAM_RTC", default=False)
    rtc_horizon = int(os.environ.get("YAM_RTC_HORIZON", "10"))
    rtc_trigger = os.environ.get("YAM_RTC_TRIGGER", "auto").strip().lower()
    rtc_schedule = os.environ.get("YAM_RTC_SCHEDULE", "linear").strip().lower()
    rtc_max_guidance = float(os.environ.get("YAM_RTC_MAX_GUIDANCE", "10.0"))
    if inference_budget_s < 0.0:
        raise ValueError("YAM_INFER_BUDGET_S must be >= 0")
    if not 0.0 <= replan_threshold <= 1.0:
        raise ValueError("YAM_REPLAN_THRESHOLD must be in [0, 1]")
    if async_plan and inference_budget_s > 0.0:
        raise ValueError(
            "async planning cannot be combined with YAM_INFER_BUDGET_S > 0 "
            "(set YAM_ASYNC_PLAN=0 for budget-padded synchronous runs)"
        )
    if rtc and not async_plan:
        raise ValueError("YAM_RTC=1 requires YAM_ASYNC_PLAN=1")
    if not 1 <= rtc_horizon <= 30:
        raise ValueError("YAM_RTC_HORIZON must be in [1, 30]")
    if rtc_trigger not in {"auto", "continuous"}:
        try:
            trigger_rows = int(rtc_trigger)
        except ValueError as exc:
            raise ValueError(
                "YAM_RTC_TRIGGER must be auto, continuous, or an integer "
                "number of remaining queue rows"
            ) from exc
        if not 1 <= trigger_rows <= 30:
            raise ValueError("numeric YAM_RTC_TRIGGER must be in [1, 30]")
        rtc_trigger = str(trigger_rows)
    if rtc_schedule not in {"zeros", "ones", "linear", "exp"}:
        raise ValueError(
            "YAM_RTC_SCHEDULE must be one of zeros, ones, linear, exp"
        )
    if not math.isfinite(rtc_max_guidance) or rtc_max_guidance < 0.0:
        raise ValueError("YAM_RTC_MAX_GUIDANCE must be finite and >= 0")
    return _PlanOptions(
        async_plan=async_plan,
        inference_budget_s=inference_budget_s,
        replan_threshold=replan_threshold,
        rtc=rtc,
        rtc_horizon=rtc_horizon,
        rtc_trigger=rtc_trigger,
        rtc_schedule=rtc_schedule,
        rtc_max_guidance=rtc_max_guidance,
    )


def _rtc_trigger_rows(
    options: _PlanOptions,
    predicted_delay: int,
    chunk_horizon: int,
) -> int:
    """Resolve when RTC may submit, expressed as remaining queue rows.

    ``auto`` waits until the queue reaches delay + guidance horizon.  This
    leaves enough work to cover inference and, when chunk capacity permits,
    executes each accepted chunk beyond its guided prefix.  ``continuous``
    preserves the original always-submit behavior for comparison runs.
    """

    chunk_horizon = max(1, int(chunk_horizon))
    if options.rtc_trigger == "continuous":
        requested = chunk_horizon
    elif options.rtc_trigger == "auto":
        requested = max(0, int(predicted_delay)) + options.rtc_horizon
    else:
        requested = int(options.rtc_trigger)
    return min(chunk_horizon, max(1, requested))


def _override_gripper_obs(
    obs: dict[str, Any], cmd: Any | None
) -> dict[str, Any]:
    """Replace the observation's gripper dims with the last commanded ones.

    Training-data semantics (see the block comment in run_one_rollout):
    the dataset's gripper state tracks the commanded action even under
    load, so the policy must see the command, not the physically blocked
    finger position. Arm joints stay measured. ``cmd`` is the last emitted
    14-d command; None (before the first command) leaves obs untouched.
    """
    if cmd is None:
        return obs
    out = dict(obs)
    joints = np.asarray(obs["joint_positions"], dtype=np.float64).copy()
    gripper = (np.arange(len(joints)) % 7) == 6
    joints[gripper] = np.asarray(cmd, dtype=np.float64)[gripper]
    out["joint_positions"] = joints
    return out


def _snapshot_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Copy mutable observation arrays before handing them to the worker."""
    snapshot: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            snapshot[key] = value.copy()
        elif torch.is_tensor(value):
            snapshot[key] = value.detach().clone()
        else:
            snapshot[key] = value
    return snapshot


def _infer_action_chunk(
    policy: MolmoAct,
    obs_snapshot: dict[str, Any],
    instruction: str,
    planned_step: int,
    rtc: dict[str, Any] | None = None,
) -> _InferenceResult:
    """Prepare and infer from an observation snapshot without robot access."""
    requested_at_s = time.monotonic()
    input_dict = policy.prepare_input(obs_snapshot, instruction)
    response = (
        policy.inference(input_dict)
        if rtc is None
        else policy.inference(input_dict, rtc=rtc)
    )
    elapsed_s = time.monotonic() - requested_at_s
    if "actions" not in response:
        raise RuntimeError("policy response is missing the 'actions' field")
    return _InferenceResult(
        actions=response["actions"],
        elapsed_s=elapsed_s,
        requested_at_s=requested_at_s,
        planned_step=planned_step,
        predicted_delay=(None if rtc is None else int(rtc["inference_delay"])),
        prefix_len=(0 if rtc is None else len(rtc["prefix_actions"])),
    )


class ActionSmoother:
    """EMA-filter arm joints and rate-limit normalized gripper commands.

    Arm dimensions are every element except the seventh element of each arm.
    Grippers skip the EMA so grasps remain decisive, but each
    command is limited to ``gripper_rate`` normalized units per control tick.

    ``alpha=1.0`` is a full compatibility bypass: the input action is returned
    bit-for-bit (including grippers) so the raw-action behavior remains
    available with ``YAM_ACTION_EMA=1.0``.
    """

    def __init__(self, alpha: float, gripper_rate: float, n_dofs: int):
        self.alpha = float(alpha)
        self.gripper_rate = float(gripper_rate)
        self.n_dofs = int(n_dofs)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"action EMA alpha must be in (0, 1], got {self.alpha}")
        if not math.isfinite(self.gripper_rate) or self.gripper_rate <= 0.0:
            raise ValueError(
                "gripper rate must be a finite positive value, got "
                f"{self.gripper_rate}"
            )
        if self.n_dofs < 1:
            raise ValueError(f"n_dofs must be positive, got {self.n_dofs}")
        self._arm_mask = (np.arange(self.n_dofs) % 7) != 6
        self._value: np.ndarray | None = None

    def reset(self, measured_pose: np.ndarray) -> None:
        """Anchor the filter state to the robot's current measured pose."""
        pose = np.asarray(measured_pose, dtype=np.float64)
        if pose.shape != (self.n_dofs,):
            raise ValueError(
                f"measured pose must have shape {(self.n_dofs,)}, got {pose.shape}"
            )
        if not np.all(np.isfinite(pose)):
            raise ValueError("measured pose contains non-finite values")
        self._value = pose.copy()

    def __call__(self, action: np.ndarray) -> np.ndarray:
        action_array = np.asarray(action)
        if action_array.shape != (self.n_dofs,):
            raise ValueError(
                f"action must have shape {(self.n_dofs,)}, got {action_array.shape}"
            )
        if not np.all(np.isfinite(action_array)):
            raise ValueError("policy action contains non-finite values")
        if self._value is None:
            raise RuntimeError("ActionSmoother.reset() must be called before use")

        # Full raw-mode bypass. Keep the original dtype and values so this
        # path is suitable for exact comparisons against the raw loop.
        if self.alpha == 1.0:
            self._value = action_array.astype(np.float64, copy=True)
            return action_array.copy()

        action_float = action_array.astype(np.float64, copy=False)
        output = self._value.copy()
        output[self._arm_mask] = (
            self.alpha * action_float[self._arm_mask]
            + (1.0 - self.alpha) * self._value[self._arm_mask]
        )
        gripper_mask = ~self._arm_mask
        gripper_delta = action_float[gripper_mask] - self._value[gripper_mask]
        output[gripper_mask] = self._value[gripper_mask] + np.clip(
            gripper_delta, -self.gripper_rate, self.gripper_rate
        )
        self._value = output
        return output.copy()


# Reference aggregation weights (lerobot async_inference "weighted_average"):
# an overlapping timestep keeps 30% of the queued action, 70% of the new plan.
_AGGREGATE_OLD_WEIGHT = 0.3
_AGGREGATE_NEW_WEIGHT = 0.7


def _merge_chunk_into_queue(
    queue: deque[tuple[int, np.ndarray]],
    result: _InferenceResult,
    last_executed_step: int,
    max_queue_len: int,
    conflict_gate: float | None = None,
) -> dict[str, Any]:
    """Merge a plan into the per-global-step action queue, lerobot-style.

    Chunk action ``i`` targets global step ``result.planned_step + i``.
    Steps already executed are dropped; steps already queued are blended
    ``0.3*old + 0.7*new`` (the reference client's weighted_average); steps
    beyond the queue are appended contiguously up to ``max_queue_len``.
    Returns merge statistics for the handoff log.
    """
    try:
        n_actions = len(result.actions)
    except TypeError as exc:
        raise RuntimeError("policy actions must be a sized sequence") from exc
    if n_actions == 0:
        raise RuntimeError("policy returned an empty action chunk")

    queued_steps = {entry[0]: idx for idx, entry in enumerate(queue)}

    # Blend overlapping rows only when the plans agree within one executable
    # clamp step. Otherwise preserve committed rows and let the new plan own
    # the appended tail.
    disagreement = 0.0
    for i in range(n_actions):
        g = result.planned_step + i
        if g in queued_steps and g > last_executed_step:
            old_action = queue[queued_steps[g]][1]
            new_action = np.asarray(result.actions[i], dtype=np.float64)
            disagreement = max(
                disagreement, _arm_seam_magnitude(old_action, new_action)
            )
    conflict = conflict_gate is not None and disagreement > conflict_gate

    stale = overlap = appended = 0
    max_revision = 0.0
    for i in range(n_actions):
        g = result.planned_step + i
        new_action = np.asarray(result.actions[i], dtype=np.float64)
        if g <= last_executed_step:
            stale += 1
            continue
        if g in queued_steps:
            idx = queued_steps[g]
            old_action = queue[idx][1]
            if conflict:
                overlap += 1
                continue  # committed rows stand; the tail is the handoff
            blended = (
                _AGGREGATE_OLD_WEIGHT * old_action
                + _AGGREGATE_NEW_WEIGHT * new_action
            )
            max_revision = max(
                max_revision, _arm_seam_magnitude(old_action, blended)
            )
            queue[idx] = (g, blended)
            overlap += 1
            continue
        expected_next = queue[-1][0] + 1 if queue else last_executed_step + 1
        if g != expected_next:
            log_collect_demos(
                f"plan step {g} would leave a gap after queued step "
                f"{expected_next - 1}; dropping the remainder of this chunk",
                "warning",
            )
            break
        if len(queue) >= max_queue_len:
            break
        queue.append((g, new_action))
        appended += 1

    return {
        "stale": stale,
        "overlap": overlap,
        "appended": appended,
        "max_revision": max_revision,
        "disagreement": disagreement,
        "conflict": conflict,
        "queue_len": len(queue),
    }


def _rtc_payload_from_queue(
    queue: deque[tuple[int, np.ndarray]],
    options: _PlanOptions,
    predicted_delay: int,
    start_pose: np.ndarray | None = None,
    arm_step: float | None = None,
    gripper_step: float | None = None,
) -> dict[str, Any] | None:
    """Build a wire RTC request from the queued actions.

    RTC conditions the next plan on the prefix expected to execute during
    inference. With the clamp limiter, roll the queue forward from the
    measured pose so guidance follows feasible commands rather than raw,
    potentially saturated targets. Ramp mode uses the raw queue.
    """

    if not queue:
        return None
    prefix_entries = list(queue)[: options.rtc_horizon]
    prefix = np.stack(
        [np.asarray(action, dtype=np.float32) for _, action in prefix_entries]
    )
    if prefix.ndim != 2 or prefix.shape[1] != 14:
        raise ValueError(
            f"queued RTC prefix must have shape (K, 14), got {prefix.shape}"
        )
    if start_pose is not None:
        if arm_step is None or gripper_step is None:
            raise ValueError("feasible prefix needs arm_step and gripper_step")
        pose = np.asarray(start_pose, dtype=np.float64)
        feasible = np.empty_like(prefix)
        for row, target in enumerate(prefix):
            pose = compute_clamped_command(pose, target, arm_step, gripper_step)
            feasible[row] = pose.astype(np.float32)
        prefix = feasible
    return {
        "prefix_actions": prefix,
        "inference_delay": int(predicted_delay),
        "execution_horizon": int(options.rtc_horizon),
        "rtc_schedule": options.rtc_schedule,
        "rtc_max_guidance": float(options.rtc_max_guidance),
    }


def _rtc_merge_into_queue(
    queue: deque[tuple[int, np.ndarray]],
    result: _InferenceResult,
    current_step: int,
    max_queue_len: int = 30,
) -> dict[str, Any]:
    """Drop actions consumed during inference, then replace the live queue."""

    try:
        n_actions = len(result.actions)
    except TypeError as exc:
        raise RuntimeError("policy actions must be a sized sequence") from exc
    if n_actions == 0:
        raise RuntimeError("policy returned an empty action chunk")
    consumed = int(current_step) - int(result.planned_step)
    if consumed < 0:
        raise ValueError(
            f"RTC result was planned for future step {result.planned_step}, "
            f"current step is {current_step}"
        )

    old_handoff = None if not queue else np.asarray(queue[0][1], dtype=np.float64)
    usable = min(max(0, n_actions - consumed), int(max_queue_len))
    replacement = [
        np.asarray(result.actions[consumed + index], dtype=np.float64)
        for index in range(usable)
    ]
    queue.clear()
    queue.extend(
        (int(current_step) + index, action)
        for index, action in enumerate(replacement)
    )

    seam = float("nan")
    if old_handoff is not None and replacement:
        seam = _arm_seam_magnitude(old_handoff, replacement[0])
    predicted = result.predicted_delay
    mismatch = predicted is not None and int(predicted) != consumed
    if mismatch:
        log_collect_demos(
            f"RTC delay mismatch: predicted {int(predicted)} ticks, actual "
            f"{consumed} ticks (prefix_len={result.prefix_len}); "
            "recalibrating from observed action steps",
            "warning",
        )
    return {
        "consumed": consumed,
        "dropped": min(consumed, n_actions),
        "queue_len": len(queue),
        "seam": seam,
        "predicted_delay": predicted,
        "actual_delay": consumed,
        "prefix_len": result.prefix_len,
        "delay_mismatch": mismatch,
    }


def _arm_dim_mask(n_dofs: int) -> np.ndarray:
    """True for arm joints, False for grippers (every 7th dim)."""
    return (np.arange(n_dofs) % 7) != 6


def _arm_seam_magnitude(previous_action: np.ndarray, next_action: Any) -> float:
    next_array = np.asarray(next_action, dtype=np.float64)
    if next_array.shape != previous_action.shape:
        return float("nan")
    arm_mask = _arm_dim_mask(len(previous_action))
    return float(np.max(np.abs(next_array[arm_mask] - previous_action[arm_mask])))


def compute_clamped_command(
    curr_joints: np.ndarray,
    target_joints: np.ndarray,
    arm_limit: float,
    gripper_limit: float,
) -> np.ndarray:
    """One reference-contract setpoint: measured pose + clamped delta.

    Arm joints move at most ``arm_limit`` rad toward the target; grippers
    (every seventh element, normalized 0..1) at most ``gripper_limit``.
    The command always lies between the measured pose and the target.
    """
    curr = np.asarray(curr_joints, dtype=np.float64)
    target = np.asarray(target_joints, dtype=np.float64)
    arm_mask = _arm_dim_mask(len(curr))
    limits = np.where(arm_mask, float(arm_limit), float(gripper_limit))
    return curr + np.clip(target - curr, -limits, limits)


def clamped_step(
    env: RobotEnv,
    target_joints: np.ndarray,
    gripper_limit: float,
    cmd_sink: list[dict[str, Any]] | None = None,
    prev_cmd: Any | None = None,
) -> dict[str, Any]:
    """Reference command contract: ONE clamped setpoint per control tick.

    Arm joints: clip once relative to the MEASURED pose (anti-sawtooth;
    ``BiYamFollower.send_action`` semantics), never blocks, never rewinds.
    The clamp doubles as the speed limit: tick * V_MAX rad per tick.

    Gripper joints are rate-limited against the previous command. A physical
    gripper can stop against an object before reaching that command, so using
    the measured position as the next anchor would prevent further closing.
    """
    curr_joints = np.asarray(
        env.get_robot_state()["joint_positions"], dtype=np.float64
    )
    anchor = curr_joints.copy()
    if prev_cmd is not None:
        gripper = (np.arange(len(anchor)) % 7) == 6
        anchor[gripper] = np.asarray(prev_cmd, dtype=np.float64)[gripper]
    cmd = compute_clamped_command(
        anchor, target_joints, env.control_period_s * V_MAX, gripper_limit
    )
    # Timestamp before the call because step_command_only sleeps for the
    # remainder of the control tick before returning.
    t_emit = time.monotonic()
    env.step_command_only(cmd)
    if cmd_sink is not None:
        cmd_sink.append({"t": t_emit, "kind": "emitted", "q": cmd.tolist()})
    return env.get_obs()


def dynamic_smoothing(
    env: RobotEnv,
    target_joints: np.ndarray,
    cmd_sink: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply one policy action the way the training rigs did, capped at V_MAX.

    Normal case: ONE position setpoint per 30 Hz control tick, no software
    interpolation, matching the teleop stacks that produced the training
    data (``BiYamFollower.send_action`` + sleep(1/30)); the motor servo's
    kp/kd does the smoothing between setpoints. Software sub-stepping here is
    actively harmful now that ``get_robot_state`` returns the MEASURED pose:
    the measured pose lags the previous command during motion, so
    interpolating from it can step the command stream backward every action, a
    30 Hz sawtooth the servo renders as jerk.

    Large jumps only (post-pause catch-up, recovery; anything the servo
    should not be asked to close in one tick): ramp from the measured pose at
    V_MAX in 0.01-radian increments. The branch condition doubles as the
    global speed limit. A single-tick setpoint is only taken when its arm
    delta stays under V_MAX x tick, so lowering YAM_MAX_JOINT_VEL slows
    everything, both branches.

    Gripper joints (every seventh element) are normalized 0..1, not radians;
    they never drive the branch choice or the cap.
    """
    curr_joints = env.get_robot_state()["joint_positions"]
    delta = np.abs(curr_joints - target_joints)
    arm = np.array([(i % 7) != 6 for i in range(len(curr_joints))])
    arm_delta = float(delta[arm].max())
    tick = env.control_period_s

    if arm_delta <= tick * V_MAX:
        # Stateful rate: paces the whole action to one control period
        # measured from the previous action's end, absorbing loop overhead
        # (get_obs, saver, live view) and keeping ``Rate.last`` coherent.
        t_emit = time.monotonic()
        env.step_command_only(target_joints)
        if cmd_sink is not None:
            cmd_sink.append(
                {
                    "t": t_emit,
                    "kind": "emitted",
                    "q": np.asarray(target_joints, dtype=np.float64).tolist(),
                }
            )
        return env.get_obs()

    exec_time = arm_delta / V_MAX
    steps = int(np.clip(np.ceil(arm_delta / 0.01), 2, 100))
    waypoints = np.linspace(curr_joints, target_joints, steps + 1)[1:]
    for jnt in waypoints[:-1]:
        t_emit = time.monotonic()
        env.step_command_only(jnt, tick_s=exec_time / steps)
        if cmd_sink is not None:
            cmd_sink.append(
                {"t": t_emit, "kind": "emitted_sub", "q": jnt.tolist()}
            )
    # Final waypoint through the stateful rate so ``Rate.last`` ends the
    # action coherent for the next one (explicit-tick sub-steps never touch
    # it); the ramp has already overshot one period, so the sleep is a no-op.
    t_emit = time.monotonic()
    env.step_command_only(waypoints[-1])
    if cmd_sink is not None:
        cmd_sink.append(
            {"t": t_emit, "kind": "emitted", "q": waypoints[-1].tolist()}
        )
    return env.get_obs()


def run_one_rollout(
    env: RobotEnv,
    policy: MolmoAct,
    saver: EvalRolloutSaver,
    instruction: str,
    rollout_idx: int,
    num_rollouts: int,
    max_steps: int,
    live_view: LiveCameraView,
) -> RolloutOutcome:
    """Execute one rollout and buffer per-step observations into ``saver``.

    End conditions:

    * ``cv2`` keypress ``y`` -> success (labeled)
    * ``cv2`` keypress ``n`` -> failure (labeled)
    * ``cv2`` keypress ``q`` -> quit (no label; rollout stays in ``eval/``)
    * step >= ``max_steps`` -> timeout (stdin prompt afterwards)

    Does not flush the saver. The caller does that so the Ctrl-C path can
    also flush the partial buffer.
    """
    configured_horizon = max(1, int(policy.get_action_horizon()))
    options = _planning_options()
    # Per-episode seeding hook: the client forwards the rollout index to the
    # server (or seeds locally); a no-op unless seeding is armed there.
    if hasattr(policy, "set_episode"):
        policy.set_episode(rollout_idx)

    # Raw actions preserve the policy's timing. EMA remains available for
    # controlled comparisons.
    alpha = float(os.environ.get("YAM_ACTION_EMA", "1.0"))
    gripper_rate = float(os.environ.get("YAM_GRIPPER_RATE", "0.15"))
    measured_pose = np.asarray(
        env.get_robot_state()["joint_positions"], dtype=np.float64
    )
    smoother = ActionSmoother(alpha, gripper_rate, len(measured_pose))
    smoother.reset(measured_pose)

    tick_s = float(env.control_period_s)
    # Stream planned and emitted commands to commands.jsonl for diagnosis and
    # crash recovery. episode.h5 stores measured state only.
    cmd_records: list[dict[str, Any]] = [
        {
            "kind": "header",
            "t": time.monotonic(),
            "wall": time.time(),
            "limiter": LIMITER,
            "alpha": alpha,
            "gripper_rate": gripper_rate,
            "v_max": V_MAX,
            "tick_s": tick_s,
            "async_plan": options.async_plan,
            "rtc": options.rtc,
            "gripper_obs": "commanded",
            "instruction": instruction,
            "rollout_idx": rollout_idx,
        }
    ]
    import json as _json

    cmd_log_fh = None
    cmd_written = 0
    cmd_last_flush = time.monotonic()
    try:
        cmd_log_path = Path(saver.rollout_dir) / "commands.jsonl"
        # This handle spans the rollout and is closed in the function's finally block.
        cmd_log_fh = cmd_log_path.open("w", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001, command logging is best-effort
        logger.warning("command log open failed: %s", exc)

    def _drain_cmd_log(force_flush: bool = False) -> None:
        nonlocal cmd_written, cmd_last_flush
        if cmd_log_fh is None:
            return
        try:
            while cmd_written < len(cmd_records):
                cmd_log_fh.write(_json.dumps(cmd_records[cmd_written]) + "\n")
                cmd_written += 1
            now = time.monotonic()
            if force_flush or now - cmd_last_flush >= 1.0:
                cmd_log_fh.flush()
                cmd_last_flush = now
        except Exception as exc:  # noqa: BLE001
            logger.warning("command log write failed: %s", exc)

    _drain_cmd_log(force_flush=True)
    lead = (
        _RTCActionDelay(tick_s=tick_s)
        if options.rtc
        else _AdaptivePlanLead(tick_s=tick_s, seed_ticks=8)
    )

    log_collect_demos(
        f"Action smoother alpha={alpha:g}, gripper_rate={gripper_rate:g}/tick "
        f"(1.0 = raw); horizon={configured_horizon}; "
        f"async_plan={'on' if options.async_plan else 'off'}"
        + (
            f", RTC=on, rtc_horizon={options.rtc_horizon}, "
            f"rtc_trigger={options.rtc_trigger}, "
            f"schedule={options.rtc_schedule}, "
            f"max_guidance={options.rtc_max_guidance:g}"
            if options.rtc
            else (
                f", replan_threshold={options.replan_threshold:g}, aggregate="
                f"{_AGGREGATE_OLD_WEIGHT:g}*old+"
                f"{_AGGREGATE_NEW_WEIGHT:g}*new"
                if options.async_plan
                else ""
            )
        )
        + (
            ", RTC=off"
            if options.async_plan and not options.rtc
            else ""
        )
        + f"; v_max={V_MAX:g} rad/s; limiter={LIMITER}"
        + f"; stage_file={os.environ.get('YAM_STAGE_FILE') or 'unset'}",
        "data_info",
    )

    # The training observations track commanded gripper position. Match that
    # convention for policy input while keeping physical measurements in the
    # rollout and command logs.
    last_gripper_cmd: dict[str, Any] = {"q": None}

    def _policy_obs(obs: dict[str, Any]) -> dict[str, Any]:
        return _override_gripper_obs(obs, last_gripper_cmd["q"])

    def sync_inference(obs: dict[str, Any], planned_step: int) -> _InferenceResult:
        result = _infer_action_chunk(
            policy, _snapshot_observation(_policy_obs(obs)), instruction,
            planned_step
        )
        if options.inference_budget_s > 0.0:
            slack = options.inference_budget_s - result.elapsed_s
            if slack > 0.0:
                time.sleep(slack)
        return result

    action_queue: deque = deque()
    last_executed_step = -1

    def merge_result(
        result: _InferenceResult, mode: str, current_step: int
    ) -> None:
        lead.observe(result.elapsed_s)
        # Keep full chunks in the command log. Per-tick records contain only
        # the stitched queue and cannot recover the original plan boundaries.
        cmd_records.append(
            {
                "t": time.monotonic(),
                "kind": "chunk",
                "mode": mode,
                "planned_step": int(result.planned_step),
                "current_step": int(current_step),
                "elapsed_s": float(result.elapsed_s),
                "actions": np.asarray(
                    result.actions, dtype=np.float64
                ).tolist(),
            }
        )
        _drain_cmd_log()
        if options.rtc:
            stats = _rtc_merge_into_queue(
                action_queue,
                result,
                current_step=current_step,
                max_queue_len=configured_horizon,
            )
            if result.predicted_delay is not None:
                lead.observe_consumed(stats["actual_delay"])
            predicted = stats["predicted_delay"]
            next_trigger = _rtc_trigger_rows(
                options,
                predicted_delay=lead.ticks,
                chunk_horizon=configured_horizon,
            )
            overlap_hz = (
                stats["actual_delay"] / result.elapsed_s
                if result.elapsed_s > 0.0
                else float("nan")
            )
            log_collect_demos(
                f"RTC splice mode={mode}, inference={result.elapsed_s:.3f}s, "
                f"consumed={stats['consumed']}, predicted_delay="
                f"{predicted if predicted is not None else 'n/a'}, "
                f"actual_delay={stats['actual_delay']}, "
                f"prefix_len={stats['prefix_len']}, arm_seam="
                f"{stats['seam']:.4f} rad, queue={stats['queue_len']}, "
                f"overlap_rate={overlap_hz:.1f} Hz, "
                f"next_delay={lead.ticks} ticks ({lead.source}), "
                f"next_trigger={next_trigger} rows",
                "data_info",
            )
        else:
            stats = _merge_chunk_into_queue(
                action_queue, result, last_executed_step, configured_horizon,
                conflict_gate=tick_s * V_MAX,
            )
            log_collect_demos(
                f"Policy merge mode={mode}, inference={result.elapsed_s:.3f}s, "
                f"stale_dropped={stats['stale']}, overlap={stats['overlap']} "
                f"({'CONFLICT: committed rows kept' if stats['conflict'] else 'blended'}, "
                f"disagreement {stats['disagreement']:.3f} rad, "
                f"max arm revision {stats['max_revision']:.4f} rad), "
                f"appended={stats['appended']}, queue={stats['queue_len']}, "
                f"next_lead={lead.ticks} ticks",
                "data_info",
            )

    initial_result = sync_inference(env.get_obs(), planned_step=0)
    log_collect_demos(
        f"Policy inference {initial_result.elapsed_s:.3f}s "
        f"({len(initial_result.actions)} actions)",
        "data_info",
    )
    merge_result(initial_result, "initial", current_step=0)

    executor = (
        ThreadPoolExecutor(max_workers=1) if options.async_plan else None
    )
    pending_plan: Future[_InferenceResult] | None = None

    try:
        for step in range(max_steps):
            # Fold a finished background plan into the queue as soon as it
            # lands (the reference client's receiver thread does the same).
            if pending_plan is not None and pending_plan.done():
                merge_result(
                    pending_plan.result(), "async", current_step=step
                )
                pending_plan = None

            # Queue starved: wait for the in-flight plan; if that is entirely
            # stale (or absent), plan synchronously from a fresh observation.
            if not action_queue:
                if pending_plan is not None:
                    merge_result(
                        pending_plan.result(),
                        "async-blocked",
                        current_step=step,
                    )
                    pending_plan = None
                if not action_queue:
                    merge_result(
                        sync_inference(env.get_obs(), planned_step=step),
                        (
                            "sync"
                            if not options.async_plan
                            else "sync-fallback"
                        ),
                        current_step=step,
                    )

            obs_pre = env.get_obs()
            needs_another_chunk = step + len(action_queue) < max_steps
            if options.rtc:
                trigger_rows = _rtc_trigger_rows(
                    options,
                    predicted_delay=lead.ticks,
                    chunk_horizon=configured_horizon,
                )
            else:
                trigger_rows = max(
                    math.ceil(options.replan_threshold * configured_horizon),
                    lead.ticks,
                )
            should_submit = len(action_queue) <= trigger_rows
            if (
                options.async_plan
                and pending_plan is None
                and executor is not None
                and needs_another_chunk
                and should_submit
            ):
                # Camera/robot I/O remains on the loop thread. Only the copied
                # observation and HTTP/local policy call enter the worker.
                obs_snapshot = _snapshot_observation(_policy_obs(obs_pre))
                rtc_payload = None
                if options.rtc:
                    clamp_kwargs = {}
                    if LIMITER == "clamp":
                        # Same anchoring as execution: measured arms,
                        # commanded gripper (_policy_obs applies the
                        # gripper override when a command exists).
                        clamp_kwargs = {
                            "start_pose": np.asarray(
                                _policy_obs(obs_pre)["joint_positions"],
                                dtype=np.float64,
                            ),
                            "arm_step": tick_s * V_MAX,
                            "gripper_step": gripper_rate,
                        }
                    rtc_payload = _rtc_payload_from_queue(
                        action_queue,
                        options,
                        predicted_delay=lead.ticks,
                        **clamp_kwargs,
                    )
                    if rtc_payload is not None:
                        raw = np.stack(
                            [
                                np.asarray(a, dtype=np.float32)
                                for _, a in list(action_queue)[
                                    : options.rtc_horizon
                                ]
                            ]
                        )
                        fiction = float(
                            np.abs(raw - rtc_payload["prefix_actions"]).max()
                        )
                        cmd_records.append(
                            {
                                "t": time.monotonic(),
                                "kind": "rtc_submit",
                                "step": step,
                                "predicted_delay": int(lead.ticks),
                                "fiction": fiction,
                                "prefix": rtc_payload[
                                    "prefix_actions"
                                ].tolist(),
                                "raw_queue": raw.tolist(),
                            }
                        )
                        if fiction > 0.02 and clamp_kwargs:
                            log_collect_demos(
                                f"RTC prefix feasibility: raw queue deviates "
                                f"{fiction:.3f} rad from feasible execution "
                                f"(sending feasible)",
                                "data_info",
                            )
                pending_plan = executor.submit(
                    _infer_action_chunk,
                    policy,
                    obs_snapshot,
                    instruction,
                    step,
                    rtc_payload,
                )

            queued_step, planned_action = action_queue.popleft()
            if queued_step != step:
                log_collect_demos(
                    f"queue head targets step {queued_step} but the loop is at "
                    f"step {step}; executing it anyway (contiguity bug?)",
                    "warning",
                )
            t_plan = time.monotonic()
            cmd_records.append(
                {
                    "t": t_plan,
                    "kind": "planned",
                    "step": step,
                    "q": np.asarray(planned_action, dtype=np.float64).tolist(),
                }
            )
            action = smoother(np.asarray(planned_action))
            cmd_records.append(
                {
                    "t": time.monotonic(),
                    "kind": "smoothed",
                    "step": step,
                    "q": np.asarray(action, dtype=np.float64).tolist(),
                }
            )
            n_before_emit = len(cmd_records)
            if LIMITER == "clamp":
                obs_post = (
                    clamped_step(
                        env,
                        action,
                        gripper_rate,
                        cmd_records,
                        prev_cmd=last_gripper_cmd["q"],
                    )
                    or obs_pre
                )
            else:
                obs_post = dynamic_smoothing(env, action, cmd_records) or obs_pre
            for record in cmd_records[n_before_emit:]:
                record["step"] = step
                if record["kind"].startswith("emitted"):
                    last_gripper_cmd["q"] = record["q"]
            _drain_cmd_log()
            last_executed_step = step

            saver.add_step(obs_pre=obs_pre, obs_post=obs_post)

            key = live_view.update(
                obs=obs_pre,
                rollout_idx=rollout_idx,
                num_rollouts=num_rollouts,
                step=step + 1,
                max_steps=max_steps,
                instruction=instruction,
            )
            # File-triggered stage advance for headless staged runs: when
            # YAM_STAGE_FILE is set and that file appears, end this rollout as
            # a success (the operator signals goal completion out-of-band).
            stage_file = os.environ.get("YAM_STAGE_FILE")
            if stage_file and os.path.exists(stage_file):
                os.unlink(stage_file)
                return RolloutOutcome(end_reason="success", last_step=step + 1)
            if key == "y":
                return RolloutOutcome(end_reason="success", last_step=step + 1)
            if key == "n":
                return RolloutOutcome(end_reason="failure", last_step=step + 1)
            if key == "q":
                return RolloutOutcome(end_reason="quit", last_step=step + 1)

        return RolloutOutcome(end_reason="timeout", last_step=max_steps)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        _drain_cmd_log(force_flush=True)
        if cmd_log_fh is not None:
            try:
                cmd_log_fh.close()
            except OSError:
                logger.warning("Failed to close command log", exc_info=True)
            log_collect_demos(
                f"command log: {cmd_written - 1} records -> "
                f"{Path(saver.rollout_dir) / 'commands.jsonl'}",
                "data_info",
            )


# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------


def run_session(
    env: RobotEnv,
    policy: MolmoAct,
    left_cfg: dict[str, Any],
    right_cfg: dict[str, Any] | None,
    bimanual: bool,
    num_rollouts: int,
) -> None:
    """Drive ``num_rollouts`` rollouts; convert the labeled set to LeRobot at the end.

    Catches ``KeyboardInterrupt`` so an in-progress rollout still gets flushed
    (as incomplete, with ``err.md``) and any rollouts already labeled in this
    session are still converted.
    """
    storage = left_cfg["storage"]
    base_save_dir = Path(storage["base_dir"]) / "data" / storage["task_directory"]
    max_steps = int(left_cfg.get("max_steps", 1000))
    last_prompt = storage.get("language_instruction") or ""

    eval_cfg = left_cfg.get("eval") or {}
    cam_srv_cfg = eval_cfg.get("camera_server") or {}
    pub_endpoint = cam_srv_cfg.get("pub_endpoint") if cam_srv_cfg.get("enabled") else None
    live_view = LiveCameraView(
        enabled=bool(eval_cfg.get("live_view_enabled", True)),
        pub_endpoint=pub_endpoint,
        recv_timeout_ms=int(cam_srv_cfg.get("recv_timeout_ms", 100)),
    )

    session_timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    labeled_rollouts: list[Path] = []
    saver: EvalRolloutSaver | None = None
    outcome: RolloutOutcome | None = None

    try:
        for rollout_idx in range(num_rollouts):
            # YAM_KEEP_POSE_BETWEEN_ROLLOUTS=1: park only before the first
            # rollout, so staged multi-instruction sequences (e.g. one arm
            # holding an object across stages) are not reset mid-sequence.
            if rollout_idx == 0 or not os.environ.get("YAM_KEEP_POSE_BETWEEN_ROLLOUTS"):
                move_to_start_position(env, bimanual, left_cfg, right_cfg)
            instruction = prompt_instruction(rollout_idx, num_rollouts, last_prompt)
            last_prompt = instruction

            rollout_timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            rollout_dir = base_save_dir / "eval" / rollout_timestamp
            saver = EvalRolloutSaver(
                rollout_dir=rollout_dir,
                instruction=instruction,
                max_workers=int(storage.get("saver_max_workers", 2)),
                png_compress_level=int(storage.get("png_compress_level", 1)),
                save_frames=bool(storage.get("save_frames", False)),
            )

            print(f"\n--- Rollout {rollout_idx + 1}/{num_rollouts} ---")
            print(f"  instruction: {instruction}")
            print(f"  rollout_dir: {rollout_dir}")

            outcome = run_one_rollout(
                env=env,
                policy=policy,
                saver=saver,
                instruction=instruction,
                rollout_idx=rollout_idx,
                num_rollouts=num_rollouts,
                max_steps=max_steps,
                live_view=live_view,
            )

            saver.flush()
            label = resolve_label(outcome)
            if label is not None:
                new_path = move_rollout(rollout_dir, label, base_save_dir)
                labeled_rollouts.append(new_path)
                print(f"  -> labeled '{label}': {new_path}")
            else:
                print(f"  -> kept in eval/: {rollout_dir}")

            saver = None
            outcome = None
    except KeyboardInterrupt:
        if _termination_requested:
            print("\n[interrupt] Stop requested; saving the incomplete rollout...")
        else:
            print(
                "\n[interrupt] Ctrl-C received; saving the incomplete rollout, "
                "then converting..."
            )
        if saver is not None:
            try:
                saver.flush()
                saver.write_err(
                    reason="KeyboardInterrupt",
                    step=outcome.last_step if outcome else saver.num_steps,
                )
                print(f"  -> incomplete rollout saved: {saver.rollout_dir}")
            except Exception:
                logger.exception("Failed to flush incomplete rollout")
    finally:
        try:
            live_view.close()
        except Exception as exc:  # noqa: BLE001 - hardware shutdown must continue
            logger.warning("Live view shutdown failed: %s", exc)
        # Park and de-energize before any optional dataset conversion. The
        # converter can take minutes and must not retain either CAN bus.
        _shutdown_robot()
        try:
            if _termination_requested:
                print("\n[session] Skipping LeRobot conversion during shutdown.")
            else:
                _convert_if_any(
                    labeled_rollouts,
                    base_save_dir,
                    session_timestamp,
                    left_cfg,
                )
        finally:
            print("[session] complete")


def _convert_if_any(
    labeled_rollouts: list[Path],
    base_save_dir: Path,
    session_timestamp: str,
    left_cfg: dict[str, Any],
) -> None:
    """Best-effort LeRobot conversion of this session's labeled rollouts."""
    if not labeled_rollouts:
        print("\n[session] No labeled rollouts this session; nothing to convert.")
        return

    if not bool((left_cfg.get("storage") or {}).get("save_frames", False)):
        print(
            "\n[session] LeRobot conversion skipped because frames were not "
            "recorded; raw rollout kept."
        )
        return
    lerobot_cfg = left_cfg.get("lerobot", {}) or {}
    output_dir = base_save_dir / "eval_lerobot_v30" / session_timestamp
    print(
        f"\n[session] Converting {len(labeled_rollouts)} labeled rollouts "
        f"to LeRobot v3.0 at {output_dir} ..."
    )
    try:
        convert_session_to_lerobot(
            session_rollout_dirs=labeled_rollouts,
            output_dir=output_dir,
            fps=int(lerobot_cfg.get("fps", left_cfg.get("hz", 30))),
            robot_type=str(lerobot_cfg.get("robot_type", "molmoact_dual_arm")),
            repo_id=str(lerobot_cfg.get("hf_repo_id", "local/eval_session")),
            action_mode=str(lerobot_cfg.get("action_mode", "next_joint_fields")),
            vcodec=str(lerobot_cfg.get("vcodec", "libsvtav1")),
            sanitize_online_viz_meta=bool(lerobot_cfg.get("sanitize_online_viz_meta", True)),
            image_writer_processes=int(lerobot_cfg.get("image_writer_processes", 0)),
            image_writer_threads=int(lerobot_cfg.get("image_writer_threads", 0)),
            parallel_encoding=bool(lerobot_cfg.get("parallel_encoding", True)),
        )
    except ModuleNotFoundError as exc:
        logger.warning(
            "LeRobot conversion skipped because %s is not installed; "
            "raw rollouts kept",
            exc.name,
        )
    except Exception:
        logger.exception("LeRobot conversion failed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    atexit.register(_shutdown_robot)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    args = tyro.cli(Args)
    if args.num_rollouts < 1:
        raise SystemExit("--num_rollouts must be >= 1")
    try:
        _planning_options()
    except ValueError as exc:
        # Refuse contradictory configurations before cameras or robot
        # buses are opened and before the arms are moved to their start pose.
        raise SystemExit(str(exc)) from exc

    env, left_cfg, right_cfg, bimanual = _build_env(args)

    global _env, _bimanual, _left_cfg, _right_cfg
    _env = env
    _bimanual = bimanual
    _left_cfg = left_cfg
    _right_cfg = right_cfg

    if bimanual:
        move_to_start_position(env, True, left_cfg, right_cfg)
    else:
        move_to_start_position(env, False, left_cfg)

    print(f"Launching robot: {env.robot().__class__.__name__}")
    print(f"Control loop: {left_cfg.get('hz', 30)} Hz")
    print(
        f"Rollouts this session: {args.num_rollouts}, "
        f"max_steps: {left_cfg.get('max_steps', 1000)}"
    )

    eval_cfg = left_cfg.get("eval") or {}
    mode = eval_cfg.get("mode", "server")
    if mode == "local":
        policy = MolmoActLocal(**(eval_cfg.get("local") or {}))
    elif mode == "server":
        policy = MolmoAct(server=eval_cfg.get("molmoact_server"))
    else:
        raise SystemExit(f"eval.mode must be 'server' or 'local', got {mode!r}")
    run_session(
        env=env,
        policy=policy,
        left_cfg=left_cfg,
        right_cfg=right_cfg,
        bimanual=bimanual,
        num_rollouts=args.num_rollouts,
    )


if __name__ == "__main__":
    main()
