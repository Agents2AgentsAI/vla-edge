"""Continuous, bounded position references for the Pi0.5 chunk client.

The hardware owner is a separate process: inference, image handling, and Python
GC in the client cannot stall its update loop. Ruckig state advances from the
previous generated reference, never from measured-position error clipping.
"""

from __future__ import annotations

import fcntl
import gc
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import signal
import time
import traceback
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ruckig import (
    ControlInterface,
    InputParameter,
    OutputParameter,
    Result,
    Ruckig,
    Synchronization,
)

logger = logging.getLogger(__name__)

ARM = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIP = np.array([6, 13])
TRACE_FIELDS = ["monotonic", "goal_seq", "phase"] + [
    f"{k}_{i}"
    for k in [
        "reference",
        "reference_velocity",
        "reference_acceleration",
        "measured",
        "measured_velocity",
        "measured_effort",
        "goal",
    ]
    for i in range(14)
]


@dataclass(frozen=True)
class MotionLimits:
    hz: float = 100.0
    velocity: float = 2.2
    acceleration: float = 6.0
    jerk: float = 60.0
    gripper_velocity: float = 4.0
    gripper_acceleration: float = 30.0
    gripper_jerk: float = 300.0


#: Shared default; MotionLimits is frozen, so one instance is safe to reuse.
DEFAULT_LIMITS = MotionLimits()


class Reference:
    def __init__(self, q, limits=DEFAULT_LIMITS, minimum_duration=None):
        self.limits = limits
        self.dt = 1 / limits.hz
        if not all(math.isfinite(x) and x > 0 for x in asdict(limits).values()):
            raise ValueError("All motion limits must be finite and positive")
        q = np.asarray(q, dtype=float)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError("Invalid initial reference")
        self.otg = Ruckig(14, self.dt)
        self.input = InputParameter(14)
        self.output = OutputParameter(14)
        self.input.current_position = q.tolist()
        self.input.current_velocity = [0.0] * 14
        self.input.current_acceleration = [0.0] * 14
        self.input.synchronization = Synchronization.Time
        self.input.target_position = q.tolist()
        self.input.target_velocity = [0.0] * 14
        self.input.target_acceleration = [0.0] * 14
        for name, a, g in [
            ("max_velocity", limits.velocity, limits.gripper_velocity),
            ("max_acceleration", limits.acceleration, limits.gripper_acceleration),
            ("max_jerk", limits.jerk, limits.gripper_jerk),
        ]:
            v = np.full(14, a)
            v[GRIP] = g
            setattr(self.input, name, v.tolist())
        if minimum_duration is not None:
            self.input.minimum_duration = minimum_duration
        self.position = q.copy()
        self.velocity = np.zeros(14)
        self.acceleration = np.zeros(14)

    def target(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError("Invalid motion goal")
        q = q.copy()
        q[GRIP] = np.clip(q[GRIP], 0, 1)
        self.input.target_position = q.tolist()

    def brake(self):
        self.input.control_interface = ControlInterface.Velocity
        self.input.synchronization = Synchronization.No
        self.input.target_velocity = [0.0] * 14
        self.input.target_acceleration = [0.0] * 14

    def tick(self):
        result = self.otg.update(self.input, self.output)
        if result not in (Result.Working, Result.Finished):
            raise RuntimeError(f"Ruckig failed: {result}")
        self.position = np.array(self.output.new_position)
        self.velocity = np.array(self.output.new_velocity)
        self.acceleration = np.array(self.output.new_acceleration)
        self.output.pass_to_input(self.input)
        return self.position.copy(), result == Result.Finished


def finish_active_robot(robot, previous_command, velocity, *, home):
    """Keep normal-stop homing on the existing controller; always close once."""
    try:
        if home:
            from home_arms import home_active_robot

            home_active_robot(
                robot, previous_command=previous_command, max_joint_vel=velocity
            )
    finally:
        robot.close()


@contextmanager
def motion_lease():
    with open("/tmp/vla-edge-yam-motion.lock", "a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _worker(
    cfgs,
    start,
    root,
    limits,
    stop_path,
    goal_shared,
    state_shared,
    ready,
    closing,
    aborted,
    ended,
    messages,
    simulate,
):
    """Only this process owns CAN; it can park independently of network inference."""
    signal.signal(signal.SIGTERM, lambda *_: closing.set())
    signal.signal(signal.SIGINT, lambda *_: closing.set())
    root = Path(root)
    root.mkdir(exist_ok=True)
    robot = None
    left = right = None
    was_gc = gc.isenabled()
    gc.collect()
    gc.disable()
    trace = (root / "motion.f64").open("wb", buffering=65536)
    (root / "motion-schema.json").write_text(
        json.dumps(
            {
                "dtype": "float64",
                "columns": TRACE_FIELDS,
                "limits": asdict(limits),
                "native_feedback": (
                    "Arm velocities rad/s and efforts per I2RT; "
                    "gripper fields retain native driver units"
                ),
            },
            indent=2,
        )
    )
    seq = -1
    lease_stack = ExitStack()
    goal = np.asarray(start, float)
    last_dispatch = None
    last_reference = None
    count = 0
    overruns = 0
    reason = "shutdown"
    fault = None
    close_succeeded = False
    owner_pid = os.getppid()

    def feedback():
        q = robot.get_joint_state().copy()
        if simulate:
            return q, np.zeros(14), np.zeros(14)
        vel = []
        eff = []
        for wrapper in [left, right]:
            o = wrapper.robot.get_observations()
            vel.extend(
                np.asarray(o["joint_vel"]).tolist()
                + np.asarray(o["gripper_vel"]).tolist()
            )
            eff.extend(
                np.asarray(o["joint_eff"]).tolist()
                + np.asarray(o["gripper_eff"]).tolist()
            )
        values = (q, np.asarray(vel), np.asarray(eff))
        if any(v.shape != (14,) or not np.isfinite(v).all() for v in values):
            raise RuntimeError("Invalid motor feedback")
        return values

    def publish(ref, done, measured, vel, eff, stamp, status=1):
        with state_shared.get_lock():
            state_shared[:] = (
                [stamp, float(status), float(seq), float(done)]
                + measured.tolist()
                + ref.position.tolist()
                + ref.velocity.tolist()
                + ref.acceleration.tolist()
                + vel.tolist()
                + eff.tolist()
            )

    def dispatch(ref, q, done, phase):
        nonlocal last_dispatch, last_reference, count, overruns
        # No catch-up bursts: timestamp the beginning of dispatch, not its end.
        if last_dispatch is not None:
            remaining = last_dispatch + ref.dt - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        stamp = time.monotonic()
        if last_dispatch is not None and stamp - last_dispatch > 0.05:
            overruns += 1
        robot.command_joint_state(q.copy())
        last_reference = q.copy()
        last_dispatch = stamp
        measured, vel, eff = feedback()
        publish(ref, done, measured, vel, eff, stamp, phase)
        row = np.concatenate(
            (
                [stamp, float(seq), float(phase)],
                q,
                ref.velocity,
                ref.acceleration,
                measured,
                vel,
                eff,
                goal,
            )
        )
        trace.write(row.astype(np.float64).tobytes())
        count += 1
        return measured

    def home_to(target, duration, interruptible=False):
        nonlocal last_dispatch, goal
        home_limits = MotionLimits(
            hz=limits.hz,
            velocity=min(limits.velocity, 0.5),
            acceleration=1.5,
            jerk=10,
            gripper_velocity=2,
            gripper_acceleration=10,
            gripper_jerk=100,
        )
        origin = robot.get_joint_state() if last_reference is None else last_reference
        ref = Reference(origin, home_limits, minimum_duration=duration)
        ref.target(target)
        goal = np.asarray(target, float)
        while True:
            if interruptible and (closing.is_set() or Path(stop_path).exists()):
                raise InterruptedError("Operator stop during initial positioning")
            q, done = ref.tick()
            dispatch(ref, q, done, 2)
            if done:
                break
        return ref

    try:
        if closing.is_set() or Path(stop_path).exists():
            reason = "operator_stop"
            return
        if simulate:

            class Robot:
                def __init__(self):
                    self.q = np.asarray(start, float).copy()

                def get_joint_state(self):
                    return self.q.copy()

                def command_joint_state(self, q):
                    if simulate != "stalled":
                        self.q = np.asarray(q).copy()

                def close(self):
                    pass

            robot = Robot()
            joint_limits = np.tile([-10.0, 10.0], (14, 1))
            joint_limits[GRIP] = [0, 1]
        else:
            from gello_min.launch_utils import instantiate_from_dict
            from gello_min.robot import BimanualRobot
            from home_arms import find_robot_drivers

            lease_stack.enter_context(motion_lease())
            # This process is the only CAN owner; its parent is an excluded ancestor.
            if find_robot_drivers():
                raise RuntimeError("Another YAM controller is active")
            left = instantiate_from_dict(cfgs[0]["robot"])
            right = instantiate_from_dict(cfgs[1]["robot"])
            robot = BimanualRobot(left, right)
            joint_limits = np.vstack(
                (
                    left.robot._joint_limits[:6],
                    [0.0, 1.0],
                    right.robot._joint_limits[:6],
                    [0.0, 1.0],
                )
            )
            messages.put({"event": "hardware_enabled"})
            if np.any(np.asarray(start) < joint_limits[:, 0]) or np.any(
                np.asarray(start) > joint_limits[:, 1]
            ):
                raise ValueError("Start pose violates configured joint limits")
            home_to(start, 4.0, interruptible=True)
            measured = robot.get_joint_state()
            if (
                not np.isfinite(measured).all()
                or np.max(abs(measured[ARM] - np.asarray(start)[ARM])) > 0.08
            ):
                raise RuntimeError(
                    "Initial positioning did not reach the configured start pose"
                )
        origin = robot.get_joint_state() if last_reference is None else last_reference
        ref = Reference(origin, limits)
        goal = ref.position.copy()
        ref.target(goal)
        q, done = ref.tick()
        dispatch(ref, q, done, 1)
        ready.set()
        messages.put({"event": "motion_ready", "wall": time.time()})
        last_goal_at = time.monotonic()
        while not closing.is_set() and not Path(stop_path).exists():
            if last_dispatch is not None:
                wait = last_dispatch + ref.dt - time.monotonic()
                if wait > 0:
                    closing.wait(wait)
            if closing.is_set() or Path(stop_path).exists():
                break
            if os.getppid() != owner_pid:
                raise RuntimeError("Policy process exited unexpectedly")
            # Advance one reference tick only. Never catch up by sending a
            # burst of commands after a scheduling delay; record timing below.
            with goal_shared.get_lock():
                data = list(goal_shared[:])
            if int(data[0]) != seq:
                last_goal_at = time.monotonic()
                seq = int(data[0])
                goal = np.clip(
                    np.array(data[1:]), joint_limits[:, 0], joint_limits[:, 1]
                )
                ref.target(goal)
            if time.monotonic() - last_goal_at > 5.0:
                raise RuntimeError("No new policy target for five seconds")
            q, done = ref.tick()
            if np.any(q[ARM] < joint_limits[ARM, 0] - 1e-6) or np.any(
                q[ARM] > joint_limits[ARM, 1] + 1e-6
            ):
                raise RuntimeError("Generated reference violates joint bounds")
            dispatch(ref, q, done, 1)
        if aborted.is_set():
            raise RuntimeError("Policy client failed; disabling without a homing move")
        reason = "operator_stop" if Path(stop_path).exists() else "shutdown"
        ref.brake()
        for _ in range(int(3 * limits.hz)):
            q, done = ref.tick()
            if np.any(q[ARM] < joint_limits[ARM, 0] - 1e-6) or np.any(
                q[ARM] > joint_limits[ARM, 1] + 1e-6
            ):
                raise RuntimeError("Braking reference violates joint bounds")
            dispatch(ref, q, done, 3)
            if done:
                break
        else:
            raise RuntimeError("Controlled braking did not finish")
    except InterruptedError:
        reason = "operator_stop"
    except BaseException as exc:
        logger.exception("Pi0.5 motion owner failed; disabling motors")
        reason = "fault"
        fault = f"{type(exc).__name__}: {exc}"
        messages.put(
            {"event": "fault", "reason": fault, "traceback": traceback.format_exc()}
        )
    finally:
        if robot is not None:
            try:
                if not simulate and not fault:
                    messages.put({"event": "homing_active"})
                finish_active_robot(
                    robot,
                    last_reference,
                    limits.velocity,
                    home=not simulate and not fault,
                )
                close_succeeded = True
            except BaseException as exc:
                logger.exception("Pi0.5 active-controller shutdown failed")
                fault = f"{fault or ''}; shutdown: {exc}"
        else:
            close_succeeded = True
            for partial in [left, right]:
                if partial is not None:
                    try:
                        partial.close()
                    except Exception as exc:
                        logger.exception("Partially initialized arm could not close")
                        close_succeeded = False
                        fault = f"{fault or ''}; partial close: {exc}"
        with state_shared.get_lock():
            state_shared[1] = -1 if fault else 0
        lease_stack.close()
        trace.close()
        (root / "motion-outcome.json").write_text(
            json.dumps(
                {
                    "reason": reason,
                    "fault": fault,
                    "reference_commands": count,
                    "timing_overruns_over_50ms": overruns,
                    "simulate": simulate,
                    "hardware_disabled": None if simulate else close_succeeded,
                },
                indent=2,
            )
        )
        messages.put({"event": "motion_finished", "fault": fault, "commands": count})
        ended.set()
        ready.set()
        if was_gc:
            gc.enable()


class MotionProcess:
    def __init__(
        self, cfgs, start, root, stop_path, limits=DEFAULT_LIMITS, simulate=False
    ):
        if Path(stop_path).exists():
            raise InterruptedError(
                "Stop marker is present before hardware initialization"
            )
        ctx = mp.get_context("spawn")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        self.goal = ctx.Array("d", [0.0] + list(start), lock=True)
        self.state = ctx.Array("d", [0.0] * (4 + 6 * 14), lock=True)
        self.ready = ctx.Event()
        self.closing = ctx.Event()
        self.aborted = ctx.Event()
        self.ended = ctx.Event()
        self.messages = ctx.Queue()
        self.events = []
        self.seq = 0
        self.outcome = None
        self.stop_path = Path(stop_path)
        self.process = ctx.Process(
            target=_worker,
            args=(
                cfgs,
                list(start),
                str(root),
                limits,
                str(stop_path),
                self.goal,
                self.state,
                self.ready,
                self.closing,
                self.aborted,
                self.ended,
                self.messages,
                simulate,
            ),
            name="yam-reference-owner",
        )
        self.process.start()
        if not self.ready.wait(40):
            self.close()
            raise RuntimeError("Motion process startup timed out")
        try:
            self.check()
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        with self.state.get_lock():
            v = np.array(self.state[:])
        return {
            "t": v[0],
            "status": int(v[1]),
            "seq": int(v[2]),
            "finished": bool(v[3]),
            **{
                name: v[4 + 14 * i : 4 + 14 * (i + 1)].copy()
                for i, name in enumerate(
                    [
                        "measured",
                        "reference",
                        "velocity",
                        "acceleration",
                        "motor_velocity",
                        "motor_effort",
                    ]
                )
            },
        }

    def check(self):
        while True:
            try:
                self.events.append(self.messages.get_nowait())
            except queue.Empty:
                break
        s = self.snapshot()
        if self.stop_path.exists():
            raise InterruptedError("Operator stop marker")
        if not self.process.is_alive() or self.ended.is_set() or s["status"] != 1:
            raise RuntimeError(f"Motion process stopped: {self.events[-3:]}")
        if time.monotonic() - s["t"] > 0.5:
            raise RuntimeError("Motion process feedback is stale")
        return s

    def set_goal(self, q):
        q = np.asarray(q, float)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError("Invalid policy goal")
        self.check()
        self.seq += 1
        with self.goal.get_lock():
            self.goal[:] = [float(self.seq)] + q.tolist()
        deadline = time.monotonic() + 0.2
        while True:
            s = self.check()
            if s["seq"] == self.seq:
                return s
            if time.monotonic() > deadline:
                raise RuntimeError("Reference owner did not acknowledge the goal")
            time.sleep(0.001)

    def abort(self):
        self.aborted.set()
        self.closing.set()

    def close(self):
        if self.outcome is not None:
            return
        self.closing.set()
        self.process.join(60)
        if self.process.is_alive():
            raise RuntimeError(
                "Hardware owner still parking; do not start another controller"
            )
        while True:
            try:
                self.events.append(self.messages.get_nowait())
            except queue.Empty:
                break
        (self.root / "motion-events.json").write_text(json.dumps(self.events, indent=2))
        outcome = self.root / "motion-outcome.json"
        if not outcome.exists():
            raise RuntimeError(
                f"Motion process exited without a shutdown receipt (exit={self.process.exitcode})"
            )
        self.outcome = json.loads(outcome.read_text())
        if self.outcome.get("fault"):
            raise RuntimeError(f"Motion process fault: {self.outcome['fault']}")
