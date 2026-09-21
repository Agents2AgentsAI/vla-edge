"""Offline controller tests. No CAN drivers or policy servers are constructed."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi05_motion import (
    ARM,
    GRIP,
    TRACE_FIELDS,
    MotionProcess,
    Reference,
)


def initial():
    q = np.zeros(14)
    q[GRIP] = 1
    return q


def test_recorded_wrist_reversal_is_continuous_and_converges():
    q = initial()
    q[10] = -0.588
    r = Reference(q)
    goal = q.copy()
    goal[10] = -0.838
    r.target(goal)
    positions = []
    velocities = []
    accelerations = []
    for i in range(350):
        if i == 25:
            goal[10] = -0.357
            r.target(goal)
        p, _done = r.tick()
        positions.append(p)
        velocities.append(r.velocity.copy())
        accelerations.append(r.acceleration.copy())
    p = np.array(positions)
    v = np.array(velocities)
    a = np.array(accelerations)
    assert np.max(abs(np.diff(p[:, ARM], axis=0))) <= 2.2 * 0.01 + 1e-10
    assert np.max(abs(v[:, ARM])) <= 2.2 + 1e-10
    assert np.max(abs(a[:, ARM])) <= 6 + 1e-10
    assert np.max(abs(np.diff(a[:, ARM], axis=0)) / 0.01) <= 60 + 1e-6
    assert abs(p[25, 10] - p[24, 10]) < 0.025
    np.testing.assert_allclose(p[-1], goal, atol=1e-10)


def test_frequent_retargets_and_inference_hold_preserve_derivative_bounds():
    r = Reference(initial())
    rng = np.random.default_rng(53)
    a = []
    v = []
    for i in range(1500):
        if i % 3 == 0 and not 600 <= i < 625:
            goal = initial()
            goal[ARM] = rng.uniform(-0.45, 0.45, 12)
            goal[GRIP] = rng.uniform(0, 1, 2)
            r.target(goal)
        p, _ = r.tick()
        assert np.isfinite(p).all()
        a.append(r.acceleration.copy())
        v.append(r.velocity.copy())
    a = np.array(a)
    v = np.array(v)
    assert abs(v[:, ARM]).max() <= 2.2 + 1e-9
    assert abs(a[:, ARM]).max() <= 6 + 1e-9
    assert abs(np.diff(a[:, ARM], axis=0) / 0.01).max() <= 60 + 1e-5


def test_braking_reaches_zero_velocity_without_a_position_jump():
    r = Reference(initial())
    goal = initial()
    goal[0] = 1
    r.target(goal)
    for _ in range(35):
        r.tick()
    before = r.position.copy()
    r.brake()
    p, done = r.tick()
    assert abs(p - before).max() <= 0.022 + 1e-9
    for _ in range(300):
        p, done = r.tick()
        if done:
            break
    assert done and abs(r.velocity).max() < 1e-9 and abs(r.acceleration).max() < 1e-9


@pytest.mark.parametrize("bad", [np.full(14, np.nan), np.zeros(13)])
def test_invalid_goal_cannot_modify_reference(bad):
    r = Reference(initial())
    before = r.input.target_position
    with pytest.raises(ValueError):
        r.target(bad)
    assert r.input.target_position == before


def test_motion_process_survives_blocked_client_and_obeys_stop(tmp_path):
    marker = tmp_path / "stop"
    motion = MotionProcess(None, initial(), tmp_path, marker, simulate=True)
    try:
        goal = initial()
        goal[0] = 0.25
        motion.set_goal(goal)
        t0 = motion.snapshot()["t"]
        time.sleep(0.12)
        assert motion.snapshot()["t"] - t0 > 0.06
        # This path must complete in the child, without a parent close/request.
        marker.touch()
        assert motion.ended.wait(4)
    finally:
        motion.close()
    outcome = json.loads((tmp_path / "motion-outcome.json").read_text())
    assert outcome["reason"] == "operator_stop" and outcome["fault"] is None
    a = np.fromfile(tmp_path / "motion.f64", np.float64).reshape(-1, len(TRACE_FIELDS))
    assert len(a) > 10 and np.all(np.diff(a[:, 0]) > 0)
    assert not motion.process.is_alive()


def test_existing_stop_marker_prevents_process_creation(tmp_path):
    marker = tmp_path / "stop"
    marker.touch()
    with pytest.raises(InterruptedError):
        MotionProcess(None, initial(), tmp_path, marker, simulate=True)
    assert not (tmp_path / "motion.f64").exists()


def test_position_error_does_not_change_goals_or_trigger_an_added_pose_gate(tmp_path):
    marker = tmp_path / "stop"
    motion = MotionProcess(None, initial(), tmp_path, marker, simulate="stalled")
    try:
        goal = initial()
        goal[0] = 0.2
        motion.set_goal(goal)
        time.sleep(1.0)
        snapshot = motion.check()
        assert not motion.ended.is_set()
        np.testing.assert_allclose(snapshot["reference"], goal, atol=1e-9)
        assert snapshot["measured"][0] == 0
    finally:
        motion.close()


def test_fault_abort_closes_worker_without_homing(tmp_path):
    motion = MotionProcess(None, initial(), tmp_path, tmp_path / "stop", simulate=True)
    motion.abort()
    assert motion.ended.wait(4)
    with pytest.raises(RuntimeError, match="Policy client failed"):
        motion.close()
    outcome = json.loads((tmp_path / "motion-outcome.json").read_text())
    assert outcome["reason"] == "fault"
    assert not any(e.get("event") == "homing_active" for e in motion.events)
    assert not motion.process.is_alive()


@pytest.mark.parametrize("normal", [False, True])
def test_shutdown_reuses_active_controller_and_closes_once(monkeypatch, normal):
    import home_arms
    from pi05_motion import finish_active_robot

    calls = []

    class Robot:
        def close(self):
            calls.append("close")

    robot = Robot()
    last = initial()

    def home(active, **kwargs):
        assert active is robot and kwargs["previous_command"] is last
        assert kwargs["max_joint_vel"] == 0.5
        calls.append("home")

    monkeypatch.setattr(home_arms, "home_active_robot", home)
    finish_active_robot(robot, last, 0.5, home=normal)
    assert calls == (["home", "close"] if normal else ["close"])


def test_home_failure_still_closes_active_controller(monkeypatch):
    import home_arms
    from pi05_motion import finish_active_robot

    calls = []

    class Robot:
        def close(self):
            calls.append("close")

    def home(*args, **kwargs):
        calls.append("home")
        raise RuntimeError("feedback failed")

    monkeypatch.setattr(home_arms, "home_active_robot", home)
    with pytest.raises(RuntimeError, match="feedback failed"):
        finish_active_robot(Robot(), initial(), 0.5, home=True)
    assert calls == ["home", "close"]


def test_full_chunk_loop_with_independent_motion_process(tmp_path):
    from pi05_rollout import run_episode

    class Camera:
        def get_obs(self):
            return {
                k: np.zeros((4, 4, 3), np.uint8)
                for k in ("front_camera", "left_camera", "right_camera")
            }

    class Client:
        def __init__(self):
            self.calls = 0

        def act(self, **kwargs):
            self.calls += 1
            time.sleep(
                0.12
            )  # Simulate network/model latency while the child continues.
            actions = np.tile(initial(), (16, 1))
            actions[:, 0] = 0.3 if self.calls == 1 else -0.2
            return actions, 120.0

    motion = MotionProcess(None, initial(), tmp_path, tmp_path / "stop", simulate=True)
    client = Client()
    try:
        assert run_episode(motion, Camera(), client, "test", max_steps=32) == 32
        assert client.calls == 2 and motion.seq == 32
    finally:
        motion.close()
    trace = np.fromfile(tmp_path / "motion.f64", np.float64).reshape(
        -1, len(TRACE_FIELDS)
    )
    refs = trace[:, 3:17]
    assert len(trace) > 100
    assert np.max(abs(np.diff(refs[:, ARM], axis=0))) <= 0.022 + 1e-8
    assert not motion.process.is_alive()
