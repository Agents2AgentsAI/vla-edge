"""No network, CUDA, cameras or CAN are opened by these rollout tests."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi05_rollout import control_settings, run_episode


class Clock:
    t = 0.0

    def now(self):
        return self.t

    def sleep(self, t):
        self.t += t


class Motion:
    def __init__(self, clock):
        self.clock = clock
        self.seq = 0
        self.goals = []
        self.reference = np.zeros(14)
        self.reference[[6, 13]] = 0.7

    def check(self):
        return {"reference": self.reference.copy(), "measured": np.zeros(14)}

    def set_goal(self, q):
        self.goals.append((self.clock.now(), q.copy()))
        self.seq += 1
        self.reference = q.copy()
        return self.check()


class Camera:
    def get_obs(self):
        return {
            key: np.full((4, 4, 3), i, np.uint8)
            for i, key in enumerate(("front_camera", "left_camera", "right_camera"))
        }


class Client:
    def __init__(self, clock):
        self.clock = clock
        self.requests = []

    def act(self, **kw):
        self.requests.append(kw)
        self.clock.sleep(0.12)
        a = np.zeros((16, 14))
        a[:, 0] = np.arange(16) + 100 * (len(self.requests) - 1)
        a[:, [6, 13]] = 0.3
        return a, 100.0


def test_sync_chunks_have_no_overlap_or_stale_row_drops():
    clock = Clock()
    motion = Motion(clock)
    client = Client(clock)
    events = []
    n = run_episode(
        motion,
        Camera(),
        client,
        "fold",
        max_steps=32,
        record=events.append,
        clock=clock.now,
        sleep=clock.sleep,
    )
    assert n == 32 and len(client.requests) == 2
    np.testing.assert_array_equal(
        [q[0] for _, q in motion.goals], np.r_[np.arange(16), 100 + np.arange(16)]
    )
    # Model/state uses commanded gripper reference, not object-limited measurement.
    assert np.allclose(client.requests[0]["state"][[6, 13]], 0.7)
    assert np.allclose(client.requests[1]["state"][[6, 13]], 0.3)
    assert all(
        b[0] - a[0] >= 1 / 30 - 1e-10 for a, b in zip(motion.goals, motion.goals[1:])
    )
    assert all(r["num_steps"] == 10 for r in client.requests)
    assert client.requests[0]["cameras"]["right_cam"][0, 0, 0] == 2


def test_stop_during_request_does_not_submit_returned_targets():
    clock = Clock()
    motion = Motion(clock)
    client = Client(clock)
    n = run_episode(
        motion,
        Camera(),
        client,
        "fold",
        stop=lambda: clock.now() > 0.1,
        clock=clock.now,
        sleep=clock.sleep,
    )
    assert n == 0 and motion.goals == []


@pytest.mark.parametrize("bad", [np.full((16, 14), np.nan), np.zeros((30, 14))])
def test_invalid_policy_chunk_never_reaches_motion(bad):
    clock = Clock()
    motion = Motion(clock)
    client = SimpleNamespace(act=lambda **kwargs: (bad, 0))
    with pytest.raises(ValueError, match="finite 16x14"):
        run_episode(
            motion, Camera(), client, "fold", clock=clock.now, sleep=clock.sleep
        )
    assert motion.goals == []


def test_short_execution_horizon_does_not_overrun_rows():
    clock = Clock()
    motion = Motion(clock)
    client = Client(clock)
    run_episode(
        motion,
        Camera(),
        client,
        "fold",
        horizon=6,
        max_steps=10,
        clock=clock.now,
        sleep=clock.sleep,
    )
    assert len(client.requests) == 2
    assert [q[0] for _, q in motion.goals] == list(range(6)) + list(range(100, 104))


def test_async_override_is_rejected(monkeypatch):
    monkeypatch.setenv("YAM_ASYNC_PLAN", "1")
    with pytest.raises(ValueError, match="synchronous"):
        control_settings()


def test_velocity_override_is_honored_without_raising_other_limits(monkeypatch):
    monkeypatch.delenv("YAM_ASYNC_PLAN", raising=False)
    monkeypatch.delenv("YAM_RTC", raising=False)
    monkeypatch.setenv("YAM_MAX_JOINT_VEL", ".5")
    _, limits = control_settings()
    assert limits.velocity == 0.5 and limits.acceleration == 6 and limits.jerk == 60
