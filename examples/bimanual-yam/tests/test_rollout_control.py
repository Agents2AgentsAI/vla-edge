"""No-hardware verification for YAM smoothing and RTC rollout control."""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

import numpy as np

YAM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(YAM_DIR))
sys.path.insert(0, str(REPO_ROOT))

import launch_yaml_eval_molmoact as rollout


class _FakeEnv:
    """Small paced environment; it never opens cameras, CAN, or a robot."""

    def __init__(
        self,
        tick_s: float,
        n_dofs: int = 14,
        actual_tick_s: float | None = None,
    ):
        self.control_period_s = tick_s
        self.actual_tick_s = (
            tick_s if actual_tick_s is None else float(actual_tick_s)
        )
        self.state = np.zeros(n_dofs, dtype=np.float64)
        self.command_times: list[float] = []
        self.commands: list[np.ndarray] = []

    def get_robot_state(self):
        return {"joint_positions": self.state.copy()}

    def get_obs(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        return {
            "joint_positions": self.state.copy(),
            "joint_velocities": np.zeros_like(self.state),
            "ee_pos_quat": np.zeros(7, dtype=np.float64),
            "gripper_position": np.array([self.state[6], self.state[13]]),
            "left_camera_rgb": image.copy(),
            "front_camera_rgb": image.copy(),
            "right_camera_rgb": image.copy(),
        }

    def apply_action(self, action: np.ndarray):
        if self.command_times:
            sleep_s = (
                self.command_times[-1]
                + self.actual_tick_s
                - time.monotonic()
            )
            if sleep_s > 0.0:
                time.sleep(sleep_s)
        self.state = np.asarray(action, dtype=np.float64).copy()
        self.command_times.append(time.monotonic())
        self.commands.append(self.state.copy())
        return self.get_obs()


class _FakeSaver:
    def __init__(self):
        self.steps = []

    def add_step(self, obs_pre, obs_post):
        self.steps.append((obs_pre, obs_post))


class _FakeView:
    def update(self, **kwargs):
        return None


class _SleepingPolicy:
    def __init__(
        self,
        configured_horizon: int,
        returned_horizon: int,
        inference_s: float,
    ):
        self.configured_horizon = configured_horizon
        self.returned_horizon = returned_horizon
        self.inference_s = inference_s
        self.call_count = 0
        self.calls: list[dict | None] = []
        self.call_steps: list[int | None] = []
        self.merge_stats: list[dict] = []
        self.step_counter = None
        self._lock = threading.Lock()
        self.episode_id = None

    def set_episode(self, episode_id: int):
        self.episode_id = int(episode_id)

    def get_action_horizon(self):
        return self.configured_horizon

    def prepare_input(self, obs, instruction):
        return {
            "state": obs["joint_positions"].copy(),
            "instruction": instruction,
        }

    def inference(self, input_dict, rtc=None):
        with self._lock:
            chunk_id = self.call_count
            self.call_count += 1
            self.call_steps.append(
                None if self.step_counter is None else self.step_counter()
            )
            if rtc is None:
                self.calls.append(None)
            else:
                copied = dict(rtc)
                copied["prefix_actions"] = np.asarray(
                    rtc["prefix_actions"]
                ).copy()
                self.calls.append(copied)
        time.sleep(self.inference_s)
        # Start at one so the first raw plan differs from the measured zero
        # pose; this exposes accidental use of the smoothed command as prefix.
        return {
            "actions": np.full(
                (self.returned_horizon, 14),
                float(chunk_id + 1),
                dtype=np.float64,
            )
        }


def _fake_dynamic_smoothing(env: _FakeEnv, action: np.ndarray, cmd_sink=None):
    return env.apply_action(action)


def _fake_clamped_step(
    env: _FakeEnv, action: np.ndarray, gripper_limit=0.15, cmd_sink=None,
    prev_cmd=None,
):
    return env.apply_action(action)


def _rtc_options(**overrides):
    values = {
        "async_plan": True,
        "inference_budget_s": 0.0,
        "replan_threshold": 0.5,
        "rtc": True,
        "rtc_horizon": 4,
        "rtc_trigger": "auto",
        "rtc_schedule": "linear",
        "rtc_max_guidance": 10.0,
    }
    values.update(overrides)
    return rollout._PlanOptions(**values)


class ActionSmootherTests(unittest.TestCase):
    def test_wiggle_variance_is_reduced_and_lag_is_bounded(self):
        smoother = rollout.ActionSmoother(
            alpha=0.5, gripper_rate=0.15, n_dofs=14
        )
        smoother.reset(np.zeros(14))
        raw = []
        filtered = []
        for index in range(100):
            action = np.zeros(14)
            action[0] = 0.09 if index % 2 else -0.09
            raw.append(action.copy())
            filtered.append(smoother(action))
        raw_delta_var = np.var(np.diff(np.asarray(raw)[20:, 0]))
        filtered_delta_var = np.var(np.diff(np.asarray(filtered)[20:, 0]))
        self.assertGreaterEqual(raw_delta_var / filtered_delta_var, 2.5)

        smoother.reset(np.zeros(14))
        action = np.zeros(14)
        action[0] = 1.0
        response = [smoother(action)[0] for _ in range(3)]
        ticks = next(
            index + 1 for index, value in enumerate(response) if value >= 0.75
        )
        self.assertLessEqual(ticks, 2)

    def test_gripper_step_is_rate_limited(self):
        smoother = rollout.ActionSmoother(
            alpha=0.5, gripper_rate=0.15, n_dofs=14
        )
        smoother.reset(np.zeros(14))
        action = np.zeros(14)
        action[[6, 13]] = 1.0
        values = np.asarray([smoother(action)[[6, 13]] for _ in range(7)])
        expected = np.minimum(np.arange(1, 8) * 0.15, 1.0)
        np.testing.assert_allclose(values[:, 0], expected)
        np.testing.assert_allclose(values[:, 1], expected)

    def test_alpha_one_is_bit_identical_raw_bypass(self):
        smoother = rollout.ActionSmoother(
            alpha=1.0, gripper_rate=0.15, n_dofs=14
        )
        smoother.reset(np.zeros(14))
        action = np.linspace(-1.0, 1.0, 14, dtype=np.float32)
        output = smoother(action)
        self.assertEqual(output.dtype, action.dtype)
        np.testing.assert_array_equal(output, action)


class PlanningOptionTests(unittest.TestCase):
    def test_rtc_requires_async_at_preflight(self):
        with mock.patch.dict(
            os.environ,
            {"YAM_RTC": "1", "YAM_ASYNC_PLAN": "0"},
            clear=True,
        ), self.assertRaisesRegex(ValueError, "requires YAM_ASYNC_PLAN"):
            rollout._planning_options()

    def test_invalid_flag_and_schedule_are_rejected(self):
        with mock.patch.dict(
            os.environ, {"YAM_ASYNC_PLAN": "perhaps"}, clear=True
        ), self.assertRaisesRegex(ValueError, "boolean flag"):
            rollout._planning_options()
        with mock.patch.dict(
            os.environ,
            {
                "YAM_ASYNC_PLAN": "1",
                "YAM_RTC": "1",
                "YAM_RTC_SCHEDULE": "triangle",
            },
            clear=True,
        ), self.assertRaisesRegex(ValueError, "YAM_RTC_SCHEDULE"):
            rollout._planning_options()

    def test_invalid_rtc_trigger_is_rejected(self):
        common = {"YAM_ASYNC_PLAN": "1", "YAM_RTC": "1"}
        with mock.patch.dict(
            os.environ,
            {**common, "YAM_RTC_TRIGGER": "whenever"},
            clear=True,
        ), self.assertRaisesRegex(ValueError, "YAM_RTC_TRIGGER"):
            rollout._planning_options()
        with mock.patch.dict(
            os.environ,
            {**common, "YAM_RTC_TRIGGER": "0"},
            clear=True,
        ), self.assertRaisesRegex(ValueError, r"in \[1, 30\]"):
            rollout._planning_options()

    def test_rtc_trigger_modes_are_parsed(self):
        common = {"YAM_ASYNC_PLAN": "1", "YAM_RTC": "1"}
        with mock.patch.dict(os.environ, common, clear=True):
            self.assertEqual(
                rollout._planning_options().rtc_trigger,
                "auto",
            )
        cases = (
            ("auto", "auto"),
            ("continuous", "continuous"),
            ("016", "16"),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw), mock.patch.dict(
                os.environ,
                {**common, "YAM_RTC_TRIGGER": raw},
                clear=True,
            ):
                self.assertEqual(
                    rollout._planning_options().rtc_trigger,
                    expected,
                )

    def test_async_and_inference_budget_remain_incompatible(self):
        with mock.patch.dict(
            os.environ,
            {"YAM_ASYNC_PLAN": "1", "YAM_INFER_BUDGET_S": "0.2"},
            clear=True,
        ), self.assertRaisesRegex(ValueError, "cannot be combined"):
            rollout._planning_options()


class DelayTrackerTests(unittest.TestCase):
    def test_max_latency_window_uses_ceil_and_last_100_samples(self):
        tracker = rollout._MaxLatencyDelay(tick_s=0.04)
        self.assertEqual(tracker.ticks, 0)
        tracker.observe(0.081)
        self.assertEqual(tracker.ticks, 3)
        tracker.observe(0.201)
        self.assertEqual(tracker.ticks, 6)
        for _ in range(100):
            tracker.observe(0.01)
        self.assertEqual(tracker.ticks, 1)

    def test_rtc_delay_switches_from_wall_time_to_consumed_actions(self):
        tracker = rollout._RTCActionDelay(tick_s=0.04)
        tracker.observe(0.201)
        self.assertEqual(tracker.ticks, 6)
        self.assertEqual(tracker.source, "wall-clock")

        # The overloaded robot loop consumed four actions, so the next RTC
        # request must guide/drop four rows even though 201 ms is six nominal
        # 40 ms ticks after ceil rounding.
        tracker.observe_consumed(4)
        self.assertEqual(tracker.ticks, 4)
        self.assertEqual(tracker.source, "observed-actions")
        tracker.observe(0.4)
        self.assertEqual(tracker.ticks, 4)
        tracker.observe_consumed(5)
        self.assertEqual(tracker.ticks, 5)

        for _ in range(100):
            tracker.observe_consumed(3)
        self.assertEqual(tracker.ticks, 3)

    def test_rtc_auto_trigger_is_delay_plus_horizon(self):
        options = _rtc_options(rtc_horizon=10)
        self.assertEqual(rollout._rtc_trigger_rows(options, 6, 30), 16)
        self.assertEqual(rollout._rtc_trigger_rows(options, 4, 30), 14)
        self.assertEqual(rollout._rtc_trigger_rows(options, 25, 30), 30)
        self.assertEqual(
            rollout._rtc_trigger_rows(
                _rtc_options(rtc_trigger="continuous"), 4, 30
            ),
            30,
        )
        self.assertEqual(
            rollout._rtc_trigger_rows(
                _rtc_options(rtc_trigger="12"), 4, 30
            ),
            12,
        )


class RTCQueueTests(unittest.TestCase):
    def test_prefix_is_true_raw_queue_rows_without_padding(self):
        queue = deque(
            (10 + index, np.full(14, index + 0.25, dtype=np.float64))
            for index in range(7)
        )
        payload = rollout._rtc_payload_from_queue(
            queue, _rtc_options(rtc_horizon=4), predicted_delay=3
        )
        self.assertEqual(payload["inference_delay"], 3)
        self.assertEqual(payload["execution_horizon"], 4)
        self.assertEqual(payload["prefix_actions"].dtype, np.float32)
        self.assertEqual(payload["prefix_actions"].shape, (4, 14))
        np.testing.assert_array_equal(
            payload["prefix_actions"][:, 0],
            np.arange(4, dtype=np.float32) + 0.25,
        )

    def test_empty_queue_produces_unguided_request(self):
        self.assertIsNone(
            rollout._rtc_payload_from_queue(
                deque(), _rtc_options(), predicted_delay=5
            )
        )

    def test_drop_consumed_then_replace_queue(self):
        queue = deque(
            (3 + index, np.full(14, 100 + index, dtype=np.float64))
            for index in range(5)
        )
        actions = np.stack(
            [np.full(14, index, dtype=np.float64) for index in range(10)]
        )
        result = rollout._InferenceResult(
            actions=actions,
            elapsed_s=0.1,
            requested_at_s=time.monotonic(),
            planned_step=0,
            predicted_delay=3,
            prefix_len=5,
        )
        stats = rollout._rtc_merge_into_queue(
            queue, result, current_step=3, max_queue_len=4
        )
        self.assertEqual(stats["consumed"], 3)
        self.assertFalse(stats["delay_mismatch"])
        self.assertEqual([step for step, _ in queue], [3, 4, 5, 6])
        np.testing.assert_array_equal(
            np.asarray([action[0] for _, action in queue]),
            np.array([3.0, 4.0, 5.0, 6.0]),
        )

    def test_delay_mismatch_warns_and_reports_splice_seam(self):
        queue = deque([(4, np.zeros(14, dtype=np.float64))])
        actions = np.ones((8, 14), dtype=np.float64)
        result = rollout._InferenceResult(
            actions=actions,
            elapsed_s=0.1,
            requested_at_s=time.monotonic(),
            planned_step=1,
            predicted_delay=2,
            prefix_len=4,
        )
        with mock.patch.object(rollout, "log_collect_demos") as log:
            stats = rollout._rtc_merge_into_queue(
                queue, result, current_step=4, max_queue_len=8
            )
        self.assertTrue(stats["delay_mismatch"])
        self.assertEqual(stats["actual_delay"], 3)
        self.assertEqual(stats["seam"], 1.0)
        log.assert_called_once()
        self.assertEqual(log.call_args.args[1], "warning")
        self.assertIn("predicted 2", log.call_args.args[0])
        self.assertIn("recalibrating", log.call_args.args[0])

    def test_merge_blends_fully_when_chunks_agree(self):
        # Agreement within the gate blends the complete action, including
        # grippers.
        old = np.zeros(14, dtype=np.float64)
        queue = deque([(1, old.copy())])
        new = np.full((2, 14), 0.05, dtype=np.float64)
        result = rollout._InferenceResult(
            new, 0.1, time.monotonic(), planned_step=1
        )
        stats = rollout._merge_chunk_into_queue(
            queue, result, last_executed_step=0, max_queue_len=4,
            conflict_gate=0.073,
        )
        self.assertFalse(stats["conflict"])
        merged = queue[0][1]
        self.assertAlmostEqual(merged[0], 0.7 * 0.05)
        self.assertAlmostEqual(merged[6], 0.7 * 0.05)   # gripper blends too

    def test_merge_keeps_committed_rows_on_conflict(self):
        # Disagreement beyond one executable step: averaging would create a
        # trajectory neither plan proposed (the canceled-descent failure).
        # Committed rows stand verbatim; the new chunk owns only the tail.
        old = np.zeros(14, dtype=np.float64)
        queue = deque([(1, old.copy())])
        new = np.ones((3, 14), dtype=np.float64)
        result = rollout._InferenceResult(
            new, 0.1, time.monotonic(), planned_step=1
        )
        stats = rollout._merge_chunk_into_queue(
            queue, result, last_executed_step=0, max_queue_len=4,
            conflict_gate=0.073,
        )
        self.assertTrue(stats["conflict"])
        self.assertGreater(stats["disagreement"], 0.9)
        np.testing.assert_array_equal(queue[0][1], old)      # untouched
        self.assertEqual(queue[1][0], 2)                     # tail appended
        np.testing.assert_array_equal(queue[1][1], np.ones(14))
        self.assertEqual(stats["max_revision"], 0.0)


class RTCRolloutTests(unittest.TestCase):
    def _run(
        self,
        *,
        inference_s: float,
        max_steps: int,
        horizon: int = 8,
        returned_horizon: int = 8,
        tick_s: float = 0.004,
        actual_tick_s: float | None = None,
        rtc_horizon: int = 4,
        rtc_trigger: str = "auto",
    ):
        env = _FakeEnv(tick_s=tick_s, actual_tick_s=actual_tick_s)
        policy = _SleepingPolicy(
            configured_horizon=horizon,
            returned_horizon=returned_horizon,
            inference_s=inference_s,
        )
        policy.step_counter = lambda: len(env.commands)
        real_rtc_merge = rollout._rtc_merge_into_queue

        def recording_rtc_merge(*args, **kwargs):
            stats = real_rtc_merge(*args, **kwargs)
            policy.merge_stats.append(dict(stats))
            return stats

        saver = _FakeSaver()
        settings = {
            "YAM_ASYNC_PLAN": "1",
            "YAM_RTC": "1",
            "YAM_RTC_HORIZON": str(rtc_horizon),
            "YAM_RTC_TRIGGER": rtc_trigger,
            "YAM_RTC_SCHEDULE": "linear",
            "YAM_RTC_MAX_GUIDANCE": "10",
            "YAM_INFER_BUDGET_S": "0",
            "YAM_ACTION_EMA": "0.5",
            "YAM_GRIPPER_RATE": "0.15",
        }
        with (
            mock.patch.dict(os.environ, settings, clear=True),
            mock.patch.object(
                rollout,
                "dynamic_smoothing",
                side_effect=_fake_dynamic_smoothing,
            ),
            mock.patch.object(
                rollout,
                "clamped_step",
                side_effect=_fake_clamped_step,
            ),
            mock.patch.object(
                rollout,
                "_rtc_merge_into_queue",
                side_effect=recording_rtc_merge,
            ),
        ):
            outcome = rollout.run_one_rollout(
                env=env,
                policy=policy,
                saver=saver,
                instruction="fake task",
                rollout_idx=7,
                num_rollouts=1,
                max_steps=max_steps,
                live_view=_FakeView(),
            )
        return env, policy, saver, outcome

    def test_auto_trigger_drains_queue_and_uses_raw_prefix(self):
        env, policy, saver, outcome = self._run(
            inference_s=0.001,
            max_steps=45,
            horizon=30,
            returned_horizon=30,
            rtc_horizon=10,
        )
        self.assertEqual(outcome.end_reason, "timeout")
        self.assertEqual(len(saver.steps), 45)
        self.assertEqual(policy.episode_id, 7)
        # Initial sync plus one delayed RTC replan. The obsolete continuous
        # trigger would issue roughly one request per inference interval.
        self.assertEqual(policy.call_count, 2)
        self.assertIsNone(policy.calls[0])  # unguided cold start
        first_rtc = next(call for call in policy.calls[1:] if call is not None)
        self.assertEqual(first_rtc["prefix_actions"].shape, (10, 14))
        # The raw first plan is 1.0, while alpha=0.5 makes the first command 0.5.
        np.testing.assert_array_equal(
            first_rtc["prefix_actions"], np.ones((10, 14), dtype=np.float32)
        )
        self.assertEqual(env.commands[0][0], 0.5)

    def test_steady_state_auto_trigger_calibrates_and_spaces_submits(self):
        _env, policy, saver, outcome = self._run(
            inference_s=0.2,
            max_steps=55,
            horizon=30,
            returned_horizon=30,
            tick_s=0.04,
            actual_tick_s=0.055,
            rtc_horizon=10,
            rtc_trigger="auto",
        )
        self.assertEqual(outcome.end_reason, "timeout")
        self.assertEqual(len(saver.steps), 55)
        self.assertEqual(policy.call_count, 3)

        # The cold-start wall-clock estimate predicts six nominal 40 ms
        # ticks. The overloaded 55 ms loop consumes only about four actions;
        # the following request must use that observed action count.
        first_rtc = policy.calls[1]
        second_rtc = policy.calls[2]
        self.assertIsNotNone(first_rtc)
        self.assertIsNotNone(second_rtc)
        first_actual = policy.merge_stats[1]["actual_delay"]
        self.assertGreater(first_actual, 0)
        self.assertGreater(first_rtc["inference_delay"], first_actual)
        self.assertEqual(second_rtc["inference_delay"], first_actual)

        # Auto submits only after the replacement queue drains to
        # observed-delay + execution-horizon rows. This rules out the old
        # continuous behavior, which submitted again as soon as inference
        # completed and kept every executed row heavily constrained.
        expected_trigger = min(30, first_actual + 10)
        expected_spacing = 30 - expected_trigger
        guided_submit_spacing = policy.call_steps[2] - policy.call_steps[1]
        self.assertGreater(expected_spacing, 0)
        self.assertGreaterEqual(guided_submit_spacing, expected_spacing)

    def test_starvation_wait_then_unguided_sync_fallback_is_intact(self):
        env, policy, saver, outcome = self._run(
            inference_s=0.03,
            max_steps=8,
            horizon=4,
            returned_horizon=4,
            tick_s=0.002,
        )
        self.assertEqual(outcome.end_reason, "timeout")
        self.assertEqual(len(saver.steps), 8)
        self.assertGreaterEqual(policy.call_count, 3)
        # initial sync, stale RTC result, then fresh unguided sync fallback
        self.assertIsNone(policy.calls[0])
        self.assertIsNotNone(policy.calls[1])
        self.assertIsNone(policy.calls[2])
        self.assertGreater(
            float(np.max(np.diff(env.command_times))),
            1.5 * env.control_period_s,
        )


class _ClampEnv:
    """Records step_command_only calls; measured state lags commands.

    The lag model is the property under test: the servo never fully reaches
    a setpoint within one tick, so ``get_robot_state`` returns a pose partway
    toward the last command. This is the situation in which the legacy ramp branch
    emits commands BEHIND its own previous command (the sawtooth).
    """

    def __init__(self, n_dofs: int = 14, lag: float = 0.5, tick_s: float = 1 / 30):
        self.control_period_s = tick_s
        self.state = np.zeros(n_dofs, dtype=np.float64)
        self.commands: list[np.ndarray] = []
        self.lag = float(lag)

    def get_robot_state(self):
        return {"joint_positions": self.state.copy()}

    def get_obs(self):
        return {"joint_positions": self.state.copy()}

    def step_command_only(self, joints, tick_s=None):
        cmd = np.asarray(joints, dtype=np.float64).copy()
        self.commands.append(cmd)
        # Servo lag: measured pose moves only part of the way to the command.
        self.state = self.state + self.lag * (cmd - self.state)


class ClampedLimiterTests(unittest.TestCase):
    """The reference command contract (YAM_LIMITER=clamp)."""

    def test_within_limit_passthrough(self):
        curr = np.zeros(14)
        target = np.full(14, 0.03)
        out = rollout.compute_clamped_command(curr, target, 0.05, 0.15)
        np.testing.assert_allclose(out, target)

    def test_arm_clamped_gripper_separate(self):
        curr = np.zeros(14)
        target = np.full(14, 1.0)
        out = rollout.compute_clamped_command(curr, target, 0.05, 0.15)
        arm = (np.arange(14) % 7) != 6
        np.testing.assert_allclose(out[arm], 0.05)
        np.testing.assert_allclose(out[~arm], 0.15)

    def test_command_between_measured_and_target(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            curr = rng.normal(size=14)
            target = rng.normal(size=14)
            out = rollout.compute_clamped_command(curr, target, 0.05, 0.15)
            lo = np.minimum(curr, target)
            hi = np.maximum(curr, target)
            self.assertTrue(np.all(out >= lo - 1e-12))
            self.assertTrue(np.all(out <= hi + 1e-12))

    def test_one_command_per_call_and_no_rewind_under_lag(self):
        env = _ClampEnv(lag=0.4)
        target = np.full(14, 2.0)  # far target: legacy ramp would block+rewind
        sink: list = []
        for i in range(20):
            rollout.clamped_step(env, target, 0.15, sink)
            self.assertEqual(len(env.commands), i + 1)  # exactly one per call
        cmds = np.stack(env.commands)
        # Monotone toward the target: no command ever steps backward relative
        # to the previous command (the sawtooth this contract eliminates).
        self.assertTrue(np.all(np.diff(cmds, axis=0) >= -1e-12))
        self.assertEqual(len(sink), 20)
        self.assertTrue(all(r["kind"] == "emitted" for r in sink))

    def test_ramp_branch_rewinds_under_lag_documenting_the_defect(self):
        # The legacy branch, same lag model: its first waypoint after a
        # full-target catch-up starts from the measured pose, which is
        # behind the previous command, so the recorded command stream goes
        # backward. This test pins the defect the clamp contract fixes.
        env = _ClampEnv(lag=0.4)
        rollout.dynamic_smoothing(env, np.full(14, 2.0))
        # Second action, target still ahead: its ramp restarts from the
        # measured pose, which lags the previous full-target command.
        rollout.dynamic_smoothing(env, np.full(14, 3.0))
        cmds = np.stack(env.commands)
        self.assertLess(float(np.min(np.diff(cmds, axis=0))), -1e-6)

    def test_limiter_env_validated(self):
        self.assertIn(rollout.LIMITER, ("ramp", "clamp"))

    def test_gripper_walks_to_deep_target_despite_block(self):
        # Object blocks the measured gripper at 0.49; the policy asks for a
        # deep 0.05 close. With commanded anchoring the emitted command must
        # WALK past the block point to the target (training grip force);
        # measured anchoring would freeze it at 0.49 - 0.15 = 0.34.
        class _BlockedEnv(_ClampEnv):
            def step_command_only(self, joints, tick_s=None):
                cmd = np.asarray(joints, dtype=np.float64).copy()
                self.commands.append(cmd)
                self.state = self.state + self.lag * (cmd - self.state)
                gripper = (np.arange(len(self.state)) % 7) == 6
                self.state[gripper] = np.maximum(self.state[gripper], 0.49)

        env = _BlockedEnv(lag=1.0)
        env.state[:] = 0.0
        env.state[[6, 13]] = 0.9
        target = np.zeros(14)
        target[[6, 13]] = 0.05
        prev = None
        for _ in range(12):
            rollout.clamped_step(env, target, 0.15, prev_cmd=prev)
            prev = env.commands[-1]
        self.assertLess(float(env.commands[-1][13]), 0.06)
        # and per-tick gripper rate stays limited (stop-slam protection)
        g = np.array([c[13] for c in env.commands])
        self.assertLessEqual(float(np.abs(np.diff(g)).max()), 0.15 + 1e-9)


class GripperObsOverrideTests(unittest.TestCase):
    """The policy must see the COMMANDED gripper, not the blocked one.

    Training observations track the gripper command. A physical gripper can
    stop against an object before reaching it.
    """

    def test_gripper_dims_replaced_arms_untouched(self):
        obs = {"joint_positions": np.linspace(0, 1.3, 14), "x": "kept"}
        cmd = np.full(14, 9.0)
        cmd[6], cmd[13] = 0.34, 0.31
        out = rollout._override_gripper_obs(obs, cmd)
        arm = (np.arange(14) % 7) != 6
        np.testing.assert_array_equal(
            out["joint_positions"][arm], obs["joint_positions"][arm]
        )
        self.assertEqual(out["joint_positions"][6], 0.34)
        self.assertEqual(out["joint_positions"][13], 0.31)
        self.assertEqual(out["x"], "kept")
        # original obs must not be mutated
        self.assertNotEqual(obs["joint_positions"][6], 0.34)

    def test_none_command_is_passthrough(self):
        obs = {"joint_positions": np.zeros(14)}
        self.assertIs(rollout._override_gripper_obs(obs, None), obs)


class RTCFeasiblePrefixTests(unittest.TestCase):
    """RTC must be conditioned on what WILL execute, not the raw queue.

    These tests exercise the clamp against a lagged servo so the prefix
    reflects feasible commands under saturation.
    """

    def _queue(self, values):
        return deque((i, np.asarray(v, dtype=np.float64))
                     for i, v in enumerate(values))

    def test_raw_prefix_preserved_without_start_pose(self):
        q = self._queue([np.full(14, 0.5), np.full(14, 1.0)])
        payload = rollout._rtc_payload_from_queue(
            q, _rtc_options(rtc_horizon=2), predicted_delay=1
        )
        np.testing.assert_allclose(payload["prefix_actions"][1], 1.0)

    def test_feasible_prefix_respects_clamp_from_lagging_pose(self):
        # Arm is at 0 (lagging); queue asks for far targets. The feasible
        # prefix must advance by at most arm_step per row, not teleport.
        q = self._queue([np.full(14, 2.0)] * 4)
        payload = rollout._rtc_payload_from_queue(
            q, _rtc_options(rtc_horizon=4), predicted_delay=1,
            start_pose=np.zeros(14), arm_step=0.0733, gripper_step=0.15,
        )
        prefix = payload["prefix_actions"]
        arm = (np.arange(14) % 7) != 6
        steps = np.diff(
            np.vstack([np.zeros(14)[arm], prefix[:, arm]]), axis=0
        )
        self.assertLessEqual(float(np.abs(steps).max()), 0.0733 + 1e-6)
        # and it must differ from the raw (infeasible) queue rows
        self.assertGreater(float(np.abs(prefix[0, arm] - 2.0).max()), 1.0)

    def test_feasible_prefix_equals_raw_when_tracking(self):
        # Within-limit queue rows: the simulation reproduces them exactly.
        rows = [np.full(14, 0.05 * (i + 1)) for i in range(3)]
        q = self._queue(rows)
        payload = rollout._rtc_payload_from_queue(
            q, _rtc_options(rtc_horizon=3), predicted_delay=1,
            start_pose=np.zeros(14), arm_step=0.0733, gripper_step=0.15,
        )
        np.testing.assert_allclose(
            payload["prefix_actions"], np.stack(rows).astype(np.float32),
            atol=1e-6,
        )

    def test_closed_loop_clamp_prefix_matches_lagged_execution(self):
        # Drive the REAL clamped_step against a lagged servo while a queue
        # of moving targets plays out; the feasible prefix computed from the
        # measured pose must match what execution then actually emits
        # (exact under raw actions + per-tick clamp, by construction).
        env = _ClampEnv(lag=0.4)
        targets = [np.full(14, 0.4 * (i + 1)) for i in range(6)]
        # execute two steps so the servo is genuinely lagging
        for t in targets[:2]:
            rollout.clamped_step(env, t, 0.15)
        pose = env.get_robot_state()["joint_positions"]
        q = self._queue(targets[2:])
        payload = rollout._rtc_payload_from_queue(
            q, _rtc_options(rtc_horizon=4), predicted_delay=1,
            start_pose=pose,
            arm_step=env.control_period_s * rollout.V_MAX,
            gripper_step=0.15,
        )
        # now actually execute the queued targets through the real limiter
        executed = []
        for t in targets[2:]:
            rollout.clamped_step(env, t, 0.15)
            executed.append(env.commands[-1])
        # perfect-servo assumption: compare against clamp-forward of the
        # commands, tolerating the servo-lag base difference per row
        prefix = payload["prefix_actions"].astype(np.float64)
        exec_arr = np.stack(executed)
        arm = (np.arange(14) % 7) != 6
        err = np.abs(prefix[:, arm] - exec_arr[:, arm]).max()
        # bounded by the per-row servo shortfall, far below the 0.7 rad
        # fiction the raw queue would carry
        self.assertLess(float(err), 0.35)
        raw_err = np.abs(
            np.stack(targets[2:])[:, arm] - exec_arr[:, arm]
        ).max()
        self.assertGreater(float(raw_err), float(err))


class ShutdownTests(unittest.TestCase):
    def tearDown(self):
        rollout._termination_requested = False
        rollout._env = None
        rollout._park_done = False
        rollout._env_close_done = False

    def test_sigterm_enters_keyboard_interrupt_cleanup_path(self):
        rollout._termination_requested = False

        with self.assertRaises(KeyboardInterrupt):
            rollout._handle_sigterm(15, None)

        self.assertTrue(rollout._termination_requested)

    def test_session_shuts_robot_down_before_conversion(self):
        events = []
        cfg = {
            "storage": {
                "base_dir": "/tmp",
                "task_directory": "yam-test",
                "language_instruction": "test",
            }
        }
        view = mock.Mock()
        view.close.side_effect = lambda: events.append("close-view")

        with (
            mock.patch.object(rollout, "LiveCameraView", return_value=view),
            mock.patch.object(
                rollout,
                "_shutdown_robot",
                side_effect=lambda: events.append("shutdown"),
            ),
            mock.patch.object(
                rollout,
                "_convert_if_any",
                side_effect=lambda *_args: events.append("convert"),
            ),
        ):
            rollout.run_session(None, None, cfg, None, False, 0)

        self.assertEqual(events, ["close-view", "shutdown", "convert"])

    def test_shutdown_parks_then_closes_once(self):
        events = []
        env = mock.Mock()
        env.close.side_effect = lambda: events.append("close")
        rollout._env = env
        rollout._bimanual = False
        rollout._left_cfg = {}
        rollout._right_cfg = None
        rollout._park_done = False
        rollout._env_close_done = False

        with mock.patch.object(
            rollout,
            "move_to_start_position",
            side_effect=lambda *_args: events.append("park"),
        ):
            rollout._shutdown_robot()
            rollout._shutdown_robot()

        self.assertEqual(events, ["park", "close"])

    def test_lerobot_conversion_requires_recorded_frames(self):
        with mock.patch.object(rollout, "convert_session_to_lerobot") as convert:
            rollout._convert_if_any(
                [Path("rollout")], Path("output"), "timestamp", {}
            )

        convert.assert_not_called()

    def test_missing_lerobot_dependency_is_a_warning_not_an_error(self):
        missing = ModuleNotFoundError(
            "No module named 'lerobot_convert'", name="lerobot_convert"
        )
        cfg = {"storage": {"save_frames": True}}
        with (
            mock.patch.object(
                rollout, "convert_session_to_lerobot", side_effect=missing
            ),
            self.assertLogs(rollout.logger, level="WARNING") as logs,
        ):
            rollout._convert_if_any(
                [Path("rollout")], Path("output"), "timestamp", cfg
            )

        self.assertIn("lerobot_convert is not installed", logs.output[0])

    def test_sigterm_shutdown_skips_conversion(self):
        rollout._termination_requested = True
        cfg = {
            "storage": {
                "base_dir": "/tmp",
                "task_directory": "yam-test",
                "language_instruction": "test",
            }
        }

        with (
            mock.patch.object(rollout, "LiveCameraView"),
            mock.patch.object(rollout, "_shutdown_robot") as shutdown,
            mock.patch.object(rollout, "_convert_if_any") as convert,
        ):
            rollout.run_session(None, None, cfg, None, False, 0)

        shutdown.assert_called_once_with()
        convert.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
