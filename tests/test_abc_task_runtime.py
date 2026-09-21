"""ABC task dispatch and action-clock checks; no CUDA, cameras or CAN."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parents[1] / "examples" / "bimanual-yam"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rollout = load("abc_rollout")
task = load("run_task")


def metadata():
    return {
        "status": "ok",
        "policy": "abcvla-bimanual-yam",
        "model_family": "abcvla",
        "embodiment": "bimanual-yam",
        "action_horizon": 30,
        "action_dim": 14,
        "state_dim": 14,
        "cameras": list(rollout.CAMERAS),
        "rtc": True,
        "rtc_mode": "hard-prefix",
        "max_prefix_length": 7,
        "gripper": {
            "state_source": "measured",
            "indices": [6, 13],
            "wire_convention": "closed_0_open_1",
        },
    }


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, delay):
        self.value += delay


class Env:
    control_period_s = 1 / 30

    def __init__(self, clock):
        self.clock, self.commands, self.times = clock, [], []
        self.state = np.zeros(14, np.float32)
        self.state[[6, 13]] = [0.4, 0.6]

    def robot(self):
        return self

    def command_joint_state(self, command):
        self.commands.append(command.copy())
        self.times.append(self.clock())
        self.state = command.copy()
        # Measured grippers differ from their last commanded values.
        self.state[[6, 13]] = [0.4, 0.6]

    def get_robot_state(self):
        return {"joint_positions": self.state.copy()}

    def get_obs(self):
        return {
            **self.get_robot_state(),
            **{
                k: np.full((2, 3, 3), [255, 0, 0], np.uint8)
                for k in rollout.CAMERAS.values()
            },
        }


class Client:
    def __init__(self):
        self.calls = []

    def act(self, **kwargs):
        self.calls.append(kwargs)
        result = np.zeros((30, 14), np.float32)
        result[:, 0] = (np.arange(30) + len(self.calls) * 30) * 0.0001
        result[:, [6, 13]] = 0.5
        return result, 32.0


@pytest.mark.parametrize("prefix", range(1, 8))
def test_exact_remaining_prefix_and_full_final_tick(prefix):
    clock = Clock()
    env = Env(clock)
    client = Client()
    recorded = []
    chunk = prefix + 1
    steps = rollout.run_episode(
        env,
        client,
        "fold",
        prefix=prefix,
        chunk=chunk,
        max_steps=2 * chunk,
        clock=clock,
        sleep=clock.sleep,
        record=lambda before, after, command, at: recorded.append(at),
    )
    assert steps == 2 * chunk and len(client.calls) == 2
    first, second = client.calls
    assert first["prefix_actions"] is None and first["prefix_length"] is None
    np.testing.assert_array_equal(
        second["prefix_actions"], np.array(env.commands[1:chunk])
    )
    assert second["prefix_length"] == prefix
    assert second["state"][0] == env.commands[0][0]
    np.testing.assert_array_equal(
        second["state"][[6, 13]], np.array([0.4, 0.6], np.float32)
    )
    assert env.commands[chunk][0] == np.float32((60 + prefix) * 0.0001)
    np.testing.assert_array_equal(first["cameras"]["top_cam"][0, 0], [255, 0, 0])
    assert all(
        b - a >= env.control_period_s - 1e-9 for a, b in zip(env.times, recorded)
    )
    assert all(
        b - a >= env.control_period_s - 1e-9 for a, b in zip(env.times, env.times[1:])
    )


def test_stop_does_not_issue_another_command():
    clock = Clock()
    env = Env(clock)
    rollout.run_episode(
        env,
        Client(),
        "fold",
        stop=lambda: len(env.commands) == 1,
        clock=clock,
        sleep=clock.sleep,
    )
    assert len(env.commands) == 1


def test_large_targets_are_limited_before_rtc_commitment():
    clock = Clock()

    class LaggingEnv(Env):
        def command_joint_state(self, command):
            super().command_joint_state(command)
            self.state[rollout.ARM_INDICES] = command[rollout.ARM_INDICES] * 0.5

    class LargeTargets(Client):
        def act(self, **kwargs):
            self.calls.append(kwargs)
            result = np.ones((30, 14), np.float32)
            if len(self.calls) == 2:
                result[:, rollout.ARM_INDICES] = -1.0
            return result, 1.0

    env = LaggingEnv(clock)
    client = LargeTargets()
    requests = []
    steps = rollout.run_episode(
        env,
        client,
        "fold",
        prefix=5,
        chunk=6,
        max_steps=12,
        max_joint_vel=3,
        clock=clock,
        sleep=clock.sleep,
        record_request=requests.append,
    )
    commands = np.asarray(env.commands)
    assert steps == 12
    np.testing.assert_allclose(commands[:6, 0], np.arange(1, 7) * 0.1, atol=1e-6)
    assert np.isclose(commands[6, 0], 0.5)  # starts from the final committed 0.6
    deltas = np.diff(np.vstack([np.zeros(14), commands]), axis=0)
    assert np.max(np.abs(deltas[:, rollout.ARM_INDICES])) <= 0.1 + 1e-6
    np.testing.assert_array_equal(client.calls[1]["prefix_actions"], commands[1:6])
    np.testing.assert_allclose(client.calls[1]["state"][rollout.ARM_INDICES], 0.05)
    np.testing.assert_array_equal(commands[:6, [6, 13]], np.ones((6, 2)))
    assert requests[0]["limited_arm_values"] > 0
    np.testing.assert_array_equal(requests[0]["command_chunk"], commands[:6])
    np.testing.assert_array_equal(requests[1]["command_anchor"], commands[5])


def test_bootstrap_continues_the_last_startup_command():
    clock = Clock()
    env = Env(clock)

    class LargeTargets(Client):
        def act(self, **kwargs):
            self.calls.append(kwargs)
            return np.ones((30, 14), np.float32), 1.0

    client = LargeTargets()
    last_startup = np.full(14, 0.5, np.float32)
    rollout.run_episode(
        env,
        client,
        "fold",
        max_steps=1,
        max_joint_vel=3,
        initial_command=last_startup,
        clock=clock,
        sleep=clock.sleep,
    )
    np.testing.assert_allclose(env.commands[0][rollout.ARM_INDICES], 0.6)
    np.testing.assert_array_equal(client.calls[0]["state"][rollout.ARM_INDICES], 0.0)


def test_failed_first_target_is_spread_over_ticks_without_raising_velocity():
    targets = np.zeros((2, 14), np.float32)
    targets[:, 8] = [0.1333, 0.14]
    targets[:, [6, 13]] = [-0.03, 1.02]
    planned = rollout.limit_chunk(targets, np.zeros(14), 1 / 30, 3.0)
    assert np.isclose(planned[0, 8], 0.1)
    assert np.isclose(planned[1, 8], 0.14)
    np.testing.assert_array_equal(planned[:, [6, 13]], targets[:, [6, 13]])
    assert np.isclose(targets[0, 8], 0.1333)  # caller's raw targets are retained


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_policy_actions_still_abort_before_motion(invalid):
    clock = Clock()
    env = Env(clock)

    class InvalidTargets(Client):
        def act(self, **kwargs):
            actions = np.ones((30, 14), np.float32)
            actions[0, 8] = invalid
            return actions, 1.0

    with pytest.raises(ValueError, match="invalid actions"):
        rollout.run_episode(
            env, InvalidTargets(), "fold", clock=clock, sleep=clock.sleep
        )
    assert not env.commands


@pytest.mark.parametrize("velocity", [0, -1, float("nan"), float("inf")])
def test_invalid_velocity_limit_is_rejected(velocity):
    with pytest.raises(ValueError, match="max_joint_vel"):
        rollout.limit_chunk(np.zeros((6, 14)), np.zeros(14), 1 / 30, velocity)


def test_startup_returns_the_final_command_for_policy_rate_limiting(monkeypatch):
    clock = Clock()
    env = Env(clock)
    configs = [{"agent": {"start_joints": [0.2] * 6 + [0.5]}} for _ in range(2)]
    monkeypatch.setattr(rollout.time, "sleep", clock.sleep)
    final = rollout.move_to_start(env, configs, 3, lambda: False)
    np.testing.assert_array_equal(final, env.commands[-1])


@pytest.mark.parametrize(
    "change",
    [
        {"gripper": {}},
        {"rtc_mode": "weighted"},
        {"cameras": ["left_cam", "top_cam", "right_cam"]},
    ],
)
def test_wrong_policy_contract_is_rejected(change):
    with pytest.raises(ValueError):
        rollout.validate_contract({**metadata(), **change}, 5, 6)


def fake_client(monkeypatch, family="abcvla"):
    class Health:
        def __init__(self, *args, **kwargs):
            pass

        def bind(self, **kwargs):
            return SimpleNamespace(model_family=family, policy=f"{family}-bimanual-yam")

        def health(self):
            return metadata()

        def close(self):
            pass

    monkeypatch.setattr(task, "ActClient", Health)


def test_check_dispatch_never_runs_the_robot(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fake_client(monkeypatch)
    calls = []
    monkeypatch.setattr(rollout, "preflight", lambda *args: calls.append("check"))
    monkeypatch.setattr(rollout, "run", lambda *args: pytest.fail("robot execution"))
    assert task.main(["fold", "--check"]) == 0
    assert calls == ["check"]


def test_abc_dispatch_uses_standard_command_and_explicit_settings(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    fake_client(monkeypatch)
    monkeypatch.setattr(rollout, "preflight", lambda *args: ["rig"])
    calls = []
    monkeypatch.setattr(
        rollout,
        "run",
        lambda args, rig, here: (
            calls.append((args.rtc_prefix_length, args.execute_chunk_dim, args.server))
            or 0
        ),
    )
    assert (
        task.main(["fold", "--rtc-prefix-length", "4", "--execute-chunk-dim", "6"]) == 0
    )
    assert calls == [(4, 6, "127.0.0.1:8202")]


def test_molmo_dispatch_preserves_legacy_launcher(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fake_client(monkeypatch, "molmoact2")
    calls = []
    monkeypatch.setattr(task.os, "execvpe", lambda *args: calls.append(args))
    task.main(["pick up the cube", "-n", "2"])
    assert calls[0][1] == [
        "bash",
        str(HERE / "run_molmoact_task.sh"),
        "pick up the cube",
        "-n",
        "2",
    ]


def test_native_gripper_values_are_not_changed_by_the_motion_guard():
    command = np.zeros(14, np.float32)
    command[[6, 13]] = [-0.03, 1.02]
    original = command.copy()
    rollout.check_command(command, np.zeros(14), 1 / 30, 2.2)
    np.testing.assert_array_equal(command, original)


@pytest.mark.parametrize(
    "close_fails,home_fails", [(False, False), (True, False), (False, True)]
)
def test_shutdown_homes_before_disabling_without_reopening_drivers(
    monkeypatch, close_fails, home_fails
):
    calls = []
    robot = object()
    last_command = np.full(14, 0.2)

    def close():
        calls.append("close")
        if close_fails:
            raise RuntimeError("close failed")

    def home(existing, **kwargs):
        assert existing is robot
        np.testing.assert_array_equal(kwargs["previous_command"], last_command)
        calls.append("home")
        if home_fails:
            raise RuntimeError("home failed")

    helpers = SimpleNamespace(
        home_active_robot=home,
        home_arms_together=lambda *args: pytest.fail("reopened drivers"),
        open_gripper=lambda *args: pytest.fail("reopened gripper"),
        disable_and_probe=lambda channels: calls.append("disable") or True,
    )
    monkeypatch.setitem(sys.modules, "home_arms", helpers)
    env = SimpleNamespace(robot=lambda: robot, close=close)
    if close_fails or home_fails:
        with pytest.raises(RuntimeError):
            rollout.shutdown(
                env, ["left", "right"], home=True, previous_command=last_command
            )
    else:
        rollout.shutdown(
            env, ["left", "right"], home=True, previous_command=last_command
        )
    assert calls == ["home", "close", "disable"]


def test_interrupted_homing_still_closes_and_disables(monkeypatch):
    calls = []

    def home(*args, **kwargs):
        calls.append("home")
        raise KeyboardInterrupt

    monkeypatch.setitem(
        sys.modules,
        "home_arms",
        SimpleNamespace(
            home_active_robot=home,
            disable_and_probe=lambda channels: calls.append("disable") or True,
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        rollout.shutdown(
            SimpleNamespace(
                robot=lambda: object(), close=lambda: calls.append("close")
            ),
            ["left", "right"],
            home=True,
        )
    assert calls == ["home", "close", "disable"]


def test_controller_fault_closes_and_disables_without_homing(monkeypatch):
    calls = []
    helpers = SimpleNamespace(
        home_active_robot=lambda *args, **kwargs: pytest.fail("home on fault"),
        disable_and_probe=lambda channels: calls.append("disable") or True,
    )
    monkeypatch.setitem(sys.modules, "home_arms", helpers)
    rollout.shutdown(
        SimpleNamespace(close=lambda: calls.append("close")),
        ["left", "right"],
        home=False,
    )
    assert calls == ["close", "disable"]


def test_last_command_is_remembered_when_stop_arrives_during_a_tick():
    clock = Clock()
    env = Env(clock)
    submitted = []
    recorded = []
    rollout.run_episode(
        env,
        Client(),
        "fold",
        clock=clock,
        sleep=clock.sleep,
        on_command=lambda command: submitted.append(command),
        stop=lambda: bool(submitted),
        record=lambda *args: recorded.append(args),
    )
    assert len(env.commands) == len(submitted) == 1 and not recorded
    np.testing.assert_array_equal(submitted[-1], env.commands[-1])


def test_startup_remembers_submitted_command_before_stop(monkeypatch):
    clock = Clock()
    env = Env(clock)
    submitted = []
    configs = [{"agent": {"start_joints": [0.3] * 6 + [0.5]}} for _ in range(2)]
    monkeypatch.setattr(rollout.time, "sleep", clock.sleep)
    with pytest.raises(KeyboardInterrupt):
        rollout.move_to_start(
            env, configs, 3, lambda: bool(submitted), on_command=submitted.append
        )
    np.testing.assert_array_equal(submitted[-1], env.commands[-1])


def test_partial_robot_startup_releases_every_resource_and_disables_both_buses(
    monkeypatch,
):
    calls = []

    class Camera:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            calls.append("camera-close")

    def construct(cfg):
        calls.append(cfg["channel"])
        if cfg["channel"] == "right":
            raise RuntimeError("right constructor failed")
        return SimpleNamespace(close=lambda: calls.append("left-close"))

    monkeypatch.setitem(
        sys.modules, "camera_client", SimpleNamespace(CameraClient=Camera)
    )
    monkeypatch.setitem(sys.modules, "gello_min.env", SimpleNamespace(RobotEnv=object))
    monkeypatch.setitem(
        sys.modules,
        "gello_min.launch_utils",
        SimpleNamespace(instantiate_from_dict=construct),
    )
    monkeypatch.setitem(
        sys.modules, "gello_min.robot", SimpleNamespace(BimanualRobot=object)
    )
    monkeypatch.setitem(
        sys.modules,
        "home_arms",
        SimpleNamespace(
            disable_and_probe=lambda channels: calls.append(tuple(channels))
        ),
    )
    cfg = [
        {
            "robot": {"channel": "left"},
            "eval": {"camera_server": {"endpoint": "tcp://example"}},
        },
        {"robot": {"channel": "right"}},
    ]
    with pytest.raises(RuntimeError, match="right constructor"):
        rollout.build_env(cfg)
    assert calls == ["left", "right", "left-close", "camera-close", ("left", "right")]
