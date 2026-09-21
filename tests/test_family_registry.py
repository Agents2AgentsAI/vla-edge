"""Model-family extension and whole-policy pipeline contract tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vla_edge.backends.families import ModelFamilyRegistry
from vla_edge.config import get_embodiment, get_policy
from vla_edge.pipeline import Pipeline


def test_family_registry_lazily_dispatches_without_pipeline_branching():
    registry = ModelFamilyRegistry()
    seen = {}

    def load(**kwargs):
        seen.update(kwargs)
        return "runtime"

    registry.register("new-vla", load)
    assert registry.load("new-vla", backend="accelerator", custom=7) == "runtime"
    assert seen == {"backend": "accelerator", "custom": 7}
    assert registry.available() == ("new-vla",)


def test_family_registry_rejects_duplicate_and_unknown_names():
    registry = ModelFamilyRegistry()
    registry.register("x", lambda **_: None)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("x", lambda **_: None)
    with pytest.raises(KeyError, match="available: x"):
        registry.load("missing")


def test_pipeline_applies_shared_robot_action_contract():
    policy = get_policy("abcvla-bimanual-yam")
    embodiment = get_embodiment(policy.embodiment)
    backend = SimpleNamespace(
        name="fake",
        action_horizon=policy.action_horizon,
        generate_actions=lambda **_: np.zeros(
            (policy.action_horizon, embodiment.action_dim), dtype=np.float32
        ),
    )
    pipeline = Pipeline(embodiment, policy, backend)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    actions = pipeline.predict(
        cameras={name: frame for name in embodiment.camera_names},
        instruction="test",
        state=np.zeros(embodiment.state_dim, dtype=np.float32),
    )

    assert actions.shape == (30, 14)


@pytest.mark.parametrize("bad", [(29, 14), (30, 13)])
def test_pipeline_rejects_wrong_action_shape_from_any_family(bad):
    policy = get_policy("abcvla-bimanual-yam")
    embodiment = get_embodiment(policy.embodiment)
    backend = SimpleNamespace(
        name="fake",
        generate_actions=lambda **_: np.zeros(bad, dtype=np.float32),
    )
    pipeline = Pipeline(embodiment, policy, backend)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="returned actions with shape"):
        pipeline.predict(
            cameras={name: frame for name in embodiment.camera_names},
            instruction="test",
            state=np.zeros(embodiment.state_dim, dtype=np.float32),
        )


def test_pipeline_rejects_nonfinite_actions_from_any_family():
    policy = get_policy("molmoact2-bimanual-yam")
    embodiment = get_embodiment(policy.embodiment)
    actions = np.zeros((policy.action_horizon, embodiment.action_dim), dtype=np.float32)
    actions[0, 0] = np.nan
    backend = SimpleNamespace(name="fake", generate_actions=lambda **_: actions)
    pipeline = Pipeline(embodiment, policy, backend)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="non-finite"):
        pipeline.predict(
            cameras={name: frame for name in embodiment.camera_names},
            instruction="test",
            state=np.zeros(embodiment.state_dim, dtype=np.float32),
        )
