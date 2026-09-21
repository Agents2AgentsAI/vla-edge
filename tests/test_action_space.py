"""Action-space contract: absolute passes through, delta adds the observed state back."""

from __future__ import annotations

import numpy as np
import pytest

from vla_edge.action_space import ActionSpace, validate_action_space
from vla_edge.config import POLICIES, PolicySpec

YAM_MASK = [True] * 6 + [False] + [True] * 6 + [False]


def test_absent_declaration_is_absolute():
    space = ActionSpace.from_mapping(None, width=14, label="host.action_space")
    assert space.kind == "absolute" and space.mask is None
    chunk = np.arange(28, dtype=np.float64).reshape(2, 14)
    np.testing.assert_array_equal(space.to_absolute(chunk, np.ones(14)), chunk)


def test_delta_adds_state_to_masked_joints_only():
    space = ActionSpace.from_mapping(
        {"kind": "delta", "mask": YAM_MASK}, width=14, label="host.action_space"
    )
    chunk = np.full((3, 14), 0.25)
    state = np.arange(14, dtype=np.float64)

    result = space.to_absolute(chunk, state)

    joints = [i for i in range(14) if i not in (6, 13)]
    np.testing.assert_allclose(result[:, joints], np.broadcast_to(0.25 + state[joints], (3, 12)))
    np.testing.assert_allclose(result[:, [6, 13]], 0.25)
    # The caller's chunk is left untouched.
    np.testing.assert_allclose(chunk, 0.25)
    assert space.to_mapping() == {"kind": "delta", "mask": YAM_MASK}


@pytest.mark.parametrize(
    "declaration, message",
    [
        ({"kind": "relative"}, "must be one of"),
        ({"kind": "delta"}, "mask of 14 booleans"),
        ({"kind": "delta", "mask": [True] * 13}, "mask of 14 booleans"),
        ({"kind": "delta", "mask": [1] * 14}, "mask of 14 booleans"),
        ({"kind": "delta", "mask": [False] * 14}, "selects no action dimension"),
        ({"kind": "absolute", "mask": YAM_MASK}, "takes no mask"),
        ("delta", "mask of 14 booleans"),
        (7, "must be an object"),
    ],
)
def test_malformed_declarations_are_rejected(declaration, message):
    with pytest.raises(ValueError, match=message):
        ActionSpace.from_mapping(declaration, width=14, label="host.action_space")


def test_delta_rejects_a_state_that_does_not_fit():
    space = ActionSpace.from_mapping(
        {"kind": "delta", "mask": YAM_MASK}, width=14, label="host.action_space"
    )
    with pytest.raises(ValueError, match="finite state of 14"):
        space.to_absolute(np.zeros((2, 14)), np.zeros(13))
    with pytest.raises(ValueError, match="finite state of 14"):
        space.to_absolute(np.zeros((2, 14)), np.full(14, np.nan))
    with pytest.raises(ValueError, match="delta chunk must be"):
        space.to_absolute(np.zeros((2, 13)), np.zeros(14))


def test_policy_specs_declare_a_known_action_space():
    assert validate_action_space("absolute", label="x") == "absolute"
    with pytest.raises(ValueError, match="action_space must be one of"):
        PolicySpec(
            name="bad",
            model_family="abcvla",
            embodiment="bimanual-yam",
            repo_id="a/b",
            norm_tag="t",
            action_horizon=16,
            action_space="relative",
        )
    kinds = {name: spec.action_space for name, spec in POLICIES.items()}
    assert kinds["abcvla-bimanual-yam"] == "absolute"
    assert kinds["pi05-libero"] == "native"
    assert all(kind == "absolute" for name, kind in kinds.items() if "delta" not in name and name != "pi05-libero")
