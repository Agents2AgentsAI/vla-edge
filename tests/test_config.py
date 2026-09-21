"""Robot and policy contract tests. No GPU, checkpoint, or network."""

from __future__ import annotations

import dataclasses

import pytest

from vla_edge.config import (
    EMBODIMENTS,
    POLICIES,
    Embodiment,
    default_policy,
    get_embodiment,
    get_policy,
    resolve_policy,
)


def test_builtin_embodiments_are_self_consistent():
    for name, emb in EMBODIMENTS.items():
        assert emb.name == name, "dict key must match the embodiment's name"
        assert emb.num_cameras == len(emb.camera_names)
        assert emb.state_dim > 0
        assert emb.action_dim > 0
        assert len(set(emb.camera_names)) == emb.num_cameras, "duplicate camera name"


def test_builtin_policies_reference_valid_embodiments():
    for name, policy in POLICIES.items():
        assert policy.name == name
        assert policy.embodiment in EMBODIMENTS
        assert policy.model_family
        assert policy.repo_id
        assert policy.norm_tag
        assert policy.action_horizon > 0
        assert policy.default_num_steps > 0


def test_libero_robot_and_checkpoint_contracts_are_separate():
    emb = get_embodiment("libero")
    policy = get_policy("molmoact2-libero")

    assert emb.state_dim == 8
    assert emb.action_dim == 8
    assert emb.camera_names == ("image", "wrist_image")
    assert policy.embodiment == emb.name
    assert policy.repo_id == "allenai/MolmoAct2-LIBERO"
    assert policy.norm_tag == "libero"






def test_get_embodiment_unknown_lists_alternatives():
    with pytest.raises(KeyError) as exc:
        get_embodiment("no-such-robot")
    assert "bimanual-yam" in str(exc.value)


def test_get_policy_unknown_lists_alternatives():
    with pytest.raises(KeyError) as exc:
        get_policy("no-such-policy")
    assert "abcvla-bimanual-yam" in str(exc.value)


@pytest.fixture
def emb():
    return Embodiment(
        name="t", state_dim=4, action_dim=3,
        camera_names=("a_cam", "b_cam"),
    )


def test_camera_validation_accepts_any_order(emb):
    """Field order in a JSON body is arbitrary; it must not be rejected.

    Ordering is applied by the pipeline from camera_names, so membership is
    the only thing this check can meaningfully enforce.
    """
    emb.validate_cameras(["a_cam", "b_cam"])
    emb.validate_cameras(["b_cam", "a_cam"])


def test_camera_validation_reports_missing_and_unexpected(emb):
    with pytest.raises(ValueError) as exc:
        emb.validate_cameras(["a_cam"])
    assert "missing" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        emb.validate_cameras(["a_cam", "b_cam", "c_cam"])
    assert "unexpected" in str(exc.value)


def test_embodiment_is_immutable(emb):
    """Robot wire values are contracts, not runtime settings."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        emb.state_dim = 7  # type: ignore[misc]




def test_abc_and_molmoact2_share_robot_geometry_but_keep_trained_state_contracts():
    abc = get_policy("abcvla-bimanual-yam")
    molmo = get_policy("molmoact2-bimanual-yam")
    emb = get_embodiment("bimanual-yam")
    assert abc.embodiment == molmo.embodiment == emb.name
    assert abc.action_horizon == molmo.action_horizon == 30
    assert abc.gripper_state == "measured"
    assert molmo.gripper_state == "commanded"
    assert emb.gripper_indices == (6, 13)
    assert emb.camera_names == ("top_cam", "left_cam", "right_cam")


def test_policy_resolution_preserves_molmo_defaults_and_checks_abc_pairing():
    assert default_policy("bimanual-yam").name == "molmoact2-bimanual-yam"
    assert resolve_policy(embodiment="bimanual-yam").name == "molmoact2-bimanual-yam"
    assert resolve_policy(policy="abcvla-bimanual-yam").embodiment == "bimanual-yam"
    with pytest.raises(ValueError, match="requires embodiment"):
        resolve_policy(policy="abcvla-bimanual-yam", embodiment="libero")


def test_public_catalog_has_no_unreleased_families():
    assert {p.model_family for p in POLICIES.values()} == {"molmoact2", "abcvla", "pi05"}
