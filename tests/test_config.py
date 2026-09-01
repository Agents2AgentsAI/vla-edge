"""Embodiment contract tests. No GPU, no checkpoint, no network."""

from __future__ import annotations

import dataclasses

import pytest

from vla_edge.config import EMBODIMENTS, Embodiment, get_embodiment


def test_builtin_embodiments_are_self_consistent():
    for name, emb in EMBODIMENTS.items():
        assert emb.name == name, "dict key must match the embodiment's name"
        assert emb.num_cameras == len(emb.camera_names)
        assert emb.state_dim > 0
        assert emb.default_num_steps > 0
        assert len(set(emb.camera_names)) == emb.num_cameras, "duplicate camera name"


def test_libero_contract_matches_checkpoint():
    emb = get_embodiment("libero")

    assert emb.repo_id == "allenai/MolmoAct2-LIBERO"
    assert emb.norm_tag == "libero"
    assert emb.state_dim == 8
    assert emb.camera_names == ("image", "wrist_image")


def test_get_embodiment_unknown_lists_alternatives():
    with pytest.raises(KeyError) as exc:
        get_embodiment("no-such-robot")
    assert "bimanual-yam" in str(exc.value)


@pytest.fixture
def emb():
    return Embodiment(
        name="t", repo_id="x/y", norm_tag="n", state_dim=4,
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
    """Embodiment values are checkpoint properties, not runtime settings."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        emb.state_dim = 7  # type: ignore[misc]
