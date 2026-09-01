"""Gate arithmetic. These thresholds decide what reaches a robot, so the
metrics themselves get tested rather than trusted."""

from __future__ import annotations

import numpy as np
import pytest

from vla_edge.gates.parity import check_stage, rel_rms, seed_spread_gate


def test_rel_rms_is_zero_for_identical_input():
    x = np.random.default_rng(0).standard_normal((4, 8))
    assert rel_rms(x, x) == pytest.approx(0.0)


def test_rel_rms_is_scale_invariant():
    """A 10x larger signal with 10x larger error is the same relative error.

    This is the property that makes one threshold meaningful across stages
    whose activation magnitudes differ by orders of magnitude.
    """
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((16, 16))
    err = rng.standard_normal((16, 16)) * 0.01
    assert rel_rms(ref + err, ref) == pytest.approx(
        rel_rms(10 * (ref + err), 10 * ref), rel=1e-9
    )


def test_rel_rms_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        rel_rms(np.zeros((2, 3)), np.zeros((3, 2)))


def test_rel_rms_handles_zero_reference():
    """A zero reference must not divide by zero and silently 'pass'."""
    assert rel_rms(np.ones(4), np.zeros(4)) == pytest.approx(1.0)


def test_check_stage_verdict_follows_threshold():
    ref = np.ones(64)
    near = ref + 0.001
    far = ref + 0.5
    assert check_stage("s", near, ref, 0.06).passed
    assert not check_stage("s", far, ref, 0.06).passed


def test_fp16_scale_corruption_is_caught_at_the_documented_threshold():
    """The failure this project exists to prevent.

    An fp16 language backbone lands near 0.55 rel_rms while raising nothing.
    The gate must reject that at the bf16 noise-floor threshold.
    """
    rng = np.random.default_rng(2)
    ref = rng.standard_normal(4096)
    corrupted = ref + rng.standard_normal(4096) * 0.55
    result = check_stage("prefill", corrupted, ref, 0.06)
    assert not result.passed
    assert result.rel_rms > 0.4


def test_seed_spread_gate_scales_to_policy_variance():
    rng = np.random.default_rng(3)
    ref_a = rng.standard_normal((30, 14))
    ref_b = ref_a + rng.standard_normal((30, 14)) * 0.8  # a different seed
    close = ref_a + 0.01
    assert seed_spread_gate(close, ref_a, ref_b).passed

    far = ref_a + 5.0
    assert not seed_spread_gate(far, ref_a, ref_b).passed


def test_seed_spread_gate_fails_when_seeding_is_not_wired_up():
    """Identical output from two seeds means the generator is not plumbed.

    Reporting a pass there would certify a gate that never ran.
    """
    ref = np.ones((4, 4))
    assert not seed_spread_gate(ref, ref, ref.copy()).passed
