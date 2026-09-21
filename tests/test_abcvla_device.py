"""Portable public hardware checks; no CUDA initialization."""
import pytest

from vla_edge.backends.abcvla.bundle import AbcVlaError
from vla_edge.backends.abcvla.device import verify_device_requirements

BASE = {"name": "NVIDIA Thor", "compute_capability": "sm_110a", "multiprocessor_count": 20, "tensorrt": "10.16.2.10"}


def test_compatible_thor_does_not_require_builder_board_uuid():
    assert verify_device_requirements(BASE, {**BASE, "compute_capability": "sm_110", "uuid": "another-board"})


@pytest.mark.parametrize("field,value", [("name", "other"), ("compute_capability", "sm_90"), ("multiprocessor_count", 16), ("tensorrt", "10.15.1")])
def test_different_hardware_or_runtime_is_rejected(field, value):
    with pytest.raises(AbcVlaError, match="device mismatch"):
        verify_device_requirements(BASE, {**BASE, field: value})


def test_incomplete_identity_is_not_treated_as_verified():
    with pytest.raises(AbcVlaError, match="incomplete"):
        verify_device_requirements({"name": "NVIDIA Thor"}, BASE)
