"""Normalized gripper convention contracts; no robot or GPU required."""

from __future__ import annotations

import numpy as np
import pytest

from vla_edge.gripper import (
    CLOSED_ONE_OPEN_ZERO,
    CLOSED_ZERO_OPEN_ONE,
    GripperContract,
)


def test_matching_checkpoint_and_wire_conventions_are_exact_noop():
    contract = GripperContract(
        indices=(6, 13),
        checkpoint_convention=CLOSED_ZERO_OPEN_ONE,
        wire_convention=CLOSED_ZERO_OPEN_ONE,
    )
    values = np.linspace(-0.25, 1.25, 28, dtype=np.float64).reshape(2, 14)

    checkpoint = contract.checkpoint_from_wire(values)
    wire = contract.wire_from_checkpoint(checkpoint)

    np.testing.assert_array_equal(checkpoint, values)
    np.testing.assert_array_equal(wire, values)
    assert checkpoint is not values


def test_opposite_conventions_invert_only_gripper_channels_without_clipping():
    contract = GripperContract(
        indices=(1, 3),
        checkpoint_convention=CLOSED_ONE_OPEN_ZERO,
        wire_convention=CLOSED_ZERO_OPEN_ONE,
    )
    wire = np.asarray(
        [[10.0, 0.0, 20.0, 1.0], [30.0, -0.2, 40.0, 1.2]],
        dtype=np.float64,
    )

    checkpoint = contract.checkpoint_from_wire(wire)

    np.testing.assert_allclose(
        checkpoint,
        [[10.0, 1.0, 20.0, 0.0], [30.0, 1.2, 40.0, -0.2]],
    )
    np.testing.assert_array_equal(checkpoint[:, [0, 2]], wire[:, [0, 2]])
    np.testing.assert_allclose(contract.wire_from_checkpoint(checkpoint), wire)


@pytest.mark.parametrize(
    "value,match",
    [
        (
            {
                "indices": [6, 13],
                "checkpoint_convention": CLOSED_ZERO_OPEN_ONE,
            },
            "fields differ",
        ),
        (
            {
                "indices": [13, 6],
                "checkpoint_convention": CLOSED_ZERO_OPEN_ONE,
                "wire_convention": CLOSED_ZERO_OPEN_ONE,
            },
            "strictly increasing",
        ),
        (
            {
                "indices": [6, 14],
                "checkpoint_convention": CLOSED_ZERO_OPEN_ONE,
                "wire_convention": CLOSED_ZERO_OPEN_ONE,
            },
            "exceed vector width",
        ),
        (
            {
                "indices": [6, 13],
                "checkpoint_convention": [CLOSED_ZERO_OPEN_ONE],
                "wire_convention": CLOSED_ZERO_OPEN_ONE,
            },
            "checkpoint gripper convention",
        ),
        (
            {
                "indices": [6, "13"],
                "checkpoint_convention": CLOSED_ZERO_OPEN_ONE,
                "wire_convention": CLOSED_ZERO_OPEN_ONE,
            },
            "list of integers",
        ),
    ],
)
def test_descriptor_gripper_contract_rejects_ambiguous_shapes(value, match):
    with pytest.raises(ValueError, match=match):
        GripperContract.from_mapping(value, width=14)
