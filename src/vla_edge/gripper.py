"""Explicit gripper-coordinate contracts shared by model-family backends.

Robot clients speak the embodiment's wire convention.  A checkpoint may have
been trained with the opposite convention, so the model-family host converts
state into checkpoint coordinates before preprocessing and converts actions
back to wire coordinates after postprocessing.  Conversion is affine and
deliberately does not clip: quantile transforms may legitimately extrapolate
slightly beyond the nominal ``[0, 1]`` endpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

CLOSED_ZERO_OPEN_ONE = "closed_0_open_1"
CLOSED_ONE_OPEN_ZERO = "closed_1_open_0"
GRIPPER_CONVENTIONS = frozenset(
    {CLOSED_ZERO_OPEN_ONE, CLOSED_ONE_OPEN_ZERO}
)

#: What the checkpoint's gripper STATE channel recorded during training.
#: ``commanded``: the dataset logged the commanded opening as the state (the
#: MolmoAct2 / robocurve YAM recordings), so the policy must be shown the last
#: command, not the finger position blocked by an object. ``measured``: the
#: dataset logged the encoder reading (ABC-130k), so a gripper closed on cloth
#: reads a small positive opening and the policy must see exactly that.
GRIPPER_STATE_COMMANDED = "commanded"
GRIPPER_STATE_MEASURED = "measured"
GRIPPER_STATE_SOURCES = frozenset({GRIPPER_STATE_COMMANDED, GRIPPER_STATE_MEASURED})


def validate_gripper_state(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or value not in GRIPPER_STATE_SOURCES:
        raise ValueError(
            f"{label} must be one of {sorted(GRIPPER_STATE_SOURCES)}, got {value!r}"
        )
    return str(value)


def validate_gripper_convention(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or value not in GRIPPER_CONVENTIONS:
        raise ValueError(
            f"{label} must be one of {sorted(GRIPPER_CONVENTIONS)}, got {value!r}"
        )
    return str(value)


@dataclass(frozen=True)
class GripperContract:
    """Checkpoint/wire mapping for normalized gripper aperture channels."""

    indices: tuple[int, ...]
    checkpoint_convention: str
    wire_convention: str

    def __post_init__(self) -> None:
        if not self.indices:
            raise ValueError("gripper contract must declare at least one index")
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in self.indices
        ):
            raise ValueError(f"gripper indices must be nonnegative integers: {self.indices}")
        if tuple(sorted(set(self.indices))) != self.indices:
            raise ValueError(
                f"gripper indices must be unique and strictly increasing: {self.indices}"
            )
        validate_gripper_convention(
            self.checkpoint_convention,
            label="checkpoint gripper convention",
        )
        validate_gripper_convention(
            self.wire_convention,
            label="wire gripper convention",
        )

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        width: int,
        label: str = "gripper contract",
    ) -> GripperContract:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be an object")  # noqa: TRY004 - preserve the wire-validation ValueError contract.
        expected = {"indices", "checkpoint_convention", "wire_convention"}
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise ValueError(
                f"{label} fields differ (missing={missing}, unexpected={extra})"
            )
        raw_indices = value["indices"]
        if not isinstance(raw_indices, list) or not all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in raw_indices
        ):
            raise ValueError(f"{label}.indices must be a list of integers")
        contract = cls(
            indices=tuple(raw_indices),
            checkpoint_convention=value["checkpoint_convention"],
            wire_convention=value["wire_convention"],
        )
        if width <= 0 or contract.indices[-1] >= width:
            raise ValueError(
                f"{label}.indices {contract.indices} exceed vector width {width}"
            )
        return contract

    def as_dict(self) -> dict[str, Any]:
        return {
            "indices": list(self.indices),
            "checkpoint_convention": self.checkpoint_convention,
            "wire_convention": self.wire_convention,
        }

    def _convert(
        self,
        values: Any,
        *,
        source: str,
        target: str,
    ) -> np.ndarray:
        validate_gripper_convention(source, label="source gripper convention")
        validate_gripper_convention(target, label="target gripper convention")
        result = np.array(values, copy=True)
        if result.ndim < 1 or result.shape[-1] <= self.indices[-1]:
            raise ValueError(
                f"gripper values have shape {result.shape}, expected final width "
                f"greater than {self.indices[-1]}"
            )
        selected = result[..., list(self.indices)]
        if not np.isfinite(selected).all():
            raise ValueError("gripper values must be finite")
        if source != target:
            result[..., list(self.indices)] = 1.0 - selected
        return result

    def checkpoint_from_wire(self, values: Any) -> np.ndarray:
        return self._convert(
            values,
            source=self.wire_convention,
            target=self.checkpoint_convention,
        )

    def wire_from_checkpoint(self, values: Any) -> np.ndarray:
        return self._convert(
            values,
            source=self.checkpoint_convention,
            target=self.wire_convention,
        )


__all__ = [
    "CLOSED_ONE_OPEN_ZERO",
    "CLOSED_ZERO_OPEN_ONE",
    "GRIPPER_CONVENTIONS",
    "GripperContract",
    "validate_gripper_convention",
]
