"""How a checkpoint's action chunk relates to the current joint state.

``absolute`` (the default, every existing policy) returns joint targets as-is.
``delta`` is openpi's ``DeltaActions`` recipe: the masked joints are predicted
as ``a[t+k] - s[t]`` and the host has to add the current state back before a
robot may execute them (``AbsoluteActions`` in openpi's output transforms).
The serving descriptor declares it under ``host.action_space``; the policy spec
declares the same kind, and the backend refuses a mismatch so a delta checkpoint
can never be served as absolute targets (which would drive the arms toward the
zero pose).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

ABSOLUTE = "absolute"
DELTA = "delta"
NATIVE = "native"  # Benchmark actions are passed through without joint-space conversion.
ACTION_SPACES = (ABSOLUTE, DELTA, NATIVE)


def validate_action_space(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or value not in ACTION_SPACES:
        raise ValueError(f"{label} must be one of {list(ACTION_SPACES)}, got {value!r}")
    return value


@dataclass(frozen=True)
class ActionSpace:
    kind: str
    #: ``delta`` only: one flag per action dimension, True where the checkpoint
    #: predicts a delta from the current state (openpi ``make_bool_mask(6, -1, 6, -1)``
    #: is ``[True]*6 + [False] + [True]*6 + [False]``: joints relative, grippers absolute).
    mask: tuple[bool, ...] | None = None

    @classmethod
    def absolute(cls) -> ActionSpace:
        return cls(ABSOLUTE)

    @classmethod
    def from_mapping(cls, value: Any, *, width: int, label: str) -> ActionSpace:
        """Parse a descriptor declaration; ``None`` means absolute (the historical contract)."""
        if value is None:
            return cls.absolute()
        if isinstance(value, str):
            value = {"kind": value}
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be an object with a kind, got {type(value).__name__}")  # noqa: TRY004 - preserve the wire-validation ValueError contract.
        kind = validate_action_space(value.get("kind"), label=f"{label}.kind")
        mask = value.get("mask")
        if kind == NATIVE:
            if mask is not None:
                raise ValueError(f"{label}: native actions take no joint delta mask")
            return cls(NATIVE)
        if kind == ABSOLUTE:
            if mask is not None:
                raise ValueError(f"{label}: an absolute action space takes no mask")
            return cls.absolute()
        if (
            not isinstance(mask, (list, tuple))
            or len(mask) != width
            or not all(isinstance(flag, bool) for flag in mask)
        ):
            raise ValueError(f"{label}: a delta action space needs a mask of {width} booleans")
        if not any(mask):
            raise ValueError(f"{label}: the delta mask selects no action dimension")
        return cls(DELTA, tuple(bool(flag) for flag in mask))

    def to_mapping(self) -> dict[str, Any]:
        if self.kind in (ABSOLUTE, NATIVE):
            return {"kind": self.kind}
        return {"kind": DELTA, "mask": list(self.mask or ())}

    def to_absolute(self, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Rebuild absolute joint targets from a ``(horizon, width)`` chunk.

        Mirrors openpi ``AbsoluteActions``: every masked dimension gets the
        current state added; the others (grippers) pass through. The state must
        be the one the policy observed, in the checkpoint's own conventions.
        """
        if self.kind == NATIVE:
            raise ValueError("native benchmark actions cannot be converted to absolute joint targets")
        chunk = np.asarray(actions, dtype=np.float64)
        if self.kind == ABSOLUTE:
            return chunk
        mask = np.asarray(self.mask, dtype=bool)
        current = np.asarray(state, dtype=np.float64).reshape(-1)
        if chunk.ndim != 2 or chunk.shape[1] != mask.shape[0]:
            raise ValueError(
                f"delta chunk must be (horizon, {mask.shape[0]}), got {chunk.shape}"
            )
        if current.shape != mask.shape or not np.isfinite(current).all():
            raise ValueError(
                f"delta action space needs a finite state of {mask.shape[0]} values"
            )
        result = chunk.copy()
        result[:, mask] += current[mask]
        return result
