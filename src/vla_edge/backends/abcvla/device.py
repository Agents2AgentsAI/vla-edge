"""Portable hardware requirements for released ABC-VLA engines."""
from __future__ import annotations

from typing import Any

from .bundle import AbcVlaError


def detect_device(torch: Any, trt: Any) -> dict:
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "name": prop.name,
        "compute_capability": f"sm_{prop.major}{prop.minor}",
        "multiprocessor_count": int(prop.multi_processor_count),
        "tensorrt": trt.__version__,
    }


def verify_device_requirements(want: dict, have: dict) -> bool:
    """Require the recorded GPU model and TensorRT version, not one board serial."""
    required = {"name", "compute_capability", "multiprocessor_count", "tensorrt"}
    if not isinstance(want, dict) or not required.issubset(want):
        raise AbcVlaError("engine metadata has incomplete hardware requirements")
    differences = []
    for field in sorted(required):
        expected, actual = want[field], have.get(field)
        if field == "compute_capability":
            expected = str(expected).rstrip("a")
            actual = str(actual).rstrip("a")
        if expected != actual:
            differences.append(f"{field}: required={expected!r}, installed={actual!r}")
    if differences:
        raise AbcVlaError("engine device mismatch; " + "; ".join(differences))
    return True
