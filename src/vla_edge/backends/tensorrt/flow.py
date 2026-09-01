"""Optional compiled flow-stage backend.

The action expert is the one stage where a TensorRT engine is not the fastest
thing available: an artifact bundle may ship a `flow/` directory holding a
compiled implementation of the same computation. When it is present this
module loads it and the backend routes the action stage through it; when it is
absent, or does not match this machine, the backend runs `action_flow.plan`
and nothing else changes.

The package is loaded from the bundle directory, not from this repository, so
a bundle can carry a build for its own device without this package pinning a
CUDA toolchain or a torch ABI.
"""

from __future__ import annotations

import json
import logging
import types
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PACKAGE_NAME = "flow"
MANIFEST = "PACKAGE.json"
DRIVER = "driver.py"


class FlowPackageError(RuntimeError):
    """The bundle's flow package is present but cannot be used here."""


def _check_environment(manifest: dict) -> None:
    requires = manifest.get("requires", {})
    want_torch = requires.get("torch")
    if want_torch:
        import torch

        # Compiled Python extensions are bound to the ABI they were built
        # against. Compare the release series only: a patch difference has not
        # broken loading, a minor one will, and the failure is an opaque
        # undefined-symbol error at import.
        have = torch.__version__.split("+")[0].split(".")[:2]
        want = want_torch.split("+")[0].split(".")[:2]
        if have != want:
            raise FlowPackageError(
                f"flow package was built against torch {want_torch}, this "
                f"environment has {torch.__version__}. Serving "
                f"action_flow.plan instead."
            )


def is_compatible(engine_dir: str | Path) -> bool:
    """Whether the optional flow package can be attempted in this process.

    The lightweight TensorRT host uses this before loading weights. If the
    package cannot load because its torch ABI differs, the fallback plan does
    not need any PyTorch action-expert parameters.
    """
    directory = Path(engine_dir) / PACKAGE_NAME
    driver_path = directory / DRIVER
    if not directory.is_dir() or not driver_path.is_file():
        return False
    try:
        manifest_path = directory / MANIFEST
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.is_file()
            else {}
        )
        _check_environment(manifest)
    except (FlowPackageError, OSError, ValueError):
        return False
    return True


def load(engine_dir: str | Path, action_expert: Any, device: Any) -> Any | None:
    """Return a callable flow runner, or None if no usable package is present.

    Never raises for an absent or mismatched package: the TensorRT action
    engine is always a correct fallback, and a bundle that cannot use its
    faster path should still serve.
    """
    directory = Path(engine_dir) / PACKAGE_NAME
    if not directory.is_dir():
        return None

    try:
        manifest_path = directory / MANIFEST
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        _check_environment(manifest)

        driver_path = directory / DRIVER
        if not driver_path.is_file():
            raise FlowPackageError(f"no {DRIVER} in {directory}")
        # Execute the checksummed driver without SourceFileLoader's bytecode
        # cache. A serving run must not mutate a verified artifact bundle and
        # make its manifest fail on the next checksum pass.
        module = types.ModuleType("vla_edge_flow_driver")
        module.__file__ = str(driver_path)
        code = compile(driver_path.read_bytes(), str(driver_path), "exec")
        exec(code, module.__dict__)  # noqa: S102 - checksummed bundle code
        runner = module.FlowKernelRunner(action_expert, str(device))
        log.info("flow package loaded from %s", directory)
        return runner
    except FlowPackageError as exc:
        log.warning("%s", exc)
    except Exception as exc:  # noqa: BLE001  # any failure falls back to TRT
        log.warning("flow package at %s could not be loaded (%s: %s); "
                    "serving action_flow.plan instead",
                    directory, type(exc).__name__, exc)
    return None
