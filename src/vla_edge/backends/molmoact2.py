"""MolmoAct2 model-family loader.

This module owns MolmoAct2 checkpoint and bundle conventions. Neither the
shared pipeline nor the HTTP server needs to know those conventions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .base import REGISTRY

log = logging.getLogger(__name__)

_REQUIRED_ENGINE_FILES = (
    "llm_prefill.plan",
    "vision.plan",
    "action_flow.plan",
)


def _declared_engine_repo(engine_dir: Path) -> str | None:
    """Return the checkpoint declared by an engine set's compact host."""

    try:
        serving = json.loads((engine_dir / "serving.json").read_text())
        host_dir = serving.get("host_dir")
        if host_dir is None:
            return None
        host_path = Path(str(host_dir))
        if not host_path.is_absolute():
            host_path = engine_dir / host_path
        host_manifest = json.loads((host_path.resolve() / "host.json").read_text())
    except (OSError, TypeError, ValueError):
        return None
    repo_id = host_manifest.get("repo_id")
    return str(repo_id) if repo_id else None


def resolve_engine_dir(
    engine_dir: str | Path,
    *,
    repo_id: str,
    fast_vision: bool,
) -> Path:
    """Accept either one MolmoAct2 engine set or a released bundle root."""

    root = Path(engine_dir).expanduser().resolve()
    if (root / "serving.json").is_file() or not (root / "MANIFEST.json").is_file():
        return root

    discovered = [
        child
        for child in sorted(root.iterdir())
        if child.is_dir()
        and (child / "serving.json").is_file()
        and all((child / name).is_file() for name in _REQUIRED_ENGINE_FILES)
    ]
    candidates = [
        child
        for child in discovered
        if _declared_engine_repo(child) in (None, repo_id)
    ]
    if fast_vision:
        accelerated = [
            child for child in candidates if (child / "vision_fp8.plan").is_file()
        ]
        if len(accelerated) == 1:
            candidates = accelerated

    if len(candidates) == 1:
        selected = candidates[0]
        log.info("bundle root %s: selected engine set %s", root, selected.name)
        return selected
    if candidates:
        choices = ", ".join(str(path) for path in candidates)
        raise ValueError(
            f"{root} contains multiple engine sets for {repo_id}. "
            f"Pass one explicitly with --engine-dir: {choices}"
        )
    found = ", ".join(
        f"{path.name} ({_declared_engine_repo(path) or 'checkpoint unknown'})"
        for path in discovered
    ) or "none"
    raise ValueError(
        f"{root} is a bundle root but contains no engine set for {repo_id}. "
        f"Found: {found}"
    )


def load_backend(
    *,
    backend: str,
    policy: Any,
    embodiment: Any,
    device: str,
    dtype: str,
    engine_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    pad_multiple: int | None = None,
    fast_vision: bool = False,
    **backend_kwargs: Any,
) -> Any:
    """Construct one MolmoAct2 eager or TensorRT whole-policy runtime."""

    checkpoint_ref = checkpoint or policy.repo_id
    checkpoint_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
    if backend == "torch":
        from ..checkpoint import load_checkpoint
        from .torch import backend as _
    elif backend == "tensorrt":
        from .tensorrt import backend as _  # noqa: F401
        from .tensorrt.host import load_checkpoint

        if engine_dir is None:
            raise TypeError("the MolmoAct2 TensorRT backend requires engine_dir=<path>")
        engine_dir = resolve_engine_dir(
            engine_dir,
            repo_id=policy.repo_id,
            fast_vision=bool(fast_vision),
        )
        checkpoint_kwargs["engine_dir"] = engine_dir
    else:
        raise ValueError(f"unsupported MolmoAct2 backend {backend!r}")

    model, processor, _local_dir = load_checkpoint(
        checkpoint_ref, **checkpoint_kwargs
    )
    return REGISTRY.create(
        backend,
        model=model,
        processor=processor,
        policy=policy,
        embodiment=embodiment,
        engine_dir=engine_dir,
        pad_multiple=pad_multiple,
        fast_vision=fast_vision,
        **backend_kwargs,
    )


__all__ = ["load_backend", "resolve_engine_dir"]
