"""The shared host path.

Everything that is not one of the three heavy stages lives here, once, for
every backend: image coercion, state validation, camera-order enforcement, and
dispatch to the backend.

Keeping this shared is the point of the architecture. See ``docs/spec.md``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import Embodiment

log = logging.getLogger("vla_edge.pipeline")


def to_pil(arr: Any) -> Any:
    """Coerce an HxWx3 uint8 RGB array to PIL, without silent rescaling."""
    from PIL import Image

    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        # Clipping rather than rescaling: a float image in 0..1 that we
        # silently multiplied by 255 would be a very hard bug to find, so
        # callers must hand us uint8-ranged data.
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a, mode="RGB")


class Pipeline:
    """One loaded checkpoint plus one backend."""

    def __init__(self, embodiment: Embodiment, backend: Any) -> None:
        self.embodiment = embodiment
        self.backend = backend

    @classmethod
    def load(
        cls,
        embodiment: Embodiment,
        backend: str = "torch",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        **backend_kwargs: Any,
    ) -> Pipeline:
        from .backends.base import REGISTRY

        # Importing a backend module is what registers it.
        if backend == "torch":
            from .backends.torch import backend as _
            from .checkpoint import load_checkpoint
        elif backend == "tensorrt":
            from .backends.tensorrt import backend as _  # noqa: F401
            from .backends.tensorrt.host import load_checkpoint
        else:
            raise ValueError(f"unsupported backend {backend!r}")

        checkpoint_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        if backend == "tensorrt":
            engine_dir = backend_kwargs.get("engine_dir")
            if not engine_dir:
                raise TypeError("the tensorrt backend requires engine_dir=<path>")
            checkpoint_kwargs["engine_dir"] = engine_dir
        model, processor, _dir = load_checkpoint(
            embodiment.repo_id, **checkpoint_kwargs
        )
        impl = REGISTRY.create(
            backend, model=model, processor=processor, embodiment=embodiment,
            **backend_kwargs,
        )
        return cls(embodiment, impl)

    def predict(
        self,
        cameras: dict[str, Any],
        instruction: str,
        state: np.ndarray,
        num_steps: int | None = None,
        enable_cuda_graph: bool = False,
    ) -> np.ndarray:
        """Run one inference and return ``(horizon, action_dim)`` float32.

        ``cameras`` is keyed by the embodiment's camera names; ordering is
        taken from the embodiment, not from dict insertion order, so a client
        that serializes its fields in a different order still gets correct
        behavior. Missing or extra keys are an error, not a warning.
        """
        emb = self.embodiment
        emb.validate_cameras(list(cameras))
        # Order comes from the embodiment, never from the caller's dict.
        images = [to_pil(cameras[name]) for name in emb.camera_names]
        steps = int(num_steps if num_steps is not None else emb.default_num_steps)

        if hasattr(self.backend, "generate_actions"):
            return self.backend.generate_actions(
                images=images,
                instruction=instruction,
                state=state,
                num_steps=steps,
                enable_cuda_graph=enable_cuda_graph,
            )
        raise NotImplementedError(
            f"backend {self.backend.name!r} does not expose "
            "generate_actions; the staged three-call interface is not "
            "driven by Pipeline yet."
        )

    def warmup(self) -> None:
        if hasattr(self.backend, "warmup"):
            self.backend.warmup()

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()
