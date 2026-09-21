"""The shared host path.

Everything that is not one of the three heavy stages lives here, once, for
every backend: image coercion, state validation, camera-order enforcement, and
dispatch to the backend.

Keeping this shared is the point of the architecture. See ``docs/spec.md``.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import numpy as np

from .config import Embodiment, PolicySpec, default_policy, get_policy_embodiment

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

    def __init__(
        self,
        embodiment: Embodiment,
        policy: PolicySpec,
        backend: Any,
    ) -> None:
        self.embodiment = embodiment
        self.policy = policy
        self.backend = backend

    @classmethod
    def load(
        cls,
        policy: PolicySpec | Embodiment,
        backend: str = "torch",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        **backend_kwargs: Any,
    ) -> Pipeline:
        # Passing an Embodiment is the legacy API and selects its default
        # policy. New callers should select a PolicySpec explicitly.
        if isinstance(policy, Embodiment):
            embodiment = policy
            selected_policy = default_policy(embodiment)
        else:
            selected_policy = policy
            embodiment = get_policy_embodiment(selected_policy)

        from .backends.families import FAMILIES

        impl = FAMILIES.load(
            selected_policy.model_family,
            backend=backend,
            policy=selected_policy,
            embodiment=embodiment,
            device=device,
            dtype=dtype,
            **backend_kwargs,
        )
        return cls(embodiment, selected_policy, impl)

    def predict(
        self,
        cameras: dict[str, Any],
        instruction: str,
        state: np.ndarray,
        num_steps: int | None = None,
        enable_cuda_graph: bool = False,
        seed: int | None = None,
        noise: Any | None = None,
    ) -> np.ndarray:
        """Run one inference and return ``(horizon, action_dim)`` float32.

        ``seed`` pins the sampler's noise for backends that draw one (ABC-VLA's
        flow-matching noise). A client sends the same seed for every chunk of
        an episode so consecutive plans differ only by the observation, not
        by the sampler; backends that do not take a seed are left untouched.

        ``cameras`` is keyed by the embodiment's camera names; ordering is
        taken from the embodiment, not from dict insertion order, so a client
        that serializes its fields in a different order still gets correct
        behavior. Missing or extra keys are an error, not a warning.
        """
        emb = self.embodiment
        emb.validate_cameras(list(cameras))
        # Order comes from the embodiment, never from the caller's dict.
        images = [to_pil(cameras[name]) for name in emb.camera_names]
        steps = int(
            num_steps
            if num_steps is not None
            else self.policy.default_num_steps
        )

        if hasattr(self.backend, "generate_actions"):
            kwargs: dict[str, Any] = {}
            if seed is not None and noise is not None:
                raise ValueError("supply either seed or explicit noise, not both")
            if noise is not None:
                if "noise" not in inspect.signature(self.backend.generate_actions).parameters:
                    raise ValueError(f"backend {self.backend.name!r} does not accept explicit noise")
                kwargs["noise"] = noise
            if seed is not None:
                if "seed" not in inspect.signature(self.backend.generate_actions).parameters:
                    raise ValueError(
                        f"backend {type(self.backend).__name__} does not accept a sampler seed"
                    )
                kwargs["seed"] = int(seed)
            actions = self.backend.generate_actions(
                images=images,
                instruction=instruction,
                state=state,
                num_steps=steps,
                enable_cuda_graph=enable_cuda_graph,
                **kwargs,
            )
            result = np.asarray(actions, dtype=np.float32)
            expected = (self.policy.action_horizon, emb.action_dim)
            if result.shape != expected:
                raise ValueError(
                    f"policy {self.policy.name!r} returned actions with shape "
                    f"{result.shape}; expected {expected}"
                )
            if not np.isfinite(result).all():
                raise ValueError(
                    f"policy {self.policy.name!r} returned non-finite actions"
                )
            return result
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
