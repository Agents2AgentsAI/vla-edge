"""The backend seam.

A backend replaces exactly three stages of a flow-matching VLA forward pass.
Everything else stays on the shared host path: tokenization, embedding scatter,
attention masks, RNG, normalization, and the checkpoint's loading quirks. It is
identical across backends. See docs/spec.md for why this is the
cut, and docs/gates.md for what a backend must prove before it is merged.

Implement these three methods, report validation results, and add a build
recipe. That is the whole contract.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VLABackend(Protocol):
    """One compute backend for one checkpoint on one device.

    Shapes below use:
        N  number of cameras
        P  patches per camera
        C  pooled image-token count
        S  prompt length in tokens (constant within an episode)
        L  transformer layers
        H  action horizon
        D  action dimension

    Implementations must be safe to call repeatedly with identical shapes;
    they may specialize on the shapes seen at warmup. They must NOT mutate
    their inputs. An in-place integrator that corrupted the caller's noise
    tensor is a real bug this contract exists to forbid (docs/gates.md).
    """

    #: Human-readable backend id, e.g. "torch", "tensorrt", "coreml-mlx".
    name: str

    def encode_vision(self, pixel_values: Any, pooling_idx: Any) -> Any:
        """Vision tower + pooling + projector.

        pixel_values : (1, N, P, hidden) float
        pooling_idx  : (1, C, k) int, identifying patches for each token
        returns      : (C, model_hidden) image tokens in the LLM's embedding
                       space, in the model's compute dtype.

        fp16 is acceptable here on every device we have measured.
        """
        ...

    def prefill(self, inputs_embeds: Any, attention_bias: Any,
                position_ids: Any) -> tuple[Any, Any]:
        """One LLM forward producing the per-layer key/value context.

        inputs_embeds  : (1, S, model_hidden)
        attention_bias : (1, 1, S, S) additive mask
        position_ids   : (1, S)
        returns        : (k_ctx, v_ctx), each (L, 1, S, kv_heads, head_dim),
                         already projected into the action expert's space if
                         the backend folds that projection into this stage.

        PRECISION IS NOT FREE HERE. This stage must run in the precision the
        model was trained in (bf16 for every checkpoint we have shipped). An
        fp16 build of this stage runs without error and returns values at 0.55
        relative RMS against the reference, a silent, total corruption. The
        parity gate is what catches it; do not weaken the gate to make a build
        pass.
        """
        ...

    def denoise(self, k_ctx: Any, v_ctx: Any, cross_mask: Any, noise: Any,
                steps: int) -> Any:
        """Flow-matching integration over the action chunk.

        k_ctx, v_ctx : as returned by prefill()
        cross_mask   : (1, 1, 1, S) additive mask over the prompt
        noise        : (1, H, D) initial sample; treat as READ-ONLY
        steps        : number of integration steps
        returns      : (1, H, D) actions, before denormalization

        `steps` is a behavior-changing parameter, not a performance knob:
        changing it changes the ODE being integrated. Reducing it requires the
        behavioral gate in docs/gates.md, not just parity.
        """
        ...

    def warmup(self) -> None:
        """Optional. Build graphs / compile / page in weights.

        Called once before the server accepts traffic. A backend that captures
        CUDA graphs must be called at least once before it is timed: capture
        without replay returns stale buffers, which looks like a garbage first
        inference and confuses everyone who meets it.
        """
        ...


class BackendRegistry:
    """Name -> factory. Backends register themselves on import."""

    def __init__(self) -> None:
        self._factories: dict[str, Any] = {}

    def register(self, name: str, factory: Any) -> None:
        if name in self._factories:
            raise ValueError(f"backend {name!r} is already registered")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> VLABackend:
        if name not in self._factories:
            raise KeyError(
                f"unknown backend {name!r}; available: "
                f"{', '.join(sorted(self._factories)) or '(none imported)'}")
        return self._factories[name](**kwargs)

    def available(self) -> list[str]:
        return sorted(self._factories)


REGISTRY = BackendRegistry()
