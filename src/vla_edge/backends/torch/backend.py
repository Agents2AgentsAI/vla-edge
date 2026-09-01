"""The reference backend.

This is the ground truth every other backend is measured against, and it is
deliberately the *least* clever thing in the repository: it calls the
checkpoint's own unmodified action-generation path.

That is why it does not implement the three-stage seam. The staged interface
in ``backends/base.py`` exists so that *compiled* backends can replace stages
individually; the reference has nothing to replace. Decomposing it would mean
reimplementing model internals in this repo, which would make the ground truth
a thing we wrote. Otherwise, a bug here could silently become the standard that
every other backend is validated against.

If you need per-stage reference tensors (and you do, to build a compiled
backend), record them with ``scripts/capture.py``, which hooks the real model
rather than reimplementing it.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ..base import REGISTRY


class TorchReferenceBackend:
    """Whole-pass reference implementation.

    Not a staged backend: it exposes ``generate_actions`` rather than
    ``encode_vision`` / ``prefill`` / ``denoise``. ``Pipeline`` dispatches on
    the presence of that method.
    """

    name = "torch"
    staged = False

    def __init__(self, model: Any, processor: Any, embodiment: Any) -> None:
        self.model = model
        self.processor = processor
        self.embodiment = embodiment
        # The action expert's CUDA-graph capture is not safe under concurrent
        # calls. A robot client polling at a few Hz does not need the
        # concurrency, so serialize coarsely rather than reason about it.
        self._lock = threading.Lock()

    def generate_actions(
        self,
        images: list[Any],
        instruction: str,
        state: np.ndarray,
        num_steps: int,
        enable_cuda_graph: bool = False,
    ) -> np.ndarray:
        import torch

        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        expected = (self.embodiment.state_dim,)
        if state_f32.shape != expected:
            raise ValueError(
                f"state must be shape {expected}, got {state_f32.shape}"
            )

        with self._lock, torch.inference_mode():
            out = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=instruction,
                state=state_f32,
                norm_tag=self.embodiment.norm_tag,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=num_steps,
                normalize_language=True,
                enable_cuda_graph=enable_cuda_graph,
            )

        raw = out.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return actions

    def warmup(self) -> None:
        """One dummy pass, to pay graph capture and lazy init before serving.

        A CUDA graph captured but never replayed returns stale buffers, so the
        first real call after load can look like garbage. Warming up is not
        just a latency nicety here.
        """
        from PIL import Image

        dummy = Image.new("RGB", (320, 180))
        self.generate_actions(
            images=[dummy] * self.embodiment.num_cameras,
            instruction="warmup",
            state=np.zeros(self.embodiment.state_dim, dtype=np.float32),
            num_steps=self.embodiment.default_num_steps,
        )


def _factory(model: Any, processor: Any, embodiment: Any, **_: Any) -> TorchReferenceBackend:
    return TorchReferenceBackend(model, processor, embodiment)


REGISTRY.register("torch", _factory)
