"""The sampler seed reaches a backend that takes one and never a backend that does not."""
from __future__ import annotations

import numpy as np
import pytest

from vla_edge.config import POLICIES, get_embodiment
from vla_edge.pipeline import Pipeline


class _SeededBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_actions(self, *, images, instruction, state, num_steps, enable_cuda_graph=False, seed=None):
        self.calls.append({"num_steps": num_steps, "seed": seed})
        policy = POLICIES["abcvla-bimanual-yam"]
        return np.zeros((policy.action_horizon, 14), dtype=np.float32)


class _UnseededBackend:
    def generate_actions(self, *, images, instruction, state, num_steps, enable_cuda_graph=False):
        policy = POLICIES["abcvla-bimanual-yam"]
        return np.zeros((policy.action_horizon, 14), dtype=np.float32)


def _pipeline(backend):
    policy = POLICIES["abcvla-bimanual-yam"]
    return Pipeline(get_embodiment(policy.embodiment), policy, backend)


def _cameras():
    emb = get_embodiment("bimanual-yam")
    return {name: np.zeros((224, 224, 3), dtype=np.uint8) for name in emb.camera_names}


def test_seed_is_forwarded_to_a_backend_that_accepts_it():
    backend = _SeededBackend()
    pipe = _pipeline(backend)
    pipe.predict(_cameras(), "fold and stack the t-shirts", np.zeros(14, np.float32), num_steps=10, seed=1234)
    pipe.predict(_cameras(), "fold and stack the t-shirts", np.zeros(14, np.float32), num_steps=10)
    assert [c["seed"] for c in backend.calls] == [1234, None]


def test_seed_is_refused_by_a_backend_that_does_not_take_one():
    pipe = _pipeline(_UnseededBackend())
    with pytest.raises(ValueError, match="sampler seed"):
        pipe.predict(_cameras(), "x", np.zeros(14, np.float32), num_steps=10, seed=1)
    # Without a seed the legacy backend is untouched.
    out = pipe.predict(_cameras(), "x", np.zeros(14, np.float32), num_steps=10)
    assert out.shape == (30, 14)
