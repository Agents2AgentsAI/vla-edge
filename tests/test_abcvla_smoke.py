"""The public comparison helper reuses inputs and closes each backend."""
import sys
from types import SimpleNamespace

import numpy as np

from vla_edge.scripts import smoke_abcvla


def test_comparison_uses_identical_noise_and_closes_models(monkeypatch, tmp_path):
    events = []
    noises = []
    class FakePipeline:
        backend = SimpleNamespace(arm_rtc=lambda *args, **kwargs: events.append("prefix"))
        def predict(self, images, prompt, state, *, noise):
            assert set(images) == {"top_cam", "left_cam", "right_cam"}
            assert state.shape == (14,)
            noises.append(noise.copy())
            return np.zeros((30, 14), np.float32)
        def close(self):
            events.append("closed")
    def load(*args, **kwargs):
        events.append(kwargs["backend"])
        return FakePipeline()
    monkeypatch.setattr(smoke_abcvla.Pipeline, "load", load)
    output = tmp_path / "actions.npy"
    monkeypatch.setattr(sys, "argv", ["smoke", "--bundle", str(tmp_path),
                                    "--prefix-length", "5", "--compare-torch", "--output", str(output)])
    smoke_abcvla.main()
    assert events == ["tensorrt", "prefix", "prefix", "closed", "torch", "prefix", "prefix", "closed"]
    assert all(np.array_equal(noises[0], x) for x in noises)
    assert np.load(output).shape == (30, 14)
