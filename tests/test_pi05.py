"""Pi0.5 wire and host contracts; these tests do not load a GPU model."""
import numpy as np
import pytest
from PIL import Image

from vla_edge.action_space import ActionSpace
from vla_edge.backends.pi05.host import Host, _resize_with_pad
from vla_edge.config import get_embodiment, get_policy, get_policy_embodiment
from vla_edge.pipeline import Pipeline


def test_libero_action_width_is_checkpoint_specific():
    pi = get_policy("pi05-libero")
    emb = get_policy_embodiment(pi)
    assert (emb.state_dim, emb.action_dim, pi.action_horizon) == (8, 7, 10)
    assert get_embodiment("libero").action_dim == 8
    assert get_policy_embodiment(get_policy("molmoact2-libero")).action_dim == 8
    with pytest.raises(ValueError, match="native benchmark"):
        ActionSpace("native").to_absolute(np.zeros((10,7)), np.zeros(8))


def test_pi05_pipeline_uses_seven_libero_actions(monkeypatch):
    from vla_edge.backends.families import FAMILIES
    class Fake:
        def generate_actions(self, **kwargs):
            return np.ones((10,7), np.float32)
    monkeypatch.setattr(FAMILIES, "load", lambda *args, **kwargs: Fake())
    pipeline = Pipeline.load(get_policy("pi05-libero"), backend="tensorrt")
    result = pipeline.predict({k:np.zeros((4,4,3), np.uint8) for k in pipeline.embodiment.camera_names}, "pick block", np.zeros(8))
    assert result.shape == (10,7)


class Tokenizer:
    def __init__(self):
        self.calls = []
    def encode(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return [2,3] if kwargs.get("add_bos") else [4]


def test_libero_prompt_does_not_discretize_state():
    host = Host.__new__(Host)
    host.desc = {"discrete_state_input": False}
    host.tokenizer = Tokenizer()
    tokens, valid = host.tokenize(" pick_red\nblock ", np.zeros(8))
    assert host.tokenizer.calls == [("pick red block", {"add_bos":True}), ("\n", {})]
    assert tokens[:3].tolist() == [2,3,4]
    assert valid.sum() == 3 and not valid[3:].any()


def test_yam_prompt_contains_fourteen_discrete_state_values():
    host = Host.__new__(Host)
    host.desc = {"discrete_state_input": True}
    host.tokenizer = Tokenizer()
    host.state_q = (np.zeros(14), np.ones(14))
    host.tokenize("fold", np.zeros(14))
    assert host.tokenizer.calls == [("Task: fold, State: " + " ".join(["0"]*14) + ";\nAction: ", {"add_bos":True})]


def test_resize_keeps_rgb_order_and_black_letterbox():
    image = np.zeros((20,40,3),np.uint8)
    image[:,:,0]=255
    resized = _resize_with_pad(image,224,224)
    assert np.all(resized[56:168,:,0] == 255)
    assert not resized[:56].any() and not resized[168:].any()
    assert not resized[:,:,1:].any()
    assert np.array_equal(resized,_resize_with_pad(Image.fromarray(image),224,224))


def test_unqualified_bundle_refuses_serving(tmp_path):
    import json

    from vla_edge.backends.pi05.bundle import Pi05Error, load_bundle
    (tmp_path/"pi05-serving.json").write_text(json.dumps({"release_status":"blocked"}))
    policy=get_policy("pi05-libero")
    with pytest.raises(Pi05Error,match="not passed numerical qualification"):
        load_bundle(tmp_path,policy,get_policy_embodiment(policy))


def test_readonly_wire_noise_is_owned_without_warnings(monkeypatch):
    import warnings
    from contextlib import nullcontext
    from threading import RLock
    from types import SimpleNamespace

    import json_numpy
    torch = pytest.importorskip("torch")
    from vla_edge.backends.pi05.backend import Pi05TensorRTBackend

    expected = np.arange(16 * 32, dtype=np.float32).reshape(1, 16, 32)
    wire_noise = json_numpy.loads(json_numpy.dumps(expected))
    assert not wire_noise.flags.writeable
    received = []

    def run(pixels, language, mask, noise):
        received.append(noise)
        return noise.clone()

    backend = Pi05TensorRTBackend.__new__(Pi05TensorRTBackend)
    backend.torch, backend.device = torch, torch.device("cpu")
    backend.lock, backend.closed = RLock(), False
    backend.stream, backend.action_horizon = None, 16
    backend.host = SimpleNamespace(
        prepare=lambda *args: (None, None, None),
        postprocess=lambda actions: actions[0, :, :14].numpy().copy(),
    )
    backend.runtime = SimpleNamespace(run=run)
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda *args: nullcontext())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        actual = backend.generate_actions(
            images=[], instruction="fold", state=np.zeros(14), num_steps=10,
            noise=wire_noise,
        )
    assert not any("not writable" in str(w.message) for w in captured)
    assert not np.shares_memory(received[0].numpy(), wire_noise)
    np.testing.assert_array_equal(actual, expected[0, :, :14])
    np.testing.assert_array_equal(wire_noise, expected)
