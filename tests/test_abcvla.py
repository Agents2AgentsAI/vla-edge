"""ABC inference contracts without loading a checkpoint or commanding hardware."""

from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from pathlib import Path

import json_numpy
import numpy as np
import pytest

from vla_edge.backends.abcvla.backend import _BaseBackend
from vla_edge.backends.abcvla.bundle import AbcVlaError, resolve, sha256, verify_file
from vla_edge.backends.abcvla.geometry import native_positions
from vla_edge.backends.abcvla.host import (
    validate_images,
    validate_prefix,
    validate_state,
)
from vla_edge.config import get_embodiment, get_policy
from vla_edge.pipeline import Pipeline
from vla_edge.serving.server import build_app


def test_abc_policy_has_native_horizon_state_and_action_conventions():
    p = get_policy("abcvla-bimanual-yam")
    assert (p.model_family, p.action_horizon, p.default_num_steps) == ("abcvla", 30, 10)
    assert p.action_space == "absolute" and p.gripper_state == "measured"
    assert p.gripper_convention == "closed_0_open_1"


@pytest.mark.parametrize("length", [0, 1, 4, 5, 7])
def test_hard_prefix_accepts_compact_or_full_chunk_without_mutation(length):
    compact = np.arange(length * 14, dtype=np.float32).reshape(length, 14)
    original = compact.copy()
    prefix = validate_prefix(compact, length)
    assert prefix.length == length and prefix.actions.shape == (30, 14)
    np.testing.assert_array_equal(prefix.actions[:length], original)
    np.testing.assert_array_equal(compact, original)
    full = validate_prefix(prefix.actions, length)
    np.testing.assert_array_equal(full.actions, prefix.actions)


@pytest.mark.parametrize("length", [-1, 8, 30, True, 3.5])
def test_untrained_or_ambiguous_prefix_lengths_are_rejected(length):
    with pytest.raises(ValueError, match="prefix_length"):
        validate_prefix(np.zeros((30, 14)), length)


def test_invalid_prefix_data_rejected_instead_of_clipping_or_reshaping():
    for data in (
        np.full((5, 14), np.nan),
        np.zeros((5, 13)),
        np.zeros((2, 5, 14)),
        np.zeros((6, 14)),
    ):
        with pytest.raises(ValueError):
            validate_prefix(data, 5)


def test_rgb_and_measured_state_are_preserved_without_gripper_clipping():
    state = np.arange(14, dtype=np.float32) / 10
    state[[6, 13]] = [-0.01, 1.01]
    np.testing.assert_array_equal(validate_state(state), state)
    images = [np.full((3, 4, 3), i, dtype=np.uint8) for i in range(3)]
    outputs = validate_images(images)
    for image, output in zip(images, outputs):
        np.testing.assert_array_equal(output, image.transpose(2, 0, 1))
        assert not np.shares_memory(output, image)
    with pytest.raises(ValueError):
        validate_images([np.zeros((3, 4, 3))] * 3)
    with pytest.raises(ValueError):
        validate_state(np.zeros((1, 14)))


def test_vision_spans_preserve_the_upstream_interleaved_layout():
    for length in (207, 213, 215, 256):
        image, other = native_positions(length)
        assert len(image) == 192 and sorted(image + other) == list(range(length))
        assert image[:64] == list(range(3, 67))
        assert image[64:128] == list(range(71, 135))
        assert image[128:] == list(range(139, 203))


def test_bundle_rejects_changed_bytes_and_outside_paths(tmp_path):
    path = tmp_path / "model"
    path.write_bytes(b"original")
    declaration = {"path": "model", "bytes": 8, "sha256": sha256(path)}
    assert verify_file(tmp_path, declaration, "model") == path
    path.write_bytes(b"changed!")
    with pytest.raises(AbcVlaError, match="mismatch"):
        verify_file(tmp_path, declaration, "model")
    with pytest.raises(AbcVlaError):
        resolve(tmp_path, str(Path(__file__).resolve()))


class FakeBackend(_BaseBackend):
    def __init__(self):
        self._closed = False
        self._prefix = ContextVar("test_abc_prefix", default=None)
        self.name = "abc-test"
        self.seen = []
        self.clears = 0

    def clear_rtc(self):
        super().clear_rtc()
        self.clears += 1

    def generate_actions(
        self,
        *,
        images,
        instruction,
        state,
        num_steps=10,
        enable_cuda_graph=False,
        seed=None,
        noise=None,
    ):
        prefix = self._prefix.get()
        self.clear_rtc()
        state = validate_state(state)
        self.seen.append((images, instruction, state, prefix, noise))
        result = np.zeros((30, 14), np.float32)
        if prefix is not None:
            result[: prefix.length] = prefix.actions[: prefix.length]
        return result

    def mount_routes(self, app, pipeline):
        # Match production's websocket transport without importing a model.
        from vla_edge.serving.abc_websocket import mount_routes

        mount_routes(app, pipeline)


def pipeline():
    policy = get_policy("abcvla-bimanual-yam")
    return Pipeline(get_embodiment("bimanual-yam"), policy, FakeBackend())


def request(payload):
    class Request:
        async def body(self):
            return json_numpy.dumps(payload).encode()

    return Request()


def endpoint(app, method):
    return next(
        x.endpoint
        for x in app.routes
        if x.path == "/act" and method in getattr(x, "methods", ())
    )


def observation():
    return {
        **{
            k: np.zeros((4, 6, 3), np.uint8)
            for k in ("top_cam", "left_cam", "right_cam")
        },
        "instruction": "fold and stack the t-shirts",
        "state": np.zeros(14, np.float32),
    }


def test_http_advertises_hard_prefix_and_uses_native_length_and_noise():
    p = pipeline()
    app = build_app(p, "cuda-graph")
    health = json.loads(asyncio.run(endpoint(app, "GET")()).body)
    assert health["model_family"] == "abcvla" and health["action_horizon"] == 30
    assert health["gripper"]["state_source"] == "measured"
    assert health["rtc_mode"] == "hard-prefix" and health["max_prefix_length"] == 7
    payload = {
        **observation(),
        "prefix_actions": np.ones((30, 14)),
        "prefix_length": 5,
        "noise": np.zeros((30, 14)),
    }
    response = asyncio.run(endpoint(app, "POST")(request(payload)))
    assert response.status_code == 200
    body = json_numpy.loads(response.body.decode())
    np.testing.assert_array_equal(body["actions"][:5], 1)
    assert body["rtc"]["mode"] == "hard-prefix" and body["rtc"]["prefix_length"] == 5
    assert p.backend.seen[-1][-1].shape == (30, 14)


def test_failed_http_request_clears_a_previously_armed_prefix():
    p = pipeline()
    app = build_app(p, "cuda-graph")
    post = endpoint(app, "POST")
    bad = {
        **observation(),
        "prefix_actions": np.ones((5, 14)),
        "prefix_length": 5,
        "num_steps": "bad",
    }
    response = asyncio.run(post(request(bad)))
    assert response.status_code == 400 and not p.backend.seen
    assert (
        p.backend.clears >= 2
    )  # arm clears first; finally clears after conversion fails
    response = asyncio.run(post(request(observation())))
    assert response.status_code == 200 and p.backend.seen[-1][3] is None


def test_abc_refuses_weighted_guidance_instead_of_silently_changing_rtc():
    p = pipeline()
    post = endpoint(build_app(p, "tensorrt"), "POST")
    for extra in (
        {"rtc_schedule": "linear"},
        {"rtc_max_guidance": 2},
        {"inference_delay": 3},
    ):
        response = asyncio.run(
            post(
                request({**observation(), "prefix_actions": np.zeros((5, 14)), **extra})
            )
        )
        assert response.status_code == 400 and not p.backend.seen


def test_websocket_observation_path_preserves_rgb_order_noise_and_prefix():
    from vla_edge.serving.abc_websocket import predict_observation

    p = pipeline()
    images = {
        name: np.full((3, 4, 6), value, np.uint8)
        for name, value in [("top", 10), ("left", 20), ("right", 30)]
    }
    obs = {
        "images": images,
        "state": np.zeros(14),
        "prompt": "fold",
        "noise": np.zeros((30, 14)),
        "action_prefix": np.full((30, 14), 0.4),
        "prefix_length": 5,
    }
    result = predict_observation(p, obs)
    np.testing.assert_array_equal(result[:5], np.float32(0.4))
    seen = p.backend.seen[-1]
    assert [int(np.asarray(image)[0, 0, 0]) for image in seen[0]] == [10, 20, 30]
    assert obs["action_prefix"].shape == (30, 14)  # wire object remains intact


def test_msgpack_wire_roundtrip_and_object_dtype_rejection():
    pytest.importorskip("msgpack")
    from vla_edge.protocol.abc_msgpack import packb, unpackb

    original = {
        "state": np.arange(14, dtype=np.float32),
        "images": {"top": np.zeros((3, 2, 2), np.uint8)},
        "prefix_length": np.int64(5),
    }
    actual = unpackb(packb(original))
    np.testing.assert_array_equal(actual["state"], original["state"])
    assert actual["prefix_length"] == 5
    with pytest.raises(ValueError):
        packb(np.array([object()], dtype=object))


def test_fp16_champion_preserves_committed_physical_prefix_and_sampled_tail():
    torch = pytest.importorskip("torch")
    from vla_edge.backends.abcvla.host import AbcHost

    host = AbcHost.__new__(AbcHost)
    mean = np.linspace(-0.2, 0.3, 14, dtype=np.float32)
    scale = np.linspace(0.2, 0.9, 14, dtype=np.float32)
    host.stats = {"actions": (mean, scale)}
    original = np.linspace(-1, 1, 30 * 14, dtype=np.float32).reshape(30, 14)
    rounded = torch.from_numpy(((original - mean) / scale)[None]).half().float()
    raw_copy = rounded.clone()
    native_finish = host.finish(rounded)
    assert not np.array_equal(native_finish[:5], original[:5])
    fixed = host.finish(rounded, restore_prefix=validate_prefix(original, 5))
    np.testing.assert_array_equal(fixed[:5], original[:5])
    np.testing.assert_array_equal(fixed[5:], native_finish[5:])
    assert torch.equal(rounded, raw_copy)


def test_websocket_wire_returns_stock_abc_dictionary_and_clears_failed_prefix():
    pytest.importorskip("httpx")
    pytest.importorskip("msgpack")
    from fastapi.testclient import TestClient

    from vla_edge.protocol.abc_msgpack import packb, unpackb

    p = pipeline()
    obs = {
        "images": {k: np.zeros((3, 4, 6), np.uint8) for k in ("top", "left", "right")},
        "state": np.zeros(14),
        "prompt": "fold",
    }
    with TestClient(build_app(p, "tensorrt")) as client:
        with client.websocket_connect("/") as ws:
            metadata = unpackb(ws.receive_bytes())
            assert metadata["model_family"] == "abc-vla"
            ws.send_bytes(
                packb({**obs, "action_prefix": np.ones((5, 14)), "prefix_length": 5})
            )
            result = unpackb(ws.receive_bytes())
            assert set(result) == {"actions"} and result["actions"].shape == (30, 14)
            np.testing.assert_array_equal(result["actions"][:5], 1)
            ws.send_bytes(
                packb({**obs, "state": np.zeros(13), "action_prefix": np.ones((5, 14))})
            )
            assert "ABC inference failed" in ws.receive_text()
        with client.websocket_connect("/") as ws:
            ws.receive_bytes()
            ws.send_bytes(packb(obs))
            np.testing.assert_array_equal(unpackb(ws.receive_bytes())["actions"], 0)
