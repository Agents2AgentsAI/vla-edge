"""Model-family-neutral `/act` client contract tests."""

from __future__ import annotations

import json_numpy
import numpy as np
import pytest

from vla_edge.protocol.client import ActClient, ServerContract


def _health(policy="example-bimanual-yam", family="example", horizon=16, rtc=False):
    return {
        "status": "ok",
        "policy": policy,
        "model_family": family,
        "embodiment": "bimanual-yam",
        "repo_id": "example/checkpoint",
        "norm_tag": "yam",
        "cameras": ["top_cam", "left_cam", "right_cam"],
        "state_dim": 14,
        "action_dim": 14,
        "action_horizon": horizon,
        "rtc": rtc,
    }


class _Response:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status
        self.text = json_numpy.dumps(body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._body


class _Session:
    def __init__(self, health, response=None):
        self.health_value = health
        self.response = response
        self.posts = []

    def get(self, _url, timeout):
        assert timeout > 0
        return _Response(self.health_value)

    def post(self, url, data, timeout):
        self.posts.append((url, data, timeout))
        return _Response(self.response)

    def close(self):
        pass


@pytest.mark.parametrize(
    ("policy", "family", "horizon", "rtc"),
    [
        ("example-bimanual-yam", "example", 16, False),
        ("molmoact2-bimanual-yam", "molmoact2", 30, True),
    ],
)
def test_contract_represents_policy_independently_of_robot(policy, family, horizon, rtc):
    value = ServerContract.from_health(_health(policy, family, horizon, rtc))

    assert value.embodiment == "bimanual-yam"
    assert value.policy == policy
    assert value.model_family == family
    assert value.action_horizon == horizon
    assert value.rtc is rtc


def test_bind_rejects_wrong_exact_policy():
    client = ActClient("localhost:8204")
    client._session = _Session(_health())

    with pytest.raises(ValueError, match="policy=.*expected"):
        client.bind(policy="molmoact2-bimanual-yam")


def test_bound_client_validates_and_returns_exact_actions():
    actions = np.zeros((16, 14), dtype=np.float32)
    client = ActClient("localhost:8204")
    client._session = _Session(_health(), {"actions": actions, "dt_ms": 100.0})
    client.bind(
        policy="example-bimanual-yam",
        embodiment="bimanual-yam",
        cameras=("top_cam", "left_cam", "right_cam"),
        state_dim=14,
        action_dim=14,
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    received, dt_ms = client.act(
        cameras={name: image for name in client.contract.cameras},
        instruction="test",
        state=np.zeros(14, dtype=np.float32),
    )

    assert received.shape == (16, 14)
    assert dt_ms == 100.0


def test_bound_client_rejects_bad_output_shape():
    client = ActClient("localhost:8204")
    client._session = _Session(
        _health(), {"actions": np.zeros((30, 14)), "dt_ms": 100.0}
    )
    client.bind()
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="invalid actions"):
        client.act(
            cameras={name: image for name in client.contract.cameras},
            instruction="test",
            state=np.zeros(14, dtype=np.float32),
        )


def test_bound_client_refuses_rtc_before_post_when_policy_lacks_it():
    client = ActClient("localhost:8204")
    session = _Session(_health(), {"actions": np.zeros((16, 14)), "dt_ms": 1.0})
    client._session = session
    client.bind()
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="does not support RTC"):
        client.act(
            cameras={name: image for name in client.contract.cameras},
            instruction="test",
            state=np.zeros(14, dtype=np.float32),
            prefix_actions=np.zeros((2, 14), dtype=np.float32),
        )
    assert session.posts == []


def test_act_sends_the_sampler_seed_only_when_given():
    actions = np.zeros((16, 14), dtype=np.float32)
    client = ActClient("localhost:8204")
    client._session = _Session(_health(), {"actions": actions, "dt_ms": 1.0})
    client.bind(policy="example-bimanual-yam")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    cameras = {name: image for name in client.contract.cameras}
    state = np.zeros(14, dtype=np.float32)

    client.act(cameras=cameras, instruction="fold", state=state, seed=7)
    client.act(cameras=cameras, instruction="fold", state=state)

    def _payload(post):
        # The fake session records (url, data, timeout).
        body = post[1]
        return json_numpy.loads(body.decode() if isinstance(body, bytes) else body)

    seeded = _payload(client._session.posts[0])
    unseeded = _payload(client._session.posts[1])
    assert seeded["seed"] == 7
    assert "seed" not in unseeded


def test_contract_carries_the_gripper_state_source_when_advertised():
    value = _health()
    assert ServerContract.from_health(value).gripper_state is None
    value["gripper"] = {"indices": [6, 13], "checkpoint_convention": "closed_0_open_1", "wire_convention": "closed_0_open_1", "state_source": "measured"}
    assert ServerContract.from_health(value).gripper_state == "measured"
    value["gripper"]["state_source"] = "encoder"
    with pytest.raises(ValueError, match="gripper.state_source is invalid"):
        ServerContract.from_health(value)


def test_abc_client_transmits_hard_prefix_and_explicit_noise_without_guidance():
    health = _health("abcvla-bimanual-yam", "abcvla", 30, True)
    health.update(rtc_mode="hard-prefix", max_prefix_length=7)
    client = ActClient("localhost:8210")
    client._session = _Session(health, {"actions": np.zeros((30, 14)), "dt_ms": 35.0})
    contract = client.bind()
    assert contract.rtc_mode == "hard-prefix" and contract.max_prefix_length == 7
    cameras = {k: np.zeros((4, 6, 3), np.uint8) for k in contract.cameras}
    noise = np.arange(420, dtype=np.float32).reshape(30, 14)
    client.act(cameras, "fold", np.zeros(14), prefix_actions=np.ones((30, 14)), prefix_length=5, noise=noise)
    payload = json_numpy.loads(client._session.posts[0][1])
    assert payload["prefix_length"] == 5 and payload["inference_delay"] == 0
    np.testing.assert_array_equal(payload["noise"], noise)
    assert "rtc_schedule" not in payload and "rtc_max_guidance" not in payload
    with pytest.raises(ValueError, match="either seed"):
        client.act(cameras, "fold", np.zeros(14), seed=2, noise=noise)
    assert len(client._session.posts) == 1
