"""Client for the ``/act`` endpoint.

Kept dependency-light on purpose: the machine driving a robot should not need
torch installed to talk to an inference server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json_numpy
import numpy as np
import requests


def _gripper_state(gripper: Any) -> str | None:
    if not isinstance(gripper, dict):
        return None
    source = gripper.get("state_source")
    if source is None:
        return None
    if source not in ("commanded", "measured"):
        raise ValueError(f"inference server gripper.state_source is invalid: {source!r}")
    return str(source)


@dataclass(frozen=True)
class ServerContract:
    """Validated robot-facing identity advertised by an inference server."""

    policy: str
    model_family: str
    embodiment: str
    cameras: tuple[str, ...]
    state_dim: int
    action_dim: int
    action_horizon: int
    rtc: bool
    repo_id: str | None = None
    norm_tag: str | None = None
    #: ``commanded`` or ``measured``: what the checkpoint expects in the gripper
    #: state channel (``gripper.state_source`` in the health payload); None for
    #: servers that predate the field.
    gripper_state: str | None = None
    rtc_mode: str | None = None
    max_prefix_length: int | None = None

    @classmethod
    def from_health(cls, value: Any) -> ServerContract:
        if not isinstance(value, dict) or value.get("status") != "ok":
            raise ValueError(f"inference server is not healthy: {value!r}")
        for field in ("policy", "model_family", "embodiment"):
            if not isinstance(value.get(field), str) or not value[field]:
                raise ValueError(f"inference server omitted {field}")
        cameras = value.get("cameras")
        if (
            not isinstance(cameras, list)
            or not cameras
            or not all(isinstance(name, str) and name for name in cameras)
            or len(set(cameras)) != len(cameras)
        ):
            raise ValueError(f"inference server cameras are invalid: {cameras!r}")
        dimensions: dict[str, int] = {}
        for field in ("state_dim", "action_dim", "action_horizon"):
            raw = value.get(field)
            if isinstance(raw, bool):
                raise ValueError(f"inference server {field} is invalid: {raw!r}")  # noqa: TRY004 - preserve the wire-validation ValueError contract.
            try:
                dimensions[field] = int(raw)
            except (TypeError, ValueError):
                raise ValueError(f"inference server {field} is invalid: {raw!r}") from None
            if dimensions[field] <= 0:
                raise ValueError(f"inference server {field} must be positive")
        return cls(
            policy=value["policy"],
            model_family=value["model_family"],
            embodiment=value["embodiment"],
            cameras=tuple(cameras),
            state_dim=dimensions["state_dim"],
            action_dim=dimensions["action_dim"],
            action_horizon=dimensions["action_horizon"],
            rtc=bool(value.get("rtc", False)),
            repo_id=value.get("repo_id"),
            norm_tag=value.get("norm_tag"),
            gripper_state=_gripper_state(value.get("gripper")),
            rtc_mode=value.get("rtc_mode"),
            max_prefix_length=value.get("max_prefix_length"),
        )


class ActClient:
    """Blocking client for one inference server.

    Args:
        endpoint: ``host:port``, or a full URL. ``/act`` is appended if absent.
        timeout_s: per-request timeout. Set this deliberately. A control loop
            that blocks indefinitely on a wedged server is worse than one that
            errors, because the arm keeps executing a stale chunk either way
            and only one of the two tells you.
    """

    def __init__(self, endpoint: str, timeout_s: float = 10.0) -> None:
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"http://{endpoint}"
        if not endpoint.rstrip("/").endswith("/act"):
            endpoint = endpoint.rstrip("/") + "/act"
        self.url = endpoint
        self.timeout_s = timeout_s
        self._session = requests.Session()
        self.contract: ServerContract | None = None

    def health(self) -> dict[str, Any]:
        resp = self._session.get(self.url, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def bind(
        self,
        *,
        policy: str | None = None,
        model_family: str | None = None,
        embodiment: str | None = None,
        cameras: tuple[str, ...] | None = None,
        state_dim: int | None = None,
        action_dim: int | None = None,
    ) -> ServerContract:
        """Pin this client to one exact server/robot contract."""

        contract = ServerContract.from_health(self.health())
        expected = {
            "policy": policy,
            "model_family": model_family,
            "embodiment": embodiment,
            "cameras": cameras,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }
        for field, wanted in expected.items():
            if wanted is not None and getattr(contract, field) != wanted:
                raise ValueError(
                    f"inference server {field}={getattr(contract, field)!r}, "
                    f"expected {wanted!r}"
                )
        self.contract = contract
        return contract

    def act(
        self,
        cameras: dict[str, np.ndarray],
        instruction: str,
        state: np.ndarray,
        num_steps: int | None = None,
        prefix_actions: np.ndarray | None = None,
        inference_delay: int = 0,
        execution_horizon: int = 10,
        rtc_schedule: str | None = None,
        rtc_max_guidance: float | None = None,
        seed: int | None = None,
        prefix_length: int | None = None,
        noise: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Request one action chunk.

        Returns ``(actions, server_dt_ms)``. ``actions`` is
        ``(horizon, action_dim)``. Do not hardcode either dimension; the
        action width comes from the checkpoint's normalization statistics and
        the horizon is a property of the model.

        ``prefix_actions`` opts into Real-Time Chunking guidance: pass the
        rows of the previous chunk the robot has not executed yet, and the
        server biases the new chunk to stay consistent with them over
        ``[inference_delay, execution_horizon)``. Servers without RTC support
        refuse the request rather than silently ignoring the prefix; the
        health endpoint's ``rtc`` field says up front whether it is available.

        ABC advertises ``rtc_mode="hard-prefix"`` instead: ``prefix_length``
        locks that many initial rows (0..7), with no weighted guidance.
        Leave inference_delay at zero and omit guidance settings for ABC.
        ``noise`` supplies explicit Gaussian samples for numerical comparisons;
        it is mutually exclusive with ``seed``.
        """
        if prefix_length is not None and prefix_actions is None:
            raise ValueError("prefix_length requires prefix_actions")
        if seed is not None and noise is not None:
            raise ValueError("supply either seed or explicit noise, not both")
        contract = self.contract
        if contract is not None:
            got = set(cameras)
            expected = set(contract.cameras)
            if got != expected:
                raise ValueError(
                    f"camera fields differ: got={sorted(got)}, "
                    f"expected={list(contract.cameras)}"
                )
            state_value = np.asarray(state, dtype=np.float32).reshape(-1)
            if state_value.shape != (contract.state_dim,):
                raise ValueError(
                    f"state must be shape ({contract.state_dim},), "
                    f"got {state_value.shape}"
                )
            if prefix_actions is not None and not contract.rtc:
                raise ValueError(
                    f"policy {contract.policy!r} does not support RTC guidance"
                )
            if prefix_length is not None and contract.rtc_mode not in (None, "hard-prefix"):
                raise ValueError("this RTC mode does not accept hard prefix lengths")

        payload: dict[str, Any] = dict(cameras)
        payload["instruction"] = instruction
        payload["state"] = np.asarray(state, dtype=np.float32)
        if num_steps is not None:
            payload["num_steps"] = int(num_steps)
        if seed is not None:
            payload["seed"] = int(seed)
        if noise is not None:
            payload["noise"] = np.asarray(noise, dtype=np.float32)
        if prefix_actions is not None:
            payload["prefix_actions"] = np.asarray(
                prefix_actions, dtype=np.float32
            )
            payload["inference_delay"] = int(inference_delay)
            payload["execution_horizon"] = int(execution_horizon)
            if prefix_length is not None:
                payload["prefix_length"] = prefix_length
            if rtc_schedule is not None:
                payload["rtc_schedule"] = str(rtc_schedule)
            if rtc_max_guidance is not None:
                payload["rtc_max_guidance"] = float(rtc_max_guidance)

        resp = self._session.post(
            self.url, data=json_numpy.dumps(payload), timeout=self.timeout_s
        )
        if resp.status_code != 200:
            raise RuntimeError(f"server returned {resp.status_code}: {resp.text[:400]}")
        body = json_numpy.loads(resp.text)
        if "error" in body:
            raise RuntimeError(f"server error: {body['error']}")
        actions = np.asarray(body["actions"], dtype=np.float32)
        if contract is not None:
            expected_shape = (contract.action_horizon, contract.action_dim)
            if actions.shape != expected_shape or not np.isfinite(actions).all():
                raise RuntimeError(
                    f"server returned invalid actions: shape={actions.shape}, "
                    f"expected={expected_shape}, finite={np.isfinite(actions).all()}"
                )
        return actions, float(body["dt_ms"])

    def close(self) -> None:
        self._session.close()
