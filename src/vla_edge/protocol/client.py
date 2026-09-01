"""Client for the ``/act`` endpoint.

Kept dependency-light on purpose: the machine driving a robot should not need
torch installed to talk to an inference server.
"""

from __future__ import annotations

from typing import Any

import json_numpy
import numpy as np
import requests


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

    def health(self) -> dict[str, Any]:
        resp = self._session.get(self.url, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

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
        """
        payload: dict[str, Any] = dict(cameras)
        payload["instruction"] = instruction
        payload["state"] = np.asarray(state, dtype=np.float32)
        if num_steps is not None:
            payload["num_steps"] = int(num_steps)
        if prefix_actions is not None:
            payload["prefix_actions"] = np.asarray(
                prefix_actions, dtype=np.float32
            )
            payload["inference_delay"] = int(inference_delay)
            payload["execution_horizon"] = int(execution_horizon)
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
        return np.asarray(body["actions"], dtype=np.float32), float(body["dt_ms"])
