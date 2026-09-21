"""ABC robot-policy client adapter for vla-edge's HTTP transport.

Uses the upstream observation/prefix schema, preserving the ABC controller.
It performs no resizing, smoothing, control, or action-unit conversion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .client import ActClient


class ABCPolicyClient:
    def __init__(
        self,
        host="127.0.0.1",
        port=8202,
        *,
        timeout_s=30.0,
        policy="abcvla-bimanual-yam",
    ):
        self.client = ActClient(f"{host}:{port}", timeout_s=timeout_s)
        self.contract = self.client.bind(
            policy=policy,
            model_family="abcvla",
            embodiment="bimanual-yam",
            cameras=("top_cam", "left_cam", "right_cam"),
            state_dim=14,
            action_dim=14,
        )
        if self.contract.rtc_mode != "hard-prefix":
            raise ValueError(
                "ABC robot control requires native hard-prefix RTC semantics"
            )
        self.metadata = self.client.health()

    def get_server_metadata(self):
        return dict(self.metadata)

    def infer(self, obs: dict[str, Any]):
        cameras = {}
        for source, wire in (
            ("top", "top_cam"),
            ("left", "left_cam"),
            ("right", "right_cam"),
        ):
            value = np.asarray(obs["images"][source])
            if value.ndim != 3 or value.shape[0] != 3 or value.dtype != np.uint8:
                raise ValueError("ABC observations must contain CHW RGB uint8 images")
            cameras[wire] = np.moveaxis(value, 0, -1).copy()
        prefix_length = obs.get("prefix_length")
        if obs.get("action_prefix") is None and prefix_length == 0:
            prefix_length = None
        actions, _ = self.client.act(
            cameras=cameras,
            instruction=obs["prompt"],
            state=obs["state"],
            prefix_actions=obs.get("action_prefix"),
            prefix_length=prefix_length,
            noise=obs.get("noise"),
            seed=obs.get("seed"),
        )
        return {"actions": actions}

    def close(self):
        self.client.close()
