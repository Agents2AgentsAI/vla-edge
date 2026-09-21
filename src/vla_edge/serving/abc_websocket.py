"""Wire-compatible endpoint for unmodified ABC robot/RTC clients.

The stock clients connect to ws://host:port/, read a metadata frame, then
exchange CHW-RGB observations and absolute action arrays via NumPy msgpack.
Inference still flows through the same vla-edge Pipeline used by HTTP.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from ..protocol import abc_msgpack

log = logging.getLogger(__name__)


def cameras_from_abc(images):
    if not isinstance(images, dict) or set(images) != {"top", "left", "right"}:
        raise ValueError("ABC requires exactly top, left and right camera fields")
    cameras = {}
    for key in ("top", "left", "right"):
        image = images[key]
        if isinstance(image, (bytes, bytearray)):
            import cv2

            decoded = cv2.imdecode(
                np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if decoded is None:
                raise ValueError(f"invalid JPEG for {key}")
            cameras[key + "_cam"] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        else:
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[0] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{key} must be CHW RGB uint8 or JPEG bytes")
            cameras[key + "_cam"] = np.moveaxis(image, 0, -1).copy()
    return cameras


def predict_observation(pipeline, observation):
    backend = pipeline.backend
    try:
        if not isinstance(observation, dict):
            raise ValueError("ABC observation must be an object")  # noqa: TRY004 - preserve the wire-validation ValueError contract.
        cameras = cameras_from_abc(observation["images"])
        prefix = observation.get("action_prefix")
        length = observation.get("prefix_length")
        if prefix is not None:
            backend.arm_rtc(prefix, prefix_length=length)
        elif length not in (None, 0):
            raise ValueError("prefix_length requires action_prefix")
        kwargs = {
            "cameras": cameras,
            "instruction": observation["prompt"],
            "state": observation["state"],
            "seed": observation.get("seed"),
        }
        if observation.get("noise") is not None:
            kwargs["noise"] = observation["noise"]
        return pipeline.predict(**kwargs)
    finally:
        backend.clear_rtc()


def mount_routes(app, pipeline):
    @app.websocket("/")
    async def abc_policy(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_bytes(
            abc_msgpack.packb(
                {
                    "model_family": "abc-vla",
                    "policy": pipeline.policy.name,
                    "backend": pipeline.backend.name,
                    "rtc_mode": "hard-prefix",
                    "action_horizon": 30,
                    "max_prefix_length": 7,
                    "runtime": pipeline.backend.serving_metadata(),
                }
            )
        )
        try:
            while True:
                request = abc_msgpack.unpackb(await websocket.receive_bytes())
                response = predict_observation(pipeline, request)
                await websocket.send_bytes(abc_msgpack.packb({"actions": response}))
        except WebSocketDisconnect:
            return
        except Exception as exc:
            log.exception("ABC websocket inference failed")
            try:
                await websocket.send_text(f"ABC inference failed: {exc}")
                await websocket.close(code=1011)
            except (WebSocketDisconnect, RuntimeError):
                pass
