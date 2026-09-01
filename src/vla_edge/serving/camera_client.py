"""ZMQ client for the camera server.

Used by eval launchers to pull the latest 3-camera observation without
holding RealSense devices in-process. See ``camera_server.py`` for the wire
protocol.
"""
from __future__ import annotations

import logging
import pickle
import time
from typing import Any

import numpy as np
import zmq

logger = logging.getLogger(__name__)


class CameraClientError(RuntimeError):
    """Raised when the camera server is unreachable, slow, or returns an error."""


class CameraClient:
    """REQ-side wrapper. ``get_obs()`` returns ``{cam_name: np.ndarray (H,W,3) uint8 RGB}``.

    On timeout the underlying REQ socket is closed and recreated. REQ sockets
    become unusable after a recv timeout without a matching reply.
    """

    def __init__(
        self,
        endpoint: str,
        request_timeout_ms: int = 500,
        max_frame_age_sec: float | None = 0.5,
    ) -> None:
        self.endpoint = endpoint
        self.request_timeout_ms = int(request_timeout_ms)
        self.max_frame_age_sec = max_frame_age_sec
        self._ctx = zmq.Context.instance()
        self._sock: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001,S110  # socket is being discarded
                pass
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self.request_timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.request_timeout_ms)
        sock.connect(self.endpoint)
        self._sock = sock

    def _request(self, cmd: str) -> dict[str, Any]:
        assert self._sock is not None
        try:
            self._sock.send(pickle.dumps({"cmd": cmd}))
            raw = self._sock.recv()
        except zmq.Again as exc:
            # REQ socket is now in a bad state; reset before raising.
            self._connect()
            raise CameraClientError(
                f"Camera server timeout ({self.request_timeout_ms} ms) on cmd={cmd!r} "
                f"at {self.endpoint}. Is the server running?"
            ) from exc
        try:
            resp = pickle.loads(raw)
        except Exception as exc:
            self._connect()
            raise CameraClientError(f"Unparseable reply from camera server: {exc!r}") from exc
        if not resp.get("ok"):
            raise CameraClientError(f"Server error: {resp.get('error')}")
        return resp

    def ping(self) -> bool:
        return bool(self._request("ping").get("pong"))

    def get_obs(self) -> dict[str, np.ndarray]:
        """Return ``{cam_name: np.ndarray (H,W,3) uint8 RGB}`` with the latest frames."""
        resp = self._request("obs")
        frames: dict[str, np.ndarray] = resp["frames"]
        if self.max_frame_age_sec is not None:
            now = time.time()
            for name, ts in (resp.get("timestamps") or {}).items():
                if ts and (now - ts) > self.max_frame_age_sec:
                    raise CameraClientError(
                        f"Stale frame from {name}: {now - ts:.3f}s old "
                        f"(>{self.max_frame_age_sec:.3f}s)."
                    )
        return frames

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001,S110  # socket is being discarded
                pass
            self._sock = None


class CameraSubscriber:
    """Optional PUB/SUB consumer for the live viewer.

    The eval inner loop should use ``CameraClient`` (REQ/REP). This subscriber
    exists so a cv2 viewer can render at camera rate without competing for the
    REP socket with the policy.
    """

    def __init__(self, endpoint: str, recv_timeout_ms: int = 100) -> None:
        self.endpoint = endpoint
        self.recv_timeout_ms = int(recv_timeout_ms)
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.connect(endpoint)

    def try_recv(self) -> dict[str, np.ndarray] | None:
        """Return the most recent frame dict if one is available, else None."""
        latest: bytes | None = None
        # Drain the queue so we hand the consumer the freshest frame.
        while True:
            try:
                latest = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is None:
            return None
        try:
            resp = pickle.loads(latest)
        except Exception as exc:  # noqa: BLE001  # drop malformed publish
            logger.warning("Dropped malformed PUB payload: %s", exc)
            return None
        if not resp.get("ok"):
            return None
        return resp.get("frames")

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:  # noqa: BLE001,S110  # socket is being discarded
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Live viewer for the YAM camera server. "
                    "Defaults to the PUB stream so it doesn't fight the policy for REP."
    )
    parser.add_argument(
        "--mode", choices=("sub", "req"), default="sub",
        help="sub: subscribe to PUB stream (default). req: poll via REQ/REP.",
    )
    parser.add_argument("--rep-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--pub-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--req-hz", type=float, default=30.0,
                        help="Polling rate when --mode=req.")
    parser.add_argument("--window", default="camera_client",
                        help="cv2 window title.")
    args = parser.parse_args()

    import cv2  # imported lazily so library users don't pay for it

    def _fetch_sub():
        return sub.try_recv()

    def _fetch_req():
        try:
            return req.get_obs()
        except CameraClientError as exc:
            logger.warning("REQ fetch failed: %s", exc)
            return None

    if args.mode == "sub":
        sub = CameraSubscriber(args.pub_endpoint)
        fetch = _fetch_sub
        period = 0.0  # PUB drives the rate; just spin with a short waitKey
    else:
        req = CameraClient(args.rep_endpoint, request_timeout_ms=1000, max_frame_age_sec=None)
        fetch = _fetch_req
        period = 1.0 / max(args.req_hz, 1e-3)

    last_fps_t = time.time()
    fps_frames = 0
    fps_disp = 0.0
    last_loop_t = 0.0

    print(f"[camera_client] mode={args.mode}; press 'q' in the window to quit.", flush=True)
    try:
        while True:
            now = time.time()
            if period and (now - last_loop_t) < period:
                time.sleep(max(0.0, period - (now - last_loop_t)))
            last_loop_t = time.time()

            frames = fetch()
            if frames:
                panes = []
                for name, img in frames.items():
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.putText(bgr, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    panes.append(bgr)
                # Match heights so hstack works even if cameras report different sizes.
                h = min(p.shape[0] for p in panes)
                panes = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panes]
                grid = np.hstack(panes)
                cv2.putText(grid, f"{fps_disp:5.1f} fps", (8, grid.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(args.window, grid)

                fps_frames += 1
                if (last_loop_t - last_fps_t) >= 1.0:
                    fps_disp = fps_frames / (last_loop_t - last_fps_t)
                    fps_frames = 0
                    last_fps_t = last_loop_t

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if args.mode == "sub":
            sub.close()
        else:
            req.close()
