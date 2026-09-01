"""ZMQ-based camera server.

Hosts the RealSense cameras in a long-lived process so eval clients can pull
the latest frames on demand without paying for pipeline startup, fighting the
policy loop for camera I/O, or coupling robot-control timing to camera I/O.

Sockets
-------
REP  ``tcp://127.0.0.1:5555``  (default)
    Pull semantics. Client sends a pickled request dict; server replies with a
    pickled response dict. Used by the policy for on-demand obs.

PUB  ``tcp://127.0.0.1:5556``  (default, optional)
    Push semantics. Server publishes the latest obs every ``pub_period_sec``.
    Intended for the cv2 live viewer so it can render at camera rate without
    burning policy-side requests.

Request protocol
----------------
    {"cmd": "obs"}   ->  {"ok": True, "frames": {cam_name: np.ndarray (H,W,3) uint8 RGB},
                          "timestamps": {cam_name: float}}
    {"cmd": "ping"}  ->  {"ok": True, "pong": True}

Errors come back as ``{"ok": False, "error": str}``. The server keeps running
across any single bad request.

CLI
---
    python camera_server.py --config configs/yam_left.yaml
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import pickle
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import zmq
from gello_min.realsense_camera import RealSenseCamera, get_device_ids
from omegaconf import OmegaConf

logger = logging.getLogger("camera_server")


DEFAULT_REP_ENDPOINT = "tcp://127.0.0.1:5555"
DEFAULT_PUB_ENDPOINT = "tcp://127.0.0.1:5556"
DEFAULT_PUB_PERIOD_SEC = 1.0 / 30.0
DEFAULT_HEARTBEAT_SEC = 10.0
DEFAULT_LOCK_PATH = Path(
    os.environ.get("YAM_CAMERA_LOCK", "/tmp/vla-edge-yam-camera-server.lock")
)


class CameraServer:
    """Owns RealSense cameras and serves their latest frames over ZMQ."""

    def __init__(
        self,
        cameras: dict[str, RealSenseCamera],
        rep_endpoint: str = DEFAULT_REP_ENDPOINT,
        pub_endpoint: str | None = None,
        pub_period_sec: float = DEFAULT_PUB_PERIOD_SEC,
        heartbeat_sec: float = DEFAULT_HEARTBEAT_SEC,
    ) -> None:
        self.cameras = cameras
        self.rep_endpoint = rep_endpoint
        self.pub_endpoint = pub_endpoint
        self.pub_period_sec = float(pub_period_sec)
        self.heartbeat_sec = float(heartbeat_sec)

        self._ctx = zmq.Context.instance()
        self._rep: zmq.Socket | None = None
        self._pub: zmq.Socket | None = None

        self._stop_event = threading.Event()
        self._pub_thread: threading.Thread | None = None
        self._shutdown_done = False

        self._req_total = 0
        self._req_window = 0
        self._last_heartbeat = time.time()

    # ------------------------------------------------------------------
    # Frame sourcing
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict[str, Any]:
        """Snapshot the latest color frame from every camera (RGB uint8)."""
        frames: dict[str, Any] = {}
        timestamps: dict[str, float] = {}
        for name, cam in self.cameras.items():
            image, _depth = cam.read()
            frames[name] = image
            # Surface the capture timestamp so the client can detect staleness.
            ts = getattr(cam, "_latest_frame_timestamp", None) or 0.0
            timestamps[name] = float(ts)
        return {"ok": True, "frames": frames, "timestamps": timestamps}

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def _handle_request(self) -> None:
        assert self._rep is not None
        raw = self._rep.recv()
        try:
            req = pickle.loads(raw)
            cmd = (req or {}).get("cmd", "obs")
        except Exception as exc:  # noqa: BLE001  # surface to client, stay alive
            self._rep.send(
                pickle.dumps({"ok": False, "error": f"bad request: {exc!r}"})
            )
            return

        try:
            if cmd == "obs":
                resp = self._snapshot()
            elif cmd == "ping":
                resp = {"ok": True, "pong": True}
            else:
                resp = {"ok": False, "error": f"unknown cmd: {cmd!r}"}
        except Exception as exc:
            logger.exception("Request failed (cmd=%r)", cmd)
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        self._rep.send(pickle.dumps(resp), copy=False)
        self._req_total += 1
        self._req_window += 1

    def _pub_loop(self) -> None:
        assert self._pub is not None
        next_tick = time.time()
        while not self._stop_event.is_set():
            now = time.time()
            if now < next_tick:
                # Tiny sleep granularity so shutdown is snappy.
                time.sleep(min(0.01, next_tick - now))
                continue
            next_tick = now + self.pub_period_sec
            try:
                resp = self._snapshot()
                self._pub.send(pickle.dumps(resp), copy=False)
            except Exception as exc:  # noqa: BLE001  # publish is best-effort
                logger.warning("PUB tick failed: %s", exc)

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        elapsed = now - self._last_heartbeat
        if elapsed < self.heartbeat_sec:
            return
        hz = self._req_window / elapsed if elapsed > 0 else 0.0
        logger.info(
            "alive: total_requests=%d window=%d (%.1f req/s) cameras=%d",
            self._req_total,
            self._req_window,
            hz,
            len(self.cameras),
        )
        self._req_window = 0
        self._last_heartbeat = now

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self._rep = self._ctx.socket(zmq.REP)
            self._rep.bind(self.rep_endpoint)
            logger.info("REP bound on %s", self.rep_endpoint)

            if self.pub_endpoint:
                self._pub = self._ctx.socket(zmq.PUB)
                self._pub.bind(self.pub_endpoint)
                logger.info(
                    "PUB bound on %s (period=%.3fs)",
                    self.pub_endpoint,
                    self.pub_period_sec,
                )
                self._pub_thread = threading.Thread(
                    target=self._pub_loop,
                    name="camera_server_pub",
                    daemon=True,
                )
                self._pub_thread.start()

            poller = zmq.Poller()
            poller.register(self._rep, zmq.POLLIN)
            while not self._stop_event.is_set():
                # 100 ms tick keeps heartbeats responsive and shutdown snappy.
                socks = dict(poller.poll(timeout=100))
                if self._rep in socks:
                    self._handle_request()
                self._maybe_heartbeat()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._stop_event.set()
        if self._pub_thread is not None:
            self._pub_thread.join(timeout=2.0)
        for sock in (self._rep, self._pub):
            if sock is not None:
                try:
                    sock.close(linger=0)
                except zmq.ZMQError:
                    logger.debug("Failed to close camera socket", exc_info=True)
        for name, cam in reversed(tuple(self.cameras.items())):
            try:
                cam.close()
            except Exception:
                logger.exception("Failed to close camera %s", name)
        logger.info("Camera server stopped.")


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def _build_cameras_from_config(cfg_path: Path) -> dict[str, RealSenseCamera]:
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    camera_cfg = cfg["sensors"]["cameras"]
    logger.info("Discovering RealSense devices...")
    ids = get_device_ids()
    logger.info("Found %d RealSense devices: %s", len(ids), ids)
    configured_ids = {name: str(spec["device_id"]) for name, spec in camera_cfg.items()}
    placeholders = {
        name: device_id
        for name, device_id in configured_ids.items()
        if "REPLACE" + "_WITH" in device_id
    }
    if placeholders:
        raise SystemExit(
            f"Camera serials still contain placeholders: {placeholders}; "
            "run `python configure_rig.py`"
        )
    if len(set(configured_ids.values())) != len(configured_ids):
        raise RuntimeError(f"Camera serials must be unique: {configured_ids}")
    missing = {
        name: device_id
        for name, device_id in configured_ids.items()
        if device_id not in ids
    }
    if missing:
        raise RuntimeError(
            f"Configured RealSense cameras were not detected: {missing}; detected={ids}"
        )

    cameras: dict[str, RealSenseCamera] = {}
    try:
        for name, device_id in configured_ids.items():
            logger.info("Opening camera %s (device_id=%s)", name, device_id)
            try:
                cameras[name] = RealSenseCamera(device_id)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Could not open camera {name!r} ({device_id}): {exc}. "
                    "Another process may own it; check `fuser -v /dev/video*`."
                ) from exc
        return cameras
    except BaseException:
        for camera in reversed(tuple(cameras.values())):
            camera.close()
        raise


def _probe_existing_server(
    endpoint: str,
    timeout_ms: int = 1500,
) -> tuple[bool, str | None]:
    """Return whether an existing server answered and any reported error."""
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.connect(endpoint)
    try:
        sock.send(pickle.dumps({"cmd": "obs"}))
        response = pickle.loads(sock.recv())
    except (zmq.Again, pickle.PickleError, EOFError):
        return False, None
    finally:
        sock.close(linger=0)
    if response.get("ok"):
        return True, None
    return True, str(response.get("error") or "unknown camera server error")


def _acquire_server_lock(path: Path = DEFAULT_LOCK_PATH) -> Any | None:
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        handle.close()
        logger.error("Another camera server is starting or running (pid %s).", owner)
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="YAM camera server (ZMQ).")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a yam_*.yaml whose sensors.cameras block lists the devices.",
    )
    parser.add_argument("--rep-endpoint", default=DEFAULT_REP_ENDPOINT)
    parser.add_argument(
        "--pub-endpoint",
        default=DEFAULT_PUB_ENDPOINT,
        help="ZMQ PUB endpoint. Pass empty string to disable the PUB stream.",
    )
    parser.add_argument("--pub-period-sec", type=float, default=DEFAULT_PUB_PERIOD_SEC)
    parser.add_argument("--heartbeat-sec", type=float, default=DEFAULT_HEARTBEAT_SEC)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    answered, existing_error = _probe_existing_server(args.rep_endpoint)
    if answered:
        if existing_error is None:
            logger.info(
                "A healthy camera server is already running at %s; leaving it in control.",
                args.rep_endpoint,
            )
            return 0
        logger.error(
            "A camera server is already running at %s but is unhealthy: %s. "
            "Stop that process before restarting it; `ss -ltnp` shows the owner.",
            args.rep_endpoint,
            existing_error,
        )
        return 1

    server_lock = _acquire_server_lock()
    if server_lock is None:
        return 1

    cameras = _build_cameras_from_config(args.config)
    server = CameraServer(
        cameras=cameras,
        rep_endpoint=args.rep_endpoint,
        pub_endpoint=(args.pub_endpoint or None),
        pub_period_sec=args.pub_period_sec,
        heartbeat_sec=args.heartbeat_sec,
    )

    def _handle(signum, _frame):
        logger.info("Signal %d received; shutting down.", signum)
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
