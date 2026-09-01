"""Serve the YAM camera stream in a browser without opening the cameras."""

from __future__ import annotations

import argparse
import logging
import pickle
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

CAMERA_ORDER = ("front_camera", "left_camera", "right_camera")
logger = logging.getLogger("yam.mjpeg_viewer")


def _ordered_camera_names(frames: dict[str, Any]) -> list[str]:
    ordered = [name for name in CAMERA_ORDER if name in frames]
    ordered.extend(sorted(name for name in frames if name not in CAMERA_ORDER))
    return ordered


def _encode_grid(frames: dict[str, Any], cv2: Any, np: Any) -> bytes | None:
    panes = []
    for name in _ordered_camera_names(frames):
        image = frames[name]
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            continue
        image = np.ascontiguousarray(image).copy()
        cv2.putText(
            image,
            name,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        panes.append(image)
    if not panes:
        return None

    height = min(pane.shape[0] for pane in panes)
    panes = [
        cv2.resize(
            pane,
            (int(pane.shape[1] * height / pane.shape[0]), height),
        )
        for pane in panes
    ]
    grid = np.hstack(panes)
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )
    return encoded.tobytes() if ok else None


def _subscriber_loop(
    latest: dict[str, bytes | None],
    endpoint: str,
    stop_event: threading.Event,
    cv2: Any,
    np: Any,
    zmq: Any,
) -> None:
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVHWM, 2)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.connect(endpoint)
    try:
        while not stop_event.is_set():
            try:
                response = pickle.loads(sock.recv())
            except zmq.Again:
                continue
            frames = response.get("frames") if isinstance(response, dict) else None
            if not isinstance(frames, dict):
                continue
            encoded = _encode_grid(frames, cv2, np)
            if encoded is not None:
                latest["jpg"] = encoded
    except Exception:
        logger.exception("camera browser subscriber stopped")
    finally:
        sock.close(linger=0)


def _handler_for(latest: dict[str, bytes | None]):
    class ViewerHandler(BaseHTTPRequestHandler):
        def log_message(self, _format, *args):
            logger.debug("browser client: %s", args)

        def do_GET(self):
            if self.path == "/":
                body = (
                    b"<html><head><title>YAM cameras</title></head>"
                    b"<body style='margin:0;background:#111'>"
                    b"<img src='/stream' style='width:100%' alt='YAM cameras'>"
                    b"</body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/snapshot.jpg":
                image = latest["jpg"]
                if image is None:
                    self.send_error(503, "waiting for camera frames")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(image)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(image)
                return

            if self.path != "/stream":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    image = latest["jpg"]
                    if image is not None:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(image)}\r\n\r\n".encode()
                            + image
                            + b"\r\n"
                        )
                    time.sleep(1.0 / 15.0)
            except (BrokenPipeError, ConnectionResetError):
                return

    return ViewerHandler


def _local_url(host: str, port: int) -> str:
    if host not in {"0.0.0.0", "::"}:
        display_host = f"[{host}]" if ":" in host else host
        return f"http://{display_host}:{port}/"
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        display_host = str(probe.getsockname()[0])
    except OSError:
        display_host = "<THOR-IP>"
    finally:
        probe.close()
    return f"http://{display_host}:{port}/"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--pub-endpoint", default="tcp://127.0.0.1:5556")
    args = parser.parse_args(argv)

    try:
        import cv2
        import numpy as np
        import zmq
    except ImportError as exc:
        parser.error(f"camera viewer dependency missing: {exc}; install .[camera]")

    latest: dict[str, bytes | None] = {"jpg": None}
    stop_event = threading.Event()
    try:
        server = ThreadingHTTPServer(
            (args.host, args.port),
            _handler_for(latest),
        )
    except OSError as exc:
        logger.error("cannot start camera browser on %s:%d: %s", args.host, args.port, exc)
        return 1
    server.daemon_threads = True

    subscriber = threading.Thread(
        target=_subscriber_loop,
        args=(latest, args.pub_endpoint, stop_event, cv2, np, zmq),
        name="camera-browser-subscriber",
        daemon=True,
    )
    subscriber.start()
    print(f"[camera_client] Open {_local_url(args.host, args.port)} in a browser.")
    print("[camera_client] Press Ctrl-C here when camera order is confirmed.")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.server_close()
        subscriber.join(timeout=1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
