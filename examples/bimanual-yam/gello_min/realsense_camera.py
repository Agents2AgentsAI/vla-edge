import logging
import os
import threading
import time

import numpy as np

from gello_min.camera import CameraDriver

logger = logging.getLogger(__name__)


def get_device_ids() -> list[str]:
    """Return connected RealSense serials without changing device state."""
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    return [dev.get_info(rs.camera_info.serial_number) for dev in devices]


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(self, device_id: str | None = None, flip: bool = False):
        import pyrealsense2 as rs

        self._device_id = device_id
        self._flip = flip
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._warmup_frames = 15
        self._read_timeout_ms = 1200
        self._read_wait_timeout_sec = 1.5
        self._max_frame_age_sec = 0.30
        self._max_read_attempts = 5
        self._latest_color_image = None
        self._latest_depth_image = None
        self._latest_frame_timestamp = None
        self._last_capture_error = None
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._closed = False

        self._rs = rs
        self._pipeline = None
        self._config = None
        self._align = rs.align(rs.stream.color)

        try:
            self._start_pipeline()
            self._start_capture_thread()
        except BaseException:
            self.close()
            raise

    def _start_capture_thread(self):
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"realsense_capture_{self._device_id or 'default'}",
            daemon=True,
        )
        self._capture_thread.start()

    def _capture_loop(self):
        consecutive_failures = 0
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    frames = self._pipeline.wait_for_frames(
                        timeout_ms=self._read_timeout_ms
                    )
                    frames = self._align.process(frames)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    raise RuntimeError("Invalid RealSense frame pair received.")

                color_image = np.asanyarray(color_frame.get_data()).copy()
                depth_image = np.asanyarray(depth_frame.get_data()).copy()
                timestamp = time.time()

                with self._frame_lock:
                    self._latest_color_image = color_image
                    self._latest_depth_image = depth_image
                    self._latest_frame_timestamp = timestamp
                    self._last_capture_error = None
                    self._frame_ready.set()

                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001  # RealSense SDK errors vary
                consecutive_failures += 1
                with self._frame_lock:
                    self._last_capture_error = exc
                if consecutive_failures >= self._max_read_attempts:
                    self._frame_ready.set()
                if self._stop_event.wait(0.05):
                    break
                try:
                    self._start_pipeline()
                except Exception as restart_exc:  # noqa: BLE001
                    with self._frame_lock:
                        self._last_capture_error = restart_exc
                    logger.warning(
                        "RealSense %s restart failed: %s",
                        self._device_id,
                        restart_exc,
                    )
                    if self._stop_event.wait(min(1.0, 0.1 * consecutive_failures)):
                        break

    def _start_pipeline(self):
        rs = self._rs

        with self._lock:
            if self._pipeline:
                try:
                    self._pipeline.stop()
                except RuntimeError:
                    logger.debug("Failed to stop RealSense pipeline", exc_info=True)

            self._pipeline = rs.pipeline()
            self._config = rs.config()

            if self._device_id is not None:
                self._config.enable_device(self._device_id)

            self._config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
            self._config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)

            self._pipeline.start(self._config)

            for _ in range(self._warmup_frames):
                self._pipeline.wait_for_frames()

    def close(self) -> None:
        """Stop capture and release all V4L2 handles held by this camera."""
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()

        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
            if pipeline is not None:
                try:
                    pipeline.stop()
                except RuntimeError:
                    logger.debug(
                        "Failed to stop RealSense %s pipeline",
                        self._device_id,
                        exc_info=True,
                    )

        capture_thread = self._capture_thread
        if (
            capture_thread is not None
            and capture_thread is not threading.current_thread()
        ):
            capture_thread.join(timeout=5.0)
            if capture_thread.is_alive():
                logger.warning(
                    "RealSense %s capture thread did not stop within 5 seconds",
                    self._device_id,
                )

    def read(
        self,
        img_size: tuple[int, int] | None = None,  # farthest: float = 0.12
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """
        import cv2

        if not self._frame_ready.wait(timeout=self._read_wait_timeout_sec):
            raise RuntimeError(
                "Timed out waiting for RealSense capture thread to produce a frame."
            )

        with self._frame_lock:
            color_image = self._latest_color_image
            depth_image = self._latest_depth_image
            frame_timestamp = self._latest_frame_timestamp
            last_error = self._last_capture_error

        if color_image is None or depth_image is None or frame_timestamp is None:
            if last_error is not None:
                raise RuntimeError(
                    "RealSense capture thread failed to produce a frame."
                ) from last_error
            raise RuntimeError("RealSense frame is unavailable.")

        frame_age = time.time() - frame_timestamp
        if frame_age > self._max_frame_age_sec:
            raise RuntimeError(
                f"RealSense frame is stale ({frame_age:.3f}s old); camera may be stalled."
            )

        if img_size is None:
            image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            depth = depth_image
        else:
            resized_color = cv2.resize(color_image, img_size)
            image = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
            depth = cv2.resize(depth_image, img_size)

        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)

        depth = depth[:, :, None]

        return image, depth


def _debug_read(camera, save_datastream=False):
    import cv2

    cv2.namedWindow("image")
    cv2.namedWindow("depth")
    counter = 0
    if not os.path.exists("images"):
        os.makedirs("images")
    if save_datastream and not os.path.exists("stream"):
        os.makedirs("stream")
    while True:
        time.sleep(0.1)
        image, depth = camera.read()
        depth = np.concatenate([depth, depth, depth], axis=-1)
        key = cv2.waitKey(1)
        cv2.imshow("image", image[:, :, ::-1])
        cv2.imshow("depth", depth)
        if key == ord("s"):
            cv2.imwrite(f"images/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"images/depth_{counter}.png", depth)
        if save_datastream:
            cv2.imwrite(f"stream/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"stream/depth_{counter}.png", depth)
        counter += 1
        if key == 27:
            break


if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    rs = RealSenseCamera(flip=True, device_id=device_ids[0])
    im, depth = rs.read()
    _debug_read(rs, save_datastream=True)
