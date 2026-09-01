"""Helpers for the eval system: per-rollout saving, live multi-camera viewer,
DROID-style labeling, end-of-session LeRobot conversion.

Each class/function is independent; the launch script wires them together.
"""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import cv2
import h5py
import numpy as np
from camera_client import CameraSubscriber
from PIL import Image

# ``lerobot_convert`` (and its ``lerobot`` dependency) is imported lazily inside
# ``convert_session_to_lerobot`` so that running rollouts does not require
# ``lerobot`` to be installed. Only the end-of-session conversion does.
# NOTE: the converter module is NOT shipped with this example (it lives in the
# upstream molmoact2 repo). Without it, the end-of-session conversion fails
# with a logged error and the raw rollouts under ``storage.base_dir`` are kept.


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Per-rollout saver
# ---------------------------------------------------------------------------


def _save_png(image: np.ndarray, path: Path, compress_level: int) -> None:
    Image.fromarray(image).save(path, compress_level=compress_level)


class EvalRolloutSaver:
    """Buffer one rollout's frames in RAM, then flush to PNG + HDF5 on disk.

    Layout per rollout::

        rollout_dir/
        ├── left_rgb/{frame:06d}.png
        ├── right_rgb/{frame:06d}.png
        ├── front_rgb/{frame:06d}.png
        ├── episode.h5
        └── err.md          # only if write_err() is called

    The HDF5 file holds the joint trajectory (state, next_state) and the
    language instruction; the PNGs hold the per-frame RGB images. The DROID
    layout converter (``load_droid_layout_data``) walks this structure.
    """

    CAMERA_OBS_TO_KEY: ClassVar[dict[str, str]] = {
        "left_camera_rgb": "left_rgb",
        "right_camera_rgb": "right_rgb",
        "front_camera_rgb": "front_rgb",
    }

    def __init__(
        self,
        rollout_dir: Path,
        instruction: str,
        max_workers: int = 2,
        png_compress_level: int = 1,
        save_frames: bool = False,
    ) -> None:
        self.rollout_dir = Path(rollout_dir)
        self.instruction = instruction
        self.max_workers = max(1, int(max_workers))
        self.png_compress_level = max(0, min(9, int(png_compress_level)))
        # Frames are ~12 GB per hour of episodes; joint states (episode.h5)
        # are always kept. Enable via storage.save_frames when a rollout's
        # video actually matters.
        self.save_frames = bool(save_frames)

        if self.rollout_dir.exists():
            raise FileExistsError(
                f"Rollout dir already exists: {self.rollout_dir}. "
                "Timestamps are expected to be unique."
            )
        self.rollout_dir.mkdir(parents=True)

        self._buffer: list[dict[str, Any]] = []

    @property
    def num_steps(self) -> int:
        return len(self._buffer)

    def add_step(
        self,
        obs_pre: dict[str, Any],
        obs_post: dict[str, Any],
    ) -> None:
        """Buffer one control-step record.

        ``obs_pre`` is the observation snapshot at the start of the step
        (the image the policy "sees" for that step) and ``obs_post`` is the
        observation after applying the action. ``next_state`` mirrors the
        data-collection convention used in ``DataSaver``: the observed post-step
        joint positions, not the commanded action.
        """
        record: dict[str, Any] = {
            "state": np.asarray(obs_pre["joint_positions"], dtype=np.float32).copy(),
            "next_state": np.asarray(obs_post["joint_positions"], dtype=np.float32).copy(),
        }
        if self.save_frames:
            for obs_key, cam_key in self.CAMERA_OBS_TO_KEY.items():
                img = obs_pre.get(obs_key)
                if img is not None:
                    record[cam_key] = np.ascontiguousarray(img).copy()
        self._buffer.append(record)

    def flush(self) -> None:
        """Write buffered PNGs and ``episode.h5`` to ``rollout_dir``."""
        if not self._buffer:
            logger.warning("Empty buffer at %s; nothing to flush.", self.rollout_dir)
            return

        cam_keys_present = sorted(
            k for k in self.CAMERA_OBS_TO_KEY.values() if k in self._buffer[0]
        )
        for cam_key in cam_keys_present:
            (self.rollout_dir / cam_key).mkdir(exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futures = []
            for i, rec in enumerate(self._buffer):
                for cam_key in cam_keys_present:
                    img_path = self.rollout_dir / cam_key / f"{i:06d}.png"
                    futures.append(
                        exe.submit(_save_png, rec[cam_key], img_path, self.png_compress_level)
                    )
            for fut in futures:
                fut.result()

        states = np.stack([rec["state"] for rec in self._buffer]).astype(np.float32)
        next_states = np.stack([rec["next_state"] for rec in self._buffer]).astype(np.float32)
        cam_names_stripped = [k.replace("_rgb", "") for k in cam_keys_present]

        h5_path = self.rollout_dir / "episode.h5"
        with h5py.File(h5_path, "w") as f:
            f.attrs["language_instruction"] = self.instruction
            f.attrs["num_steps"] = len(self._buffer)
            f.attrs["camera_names"] = np.array(
                cam_names_stripped, dtype=h5py.string_dtype()
            )
            f.create_dataset("state", data=states, compression="gzip", compression_opts=4)
            f.create_dataset(
                "next_state", data=next_states, compression="gzip", compression_opts=4
            )

        logger.info(
            "Saved rollout: %s (%d steps, cameras=%s)",
            self.rollout_dir,
            len(self._buffer),
            cam_names_stripped,
        )

    def write_err(self, reason: str, step: int) -> None:
        """Drop a marker file explaining why this rollout is incomplete."""
        err_path = self.rollout_dir / "err.md"
        with open(err_path, "w") as f:
            f.write("# Incomplete rollout\n\n")
            f.write(f"- Reason: {reason}\n")
            f.write(f"- Step at interruption: {step}\n")
            f.write(f"- Steps actually saved: {self.num_steps}\n")
            f.write(f"- Instruction: {self.instruction}\n")
            written_at = datetime.now().astimezone().isoformat(timespec="seconds")
            f.write(f"- Written at: {written_at}\n")


# ---------------------------------------------------------------------------
# Live cv2 viewer
# ---------------------------------------------------------------------------


class LiveCameraView:
    """Single cv2 window showing the three policy-input frames hconcat'd, with
    a text header and key polling.

    Two modes:

    * ``thread`` (when ``pub_endpoint`` is given): a daemon background thread
      owns the cv2 window and a ``CameraSubscriber`` connected to the camera
      server's PUB stream. The thread fetches frames and repaints at camera
      rate, so the window keeps updating even while the rollout loop is
      blocked inside ``policy.inference()``. ``update()`` becomes a non-blocking
      header push + key poll.
    * ``obs`` (no ``pub_endpoint``): legacy path; cv2 runs on the calling
      thread, frames come from the ``obs`` dict passed to ``update()``.

    ``update()`` returns the lowercase key character ``'y' | 'n' | 'q'`` if one
    of those was captured since the last call, otherwise ``None``.
    """

    WINDOW_NAME = "YAM Eval"
    OBS_KEYS = ("left_camera_rgb", "front_camera_rgb", "right_camera_rgb")
    PUB_CAM_NAMES = ("left_camera", "front_camera", "right_camera")
    OBS_LABELS = ("LEFT", "FRONT", "RIGHT")
    # Window grows 2x in each linear dimension on first frame -> 4x screen area.
    SCALE = 2

    def __init__(
        self,
        enabled: bool = True,
        pub_endpoint: str | None = None,
        recv_timeout_ms: int = 100,
        target_fps: float = 30.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.pub_endpoint = pub_endpoint if (self.enabled and pub_endpoint) else None
        self.recv_timeout_ms = int(recv_timeout_ms)
        self.target_fps = float(target_fps)

        if not self.enabled:
            self._mode = "off"
        elif self.pub_endpoint:
            self._mode = "thread"
        else:
            self._mode = "obs"

        # obs-mode state (cv2 on calling thread)
        self._initialized = False
        self._sized = False

        # thread-mode state
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sub: Any | None = None  # CameraSubscriber, lazily constructed
        self._key_buf: deque[str] = deque(maxlen=8)
        self._header: dict[str, Any] = {
            "rollout_idx": 0, "num_rollouts": 1,
            "step": 0, "max_steps": 1,
            "instruction": "",
        }

    # ------------------------------------------------------------------
    # Thread mode
    # ------------------------------------------------------------------

    def _start_thread(self) -> None:
        self._sub = CameraSubscriber(self.pub_endpoint, recv_timeout_ms=self.recv_timeout_ms)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._render_loop, name="LiveCameraView", daemon=True,
        )
        self._thread.start()

    def _render_loop(self) -> None:
        last_frames: dict[str, np.ndarray] | None = None
        last_recv_t: float = 0.0
        wait_ms = max(1, int(1000.0 / max(self.target_fps, 1.0)))
        window_ready = False

        while not self._stop.is_set():
            try:
                frames = self._sub.try_recv() if self._sub is not None else None
                if frames:
                    last_frames = frames
                    last_recv_t = time.monotonic()

                with self._lock:
                    hdr = dict(self._header)

                canvas = self._build_panes(last_frames)
                stale = (time.monotonic() - last_recv_t) if last_recv_t else None
                header = self._build_header(canvas.shape[1], hdr, stale)
                final = cv2.vconcat([header, canvas])

                if not window_ready:
                    cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(self.WINDOW_NAME, final)
                if not window_ready:
                    h, w = final.shape[:2]
                    cv2.resizeWindow(self.WINDOW_NAME, w * self.SCALE, h * self.SCALE)
                    window_ready = True

                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("y"), ord("n"), ord("q")):
                    with self._lock:
                        self._key_buf.append(chr(key))
            except Exception:
                logger.exception("LiveCameraView render tick failed")
                self._stop.wait(0.1)

        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except cv2.error:
            pass
        if self._sub is not None:
            try:
                self._sub.close()
            except Exception:
                logger.debug("Failed to close camera subscriber", exc_info=True)

    def _build_panes(
        self, frames: dict[str, np.ndarray] | None,
    ) -> np.ndarray:
        if not frames:
            return self._placeholder_canvas("Waiting for camera server...")
        panes: list[np.ndarray] = []
        for name, label in zip(self.PUB_CAM_NAMES, self.OBS_LABELS):
            img = frames.get(name)
            if img is None:
                img = frames.get(name + "_rgb")
            if img is None:
                continue
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
            cv2.putText(
                bgr, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
            )
            panes.append(bgr)
        if not panes:
            return self._placeholder_canvas("No camera frames received yet.")
        max_h = max(p.shape[0] for p in panes)
        padded = [
            np.pad(p, ((0, max_h - p.shape[0]), (0, 0), (0, 0))) if p.shape[0] < max_h else p
            for p in panes
        ]
        return cv2.hconcat(padded)

    def _placeholder_canvas(self, msg: str) -> np.ndarray:
        canvas = np.zeros((360, 1280, 3), dtype=np.uint8)
        cv2.putText(
            canvas, msg, (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA,
        )
        return canvas

    def _build_header(
        self, width: int, hdr: dict[str, Any], stale_sec: float | None,
    ) -> np.ndarray:
        header_h = 90
        header = np.zeros((header_h, width, 3), dtype=np.uint8)
        rollout_idx = int(hdr.get("rollout_idx", 0))
        num_rollouts = int(hdr.get("num_rollouts", 1))
        step = int(hdr.get("step", 0))
        max_steps = int(hdr.get("max_steps", 1))
        instruction = str(hdr.get("instruction", ""))
        first_line = f"Rollout {rollout_idx + 1}/{num_rollouts}    Step {step}/{max_steps}"
        if stale_sec is not None and stale_sec > 1.0:
            first_line = f"{first_line}    [stale {stale_sec:.1f}s]"
        lines = [
            first_line,
            f"Instruction: {instruction}",
            "Keys:  y = success    n = failure    q = quit rollout (saves as eval)",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                header, line, (10, 24 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return header

    # ------------------------------------------------------------------
    # Obs mode (legacy fallback)
    # ------------------------------------------------------------------

    def _ensure_window(self) -> None:
        if self._initialized or not self.enabled:
            return
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        self._initialized = True

    def _update_obs_mode(
        self,
        obs: dict[str, Any],
        rollout_idx: int,
        num_rollouts: int,
        step: int,
        max_steps: int,
        instruction: str,
    ) -> str | None:
        self._ensure_window()
        panes: list[np.ndarray] = []
        for obs_key, label in zip(self.OBS_KEYS, self.OBS_LABELS):
            rgb = obs.get(obs_key)
            if rgb is None:
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            cv2.putText(
                bgr, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
            )
            panes.append(bgr)
        if not panes:
            return None
        max_h = max(p.shape[0] for p in panes)
        padded = [
            np.pad(p, ((0, max_h - p.shape[0]), (0, 0), (0, 0))) if p.shape[0] < max_h else p
            for p in panes
        ]
        canvas = cv2.hconcat(padded)
        header = self._build_header(
            canvas.shape[1],
            {
                "rollout_idx": rollout_idx, "num_rollouts": num_rollouts,
                "step": step, "max_steps": max_steps,
                "instruction": instruction,
            },
            stale_sec=None,
        )
        final = cv2.vconcat([header, canvas])
        cv2.imshow(self.WINDOW_NAME, final)
        if not self._sized:
            h, w = final.shape[:2]
            cv2.resizeWindow(self.WINDOW_NAME, w * self.SCALE, h * self.SCALE)
            self._sized = True
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("y"), ord("n"), ord("q")):
            return chr(key)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        obs: dict[str, Any],
        rollout_idx: int,
        num_rollouts: int,
        step: int,
        max_steps: int,
        instruction: str,
    ) -> str | None:
        if not self.enabled:
            return None

        if self._mode == "thread" and self._thread is None:
            try:
                self._start_thread()
            except Exception:
                logger.exception(
                    "LiveCameraView: failed to start PUB thread; falling back to obs mode"
                )
                self._mode = "obs"

        if self._mode == "thread":
            with self._lock:
                self._header = {
                    "rollout_idx": rollout_idx,
                    "num_rollouts": num_rollouts,
                    "step": step,
                    "max_steps": max_steps,
                    "instruction": instruction,
                }
                key = self._key_buf.popleft() if self._key_buf else None
            return key

        return self._update_obs_mode(
            obs=obs,
            rollout_idx=rollout_idx,
            num_rollouts=num_rollouts,
            step=step,
            max_steps=max_steps,
            instruction=instruction,
        )

    def close(self) -> None:
        if self._thread is not None:
            self._stop.set()
            try:
                self._thread.join(timeout=2.0)
            except RuntimeError:
                logger.debug("Failed to join camera-view thread", exc_info=True)
            self._thread = None
        elif self._initialized:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
            except cv2.error:
                pass
            self._initialized = False


# ---------------------------------------------------------------------------
# Interactive prompts (stdin)
# ---------------------------------------------------------------------------


def prompt_instruction(rollout_idx: int, num_rollouts: int, last_prompt: str) -> str:
    """Ask user for the task instruction for this rollout.

    Empty input → reuse ``last_prompt``.
    """
    text = input(
        f"\n[rollout {rollout_idx + 1}/{num_rollouts}] "
        f"Task instruction (Enter to reuse '{last_prompt}'): "
    ).strip()
    return text if text else last_prompt


def prompt_label() -> str | None:
    """DROID-style label prompt for timeout-ended rollouts.

    Returns ``"success"`` / ``"failure"`` / ``None`` (keep as eval).
    Reprompts on any other input.
    """
    while True:
        text = input(
            "Label rollout (y = success / n = failure / Enter = keep as eval): "
        ).strip().lower()
        if text == "y":
            return "success"
        if text == "n":
            return "failure"
        if text == "":
            return None
        print("Invalid input. Type y, n, or press Enter.")


# ---------------------------------------------------------------------------
# Rollout end-result -> filesystem move
# ---------------------------------------------------------------------------


@dataclass
class RolloutOutcome:
    """The outcome of one rollout, used by the labeling and move logic."""

    end_reason: str  # 'success' | 'failure' | 'quit' | 'timeout'
    last_step: int

    def implicit_label(self) -> str | None:
        """Label inferred from end_reason; ``None`` means 'ask the user'."""
        if self.end_reason == "success":
            return "success"
        if self.end_reason == "failure":
            return "failure"
        return None  # 'quit' or 'timeout' need either a stay-in-eval or stdin prompt

    def keep_in_eval(self) -> bool:
        """Quit means user explicitly wanted no label."""
        return self.end_reason == "quit"


def resolve_label(outcome: RolloutOutcome) -> str | None:
    """Decide where a rollout goes given its end reason.

    Returns the label (``"success"`` / ``"failure"``) or ``None`` for stay-in-eval.
    Timeout triggers the stdin prompt; quit is treated as no-label.
    """
    if outcome.keep_in_eval():
        return None
    implicit = outcome.implicit_label()
    if implicit is not None:
        return implicit
    # Timeout: ask the user.
    return prompt_label()


def move_rollout(rollout_dir: Path, label: str, base_save_dir: Path) -> Path:
    """Move ``rollout_dir`` (under ``base_save_dir/eval/``) to
    ``base_save_dir/{label}/{YYYY-MM-DD}/{name}/``. Returns the new path.
    """
    if label not in ("success", "failure"):
        raise ValueError(f"Unknown label: {label!r}")
    date_str = datetime.now().astimezone().strftime("%Y-%m-%d")
    dest_parent = Path(base_save_dir) / label / date_str
    dest_parent.mkdir(parents=True, exist_ok=True)
    dest = dest_parent / rollout_dir.name
    if dest.exists():
        # Defensive: same timestamp twice shouldn't happen, but don't clobber.
        suffix = datetime.now().astimezone().strftime("_%f")
        dest = dest_parent / (rollout_dir.name + suffix)
    shutil.move(str(rollout_dir), str(dest))
    return dest


# ---------------------------------------------------------------------------
# End-of-session LeRobot conversion
# ---------------------------------------------------------------------------


def convert_session_to_lerobot(
    session_rollout_dirs: Sequence[Path],
    output_dir: Path,
    fps: int,
    robot_type: str,
    repo_id: str = "local/eval_session",
    action_mode: str = "next_joint_fields",
    vcodec: str = "libsvtav1",
    sanitize_online_viz_meta: bool = True,
    image_writer_processes: int = 0,
    image_writer_threads: int = 0,
    parallel_encoding: bool = True,
) -> Path | None:
    """Convert the labeled rollouts from this eval session into one LeRobot v3.0 dataset.

    Calls into the existing ``create_lerobot_dataset_v30`` so the dataset schema
    stays identical to the data-collection pipeline. Returns the final output
    path (may differ from ``output_dir`` if a uniqueness suffix was applied to
    avoid a non-empty-directory collision).
    """
    if not session_rollout_dirs:
        logger.info("No labeled rollouts to convert.")
        return None

    # Imported here (not at module load) so rollouts run without `lerobot`.
    from lerobot_convert import create_lerobot_dataset_v30, load_droid_layout_data

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        suffix = datetime.now().astimezone().strftime("_%H%M%S")
        output_dir = output_dir.parent / f"{output_dir.name}{suffix}"
        logger.warning("Output dir non-empty; using %s instead.", output_dir)

    episodes = load_droid_layout_data(
        base_dir=None,
        explicit_paths=[Path(p) for p in session_rollout_dirs],
    )
    if not episodes:
        logger.error("No usable episodes from this session; skipping conversion.")
        return None

    create_lerobot_dataset_v30(
        episodes=episodes,
        output_dir=str(output_dir),
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        action_mode=action_mode,
        sanitize_online_viz_meta=sanitize_online_viz_meta,
        vcodec=vcodec,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        parallel_encoding=parallel_encoding,
    )
    logger.info("LeRobot dataset written: %s", output_dir)
    return output_dir
