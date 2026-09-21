"""Run controller tests quietly, without GPU or robot access."""

import os
import subprocess
import sys
from pathlib import Path


def run(*paths: Path) -> int:
    """Use the same captured pytest output for individual and combined suites."""
    root = Path(__file__).resolve().parents[3]
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *(str(path) for path in paths),
            *sys.argv[1:],
        ],
        cwd=root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    raise SystemExit(
        run(
            here / "test_rollout_control.py",
            here / "test_pi05_motion.py",
            here / "test_pi05_rollout.py",
            root / "tests/test_abc_task_runtime.py",
            root / "tests/test_pi05_task.py",
            root / "tests/test_home_arms.py",
            root / "tests/test_rollout_shutdown_wrapper.py",
        )
    )
