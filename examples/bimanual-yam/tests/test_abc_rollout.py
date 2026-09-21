"""Run ABC controller tests without opening hardware."""

from pathlib import Path

if __name__ == "__main__":
    from run_tests import run

    root = Path(__file__).resolve().parents[3]
    raise SystemExit(run(root / "tests/test_abc_task_runtime.py"))
