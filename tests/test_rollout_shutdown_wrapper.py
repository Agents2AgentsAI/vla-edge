"""The shell fallback must never re-enable motors after its launcher exits."""

import json
import os
import subprocess
from pathlib import Path

import pytest

WRAPPER = Path(__file__).parents[1] / "examples/bimanual-yam/run_rollout.sh"


@pytest.mark.parametrize("launcher_status", [0, 7])
def test_wrapper_fallback_only_disables(tmp_path, launcher_status):
    wrapper = tmp_path / "run_rollout.sh"
    wrapper.write_bytes(WRAPPER.read_bytes())
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
script = Path(sys.argv[2]).name
if script == 'launch_yaml_eval_molmoact.py':
    sys.exit(int(os.environ['TEST_LAUNCHER_STATUS']))
assert script == 'home_arms.py'
Path(os.environ['TEST_CALLS']).write_text(json.dumps(sys.argv[3:]))
""")
    fake_python.chmod(0o755)
    calls = tmp_path / "calls.json"
    result = subprocess.run(
        ["bash", str(wrapper)],
        env={
            **os.environ,
            "YAM_PYTHON": str(fake_python),
            "TEST_LAUNCHER_STATUS": str(launcher_status),
            "TEST_CALLS": str(calls),
        },
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    assert result.returncode == launcher_status, result.stderr
    assert json.loads(calls.read_text()) == ["--status"]
