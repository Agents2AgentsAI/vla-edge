"""Compiled flow-package loading behavior."""

from __future__ import annotations

from vla_edge.backends.tensorrt import flow


def test_driver_load_does_not_mutate_verified_bundle(tmp_path):
    package = tmp_path / "flow"
    package.mkdir()
    (package / "driver.py").write_text(
        "class FlowKernelRunner:\n"
        "    def __init__(self, action_expert, device):\n"
        "        self.action_expert = action_expert\n"
        "        self.device = device\n"
    )

    runner = flow.load(tmp_path, action_expert="weights", device="cuda:0")

    assert runner.action_expert == "weights"
    assert runner.device == "cuda:0"
    assert not (package / "__pycache__").exists()
