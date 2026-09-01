"""Config instantiation + start-position helpers used by the eval launcher.

Trimmed from the upstream ``gello.utils.launch_utils`` to only the two helpers
the MolmoAct eval path needs: ``instantiate_from_dict`` (build a robot from a
``_target_`` config block) and ``move_to_start_position`` (interpolate the
arm(s) to ``agent.start_joints`` between rollouts). The teleop launch manager
and its Dynamixel/ZMQ dependencies are intentionally left out.
"""

import importlib
import time
from typing import Any

import numpy as np


def instantiate_from_dict(cfg):
    """Recursively instantiate objects from a ``_target_``-style config dict."""
    if isinstance(cfg, dict) and "_target_" in cfg:
        module_path, class_name = cfg["_target_"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        kwargs = {k: v for k, v in cfg.items() if k != "_target_"}
        return cls(**{k: instantiate_from_dict(v) for k, v in kwargs.items()})
    elif isinstance(cfg, dict):
        return {k: instantiate_from_dict(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [instantiate_from_dict(v) for v in cfg]
    else:
        return cfg


def move_to_start_position(
    env,
    bimanual: bool = False,
    left_cfg: dict[str, Any] | None = None,
    right_cfg: dict[str, Any] | None = None,
):
    """Interpolate the robot to ``agent.start_joints`` if specified in config."""
    if bimanual:
        if right_cfg is None:
            return
        left_start = left_cfg["agent"].get("start_joints")
        right_start = right_cfg["agent"].get("start_joints")
        if left_start is None or right_start is None:
            return
        reset_joints = np.concatenate([np.array(left_start), np.array(right_start)])
    else:
        if (
            "start_joints" not in left_cfg["agent"]
            or left_cfg["agent"]["start_joints"] is None
        ):
            return
        reset_joints = np.array(left_cfg["agent"]["start_joints"])

    curr_joints = env.get_obs()["joint_positions"]
    if reset_joints.shape != curr_joints.shape:
        print("Warning: Mismatch in joint shapes, skipping move_to_start_position.")
        return

    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.01), 100)

    print(f"Moving robot to start position: {reset_joints}")
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt, reset=True)
        time.sleep(0.001)
