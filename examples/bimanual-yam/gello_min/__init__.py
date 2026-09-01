"""Minimal subset of the bimanual-YAM ``gello`` runtime needed to run MolmoAct2
closed-loop eval: the robot abstraction (``robot``, ``yam``), the obs/command
environment (``env``), the camera drivers (``camera``, ``realsense_camera``),
and config/logging helpers (``launch_utils``, ``logging_utils``).

This is a vendored, trimmed copy of https://github.com/williamtsai726/YAM.
only the eval-relevant pieces are kept. Teleop, data collection, and the
Dynamixel/Gello leader-arm code are intentionally omitted.
"""
