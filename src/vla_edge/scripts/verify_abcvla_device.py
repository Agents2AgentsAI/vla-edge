"""Verify ABC-VLA hardware compatibility and CUDA plugins; no model execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_edge.backends.abcvla.bundle import engine_declaration, read_json
from vla_edge.backends.abcvla.device import detect_device, verify_device_requirements
from vla_edge.backends.abcvla.trt_runtime import _load_plugins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    import tensorrt as trt
    import torch

    root = args.bundle.resolve(strict=True)
    desc = read_json(root / "abcvla-serving.json")
    desc["root"] = root
    current = detect_device(torch, trt)
    entries = {"vision": desc["engines"]["vision"], "flow": desc["engines"]["flow"]}
    entries.update({f"prefill-{k}": v for k, v in desc["engines"]["prefill"].items()})
    for label, raw in entries.items():
        entry = engine_declaration(desc, raw, label)
        metadata = read_json(Path(entry["metadata"]))
        verify_device_requirements(metadata["requires"], current)
        _load_plugins(trt, entry)
        print(
            f"{label}: checksums, {current['name']} / TensorRT {current['tensorrt']} and CUDA plugin initialization pass"
        )


if __name__ == "__main__":
    main()
