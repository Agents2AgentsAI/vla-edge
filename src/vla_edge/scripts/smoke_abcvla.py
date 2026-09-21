"""Run ABC-VLA on one saved or synthetic observation; never opens a robot or camera."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from vla_edge.config import get_policy
from vla_edge.pipeline import Pipeline


def load_observation(path: Path | None):
    if path is None:
        rng = np.random.default_rng(0)
        return ({name: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
                 for name in ("top_cam", "left_cam", "right_cam")},
                np.zeros(14, np.float32), np.zeros((30, 14), np.float32))
    with np.load(path, allow_pickle=False) as sample:
        images = {key: sample[key].copy() for key in ("top_cam", "left_cam", "right_cam")}
        state = sample["state"].copy()
        prefix = sample["action_prefix"].copy() if "action_prefix" in sample else np.zeros((30, 14), np.float32)
    for image in images.values():
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("saved images must be HWC RGB uint8 arrays")
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("saved state must be a finite (14,) array")
    return images, state, prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--observation", type=Path, help="NPZ: top_cam, left_cam, right_cam, state; optional action_prefix")
    parser.add_argument("--backend", choices=("tensorrt", "torch", "cuda-graph"), default="tensorrt")
    parser.add_argument("--prompt", default="fold and stack the t-shirts")
    parser.add_argument("--prefix-length", type=int, choices=range(8), default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="save actions as a NumPy array")
    parser.add_argument("--compare-torch", action="store_true", help="also report errors against the native checkpoint")
    args = parser.parse_args()
    images, state, prefix = load_observation(args.observation)
    # Explicit noise is shared across backends; no dependence on their RNG state.
    noise = np.random.default_rng(args.seed).standard_normal((1, 30, 14)).astype(np.float32)
    outputs = {}
    backends = [args.backend]
    if args.compare_torch and args.backend != "torch":
        backends.append("torch")
    for backend in backends:
        pipeline = Pipeline.load(get_policy("abcvla-bimanual-yam"), backend=backend, engine_dir=args.bundle)
        try:
            def predict(active_pipeline):
                if args.prefix_length:
                    active_pipeline.backend.arm_rtc(prefix[:args.prefix_length], prefix_length=args.prefix_length)
                return active_pipeline.predict(images, args.prompt, state, noise=noise)
            predict(pipeline)  # warm this exact prompt shape and prefix
            start = perf_counter()
            actions = predict(pipeline)
            elapsed = (perf_counter() - start) * 1000
            outputs[backend] = actions.copy()
            print(json.dumps({"backend": backend, "shape": list(actions.shape),
                              "finite": bool(np.isfinite(actions).all()), "host_to_actions_ms": elapsed}))
        finally:
            pipeline.close()
            del pipeline
            gc.collect()
    if args.output:
        np.save(args.output, outputs[args.backend], allow_pickle=False)
    if len(outputs) == 2:
        reference = outputs["torch"].astype(np.float64)
        actual = outputs[args.backend].astype(np.float64)
        diff = actual - reference
        print(json.dumps({"comparison": "native checkpoint", "max_abs": float(np.abs(diff).max()),
                          "rms": float(np.sqrt(np.mean(diff**2))),
                          "rel_rms": float(np.linalg.norm(diff) / max(np.linalg.norm(reference), 1e-12))}))


if __name__ == "__main__":
    main()
