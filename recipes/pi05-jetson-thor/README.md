# π0.5 BimanualYAM on Jetson AGX Thor

This release serves the public [robocurve/pi0.5-yam](https://huggingface.co/robocurve/pi0.5-yam)
checkpoint with three RGB cameras, 16×14 absolute joint actions and ten
diffusion steps. It uses BF16/FP16 execution with lossless weight storage.
The π0.5 release covers **BimanualYAM only**.

## Setup and serving

Use the [shared Thor setup](../../examples/bimanual-yam/README.md#1-shared-setup).
The bundle requires NVIDIA Thor (20 SMs), CUDA 13.2, TensorRT 10.16.2.10 and
PyTorch 2.10.0. It includes engines, plugins, tokenizer, normalization data,
language embeddings and checksums; no upstream checkpoint download is needed.

Download the bundle, then point `--engine-dir` to its directory:

```bash
hf download agents2agents/Pi0.5-BimanualYAM-Jetson-Thor --local-dir ./pi05-yam-thor
vla-edge-serve --policy pi05-bimanual-yam --backend tensorrt \
  --engine-dir ./pi05-yam-thor
```

The default port is 8202. This release uses ten diffusion steps and does not
support RTC. Gripper state is the last commanded opening.

## Check inference without robot hardware

```bash
python -m vla_edge.scripts.verify_release --bundle ./pi05-yam-thor
python -m vla_edge.scripts.smoke_pi05 --bundle ./pi05-yam-thor
```

These commands verify checksums and execute the GPU stages without opening
cameras or robot interfaces. The smoke check reports the action shape and
host-to-actions time. See the bundle's `VALIDATION.json` for test coverage.

To use a saved observation, provide an NPZ containing HWC RGB `uint8` arrays
named `top_cam`, `left_cam` and `right_cam`, plus a 14-value float32 `state`:

```bash
python -m vla_edge.scripts.smoke_pi05 --bundle ./pi05-yam-thor \
  --observation sample.npz --prompt "fold the t-shirt" --output actions.npy
```

Robot execution uses synchronous policy chunks and an independent 100 Hz
Ruckig motion process with velocity, acceleration and jerk limits.
For camera setup and robot tasks, follow the
[Bimanual YAM guide](../../examples/bimanual-yam/README.md#pi05).

## Startup diagnostics

Engine initialization traces are opt-in with `VLA_EDGE_DEBUG=1`. Runtime
warnings and errors remain visible with or without this setting.

## Model terms

Weights and tokenizer retain their upstream terms, including the Gemma
terms. Read the model card and notices included in the bundle before
redistribution.
