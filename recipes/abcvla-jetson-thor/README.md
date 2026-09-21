# ABC-VLA on Jetson AGX Thor

The bundle contains the `amazon-far/abc:vla_abc130k_v2/200000` checkpoint,
its tokenizer and normalization data, and TensorRT engines for three RGB
cameras and the original 10-step sampler. Outputs are 30 absolute actions
with 14 dimensions. Grippers at indices 6 and 13 use closed=0, open=1;
observations use **measured** gripper positions.

## Install and download

Follow the repository's Jetson PyTorch setup first, then install:

```bash
python -m pip install ".[abcvla]" huggingface-hub
hf download agents2agents/ABC-VLA-Jetson-Thor --local-dir ./abcvla-thor
python -m vla_edge.scripts.verify_release --bundle ./abcvla-thor
python -m vla_edge.scripts.verify_abcvla_device --bundle ./abcvla-thor
```

This release requires NVIDIA Thor (20 SMs, SM 11.0), Linux aarch64,
CUDA 13.2, cuDNN 9 and TensorRT **10.16.2.10**. Use the matching Jetson
TensorRT installation. Plans are specific to this device/software stack;
installing a newer PyPI TensorRT package does not make them portable.
The bundle is approximately 25 GB, including the native reference weights.

The model incorporates Gemma-derived weights. Read the bundle's Gemma
Terms of Use, Prohibited Use Policy and third-party notices before use.
The Python runtime's Apache license does not replace the model's terms.

## Test inference without hardware motion

```bash
python -m vla_edge.scripts.smoke_abcvla --bundle ./abcvla-thor
python -m vla_edge.scripts.smoke_abcvla --bundle ./abcvla-thor \
  --prompt "fold and stack the t-shirts" --prefix-length 5 --compare-torch
```

These commands use synthetic observations and never open cameras or robot
interfaces. To test a recorded observation, add `--observation sample.npz`.
It must contain `top_cam`, `left_cam`, `right_cam` as HWC **RGB uint8** arrays,
`state` as a 14-element vector, and optionally `action_prefix` in absolute
robot coordinates. Images retain their native size: the runtime performs
ABC's resize and padding. Convert OpenCV BGR frames to RGB once on input.
`--output actions.npy` saves the result. Comparison reports errors against
the native checkpoint; synthetic agreement is not a robot-task evaluation.
Timing includes the host path and output transfer, not only engine execution.

## Start the policy server

```bash
./examples/bimanual-yam/start_policy_server.sh --engine-dir ./abcvla-thor
# Equivalent:
vla-edge-serve --policy abcvla-bimanual-yam --backend tensorrt \
  --engine-dir ./abcvla-thor --port 8202
```

`ABCVLA_BUNDLE` can supply the bundle directory; `VLA_PYTHON` selects Python.
The launcher starts only inference. `--host 127.0.0.1` restricts it to the
local machine. Check it from another terminal:

```bash
curl -fsS http://127.0.0.1:8202/act | python -m json.tool
python -m vla_edge.scripts.check_abc_rollout --port 8202 --prefix 5 --chunk 6
```

The server supports vla-edge HTTP at `/act` and the upstream ABC NumPy
msgpack WebSocket protocol at `ws://127.0.0.1:8202/`. The latter accepts
`images` keyed by `top`, `left`, `right` (CHW RGB or JPEG), `state`, `prompt`,
and optional `action_prefix`, `prefix_length`, `seed` or explicit `noise`.
Use either `seed` or `noise`, not both. For HTTP integration,
`vla_edge.protocol.abc.ABCPolicyClient` adapts the same observation schema.

The complete setup and robot workflow is in
[examples/bimanual-yam/README.md](../../examples/bimanual-yam/README.md).
The shared `run_task.sh` selects ABC's hard-prefix controller from server
metadata. It defaults to a prefix of 5 and executes 6 rows per cycle; both
are command-line options. The inference server itself accepts prefixes 0–7.
Rig configuration, camera ownership and rollout storage use the same example
as MolmoAct. Machine-specific calibration and recordings remain local.

## Native reference

Use the same bundle with `--backend torch` for native checkpoint inference,
or `--backend cuda-graph` for native operators with CUDA graph capture.
The TensorRT path uses FP16 vision and flow plus BF16 prefill. It preserves
the checkpoint, image size, action dimensions, camera count and 10 steps.
Static prefill plans cover lengths 213 and 215; the dynamic plan handles
other valid token lengths from 207 through 256. Unsupported prompts fail
explicitly rather than being silently truncated.
