# vla-edge

Run flow-matching vision-language-action policies at low latency on edge
hardware.

`vla-edge` is an independent **deployment runtime**, not a model. It was built
around Ai2's [MolmoAct2](https://github.com/allenai/molmoact2) and runs existing
checkpoints at low latency on edge devices without quantizing, distilling, or
dropping cameras. See the upstream [paper](https://arxiv.org/abs/2605.02881)
and [released checkpoints](https://huggingface.co/collections/allenai/molmoact2-models).

<p align="center">
  <a href="https://agents2agents.ai/blog-assets/molmoact2-thor-cube-basket.mp4">
    <img src="https://agents2agents.ai/blog-assets/molmoact2-thor-cube-basket-poster.jpg" width="720" alt="MolmoAct2 running on Jetson AGX Thor and placing a Rubik's cube in a black basket with two YAM arms">
  </a>
  <br>
  <a href="https://agents2agents.ai/blog-assets/molmoact2-thor-cube-basket.mp4">Watch the real-time robot demo</a>
  ·
  <a href="https://agents2agents.ai/blog/molmoact2-jetson-thor">Read the performance writeup</a>
</p>

```
                      ┌─────────────────────────────────────┐
   cameras ──────────►│  shared host path (identical        │
   instruction ──────►│  across every backend)              │
   robot state ──────►│  tokenize · scatter · mask · RNG    │
                      └───────────────┬─────────────────────┘
                                      │  three replaceable stages
                      ┌───────────────▼─────────────────────┐
                      │  VISION  ─►  PREFILL  ─►  FLOW      │
                      │  torch │ tensorrt │ coreml+mlx │ npu│
                      └───────────────┬─────────────────────┘
                                      ▼
                             actions (horizon × dim)
```

## Status

| backend | device | state |
|---|---|---|
| `torch` | anything with the checkpoint's deps | **working**: the reference implementation and the parity ground truth |
| `tensorrt` | NVIDIA Jetson Thor, dGPU | **working**: serves built or prebuilt plans; validated against `torch` on-device |
| `coreml-mlx` | Apple silicon | coming soon |
| `npu` | mobile NPU | coming soon |

**This repository ships no weights and no compiled artifacts.**
Prebuilt Jetson AGX Thor bundles are published separately on Hugging Face:

- [fastest BimanualYAM](https://huggingface.co/agents2agents/MolmoAct2-Jetson-Thor)
- [dynamic-prompt BimanualYAM](https://huggingface.co/agents2agents/MolmoAct2-BimanualYAM-Dynamic-Jetson-Thor)
- [LIBERO](https://huggingface.co/agents2agents/MolmoAct2-LIBERO-Jetson-Thor)

Each bundle includes its provenance, hardware requirements, and checksums.

## Start here

- **[docs/spec.md](docs/spec.md)**: the three-stage seam.
- **[docs/gates.md](docs/gates.md)**: the validation methodology used for
  backend releases.
- **[examples/bimanual-yam/](examples/bimanual-yam/)**: the complete robot
  side for the standard MolmoAct2 BimanualYAM rig: camera server, rollout
  client (sync, async planning, and Real-Time Chunking), preflight, homing.
  Point it at this server and the arms move.
- **[Optimization writeup](https://agents2agents.ai/blog/molmoact2-jetson-thor)**:
  Jetson Thor results, methodology, and findings.

## Quick start on Jetson AGX Thor

`pyproject.toml` covers portable Python dependencies. It cannot select the
Jetson SBSA PyTorch index or configure the dynamic loader. On Thor, install
the package in a fresh environment:

```bash
git clone https://github.com/Agents2AgentsAI/vla-edge.git
cd vla-edge
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision \
  --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple
python -m pip install nvpl-blas nvpl-lapack nvidia-cudss-cu13
python scripts/jetson_thor_postinstall.py
python -m pip install ".[torch]"
python -c "import torch; assert torch.version.cuda and 'sm_110' in torch.cuda.get_arch_list(), 'CPU-only torch: reinstall from the SBSA index'"
```

The released compiled flow package requires the PyTorch 2.10.0 ABI. Other
versions use the `action_flow.plan` fallback, which adds about 30 ms per chunk.

Download the fastest BimanualYAM bundle and start the server:

```bash
hf download agents2agents/MolmoAct2-Jetson-Thor --local-dir vla-edge-thor
vla-edge-serve --embodiment bimanual-yam --backend tensorrt \
  --engine-dir ./vla-edge-thor --fast-vision
```

The download is about 11.6 GB. `vla-edge` selects the matching engine set from
the bundle and checks the hardware before loading it. In another terminal:

```bash
source .venv/bin/activate
curl -fsS http://127.0.0.1:8202/act | python -m json.tool
```

The response should report `"backend": "tensorrt"` and `"rtc": true`.
Serving this bundle does not download the upstream checkpoint.

## PyTorch reference backend

Use the PyTorch backend as a correctness reference or when porting to other
hardware. It downloads and caches the upstream checkpoint on first launch:

```bash
source .venv/bin/activate
vla-edge-serve --embodiment bimanual-yam --backend torch
```

## Call the server

```python
from vla_edge.protocol.client import ActClient

client = ActClient("127.0.0.1:8202")
actions, dt_ms = client.act(
    cameras={"top_cam": top, "left_cam": left, "right_cam": right},
    instruction="pick up the cup and put it on the plate",
    state=joint_positions,          # (14,) float32 for bimanual-yam
)                                   # actions: (horizon, action_dim)
```

The server exposes one endpoint, `POST /act`, json_numpy encoded. Camera order
in the request **must match training order**: a policy fed its cameras in the
wrong order produces confident, plausible, wrong actions, and nothing in the
stack will tell you. Assert it at startup.

## Contributing a backend

Implement three methods in `src/vla_edge/backends/base.py`, report the
validation results described in `docs/gates.md`, and add a reproducible recipe
under `recipes/`.

## Scope

This repository covers the runtime, wire protocol, backend seam, validation
methodology, deployment examples, and recipes. Model training and fine-tuning
live in the upstream project.

## License

The original code is Apache-2.0. Adapted GELLO files remain under MIT. See
[LICENSE](LICENSE), [NOTICE](NOTICE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
