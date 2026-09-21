# vla-edge

Deploy **MolmoAct2, ABC-VLA and π0.5** on NVIDIA Jetson AGX Thor with our
optimized custom CUDA kernels, a shared serving API, and tools for robot
control and recording.

## [Read the benchmarks and optimization writeups →](https://agents2agents.ai/blog)

End-to-end latency, stage-by-stage results, and how we optimized these models
for Jetson Thor.

<p align="center">
  <a href="https://agents2agents.ai/blog-assets/abcvla-thor-tshirt-folding.mp4">
    <img src="https://agents2agents.ai/blog-assets/abcvla-thor-tshirt-folding-poster.jpg" width="720" alt="ABC-VLA folding a t-shirt with two YAM arms on Jetson AGX Thor">
  </a>
  <br>
  <a href="https://agents2agents.ai/blog-assets/abcvla-thor-tshirt-folding.mp4">Watch the folding demo</a>
  <br>
  <sub>First and last 15 seconds at 1×; middle section at 5×.</sub>
</p>

## Performance on Jetson AGX Thor

Latency per action chunk, using the same checkpoint within each row:

| Model | PyTorch eager | TensorRT baseline | Our optimized kernels | vs. PyTorch | vs. TensorRT |
|---|---:|---:|---:|---:|---:|
| ABC-VLA | 90.0 ms | 63.4 ms | **32.4 ms** | **2.78×** | **1.96×** |
| MolmoAct2 | 611.3 ms | 174.8 ms | **113.4 ms** | **5.39×** | **1.54×** |
| π0.5 | 230.6 ms | 112.0 ms | **62.2 ms** | **3.71×** | **1.80×** |

TensorRT baselines are working conversions. ABC-VLA and π0.5 measure the
prepared model pipeline; MolmoAct2 reports the full action call. ABC-VLA and MolmoAct2 use 30-action
chunks; π0.5 uses 16. Our ABC-VLA and π0.5 kernels use BF16/FP16; the optimized
MolmoAct2 stack uses FP8 vision with a BF16 language backbone. See the writeups
above for measurement details and validation.

## Models and bundles

| Model | Jetson Thor bundles | Setup |
|---|---|---|
| MolmoAct2 | [BimanualYAM](https://huggingface.co/agents2agents/MolmoAct2-Jetson-Thor) · [Dynamic prompts](https://huggingface.co/agents2agents/MolmoAct2-BimanualYAM-Dynamic-Jetson-Thor) · [LIBERO](https://huggingface.co/agents2agents/MolmoAct2-LIBERO-Jetson-Thor) | [Guide](recipes/tensorrt-thor/README.md) |
| ABC-VLA | [BimanualYAM](https://huggingface.co/agents2agents/ABC-VLA-Jetson-Thor) | [Guide](recipes/abcvla-jetson-thor/README.md) |
| π0.5 | [BimanualYAM](https://huggingface.co/agents2agents/Pi0.5-BimanualYAM-Jetson-Thor) | [Guide](recipes/pi05-jetson-thor/README.md) |

Weights and compiled artifacts are distributed separately from this repository.
Each bundle specifies its hardware and software requirements and includes checksums.
PyTorch reference backends are available for MolmoAct2 and ABC-VLA.

## Quick start on Jetson AGX Thor

Install the shared environment, including the matching Jetson PyTorch build:

```bash
git clone https://github.com/Agents2AgentsAI/vla-edge.git
cd vla-edge
python3 -m venv .venv
source .venv/bin/activate
./examples/bimanual-yam/setup_jetson_thor.sh
```

Choose a model using the guides above. For π0.5 YAM:

```bash
hf download agents2agents/Pi0.5-BimanualYAM-Jetson-Thor --local-dir ./pi05-yam-thor
vla-edge-serve --policy pi05-bimanual-yam --backend tensorrt \
  --engine-dir ./pi05-yam-thor
```

In another terminal, check the loaded policy:

```bash
curl -fsS http://127.0.0.1:8202/act | python -m json.tool
```

For camera setup, calibration, task execution and video recording, follow the
[Bimanual YAM guide](examples/bimanual-yam/README.md).

## Call the server

```python
from vla_edge.protocol.client import ActClient

client = ActClient("127.0.0.1:8202")
client.bind(policy="pi05-bimanual-yam", embodiment="bimanual-yam")
actions, dt_ms = client.act(
    cameras={"top_cam": top, "left_cam": left, "right_cam": right},
    instruction="fold the t-shirt",
    state=joint_positions,  # 14 values for BimanualYAM
)
```

Images are HWC RGB `uint8` arrays. Use the selected policy's camera names and
training order. The HTTP API uses json_numpy at `POST /act` and returns an
action chunk shaped `(horizon, action_dim)`.

## Development

See the [runtime design](docs/spec.md), [validation gates](docs/gates.md) and
[deployment recipes](recipes/). To add a backend, implement the
[backend interface](src/vla_edge/backends/base.py) and include validation
results and a reproducible recipe.

## License

Original code is Apache-2.0; adapted GELLO files remain under MIT. Model
weights retain their upstream terms. See [LICENSE](LICENSE), [NOTICE](NOTICE)
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
