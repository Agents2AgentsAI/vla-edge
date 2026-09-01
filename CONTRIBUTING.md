# Contributing

`vla-edge` has one job: run flow-matching VLA policies quickly without
changing their behavior. A backend contribution must include correctness
results, not only a latency number.

## Scope

Contributions are welcome across the runtime, protocol, backend seam,
deployment examples, and recipes. Model training and fine-tuning belong in
the upstream project.

MolmoAct2 is currently the only supported model family. Open an issue before
adding another one so we can agree on the shared interface first.

## Development setup

```bash
git clone https://github.com/Agents2AgentsAI/vla-edge.git
cd vla-edge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest
```

The default test suite is CPU-only. Keep accelerator imports inside their
backend packages so the shared runtime and tests remain portable.

Optional dependencies are grouped by use:

| extra | purpose |
|---|---|
| `torch` | PyTorch reference backend and tensor capture |
| `tensorrt` | NVIDIA backend development |
| `coreml` | Apple backend development |
| `eval` | behavioral evaluation video support |
| `camera` | live camera clients |

## Backend changes

A backend implements `encode_vision`, `prefill`, and `denoise` from
`src/vla_edge/backends/base.py`. Tokenization, masks, RNG, normalization, and
embedding scatter stay on the shared host path.

Record reference tensors for the exact checkpoint and input shape you use:

```bash
python -m vla_edge.scripts.capture \
  --embodiment bimanual-yam --out capture.pt
```

Use `vla_edge.gates.parity` to report:

- the reference noise floor in the model's trained precision
- relative RMS and maximum absolute error for each compiled stage
- the end-to-end candidate delta relative to the policy's seed-to-seed spread

Run behavioral evaluation when a change can alter the policy, including
quantization, fewer denoising steps, or a different integrator. Report each
suite separately and compare it with a matched control. See
[docs/gates.md](docs/gates.md) for the release methodology.

Add a reproducible recipe under `recipes/<platform>/` for any new backend or
artifact format.

## Performance claims

Report enough information for someone else to reproduce the comparison:

- hardware, software versions, precision, shapes, and warmup
- whether candidates ran in one process and were interleaved
- clock settings, checked before and after the run
- an A/A measurement of the timing noise floor

Do not use separate fresh processes to support small latency claims. On our
reference device, identical configurations in four processes spanned 17.8 ms.

## Keep generated artifacts out of Git

Do not commit model weights or generated inference artifacts:

- TensorRT plans and engines
- ONNX exports
- Core ML packages
- checkpoints, captures, and tensor dumps

Compiled inference artifacts embed checkpoint weights. Publish them only as a
deliberate model release with the original license and provenance. This source
repository excludes them through `.gitignore` and CI checks.

Do not commit absolute local paths, credentials, logs, camera serial numbers,
CAN assignments, or other rig-specific configuration.

## Pull requests

- Keep each pull request focused.
- Run `ruff check .` and `pytest`.
- Include validation numbers for backend or host-path changes.
- Include the measurement setup for performance claims.
- State what hardware and configurations you did not test.

By contributing, you agree that your contribution is licensed under
Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
