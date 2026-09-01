# TensorRT on NVIDIA Jetson Thor

The released bundles contain prebuilt TensorRT engines and the compact host
runtime needed to serve them without downloading the full PyTorch checkpoint.

## Requirements

- NVIDIA Jetson AGX Thor Developer Kit
- JetPack R39 rev 2.1
- TensorRT 10.16.2.10

TensorRT plans are tied to the GPU architecture and TensorRT version. The
server checks the bundle manifest against the machine before loading an
engine.

## Fastest BimanualYAM configuration

Complete the [Thor setup](../../README.md#quick-start-on-jetson-agx-thor), then:

```bash
hf download agents2agents/MolmoAct2-Jetson-Thor \
  --local-dir vla-edge-thor

vla-edge-serve --embodiment bimanual-yam --backend tensorrt \
  --engine-dir ./vla-edge-thor --fast-vision
```

The bundle uses a fixed 704-token prefill engine and includes the FP8 vision
engine selected by `--fast-vision`. It produces a 30-action chunk in 113.4 ms
on our reference Thor.

## Dynamic prompts

Use the dynamic-prompt bundle when instructions can exceed the fixed engine's
prompt capacity:

```bash
hf download agents2agents/MolmoAct2-BimanualYAM-Dynamic-Jetson-Thor \
  --local-dir vla-edge-thor-dynamic

vla-edge-serve --embodiment bimanual-yam --backend tensorrt \
  --engine-dir ./vla-edge-thor-dynamic
```

The fixed bundle rejects an over-capacity prompt with a clear error. It never
truncates the instruction silently.

## Verify the bundle

```bash
python -c "from vla_edge.backends.tensorrt import artifacts; \
artifacts.check_compatible('vla-edge-thor'); \
artifacts.verify_checksums('vla-edge-thor'); print('bundle verified')"
```

Checksum verification reads the full bundle and can take a few minutes. See
[docs/gates.md](../../docs/gates.md) for the correctness and latency
methodology used for the release.

Each plan embeds checkpoint weights and is distributed separately under the
checkpoint's Apache-2.0 license. This source repository excludes plans,
exports, checkpoints, and other generated model artifacts. See
[NOTICE](../../NOTICE).
