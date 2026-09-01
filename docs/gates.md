# Validation methodology

We validate a compiled backend at three levels before release: stage parity,
deterministic end-to-end comparison, and behavioral evaluation.

| layer | catches | limitation |
|---|---|---|
| stage parity | kernel, precision, and tensor-layout errors | cannot detect errors that cancel across stages |
| end-to-end seed gate | composition, RNG, and state-plumbing errors | does not measure task success |
| behavioral evaluation | task-level regressions | slower and subject to sampling noise |

## 1. Stage parity

`vla_edge.scripts.capture` records the PyTorch backend's intermediate inputs
and outputs for a specific checkpoint and input shape. Backend development
tooling compares each compiled stage against those tensors using the metrics
in `vla_edge.gates.parity`.

Set thresholds from a reference comparison in the checkpoint's trained
precision. On our reference checkpoint, the `bf16` noise floor is
**0.04–0.06 relative RMS**.

| comparison | relative RMS | result |
|---|---:|---|
| `bf16` reference noise floor | 0.04–0.06 | expected |
| compiled `bf16` prefill | 0.04–0.06 | pass |
| compiled `fp16` prefill | **0.55** | fail |

Measure the noise floor again when the checkpoint, precision, or input shape
changes.

## 2. End-to-end seed gate

Flow-matching policies are stochastic, so candidate error is measured against
the policy's variation across noise seeds.

1. Run the PyTorch reference with two different seeds and record the maximum
   absolute action difference.
2. Run the reference and candidate with the same seed. Their maximum absolute
   difference must be no more than **0.5× the seed-to-seed difference**.

## 3. Behavioral evaluation

Changes that alter policy behavior, including fewer denoising steps,
quantization, or a different integrator, require simulation evaluation.

The matched evaluation uses 4 suites × 10 tasks × 10 episodes, for 400
episodes per configuration. Candidate and control runs use the same evaluation
and episode seeds.

Report each suite's success rate and its difference from the control. Sampling
uncertainty over 400 episodes is approximately ±1.7 percentage points. The
deployed LIBERO configuration scored 97.2% over the full 2,000-episode
protocol, matching the result in the
[MolmoAct2 paper](https://arxiv.org/abs/2605.02881).

Use the 400-episode evaluation for per-change validation and the full protocol
for release validation.

## Measuring latency

Measure performance separately from correctness. On a unified-memory edge
SoC, process and power state can move latency enough to hide small changes.

- Pin the clocks and verify them before and after the run.
- Compare candidates within one process and interleave the configurations.
- Run an A/A comparison to measure the timing noise floor.
- Report the process count, interleaving method, and A/A result.

Four fresh processes running the same configuration differed by 17.8 ms on
our reference device. Use paired, in-process measurements for smaller effects.
