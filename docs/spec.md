# The three-stage seam

This is the whole idea of the repository, so it comes first.

A flow-matching VLA policy of the MolmoAct2 family does one forward pass per
action chunk. There is no autoregressive decoding. That pass decomposes into
exactly three compute-heavy stages with stable, static shapes:

```
  cameras ─► [1] VISION ─► image tokens ─┐
                                          ├─► [2] PREFILL ─► per-layer K/V ─┐
  instruction + state ──► tokenizer ─────┘                                  │
                                                                            ▼
                                    noise ──────────────────► [3] FLOW ─► actions
```

Everything else (tokenization, embedding scatter, attention masks, RNG,
normalization, the checkpoint's own loading quirks) stays in PyTorch on the
CPU/host path and is **identical across every backend**. It is a small
fraction of the wall time and it is where correctness bugs hide, so we do not
reimplement it per platform.

A backend therefore implements three methods and nothing else:

```python
class Backend(Protocol):
    def encode_vision(self, pixel_values, pooling_idx) -> Tensor: ...
    def prefill(self, inputs_embeds, attention_bias, position_ids) -> tuple[Tensor, Tensor]: ...
    def denoise(self, kv_context, cross_mask, noise, steps: int) -> Tensor: ...
```

`src/vla_edge/backends/base.py` is the authoritative definition.

## Why this particular seam

1. **The stage boundaries are where the shapes are static.** Prompt length is
   constant within an episode (the robot state is encoded as discrete tokens,
   not a variable-length string), the image token count is fixed by the camera
   count, and the action chunk is a fixed `[horizon, dim]`. Static shapes are
   what make ahead-of-time compilation, graph capture, and kernel
   specialization possible at all.

2. **The stages are independently replaceable.** You can run vision on a
   compiled engine, prefill in PyTorch, and flow on hand-written kernels, in
   any combination.

## The wire protocol

Serving is simple: one `POST /act` endpoint, `json_numpy` encoding, batch 1.

Request:

| field | type | note |
|---|---|---|
| camera fields | `(H, W, 3)` uint8 RGB | names and order are embodiment-specific and **must match training order** |
| `instruction` | str | |
| `state` | `(D,)` float32 | joint positions + gripper |
| `num_steps` | int, optional | flow denoising steps |

Response: `actions` `(horizon, action_dim)` float32, and `dt_ms` float.

Action shape is driven by the checkpoint's `norm_stats.json`; never hardcode
it. Camera *order* is a silent-failure surface: a policy fed its cameras in the
wrong order produces confident, plausible, wrong actions. Assert it at startup
against the embodiment config.

## Adding a backend

1. Capture reference intermediates once from the PyTorch backend
   (`scripts/capture.py`). This gives you per-stage ground truth for the exact
   checkpoint and shapes you are targeting.
2. Implement the three methods.
3. Pass the stage parity gates, then the end-to-end gate (`docs/gates.md`).
4. Add a recipe under `recipes/<platform>/` that builds your artifacts from a
   checkpoint the user supplies. Do not commit the artifacts; see NOTICE.
