"""Record reference intermediates from the torch backend.

Every compiled backend is verified against tensors produced by this script.
It hooks the real model rather than reimplementing it, so the reference stays
the checkpoint's own behavior and not our interpretation of it.

Run once per (checkpoint, embodiment, input shape):

    python -m vla_edge.scripts.capture --embodiment bimanual-yam --out capture.pt

The output is a flat dict of named tensors, CPU float32, plus a `meta` entry
recording the checkpoint revision and shapes. Backend development tooling can
compare candidate outputs with these tensors using ``vla_edge.gates.parity``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from ..config import EMBODIMENTS, get_embodiment

log = logging.getLogger("vla_edge.capture")

#: Attribute paths hooked on the loaded model. Each entry is
#: (name, dotted-path-from-backbone). Missing paths are reported, not fatal:
#: checkpoint revisions move things, and a partial capture still gates the
#: stages it covers.
_HOOK_POINTS = [
    ("vision", "vision_backbone"),
    ("prefill", "transformer"),
    ("action_expert", "action_expert"),
]


def _resolve(root, dotted: str):
    obj = root
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _backbone(model):
    """The inner module that owns the stages, under any wrapper."""
    core = model
    seen = 0
    while not hasattr(core, "generate_actions_from_inputs") and hasattr(core, "model"):
        core = core.model
        seen += 1
        if seen > 8:
            break
    return core


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embodiment", default="bimanual-yam", choices=sorted(EMBODIMENTS))
    ap.add_argument("--checkpoint", default=None,
                    help="override the embodiment's repo id or local path")
    ap.add_argument("--out", default="capture.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--instruction", default="pick up the object")
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torch

    from ..checkpoint import load_checkpoint
    from ..pipeline import to_pil

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    emb = get_embodiment(args.embodiment)
    repo = args.checkpoint or emb.repo_id

    model, processor, local_dir = load_checkpoint(repo, args.device, args.dtype)
    core = _backbone(model)
    checkpoint_dir = Path(local_dir)
    revision = (
        checkpoint_dir.name if checkpoint_dir.parent.name == "snapshots" else None
    )
    repo_path = Path(str(repo)).expanduser()
    source = "<local-checkpoint>" if repo_path.exists() else str(repo)

    captured: dict[str, torch.Tensor] = {}
    handles = []

    def _save(prefix: str):
        def hook(_module, inputs, output):
            def store(tag, val):
                if torch.is_tensor(val):
                    captured[tag] = val.detach().to("cpu", torch.float32)
            for i, val in enumerate(inputs):
                store(f"{prefix}.in{i}", val)
            if isinstance(output, (tuple, list)):
                for i, val in enumerate(output):
                    store(f"{prefix}.out{i}", val)
            else:
                store(f"{prefix}.out", output)
        return hook

    for name, path in _HOOK_POINTS:
        mod = _resolve(core, path)
        if mod is None:
            log.warning("hook point %r (%s) not found on this checkpoint "
                        "revision; stages downstream of it will not be gated",
                        name, path)
            continue
        handles.append(mod.register_forward_hook(_save(name)))

    # Deterministic inputs so two captures of the same checkpoint agree.
    rng = np.random.default_rng(args.seed)
    frame = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    images = [to_pil(frame) for _ in emb.camera_names]
    state = rng.standard_normal(emb.state_dim).astype(np.float32)

    torch.manual_seed(args.seed)
    with torch.inference_mode():
        out = model.predict_action(
            processor=processor,
            images=images,
            task=args.instruction,
            state=state,
            norm_tag=emb.norm_tag,
            inference_action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=emb.default_num_steps,
            normalize_language=True,
            enable_cuda_graph=False,
        )

    for handle in handles:
        handle.remove()

    actions = out.actions
    if torch.is_tensor(actions):
        actions = actions.detach().to("cpu", torch.float32)
    captured["final.actions"] = torch.as_tensor(np.asarray(actions))

    payload = {
        "tensors": captured,
        "meta": {
            "embodiment": emb.name,
            "repo_id": source,
            "revision": revision,
            "dtype": args.dtype,
            "num_steps": emb.default_num_steps,
            "seed": args.seed,
            "instruction": args.instruction,
            "image_hw": [args.height, args.width],
            "shapes": {k: list(v.shape) for k, v in captured.items()},
        },
    }
    torch.save(payload, args.out)

    log.info("captured %d tensors -> %s", len(captured), args.out)
    for key in sorted(captured):
        log.info("  %-28s %s", key, tuple(captured[key].shape))
    if not captured:
        raise SystemExit(
            "no tensors captured: none of the hook points resolved. The "
            "checkpoint layout has changed; update _HOOK_POINTS."
        )


if __name__ == "__main__":
    main()
