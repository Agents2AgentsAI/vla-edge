"""Package the checkpoint subset used by a TensorRT engine set.

Usage:

    python scripts/package_trt_host.py \
        --checkpoint allenai/MolmoAct2-BimanualYAM \
        --out /path/to/bundle/host/yam \
        --with-action-expert

The action expert is needed only when an engine set ships the optional
compiled ``flow/`` package. The fallback ``action_flow.plan`` already embeds
those weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HOST_FILES = (
    "chat_template.jinja",
    "config.json",
    "configuration_molmoact2.py",
    "generation_config.json",
    "image_processing_molmoact2.py",
    "inference.py",
    "modeling_molmoact2.py",
    "norm_stats.json",
    "processing_molmoact2.py",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_processing_molmoact2.py",
)


def _resolve_checkpoint(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_dir():
        return candidate.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=value))


def _snapshot_revision(checkpoint: Path) -> str | None:
    if checkpoint.parent.name == "snapshots":
        return checkpoint.name
    return None


def _snapshot_repo_id(checkpoint: Path) -> str | None:
    if checkpoint.parent.name != "snapshots":
        return None
    cache_dir = checkpoint.parent.parent.name
    if not cache_dir.startswith("models--"):
        return None
    return cache_dir.removeprefix("models--").replace("--", "/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--repo-id",
        default=None,
        help="checkpoint repository ID recorded in host.json. Required when "
        "a local checkpoint is not in the standard Hugging Face cache layout",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-action-expert", action="store_true")
    args = ap.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from vla_edge.backends.tensorrt.host import (
        ACTION_EXPERT_PREFIX,
        EMBEDDING_PREFIX,
        HOST_FORMAT_VERSION,
        HOST_MANIFEST_NAME,
        HOST_WEIGHTS_NAME,
    )
    from vla_edge.checkpoint import _patch_modeling_for_bf16

    checkpoint_arg = Path(args.checkpoint)
    checkpoint = _resolve_checkpoint(args.checkpoint)
    repo_id = args.repo_id
    if repo_id is None and not checkpoint_arg.is_dir():
        repo_id = args.checkpoint
    if repo_id is None:
        repo_id = _snapshot_repo_id(checkpoint)
    if repo_id is None:
        raise SystemExit(
            "--repo-id is required when --checkpoint is a local directory"
        )
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty host runtime directory: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)

    for name in HOST_FILES:
        source = checkpoint / name
        if not source.is_file():
            raise SystemExit(f"checkpoint is missing required host file: {source}")
        shutil.copy2(source, out / name)
    _patch_modeling_for_bf16(str(out))

    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"checkpoint is missing {index_path.name}: {checkpoint}")
    weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    prefixes = [EMBEDDING_PREFIX]
    if args.with_action_expert:
        prefixes.append(ACTION_EXPERT_PREFIX)
    names = sorted(name for name in weight_map if name.startswith(tuple(prefixes)))
    if not names:
        raise SystemExit("checkpoint has no parameters for the TensorRT host runtime")

    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)
    state: dict[str, torch.Tensor] = {}
    for shard, shard_names in sorted(by_shard.items()):
        path = checkpoint / shard
        if not path.is_file():
            raise SystemExit(f"checkpoint is missing weight shard: {path}")
        with safe_open(path, framework="pt", device="cpu") as weights:
            for name in shard_names:
                state[name] = weights.get_tensor(name).to(torch.bfloat16).contiguous()
        print(f"loaded {len(shard_names)} host tensors from {shard}", flush=True)

    weights_path = out / HOST_WEIGHTS_NAME
    revision = _snapshot_revision(checkpoint)
    save_file(
        state,
        weights_path,
        metadata={
            "format": "pt",
            "source": repo_id,
            "revision": revision or "unknown",
        },
    )
    weight_bytes = sum(t.numel() * t.element_size() for t in state.values())
    manifest = {
        "format_version": HOST_FORMAT_VERSION,
        "repo_id": repo_id,
        "revision": revision,
        "dtype": "bfloat16",
        "parameter_prefixes": prefixes,
        "parameters": len(state),
        "weight_bytes": weight_bytes,
    }
    (out / HOST_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}: {len(state)} parameters, {weight_bytes / (1 << 30):.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
