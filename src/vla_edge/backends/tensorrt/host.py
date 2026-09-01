"""Lightweight checkpoint host for TensorRT execution.

TensorRT plans contain the vision, language-backbone, and fallback flow
weights. The serving path still needs the checkpoint's processor, host-side
action semantics, token embeddings, and, when the compiled flow package is
usable, action-expert weights. Loading the entire PyTorch checkpoint duplicates
the two largest compiled stages and makes a local engine bundle depend on a
second 22 GB download.

This module instantiates the upstream model structure on the ``meta`` device
and materializes only the parameters that the TensorRT path executes. Released
bundles carry those parameters and the processor files in a compact host
runtime. A source-built engine without that directory can still resolve the
same subset from the upstream checkpoint.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ...checkpoint import _patch_modeling_for_bf16, load_processor
from .artifacts import MANIFEST_NAME, check_compatible, load_serving_config

log = logging.getLogger(__name__)

HOST_MANIFEST_NAME = "host.json"
HOST_WEIGHTS_NAME = "host.safetensors"
HOST_FORMAT_VERSION = 1

EMBEDDING_PREFIX = "model.transformer.wte."
ACTION_EXPERT_PREFIX = "model.action_expert."

# Everything needed by AutoConfig, the dynamic model class, and the processor.
# Weight shards are resolved separately from model.safetensors.index.json.
UPSTREAM_HOST_FILES = (
    "chat_template.jinja",
    "config.json",
    "configuration_molmoact2.py",
    "generation_config.json",
    "image_processing_molmoact2.py",
    "inference.py",
    "model.safetensors.index.json",
    "modeling_molmoact2.py",
    "norm_stats.json",
    "processing_molmoact2.py",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_processing_molmoact2.py",
)


class HostRuntimeError(RuntimeError):
    """The local TensorRT host runtime is absent, incomplete, or mismatched."""


def required_prefixes(*, action_expert: bool) -> tuple[str, ...]:
    prefixes = [EMBEDDING_PREFIX]
    if action_expert:
        prefixes.append(ACTION_EXPERT_PREFIX)
    return tuple(prefixes)


def _flow_needs_weights(engine_dir: Path) -> bool:
    """Whether this process can load the bundle's compiled flow package."""
    from .flow import is_compatible

    return is_compatible(engine_dir)


def _validate_host_manifest(host_dir: Path, repo_id: str) -> dict[str, Any]:
    path = host_dir / HOST_MANIFEST_NAME
    if not path.is_file():
        raise HostRuntimeError(
            f"local TensorRT host runtime has no {path.name}: {path}"
        )
    payload = json.loads(path.read_text())
    version = payload.get("format_version")
    if version != HOST_FORMAT_VERSION:
        raise HostRuntimeError(
            f"{path} has host format {version!r}; this runtime needs "
            f"{HOST_FORMAT_VERSION}"
        )
    packaged_repo = payload.get("repo_id")
    if packaged_repo != repo_id:
        raise HostRuntimeError(
            f"{path} is for {packaged_repo!r}, but this embodiment uses {repo_id!r}"
        )
    if payload.get("dtype") != "bfloat16":
        raise HostRuntimeError(
            f"{path} declares dtype {payload.get('dtype')!r}; TensorRT host "
            "weights must be bfloat16"
        )
    weights = host_dir / HOST_WEIGHTS_NAME
    if not weights.is_file():
        raise HostRuntimeError(f"local TensorRT host runtime is missing {weights}")
    return payload


def resolve_local_host(engine_dir: str | Path, repo_id: str) -> Path | None:
    """Resolve a compact host directory declared by ``serving.json``."""
    engine_dir = Path(engine_dir).resolve()
    serving = load_serving_config(engine_dir)
    declared = serving.get("host_dir")
    if declared is None:
        candidate = engine_dir / "host"
        if not candidate.is_dir():
            return None
    else:
        candidate = Path(str(declared))
        if not candidate.is_absolute():
            candidate = engine_dir / candidate
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise HostRuntimeError(
            f"{engine_dir / 'serving.json'} declares host_dir={declared!r}, "
            f"but {candidate} is not a directory"
        )
    _validate_host_manifest(candidate, repo_id)
    return candidate


def _select_upstream_shards(
    local_dir: Path, prefixes: tuple[str, ...]
) -> tuple[str, ...]:
    index_path = local_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise HostRuntimeError(
            f"upstream checkpoint has no {index_path.name}: {local_dir}"
        )
    weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    names = [name for name in weight_map if name.startswith(prefixes)]
    if not names:
        raise HostRuntimeError(
            f"upstream checkpoint {local_dir} has none of the TensorRT host weights"
        )
    return tuple(sorted({weight_map[name] for name in names}))


def _resolve_upstream_source(repo_id: str, prefixes: tuple[str, ...]) -> Path:
    if os.path.isdir(repo_id):
        return Path(repo_id).resolve()

    from huggingface_hub import snapshot_download

    # Resolve the small files and index first, then only the shards containing
    # parameters used by this serving path. This is the compatibility path for
    # locally built plans; released bundles should carry a compact host runtime
    # and never enter here.
    local_dir = Path(
        snapshot_download(repo_id=repo_id, allow_patterns=list(UPSTREAM_HOST_FILES))
    )
    shards = _select_upstream_shards(local_dir, prefixes)
    local_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[*UPSTREAM_HOST_FILES, *shards],
        )
    )
    return local_dir


def _expected_parameter_names(model: Any, prefixes: tuple[str, ...]) -> set[str]:
    return {
        name
        for name, _parameter in model.named_parameters()
        if name.startswith(prefixes)
    }


def _set_parameter(model: Any, name: str, value: Any) -> None:
    import torch

    parent_name, leaf = name.rsplit(".", 1)
    module = model.get_submodule(parent_name)
    old = module._parameters.get(leaf)
    if old is None:
        raise HostRuntimeError(f"host weight {name!r} is not a model parameter")
    module._parameters[leaf] = torch.nn.Parameter(
        value, requires_grad=old.requires_grad
    )


def _load_compact_weights(
    model: Any,
    host_dir: Path,
    names: set[str],
    device: str,
    torch_dtype: Any,
) -> int:
    from safetensors import safe_open

    path = host_dir / HOST_WEIGHTS_NAME
    loaded_bytes = 0
    with safe_open(path, framework="pt", device="cpu") as weights:
        available = set(weights.keys())
        missing = sorted(names - available)
        if missing:
            sample = ", ".join(missing[:3])
            raise HostRuntimeError(
                f"{path} is missing {len(missing)} required parameters "
                f"(first: {sample})"
            )
        for name in sorted(names):
            value = weights.get_tensor(name).to(device=device, dtype=torch_dtype)
            _set_parameter(model, name, value)
            loaded_bytes += value.numel() * value.element_size()
    return loaded_bytes


def _load_upstream_weights(
    model: Any,
    local_dir: Path,
    names: set[str],
    device: str,
    torch_dtype: Any,
) -> int:
    from safetensors import safe_open

    index = json.loads((local_dir / "model.safetensors.index.json").read_text()).get(
        "weight_map", {}
    )
    missing = sorted(names - set(index))
    if missing:
        sample = ", ".join(missing[:3])
        raise HostRuntimeError(
            f"upstream checkpoint is missing {len(missing)} required "
            f"parameters (first: {sample})"
        )
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(index[name], []).append(name)

    loaded_bytes = 0
    for shard, shard_names in sorted(by_shard.items()):
        path = local_dir / shard
        if not path.is_file():
            raise HostRuntimeError(f"upstream checkpoint is missing shard {path}")
        with safe_open(path, framework="pt", device="cpu") as weights:
            for name in sorted(shard_names):
                value = weights.get_tensor(name).to(device=device, dtype=torch_dtype)
                _set_parameter(model, name, value)
                loaded_bytes += value.numel() * value.element_size()
    return loaded_bytes


def _patch_input_mover(model: Any, torch_dtype: Any) -> None:
    import torch

    def _move_and_cast(inputs: Any, device: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                value = value.to(device)
                if value.is_floating_point() and value.dtype != torch_dtype:
                    value = value.to(torch_dtype)
            out[key] = value
        return out

    model._move_inputs_to_device = _move_and_cast


def _ensure_materialized(model: Any, names: Iterable[str], device: str) -> None:
    import torch

    expected = torch.device(device)
    if expected.type == "cuda" and expected.index is None:
        expected = torch.device("cuda", torch.cuda.current_device())
    expected_device = str(expected)
    wrong = [
        name
        for name in names
        if str(model.get_parameter(name).device) != expected_device
    ]
    if wrong:
        sample = ", ".join(wrong[:3])
        raise HostRuntimeError(
            f"{len(wrong)} TensorRT host parameters were not materialized on "
            f"{expected_device} (first: {sample})"
        )


def load_checkpoint(
    repo_id: str,
    engine_dir: str | Path,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
) -> tuple[Any, Any, str]:
    """Load only the PyTorch state executed around the TensorRT plans."""
    if dtype != "bfloat16":
        raise HostRuntimeError(
            "the TensorRT artifacts require --dtype bfloat16; their prefill "
            f"inputs were built for bf16, got {dtype!r}"
        )

    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText

    engine_dir = Path(engine_dir).resolve()
    for artifact_dir in (engine_dir, engine_dir.parent):
        if (artifact_dir / MANIFEST_NAME).is_file():
            check_compatible(artifact_dir)
            break
    need_action_expert = _flow_needs_weights(engine_dir)
    prefixes = required_prefixes(action_expert=need_action_expert)
    local_host = resolve_local_host(engine_dir, repo_id)
    if local_host is not None:
        local_dir = local_host
        source_kind = "bundle"
    else:
        log.warning(
            "%s has no compact host runtime; resolving only the required "
            "weights from %s. Package a host runtime to make this engine set "
            "fully local.",
            engine_dir,
            repo_id,
        )
        local_dir = _resolve_upstream_source(repo_id, prefixes)
        source_kind = "upstream cache"

    _patch_modeling_for_bf16(str(local_dir))
    processor = load_processor(local_dir)
    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)

    names = _expected_parameter_names(model, prefixes)
    if source_kind == "bundle":
        loaded_bytes = _load_compact_weights(
            model, local_dir, names, device, torch.bfloat16
        )
    else:
        loaded_bytes = _load_upstream_weights(
            model, local_dir, names, device, torch.bfloat16
        )
    _ensure_materialized(model, names, device)

    model.eval()
    model._vla_edge_tensorrt_host = True
    model._vla_edge_torch_fallback = False
    model._vla_edge_dtype = torch.bfloat16
    _patch_input_mover(model, torch.bfloat16)
    log.info(
        "loaded local TensorRT host runtime from %s: %d parameters, %.2f GiB "
        "materialized; replaced vision and language weights remain unloaded",
        local_dir,
        len(names),
        loaded_bytes / (1 << 30),
    )
    return model, processor, str(local_dir)
