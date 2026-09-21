"""Portable, SHA-bound ABC serving bundle loading (no CUDA imports)."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any


class AbcVlaError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AbcVlaError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise AbcVlaError(f"expected an object in {path}")
    return value


def resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise AbcVlaError(
            f"bundle dependency escapes its root or is not a file: {value}"
        )
    return path


def verify_file(root: Path, declaration: dict, label: str) -> Path:
    if not isinstance(declaration, dict) or not isinstance(
        declaration.get("path"), str
    ):
        raise AbcVlaError(f"missing file declaration: {label}")
    path = resolve(root, declaration["path"])
    if path.stat().st_size != declaration.get("bytes") or sha256(
        path
    ) != declaration.get("sha256"):
        raise AbcVlaError(f"{label} size or SHA-256 mismatch: {path}")
    return path


def load_bundle(directory: str | Path, policy: Any, embodiment: Any) -> dict:
    root = Path(directory).expanduser().resolve(strict=True)
    desc = read_json(root / "abcvla-serving.json")
    expected = {
        "schema_version": 2,
        "kind": "vla-edge-abcvla-serving",
        "model_family": "abcvla",
        "policy": policy.name,
        "embodiment": embodiment.name,
        "repo_id": policy.repo_id,
        "norm_tag": policy.norm_tag,
        "action_horizon": 30,
        "action_dim": 14,
        "state_dim": 14,
        "camera_names": ["top_cam", "left_cam", "right_cam"],
        "steps": 10,
        "gripper": {
            "indices": [6, 13],
            "convention": "closed_0_open_1",
            "state_source": "measured",
        },
    }
    for key, value in expected.items():
        if desc.get(key) != value:
            raise AbcVlaError(
                f"ABC bundle {key}: expected {value!r}, got {desc.get(key)!r}"
            )
    if (
        policy.action_horizon != 30
        or policy.default_num_steps != 10
        or policy.action_space != "absolute"
        or policy.gripper_state != "measured"
        or policy.gripper_convention != "closed_0_open_1"
        or tuple(embodiment.camera_names) != tuple(expected["camera_names"])
        or embodiment.gripper_convention != "closed_0_open_1"
    ):
        raise AbcVlaError(
            "ABC checkpoint does not match the selected robot/policy contract"
        )
    desc["root"] = root
    desc["config_path"] = verify_file(
        root, desc["checkpoint"]["config"], "checkpoint config"
    )
    desc["config"] = read_json(desc["config_path"])
    config = desc["config"]
    if (
        config.get("family") != "abcvla"
        or config.get("diffusion_steps") != 10
        or config.get("backbone_dtype") != "bfloat16"
        or config.get("head_dtype") != "float32"
        or config["model_config"]["camera_keys"] != ["top", "left", "right"]
        or config["model_config"]["backbone"]["image_size"] != 224
        or config["model_config"]["backbone"]["fixed_seq_len"] != 256
        or config["model_config"]["dit"]["chunk_length"] != 30
        or config["model_config"]["dit"]["state_dim"] != 14
        or config["model_config"]["dit"]["action_dim"] != 14
    ):
        raise AbcVlaError("unsupported ABC checkpoint geometry or precision")
    return desc


def install_upstream(desc: dict) -> None:
    root = desc["root"]
    for relative, declaration in desc["upstream"]["files"].items():
        verify_file(root, declaration, f"ABC source {relative}")
    source = (root / desc["upstream"]["root"]).resolve(strict=True)
    loaded = sys.modules.get("abc_minimal")
    if loaded is not None:
        origin = Path(loaded.__file__).resolve().parent
        if origin != source / "abc_minimal":
            # Reusing identical upstream sources is safe for reference comparisons.
            for name, module in tuple(sys.modules.items()):
                if name.startswith("abc_minimal") and getattr(module, "__file__", None):
                    path = Path(module.__file__).resolve()
                    try:
                        rel = path.relative_to(origin.parent).as_posix()
                        expected = desc["upstream"]["files"][rel]["sha256"]
                    except (ValueError, KeyError) as exc:
                        raise AbcVlaError(
                            "a different ABC source tree is already imported"
                        ) from exc
                    if sha256(path) != expected:
                        raise AbcVlaError(
                            f"imported ABC source differs from bundle: {path}"
                        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    importlib.import_module("abc_minimal")


def engine_declaration(desc: dict, raw: dict, label: str) -> dict:
    root = desc["root"]
    entry = dict(raw)
    entry["path"] = str(verify_file(root, raw, f"{label} engine"))
    metadata_path = verify_file(root, raw["metadata"], f"{label} metadata")
    metadata = read_json(metadata_path)
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "vla-edge-engine"
        or metadata.get("sha256") != raw["sha256"]
        or metadata.get("bytes") != raw["bytes"]
        or metadata.get("file") != Path(raw["path"]).name
    ):
        raise AbcVlaError(f"{label} metadata does not match its released engine")
    entry["metadata"] = str(metadata_path)
    entry["metadata_sha256"] = raw["metadata"]["sha256"]
    entry["plugins"] = []
    for i, plugin in enumerate(raw.get("plugins", [])):
        resolved = dict(plugin)
        resolved["path"] = str(verify_file(root, plugin, f"{label} plugin {i}"))
        entry["plugins"].append(resolved)
    return entry
