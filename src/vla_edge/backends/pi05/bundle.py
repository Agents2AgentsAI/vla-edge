"""Verified, portable Pi0.5 release descriptors."""
from pathlib import Path

from ..abcvla.bundle import read_json, verify_file


class Pi05Error(ValueError):
    pass

def load_bundle(directory, policy, embodiment):
    root = Path(directory).expanduser().resolve(strict=True)
    desc = read_json(root / "pi05-serving.json")
    if desc.get("release_status") == "blocked":
        raise Pi05Error("this bundle has not passed numerical qualification; see VALIDATION.json")
    expected = {
        "schema_version": 1, "kind": "vla-edge-pi05-serving",
        "policy": policy.name, "model_family": "pi05",
        "repo_id": policy.repo_id, "norm_tag": policy.norm_tag,
        "embodiment": embodiment.name, "state_dim": embodiment.state_dim,
        "action_dim": embodiment.action_dim,
        "camera_names": list(embodiment.camera_names),
        "action_horizon": policy.action_horizon, "steps": 10,
        "precision": "bfloat16", "model_action_dim": 32,
        "prefix_length": 968, "language_length": 200,
        "discrete_state_input": policy.embodiment == "bimanual-yam",
    }
    for key, value in expected.items():
        if desc.get(key) != value:
            raise Pi05Error(f"Pi0.5 bundle {key}: expected {value!r}, got {desc.get(key)!r}")
    if policy.default_num_steps != 10:
        raise Pi05Error("Pi0.5 releases require ten diffusion steps")
    desc["assets"] = {
        name: verify_file(root, declaration, name)
        for name, declaration in desc["host_assets"].items()
    }
    if set(desc["assets"]) != {"embedding", "tokenizer", "norm_stats"}:
        raise Pi05Error("Pi0.5 host assets must be embedding, tokenizer and norm_stats")
    for stage in ("vision", "prefill", "flow"):
        entry = desc["engines"][stage]
        entry["resolved"] = verify_file(root, entry, stage)
        for plugin in entry["plugins"]:
            plugin["resolved"] = verify_file(root, plugin, f"{stage} plugin")
    return desc
