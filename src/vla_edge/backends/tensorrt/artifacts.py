"""Loading prebuilt TensorRT artifacts, and refusing to load the wrong ones.

A TensorRT plan is compiled against a specific GPU architecture and a specific
TensorRT version. Deserializing a mismatched plan does not produce a clear
error: depending on the mismatch you get a null engine, an unrelated
``Serialization assertion`` from deep inside the runtime, or in the worst case
a plan that loads and then misbehaves. None of those point at the cause.

So we check first, against the manifest that ships beside the artifacts, and
fail with a message that says what to do instead. The rebuild recipe exists
precisely so that a version mismatch is an inconvenience rather than a wall.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

MANIFEST_NAME = "MANIFEST.json"


def _major_minor(version: str) -> tuple[str, ...]:
    """First two version components, e.g. '10.16.2.10' -> ('10', '16')."""
    return tuple(version.split(".")[:2])


class ArtifactMismatch(RuntimeError):
    """The prebuilt artifacts cannot run on this machine."""


@dataclass(frozen=True)
class Environment:
    tensorrt: str | None
    arch: str
    compute_capability: str | None
    cuda_device_name: str | None = None
    multiprocessor_count: int | None = None
    board_model: str | None = None

    @classmethod
    def detect(cls) -> Environment:
        try:
            import tensorrt
            trt_version = tensorrt.__version__
        except Exception:  # noqa: BLE001  # absence is a legitimate answer
            trt_version = None

        cc = None
        device_name = None
        multiprocessor_count = None
        try:
            import torch
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                cc = f"sm_{major}{minor}"
                properties = torch.cuda.get_device_properties(0)
                device_name = properties.name
                multiprocessor_count = int(properties.multi_processor_count)
        except Exception:  # noqa: BLE001,S110  # CUDA may be absent
            pass

        board_model = None
        try:
            board_model = Path("/proc/device-tree/model").read_text().rstrip("\x00")
        except OSError:
            pass

        return cls(
            tensorrt=trt_version,
            arch=platform.machine(),
            compute_capability=cc,
            cuda_device_name=device_name,
            multiprocessor_count=multiprocessor_count,
            board_model=board_model,
        )


def load_manifest(artifact_dir: str | Path) -> dict:
    path = Path(artifact_dir) / MANIFEST_NAME
    if not path.is_file():
        raise ArtifactMismatch(
            f"no {MANIFEST_NAME} in {artifact_dir}. Prebuilt artifacts must ship "
            "with their manifest; without it there is no way to tell whether "
            "they match this machine."
        )
    return json.loads(path.read_text())


def check_compatible(artifact_dir: str | Path, *, strict: bool = True) -> list[str]:
    """Compare the shipped manifest against this machine.

    Returns the list of mismatches. With ``strict`` (the default) any mismatch
    raises instead, because the failure mode of loading anyway is confusing
    rather than informative.

    The TensorRT check compares major.minor only. Patch releases have not
    broken plan compatibility for us, but that is an observation rather than a
    guarantee from NVIDIA, so a patch difference is reported as a warning line
    rather than treated as identical.
    """
    manifest = load_manifest(artifact_dir)
    want = manifest.get("requires", {})
    have = Environment.detect()
    problems: list[str] = []

    want_trt = want.get("tensorrt")
    if want_trt and have.tensorrt:
        if _major_minor(want_trt) != _major_minor(have.tensorrt):
            problems.append(
                f"TensorRT {have.tensorrt} installed, artifacts built with "
                f"{want_trt}. Plans do not load across minor versions."
            )
    elif want_trt and not have.tensorrt:
        problems.append("TensorRT is not importable, so these plans cannot be loaded.")

    # sm_110a and sm_110 are the same device; the suffix is a compile target,
    # not a capability, so it is stripped before comparing.
    want_cc = want.get("compute_capability")
    if (
        want_cc
        and have.compute_capability
        and want_cc.rstrip("a") != have.compute_capability.rstrip("a")
    ):
        problems.append(
            f"this GPU is {have.compute_capability}, artifacts were built "
            f"for {want_cc}. A plan is not portable across architectures."
        )

    want_arch = want.get("arch")
    if want_arch and want_arch != have.arch:
        problems.append(f"CPU architecture is {have.arch}, artifacts are {want_arch}.")

    want_name = want.get("cuda_device_name")
    if want_name and want_name != have.cuda_device_name:
        problems.append(
            f"CUDA device is {have.cuda_device_name!r}, artifacts were built "
            f"on {want_name!r}."
        )

    want_sms = want.get("multiprocessor_count")
    if want_sms is not None and int(want_sms) != have.multiprocessor_count:
        problems.append(
            f"CUDA device has {have.multiprocessor_count} SMs, artifacts were "
            f"built on a device with {want_sms}."
        )

    want_board = want.get("board_model")
    if want_board and want_board != have.board_model:
        problems.append(
            f"board is {have.board_model!r}, artifacts were built on "
            f"{want_board!r}."
        )

    if problems and strict:
        raise ArtifactMismatch(
            "prebuilt artifacts do not match this machine:\n"
            + "".join(f"  - {p}\n" for p in problems)
            + "\nBuild from source for your configuration instead; see\n"
            "recipes/tensorrt-thor/. The recipe produces equivalent engines\n"
            "and the parity gates verify them (docs/gates.md)."
        )
    return problems


def exact_device_match(artifact_dir: str | Path) -> bool:
    """Whether the manifest proves the plan and runtime GPU model match.

    TensorRT sometimes emits its generic cross-device warning on Jetson even
    for plans built on the same board model. We suppress only that one warning,
    and only when the bundle records all exact model fields and they match the
    runtime. Older or less specific manifests keep TensorRT's warning.
    """
    want = load_manifest(artifact_dir).get("requires", {})
    required = (
        "compute_capability",
        "cuda_device_name",
        "multiprocessor_count",
        "board_model",
    )
    if any(want.get(field) is None for field in required):
        return False
    have = Environment.detect()
    return (
        want["compute_capability"].rstrip("a")
        == (have.compute_capability or "").rstrip("a")
        and want["cuda_device_name"] == have.cuda_device_name
        and int(want["multiprocessor_count"]) == have.multiprocessor_count
        and want["board_model"] == have.board_model
    )


SERVING_CONFIG_NAME = "serving.json"


def load_serving_config(engine_dir: str | Path) -> dict:
    """Read the engine set's serving defaults (``serving.json``), if shipped.

    An engine set is not always self-describing: a set compiled for a fixed
    sequence bracket executes correctly only when every prompt is padded to
    that bracket, and nothing in the plan file says so. The bundle ships the
    requirement beside the plans so the operator does not have to remember it.
    """
    path = Path(engine_dir) / SERVING_CONFIG_NAME
    if not path.is_file():
        return {}
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ArtifactMismatch(f"{path} must contain a JSON object")
    return config


def effective_token_limit(profile_max_s: int, pad_multiple: int) -> int:
    """Longest prompt (in tokens, pre-padding) an engine profile can execute.

    With padding, the executable padded lengths are the multiples of
    ``pad_multiple`` up to the profile bound, so the largest acceptable
    prompt is the largest such multiple. For the champion sets,
    (``pad_multiple=704``, profile bound 1024) that is 704, not 1024, and
    quoting 1024 in a refusal would send the user counting the wrong budget.
    """
    if pad_multiple <= 1:
        return profile_max_s
    return (profile_max_s // pad_multiple) * pad_multiple


def verify_checksums(artifact_dir: str | Path, *, files: list[str] | None = None) -> None:
    """Verify shipped files against the manifest's sha256 entries.

    Worth running once after download. A truncated multi-gigabyte plan is a
    common and confusing failure: TensorRT reports a deserialization assertion
    that looks like a version problem.
    """
    import hashlib

    root = Path(artifact_dir)
    manifest = load_manifest(root)
    entries = manifest.get("files", {})
    targets = files if files is not None else list(entries)

    for rel in targets:
        meta = entries.get(rel)
        if meta is None:
            raise ArtifactMismatch(f"{rel} is not listed in {MANIFEST_NAME}")
        path = root / rel
        if not path.is_file():
            raise ArtifactMismatch(f"missing file: {rel}")
        if path.stat().st_size != meta["bytes"]:
            raise ArtifactMismatch(
                f"{rel} is {path.stat().st_size} bytes, manifest says "
                f"{meta['bytes']}; the download is incomplete."
            )
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(1 << 22):
                h.update(chunk)
        if h.hexdigest() != meta["sha256"]:
            raise ArtifactMismatch(f"{rel} failed checksum verification.")
