"""ABC-VLA TensorRT stage execution with verified release artifacts.

Static engines use CUDA graphs over persistent buffers. Dynamic prefill plans
bind the current prompt shape before each execution. Every request copies its
current images, state, prompt and noise; only input-independent layouts and
compiled execution graphs are cached.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..tensorrt.logger import deserialization_scope
from .device import detect_device, verify_device_requirements
from .geometry import IMAGE_SPANS, TEXT_START, VISION_TOKENS, native_positions


class TensorRTRuntimeError(RuntimeError):
    """A plan, binding, or execution result violates the engine contract."""


def _cuda_device(torch: Any) -> Any:
    return torch.device("cuda", torch.cuda.current_device())


def _read_json_nofollow(
    path: Path, *, maximum_bytes: int = 4 * 1024 * 1024
) -> tuple[dict[str, Any], str]:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TensorRTRuntimeError(f"cannot open metadata {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum_bytes:
            raise TensorRTRuntimeError(
                f"metadata is not a bounded non-empty regular file: {path}"
            )
        payload = os.pread(descriptor, info.st_size + 1, 0)
        if len(payload) != info.st_size:
            raise TensorRTRuntimeError(f"metadata changed while reading: {path}")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise TensorRTRuntimeError(
            f"cannot decode engine metadata {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TensorRTRuntimeError(f"engine metadata must contain an object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _deserialize_sha_bound_plan(
    path: str | Path, *, expected_sha256: str, trt: Any,
    verified_device: bool = False,
) -> tuple[Path, str, os.stat_result, Any, Any]:
    """Hash and deserialize one immutable open file description (no path re-open race)."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise TensorRTRuntimeError(f"TensorRT engine may not be a symlink: {requested}")
    resolved = requested.resolve(strict=True)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise TensorRTRuntimeError(
            f"cannot open TensorRT engine {resolved}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise TensorRTRuntimeError(
                f"TensorRT engine is not a non-empty regular file: {resolved}"
            )
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor, min(8 * 1024 * 1024, before.st_size - offset), offset
            )
            if not chunk:
                raise TensorRTRuntimeError(
                    f"short read while hashing TensorRT engine: {resolved}"
                )
            digest.update(chunk)
            offset += len(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise TensorRTRuntimeError(
                f"TensorRT engine SHA-256 mismatch: expected {expected_sha256}, got {actual}"
            )
        with (
            deserialization_scope(trt, verified_device=verified_device) as logger,
            mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ) as payload,
        ):
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(payload)
        after = os.fstat(descriptor)
        fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, name) != getattr(after, name) for name in fields):
            raise TensorRTRuntimeError(
                f"TensorRT engine changed during load: {resolved}"
            )
        named = os.stat(resolved, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (after.st_dev, after.st_ino):
            raise TensorRTRuntimeError(
                f"TensorRT engine path was replaced during load: {resolved}"
            )
    finally:
        os.close(descriptor)
    if engine is None:
        raise TensorRTRuntimeError(f"TensorRT could not deserialize {resolved}")
    return resolved, actual, before, runtime, engine


def _torch_dtype(torch: Any, dtype: Any) -> Any:
    mapping = {
        "DataType.FLOAT": torch.float32,
        "DataType.HALF": torch.float16,
        "DataType.BF16": torch.bfloat16,
        "DataType.INT8": torch.int8,
        "DataType.INT32": torch.int32,
        "DataType.INT64": torch.int64,
        "DataType.BOOL": torch.bool,
    }
    try:
        return mapping[str(dtype)]
    except KeyError as exc:
        raise TensorRTRuntimeError(
            f"unsupported TensorRT binding dtype: {dtype}"
        ) from exc


_PLUGIN_HANDLES: dict[str, Any] = {}


def _load_plugins(trt: Any, declaration: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Register declared plugin libraries unless their creators are already present.

    Registering a library twice can replace a plugin creator. Reuse an existing
    registration only when its initializer is already loaded.
    """

    loaded = []
    registry = trt.get_plugin_registry()
    for plugin in declaration.get("plugins", []) or []:
        path, digest, initialize = (
            plugin.get("path"),
            plugin.get("sha256"),
            plugin.get("initialize"),
        )
        creators = plugin.get("creators") or []
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or not isinstance(initialize, str)
        ):
            raise TensorRTRuntimeError(
                "plugin declaration needs path, sha256 and initialize"
            )
        present = bool(creators) and all(
            registry.get_creator(
                str(c["name"]), str(c.get("version", "1")), str(c.get("namespace", ""))
            )
            is not None
            for c in creators
        )
        if not present:
            # A process-global initializer identifies an already loaded library.
            try:
                present = getattr(ctypes.CDLL(None), initialize, None) is not None
            except (OSError, AttributeError):
                present = False
        if not present and digest not in _PLUGIN_HANDLES:
            resolved = Path(path).expanduser().resolve(strict=True)
            if hashlib.sha256(resolved.read_bytes()).hexdigest() != digest:
                raise TensorRTRuntimeError(f"plugin SHA-256 mismatch: {resolved}")
            handle = ctypes.CDLL(str(resolved), mode=ctypes.RTLD_GLOBAL)
            entry = getattr(handle, initialize)
            entry.restype = ctypes.c_int
            if entry():
                raise TensorRTRuntimeError(f"plugin initialization failed: {resolved}")
            _PLUGIN_HANDLES[digest] = handle
        loaded.append({"path": path, "sha256": digest, "initialize": initialize})
    return loaded


class _EngineRuntime:
    """One SHA-bound plan, its release metadata, direct enqueue and optional CUDA-graph replay."""

    stage = ""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        expected_sha256: str,
        expected_identity: Mapping[str, Any] | None = None,
    ) -> None:
        import tensorrt as trt
        import torch

        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise TensorRTRuntimeError(
                f"{self.stage} engine requires a SHA-256 identity"
            )
        identity = dict(expected_identity or {})
        metadata_path = identity.get("metadata")
        if not isinstance(metadata_path, str):
            raise TensorRTRuntimeError("engine declaration is missing release metadata")
        metadata_path = Path(metadata_path).resolve(strict=True)
        metadata, metadata_sha256 = _read_json_nofollow(metadata_path)
        if identity.get("metadata_sha256") != metadata_sha256:
            raise TensorRTRuntimeError("engine metadata SHA-256 mismatch")
        verified_device = verify_device_requirements(
            metadata.get("requires"), detect_device(torch, trt)
        )
        self._plugins = _load_plugins(trt, identity)
        path, sha256, plan_stat, runtime, engine = _deserialize_sha_bound_plan(
            engine_path, expected_sha256=expected_sha256, trt=trt,
            verified_device=verified_device,
        )
        if (
            metadata.get("schema_version") != 1
            or metadata.get("kind") != "vla-edge-engine"
            or metadata.get("sha256") != sha256
            or metadata.get("bytes") != plan_stat.st_size
            or metadata.get("file") != path.name
        ):
            raise TensorRTRuntimeError(f"{self.stage} metadata does not match the plan")
        context = engine.create_execution_context()
        if context is None:
            raise TensorRTRuntimeError(
                f"TensorRT could not create a {self.stage} execution context"
            )
        options = identity.get("runtime_options") or {}
        if not isinstance(options, Mapping) or set(options) - {"cuda_graph"}:
            raise TensorRTRuntimeError("runtime_options supports only cuda_graph")
        requested_graph = options.get("cuda_graph")
        self.cuda_graph = False
        self._trt, self._torch, self._runtime = trt, torch, runtime
        self.engine, self.ctx = engine, context
        self.engine_path, self.engine_sha256 = path, sha256
        self.engine_metadata, self.engine_metadata_sha256 = metadata, metadata_sha256
        self.device = _cuda_device(torch)
        self.names = [
            engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
        ]
        self.inputs = [
            name
            for name in self.names
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.outputs = [name for name in self.names if name not in self.inputs]
        self._bindings = [
            {
                "name": name,
                "mode": str(engine.get_tensor_mode(name)),
                "dtype": str(engine.get_tensor_dtype(name)),
                "shape": [int(v) for v in engine.get_tensor_shape(name)],
            }
            for name in self.names
        ]
        self.execution_count = 0
        self._stream = None
        self._graph = None
        self._static: dict[str, Any] = {}
        self._graph_shapes: dict[str, tuple[int, ...]] | None = None
        static_plan = all(
            min(binding["shape"], default=1) > 0 for binding in self._bindings
        )
        if requested_graph is True or (requested_graph is None and static_plan):
            if not static_plan:
                self.cuda_graph = (
                    True  # dynamic plan, explicitly requested: captured per bound shape
                )
            else:
                try:
                    self.cuda_graph = True
                    self._capture(
                        {
                            name: torch.zeros(
                                tuple(engine.get_tensor_shape(name)),
                                device=self.device,
                                dtype=self.dtype(name),
                            )
                            for name in self.inputs
                        }
                    )
                except Exception:
                    if requested_graph is True:
                        raise
                    self.cuda_graph = False
                    self._graph, self._static, self._graph_shapes = None, {}, None

    # -- identity -----------------------------------------------------------------------


    def synchronize(self) -> None:
        self._torch.cuda.synchronize(self.device)

    def dtype(self, name: str) -> Any:
        return _torch_dtype(self._torch, self.engine.get_tensor_dtype(name))

    def input_shape(self, name: str) -> tuple[int, ...]:
        return tuple(int(v) for v in self.ctx.get_tensor_shape(name))

    def output_shape(self, name: str) -> tuple[int, ...]:
        shape = tuple(int(v) for v in self.ctx.get_tensor_shape(name))
        if any(v < 0 for v in shape):
            raise TensorRTRuntimeError(
                f"{self.stage} output {name} is unresolved; bind the input shapes first"
            )
        return shape

    def _cast(self, name: str, value: Any) -> Any:
        torch = self._torch
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        return tensor.to(device=self.device, dtype=self.dtype(name)).contiguous()

    # -- execution ----------------------------------------------------------------------

    def _bind_shapes(self, values: Mapping[str, Any]) -> None:
        for name in self.inputs:
            shape = tuple(values[name].shape)
            if tuple(
                self.ctx.get_tensor_shape(name)
            ) != shape and not self.ctx.set_input_shape(name, shape):
                raise TensorRTRuntimeError(
                    f"TensorRT rejected {self.stage} input shape {name}={shape}"
                )

    def _enqueue(self, tensors: Mapping[str, Any]) -> None:
        for name in self.names:
            if not self.ctx.set_tensor_address(name, tensors[name].data_ptr()):
                raise TensorRTRuntimeError(
                    f"TensorRT rejected the {self.stage} address of {name}"
                )
        torch = self._torch
        current = torch.cuda.current_stream(self.device)
        if current.cuda_stream != 0:
            if not self.ctx.execute_async_v3(stream_handle=current.cuda_stream):
                raise TensorRTRuntimeError(f"TensorRT {self.stage} enqueue failed")
            return
        # TensorRT adds cudaStreamSynchronize calls when it is handed the default stream. Run the
        # plan on a runtime-owned stream that is ordered after the caller's work, and order the
        # caller's stream after it, so producers and consumers keep their single-stream semantics.
        if self._stream is None:
            self._stream = torch.cuda.Stream(self.device)
        self._stream.wait_stream(current)
        if not self.ctx.execute_async_v3(stream_handle=self._stream.cuda_stream):
            raise TensorRTRuntimeError(f"TensorRT {self.stage} enqueue failed")
        current.wait_stream(self._stream)

    def _capture(self, values: Mapping[str, Any]) -> None:
        torch = self._torch
        self._static = {name: values[name].clone() for name in self.inputs}
        self._bind_shapes(self._static)
        for name in self.outputs:
            self._static[name] = torch.empty(
                self.output_shape(name), device=self.device, dtype=self.dtype(name)
            )
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            self._enqueue(
                self._static
            )  # lazy TensorRT resources are created outside the capture
        side.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            self._enqueue(self._static)
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph.replay()  # capture records without executing
        torch.cuda.synchronize(self.device)
        self._graph = graph
        self._graph_shapes = {
            name: tuple(self._static[name].shape) for name in self.inputs
        }

    def execute(
        self, values: Mapping[str, Any], outputs: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run the plan once on device tensors that already have the binding dtypes."""

        torch = self._torch
        for name in self.inputs:
            value = values[name]
            if (
                not torch.is_tensor(value)
                or value.device != self.device
                or value.dtype != self.dtype(name)
                or not value.is_contiguous()
            ):
                raise TensorRTRuntimeError(
                    f"{self.stage} input {name} must be a contiguous {self.dtype(name)} tensor on {self.device}"
                )
        if self.cuda_graph:
            shapes = {name: tuple(values[name].shape) for name in self.inputs}
            if self._graph is None or shapes != self._graph_shapes:
                self._capture(values)
            for name in self.inputs:
                if values[name].data_ptr() != self._static[name].data_ptr():
                    self._static[name].copy_(values[name])
            self._graph.replay()
            self.execution_count += 1
            result = {name: self._static[name] for name in self.outputs}
            if outputs:
                for name, target in outputs.items():
                    target.copy_(result[name])
                    result[name] = target
            return result
        self._bind_shapes(values)
        tensors = dict(values)
        for name in self.outputs:
            target = (outputs or {}).get(name)
            shape, dtype = self.output_shape(name), self.dtype(name)
            if target is None:
                target = torch.empty(shape, device=self.device, dtype=dtype)
            elif (
                tuple(target.shape) != shape
                or target.dtype != dtype
                or target.device != self.device
                or not target.is_contiguous()
            ):
                raise TensorRTRuntimeError(
                    f"{self.stage} output buffer {name} must be a contiguous exact-shape/device/dtype tensor"
                )
            tensors[name] = target
        self._enqueue(tensors)
        self.execution_count += 1
        return {name: tensors[name] for name in self.outputs}

    def static_input(self, name: str) -> Any | None:
        """The runtime-owned graph input buffer, once captured (producers may write into it)."""

        return self._static.get(name) if self._graph is not None else None

    def close(self) -> None:
        if self.engine is None:
            return
        self.synchronize()
        self._graph = None
        self._static = {}
        self._stream = None
        # TensorRT contexts retain workspace and engine weight allocations.
        # Releasing only the CUDA graph keeps those allocations alive for an
        # otherwise closed in-process Pipeline.
        self.ctx = None
        self.engine = None
        self._runtime = None


# ---------------------------------------------------------------------------------------------


class VisionTensorRTRuntime(_EngineRuntime):
    stage = "vision"

    def prepare_inputs(
        self, method: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        del method
        return {"pixel_values": self._cast("pixel_values", arguments["pixel_values"])}

    def run(self, pixel_values: Any, *, output: Any | None = None) -> Any:
        values = {"pixel_values": self._cast("pixel_values", pixel_values)}
        return self.execute(values, {"vision": output} if output is not None else None)[
            "vision"
        ]

    encode_vision = run


def canonical_attention_bias(
    torch: Any, length: int, *, dtype: Any, device: Any
) -> Any:
    """Causal mask with bidirectional attention inside each image block, as additive ``dtype`` bias."""

    allowed = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    for start, stop in IMAGE_SPANS:
        allowed[start:stop, start:stop] = True
    bias = torch.where(allowed, 0.0, torch.finfo(dtype).min).to(dtype)
    return bias[None, None]


class PrefillTensorRTRuntime(_EngineRuntime):
    """``(inputs_embeds, attention_bias, position_ids) -> (cond, hidden)`` at the valid length.

    A plan may omit ``attention_bias`` and/or ``position_ids``: both are fixed functions of the
    prefix length for this model (causal + image-block mask, ``arange`` positions).  The runtime
    then refuses any request whose values differ from those canonical tensors.
    """

    stage = "prefill"

    def _check_canonical(self, name: str, value: Any, length: int) -> None:
        torch = self._torch
        value = value.to(self.device)
        if name == "position_ids":
            expected = torch.arange(length, device=self.device, dtype=value.dtype)[None]
            same = tuple(value.shape) == (1, length) and bool(
                torch.equal(value, expected)
            )
        else:
            expected = canonical_attention_bias(
                torch, length, dtype=torch.bfloat16, device=self.device
            )
            same = tuple(value.shape) == tuple(expected.shape) and bool(
                torch.equal(value.to(torch.bfloat16) < 0, expected < 0)
            )
        if not same:
            raise TensorRTRuntimeError(
                f"prefill plan bakes the canonical {name}; the request supplies a different one"
            )

    def prepare_inputs(
        self, method: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        del method
        prepared = {
            "inputs_embeds": self._cast("inputs_embeds", arguments["inputs_embeds"])
        }
        length = int(prepared["inputs_embeds"].shape[1])
        for name in ("attention_bias", "position_ids"):
            if name in self.inputs:
                prepared[name] = self._cast(name, arguments[name])
            else:
                self._check_canonical(name, arguments[name], length)
                prepared[name] = arguments[name]
        return prepared

    def run(
        self,
        inputs_embeds: Any,
        attention_bias: Any = None,
        position_ids: Any = None,
        *,
        outputs: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        supplied = {
            "inputs_embeds": inputs_embeds,
            "attention_bias": attention_bias,
            "position_ids": position_ids,
        }
        values = {name: self._cast(name, supplied[name]) for name in self.inputs}
        result = self.execute(values, outputs)
        return result["cond"], result["hidden"]

    prefill = run


class FlowTensorRTRuntime(_EngineRuntime):
    """Static unrolled sampler: ``(cond, state, noise, action_prefix, prefix_length) -> actions``."""

    stage = "flow"

    def __init__(
        self,
        engine_path: str | Path,
        *,
        expected_sha256: str,
        steps: int,
        expected_identity: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            engine_path,
            expected_sha256=expected_sha256,
            expected_identity=expected_identity,
        )
        self.steps = int(steps)
        shape = tuple(self.engine.get_tensor_shape("noise"))
        self._zero_prefix = self._torch.zeros(
            shape, device=self.device, dtype=self.dtype("action_prefix")
        )
        self._zero_length = self._torch.zeros(
            1, device=self.device, dtype=self.dtype("prefix_length")
        )

    def _context(self, cross_mask: Any) -> dict[str, Any]:
        if hasattr(cross_mask, "state"):
            cross_mask = {
                "state": cross_mask.state,
                "action_prefix": cross_mask.action_prefix,
                "prefix_length": cross_mask.prefix_length,
            }
        if not isinstance(cross_mask, Mapping) or cross_mask.get("state") is None:
            raise TensorRTRuntimeError("flow context must carry the normalized state")
        prefix, length = (
            cross_mask.get("action_prefix"),
            cross_mask.get("prefix_length"),
        )
        if (prefix is None) != (length is None):
            raise TensorRTRuntimeError(
                "action_prefix and prefix_length must be supplied together"
            )
        return {
            "state": self._cast("state", cross_mask["state"]),
            "action_prefix": self._zero_prefix
            if prefix is None
            else self._cast("action_prefix", prefix),
            "prefix_length": self._zero_length
            if length is None
            else self._cast("prefix_length", length).reshape(1),
        }

    def prepare_inputs(
        self, method: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        del method
        prepared = dict(arguments)
        prepared["cond"] = self._cast("cond", arguments["cond"])
        prepared["noise"] = self._cast("noise", arguments["noise"])
        prepared["cross_mask"] = self._context(arguments["cross_mask"])
        return prepared

    def run(
        self,
        cond: Any,
        state: Any,
        noise: Any,
        action_prefix: Any = None,
        prefix_length: Any = None,
        *,
        output: Any | None = None,
    ) -> Any:
        """Flat binding order of the plan; an absent prefix selects the unprefixed sampler."""

        context = self._context(
            {
                "state": state,
                "action_prefix": action_prefix,
                "prefix_length": prefix_length,
            }
        )
        values = {
            "cond": self._cast("cond", cond),
            "noise": self._cast("noise", noise),
            **context,
        }
        return self.execute(
            values, {"actions": output} if output is not None else None
        )["actions"]

    def denoise(
        self,
        cond: Any,
        hidden: Any = None,
        cross_mask: Any = None,
        noise: Any = None,
        steps: int | None = None,
    ) -> Any:
        """The seam call: ``cross_mask`` carries state and the optional real-time-chunking prefix."""

        del hidden
        if steps is not None and int(steps) != self.steps:
            raise TensorRTRuntimeError(
                f"flow plan is unrolled for {self.steps} steps, not {steps}"
            )
        context = self._context(cross_mask)
        return self.run(
            cond,
            context["state"],
            noise,
            context["action_prefix"],
            context["prefix_length"],
        )


class PipelineTensorRTRuntime:
    """Causal vision -> native prefix -> condition -> static flow pipeline on one CUDA stream."""

    def __init__(
        self,
        vision: VisionTensorRTRuntime,
        prefill: PrefillTensorRTRuntime,
        flow: FlowTensorRTRuntime,
    ) -> None:
        torch = prefill._torch
        self._torch = torch
        self.vision_runtime, self.prefill_runtime, self.flow_runtime = (
            vision,
            prefill,
            flow,
        )
        self.steps = flow.steps
        self.device = prefill.device
        if tuple(vision.engine.get_tensor_shape("vision"))[1] != VISION_TOKENS:
            raise TensorRTRuntimeError(
                "vision plan does not produce the 192-token span"
            )
        if vision.dtype("vision") != prefill.dtype("inputs_embeds"):
            raise TensorRTRuntimeError("vision/prefix binding dtype mismatch")
        if prefill.dtype("cond") != flow.dtype("cond"):
            raise TensorRTRuntimeError("prefill/flow condition dtype mismatch")
        self._prefix: Any = None
        self._orders: tuple[Any, Any] | None = None
        self.execution_count = 0


    def synchronize(self) -> None:
        self._torch.cuda.synchronize(self.device)

    def prepare_inputs(
        self, method: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        del method
        prepared = dict(arguments)
        prepared["pixel_values"] = self.vision_runtime._cast(
            "pixel_values", arguments["pixel_values"]
        )
        prepared["inputs_embeds"] = self.prefill_runtime._cast(
            "inputs_embeds", arguments["inputs_embeds"]
        )
        prefill = self.prefill_runtime.prepare_inputs("prefill", arguments)
        prepared["attention_bias"], prepared["position_ids"] = (
            prefill["attention_bias"],
            prefill["position_ids"],
        )
        prepared["noise"] = self.flow_runtime._cast("noise", arguments["noise"])
        prepared["cross_mask"] = self.flow_runtime._context(arguments["cross_mask"])
        return prepared

    def encode_vision(self, pixel_values: Any) -> Any:
        return self.vision_runtime.run(pixel_values)

    def assemble_prefix(self, vision: Any, inputs_embeds: Any) -> Any:
        """Scatter the live vision span and the template tail into the native-order prefix buffer."""

        torch = self._torch
        length = int(inputs_embeds.shape[1])
        if length <= TEXT_START or tuple(vision.shape[1:]) != (
            VISION_TOKENS,
            inputs_embeds.shape[2],
        ):
            raise TensorRTRuntimeError(
                "vision/template shapes cannot form the native prefix"
            )
        if self._prefix is None or tuple(self._prefix.shape) != tuple(
            inputs_embeds.shape
        ):
            self._prefix = torch.empty_like(inputs_embeds)
            image, other = native_positions(length)
            self._orders = (
                torch.as_tensor(image, device=self.device),
                torch.as_tensor(other, device=self.device),
            )
        image, other = self._orders
        self._prefix.index_copy_(1, image, vision)
        self._prefix.index_copy_(1, other, inputs_embeds[:, VISION_TOKENS:])
        return self._prefix

    def prefill(
        self, inputs_embeds: Any, attention_bias: Any, position_ids: Any
    ) -> tuple[Any, Any]:
        return self.prefill_runtime.run(inputs_embeds, attention_bias, position_ids)

    def denoise(
        self,
        cond: Any,
        hidden: Any,
        cross_mask: Any,
        noise: Any,
        steps: int | None = None,
    ) -> Any:
        return self.flow_runtime.denoise(cond, hidden, cross_mask, noise, steps)

    def run(
        self,
        pixel_values: Any,
        inputs_embeds: Any,
        attention_bias: Any,
        position_ids: Any,
        cross_mask: Any,
        noise: Any,
        steps: int | None = None,
    ) -> dict[str, Any]:
        vision = self.encode_vision(pixel_values)
        template = self.prefill_runtime._cast("inputs_embeds", inputs_embeds)
        prefix = self.assemble_prefix(vision, template)
        cond, hidden = self.prefill(prefix, attention_bias, position_ids)
        actions = self.denoise(cond, hidden, cross_mask, noise, steps)
        self.execution_count += 1
        return {"vision": vision, "actions": actions}

    pipeline = run

    def close(self) -> None:
        for runtime in (self.vision_runtime, self.prefill_runtime, self.flow_runtime):
            runtime.close()


def runtime_from_declaration(
    stage: str, declaration: Mapping[str, Any], *, steps: int | None = None
) -> Any:
    cls = {
        "vision": VisionTensorRTRuntime,
        "prefill": PrefillTensorRTRuntime,
        "flow": FlowTensorRTRuntime,
    }[stage]
    kwargs: dict[str, Any] = {
        "expected_sha256": declaration["sha256"],
        "expected_identity": declaration,
    }
    if stage == "flow":
        kwargs["steps"] = int(
            steps if steps is not None else declaration.get("steps", 10)
        )
    return cls(declaration["path"], **kwargs)
