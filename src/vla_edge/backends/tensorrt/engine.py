"""Minimal TensorRT 10 engine wrapper operating on torch CUDA tensors.

Two latency features, both optional:

* dedicated CUDA stream: enqueueV3 on the default stream makes TRT insert
  extra synchronizations; a side stream bracketed by event waits avoids them
  without any host sync.
* CUDA-graph replay: engines dominated by many small kernels (the unrolled
  flow loop is ~3.6k launches) are launch-bound; capturing the execution once
  per input shape and replaying collapses the launch overhead. Inputs are
  copied into persistent buffers before replay, outputs are returned as the
  persistent buffers (clone if you need to hold them across calls).
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt
import torch

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
}

_LOGGER = trt.Logger(trt.Logger.WARNING)
_DEVICE_MISMATCH_WARNING = (
    "Using an engine plan file across different models of device"
)


class _VerifiedDeviceLogger(trt.ILogger):
    """Drop only TensorRT's generic device warning after an exact check."""

    def __init__(self) -> None:
        super().__init__()

    def log(self, severity: trt.ILogger.Severity, message: str) -> None:
        if _DEVICE_MISMATCH_WARNING in message:
            return
        _LOGGER.log(severity, message)


_VERIFIED_DEVICE_LOGGER = _VerifiedDeviceLogger()


class TrtEngine:
    """Deserialized engine + execution context with a torch-tensor interface.

    Call with keyword tensors matching the ONNX input names; returns a dict of
    output name -> torch CUDA tensor. Not thread-safe (one execution context).
    """

    def __init__(
        self,
        path: str | Path,
        device: str = "cuda:0",
        stream: torch.cuda.Stream | None = None,
        use_graph: bool = False,
        suppress_device_mismatch_warning: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self._logger = (
            _VERIFIED_DEVICE_LOGGER
            if suppress_device_mismatch_warning
            else _LOGGER
        )
        runtime = trt.Runtime(self._logger)
        self.engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {path}")
        self.context = self.engine.create_execution_context()
        self.stream = stream or torch.cuda.Stream(device=self.device)
        self.use_graph = use_graph
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        # per-shape-signature graph state
        self._graphs: dict[tuple, dict] = {}

    def input_dtype(self, name: str) -> torch.dtype:
        return _TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]

    def _prepare(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prepared = {}
        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"missing engine input {name!r}")
            t = inputs[name]
            want = self.input_dtype(name)
            if t.dtype != want:
                t = t.to(want)
            prepared[name] = t.to(self.device).contiguous()
        return prepared

    def _alloc_outputs(self) -> dict[str, torch.Tensor]:
        outputs = {}
        for name in self.output_names:
            outputs[name] = torch.empty(
                tuple(self.context.get_tensor_shape(name)),
                dtype=_TRT_TO_TORCH[self.engine.get_tensor_dtype(name)],
                device=self.device,
            )
        return outputs

    def profile_max(self, name: str) -> tuple:
        """Max shape the engine accepts for input ``name`` (profile 0)."""
        return tuple(self.engine.get_tensor_profile_shape(name, 0)[2])

    def close(self) -> None:
        """Release CUDA graphs and TensorRT objects before interpreter exit."""
        if self.context is None:
            return
        torch.cuda.synchronize(self.device)
        self._graphs.clear()
        self.context = None
        self.engine = None
        self.stream = None

    def _bind(self, tensors: dict[str, torch.Tensor], input_shapes: bool) -> None:
        for name, t in tensors.items():
            # set_input_shape returns False on a profile violation and TRT
            # only LOGS the error. Ignoring it leaves the context on the
            # previous shapes while the new addresses are bound, so
            # enqueueV3 silently computes on a truncated/stale view of the
            # input. On a VLA that can cut off the prompt tail (the
            # robot-state tokens) and produce state-blind actions with no
            # exception anywhere. Fail loudly instead.
            if input_shapes and not self.context.set_input_shape(name, tuple(t.shape)):
                try:
                    mn, _opt, mx = self.engine.get_tensor_profile_shape(name, 0)
                    valid = f"valid range {tuple(mn)}..{tuple(mx)}"
                except Exception:  # noqa: BLE001  # best-effort introspection
                    valid = "profile introspection unavailable"
                raise RuntimeError(
                    f"TensorRT rejected input shape {tuple(t.shape)} for "
                    f"{name!r} ({valid}). Refusing to execute: proceeding "
                    f"would silently truncate this input."
                )
            self.context.set_tensor_address(name, t.data_ptr())

    def _execute(self) -> None:
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        prepared = self._prepare(inputs)
        if getattr(self, "external_graph", False):
            # caller is capturing an outer CUDA graph: bind + enqueue on the
            # current stream, no internal graphs or stream hops.
            self._bind(prepared, input_shapes=True)
            out = self._alloc_outputs()
            self._bind(out, input_shapes=False)
            self.stream_holder = prepared  # keep casts alive until execution
            if not self.context.execute_async_v3(
                torch.cuda.current_stream(self.device).cuda_stream
            ):
                raise RuntimeError("TensorRT execution failed")
            return out
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        if self.use_graph:
            out = self._call_graph(prepared)
        else:
            self._bind(prepared, input_shapes=True)
            out = self._alloc_outputs()
            self._bind(out, input_shapes=False)
            with torch.cuda.stream(self.stream):
                self._execute()
        current.wait_stream(self.stream)
        return out

    def _call_graph(self, prepared: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        key = tuple((n, tuple(t.shape)) for n, t in sorted(prepared.items()))
        state = self._graphs.get(key)
        if state is None:
            static_in = {n: t.clone() for n, t in prepared.items()}
            self._bind(static_in, input_shapes=True)
            static_out = self._alloc_outputs()
            self._bind(static_out, input_shapes=False)
            with torch.cuda.stream(self.stream):
                for _ in range(2):  # warmup enqueues allocate TRT scratch
                    self._execute()
            self.stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                self._execute()
            state = {"graph": graph, "in": static_in, "out": static_out}
            self._graphs[key] = state
        else:
            with torch.cuda.stream(self.stream):
                for name, t in prepared.items():
                    state["in"][name].copy_(t, non_blocking=True)
            with torch.cuda.stream(self.stream):
                state["graph"].replay()
        return state["out"]
