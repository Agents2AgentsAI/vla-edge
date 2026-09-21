"""Three TensorRT stages sharing persistent device buffers."""
import ctypes
import fcntl
import hashlib
import os
import threading

import numpy as np

from ..abcvla.device import detect_device, verify_device_requirements
from ..abcvla.trt_runtime import _deserialize_sha_bound_plan, _torch_dtype
from ..tensorrt.logger import deserialization_scope
from .bundle import Pi05Error

_LIBRARIES = {}
_INITIALIZERS = {}
_LIBRARY_LOCK = threading.Lock()


def load_plugins(plugins):
    # Keep the exact verified library bytes and their descriptors alive for the process.
    with _LIBRARY_LOCK:
        for plugin in plugins:
            digest, symbol = plugin["sha256"], plugin["initialize"]
            if symbol in _INITIALIZERS and _INITIALIZERS[symbol] != digest:
                raise Pi05Error("a different plugin version is already loaded; restart the process")
            if digest in _LIBRARIES:
                if not _LIBRARIES[digest][2]:
                    raise Pi05Error("plugin initialization previously failed; restart the process")
                continue
            payload = plugin["resolved"].read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise Pi05Error("plugin changed after bundle verification")
            fd = os.memfd_create("vla-edge-plugin", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
            try:
                with os.fdopen(os.dup(fd), "wb") as stream:
                    stream.write(payload)
                fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033), 0x0008 | 0x0004 | 0x0002 | 0x0001)
                handle = ctypes.CDLL(f"/proc/self/fd/{fd}", mode=ctypes.RTLD_GLOBAL)
            except BaseException:
                os.close(fd)
                raise
            _LIBRARIES[digest] = (handle, fd, False)
            _INITIALIZERS[symbol] = digest
            initialize = getattr(handle, symbol)
            initialize.restype = ctypes.c_int
            initialize.argtypes = []
            if initialize():
                raise Pi05Error("plugin initialization failed")
            _LIBRARIES[digest] = (handle, fd, True)


class Engine:
    def __init__(self, declaration, requirements):
        import tensorrt as trt
        import torch
        verify_device_requirements(requirements, detect_device(torch, trt))
        load_plugins(declaration["plugins"])
        with deserialization_scope(trt, verified_device=True) as logger:
            trt.init_libnvinfer_plugins(logger, "")
        _, _, _, self.runtime, self.engine = _deserialize_sha_bound_plan(
            declaration["resolved"], expected_sha256=declaration["sha256"],
            trt=trt, verified_device=True)
        self.ctx = self.engine.create_execution_context()
        if self.ctx is None:
            raise Pi05Error("could not create TensorRT execution context")
        self.names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        for name in self.names:
            if (self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
                or self.engine.get_tensor_vectorized_dim(name) != -1):
                raise Pi05Error(f"unsupported tensor layout: {name}")
        self.trt, self.torch = trt, torch

    def run(self, tensors, stream):
        if set(tensors) != self.names:
            raise Pi05Error("TensorRT binding names differ from the release contract")
        for name, tensor in tensors.items():
            if tensor.dtype != _torch_dtype(self.torch, self.engine.get_tensor_dtype(name)):
                raise Pi05Error(f"wrong binding dtype: {name}")
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise Pi05Error(f"binding must be contiguous on CUDA: {name}")
            if (self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT
                and not self.ctx.set_input_shape(name, tuple(tensor.shape))):
                raise Pi05Error(f"unsupported input shape: {name}")
        if self.ctx.infer_shapes():
            raise Pi05Error("TensorRT input shapes are incomplete")
        for name, tensor in tensors.items():
            if tuple(self.ctx.get_tensor_shape(name)) != tuple(tensor.shape):
                raise Pi05Error(f"wrong binding shape: {name}")
            if not self.ctx.set_tensor_address(name, tensor.data_ptr()):
                raise Pi05Error(f"could not bind tensor: {name}")
        if not self.ctx.execute_async_v3(stream.cuda_stream):
            raise Pi05Error("TensorRT enqueue failed")


class Runtime:
    def __init__(self, desc, device):
        import torch
        self.torch, self.device = torch, torch.device(device)
        self.stages = {stage: Engine(desc["engines"][stage], desc["requires"])
                       for stage in ("vision", "prefill", "flow")}
        self.horizon = desc["action_horizon"]
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=self.device)
        self.pixels = empty((1, 3, 3, 224, 224), torch.float32)
        self.prefix = empty((1, 968, 2048), torch.bfloat16)
        self.cache = {name: empty((18, 1, 968, 1, 256), torch.bfloat16) for name in ("k_ctx", "v_ctx")}
        self.mask = empty((1, 968), torch.bool)
        self.positions = empty((1, 968), torch.int64)
        self.bias = empty((968 * 968,), torch.float32)
        self.noise = empty((1, self.horizon, 32), torch.float32)
        self.actions = empty((1, self.horizon, 32), torch.float32)
        self.previous_stream = None

    def run(self, pixels, language, mask, noise):
        torch = self.torch
        stream = torch.cuda.current_stream(self.device)
        if self.previous_stream is not None:
            stream.wait_stream(self.previous_stream)
        self.previous_stream = stream
        mask = np.asarray(mask, dtype=np.bool_)
        if mask.shape != (968,) or not mask.any():
            raise Pi05Error("invalid prefix mask")
        extent = max(830, int(np.flatnonzero(mask)[-1]) + 1)
        self.pixels.copy_(pixels)
        self.prefix[:, 768:].copy_(language)
        self.mask.copy_(torch.from_numpy(mask).to(self.device)[None])
        self.positions.copy_(torch.from_numpy(np.cumsum(mask, dtype=np.int64) - 1).to(self.device)[None])
        self.noise.copy_(noise)
        self.stages["vision"].run({"pixel_values": self.pixels, "vision": self.prefix[:, :768]}, stream)
        bias = self.bias[:extent * extent].view(1, 1, extent, extent)
        visible = self.mask[:, :extent]
        bias.copy_(torch.where((visible[:, :, None] & visible[:, None, :])[:, None], 0., -2.3819763e38))
        bindings = {"inputs_embeds": self.prefix[:, :extent], "attention_bias": bias,
                    "position_ids": self.positions[:, :extent]}
        for kind in ("k_ctx", "v_ctx"):
            for layer in range(18):
                bindings[f"{kind}.layer.{layer:02d}"] = self.cache[kind][layer, :, :extent].view(1, 1, extent, 256)
        self.stages["prefill"].run(bindings, stream)
        # Invalidate masked and unused rows on every request, including shrinking prompts.
        for cache in self.cache.values():
            cache.masked_fill_(~self.mask[None, :, :, None, None], 0)
        self.stages["flow"].run(dict(self.cache, prefix_pad_masks=self.mask,
                                      noise=self.noise, actions=self.actions), stream)
        return self.actions

    def close(self):
        if self.previous_stream is not None:
            self.previous_stream.synchronize()
        for engine in self.stages.values():
            engine.ctx = None
        self.stages.clear()
