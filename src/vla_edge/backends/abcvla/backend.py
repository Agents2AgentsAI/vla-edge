"""ABC-VLA native, native-operator CUDA graph, and packed TensorRT policy runtimes."""

from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar
from pathlib import Path
from threading import RLock

import numpy as np

from .bundle import (
    AbcVlaError,
    engine_declaration,
    install_upstream,
    load_bundle,
    verify_file,
)
from .host import AbcHost, validate_prefix


class _BaseBackend:
    action_horizon = 30
    rtc_available = True
    rtc_mode = "hard-prefix"
    max_prefix_length = 7
    restore_output_prefix = False

    def __init__(self, desc: dict, device: str, dtype: str):
        if dtype != "bfloat16":
            raise AbcVlaError(
                "ABC requires its trained BF16 backbone and FP32 native head"
            )
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.descriptor = desc
        self._closed = False
        self._lock = RLock()
        self._prefix = ContextVar(f"abc_prefix_{id(self)}", default=None)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        install_upstream(desc)
        if int(desc["config"].get("max_action_prefix", 0)) != 8:
            raise AbcVlaError(
                "ABC serving expects the checkpoint trained with prefixes 0..7"
            )
        self.host = AbcHost(desc, self.device)

    def arm_rtc(
        self,
        prefix_actions,
        *,
        prefix_length=None,
        inference_delay=0,
        execution_horizon=10,
        rtc_schedule=None,
        rtc_max_guidance=None,
    ):
        self.clear_rtc()
        if self._closed:
            raise AbcVlaError("ABC backend is closed")
        if rtc_schedule is not None or rtc_max_guidance is not None:
            raise ValueError("ABC uses a hard action prefix, not weighted RTC guidance")
        if inference_delay != 0:
            raise ValueError(
                "ABC prefix rows start at observation time; use prefix_length, not inference_delay guidance"
            )
        prefix = validate_prefix(prefix_actions, prefix_length, self.max_prefix_length)
        self._prefix.set(prefix)
        return {
            "rtc_armed": True,
            "mode": self.rtc_mode,
            "prefix_length": prefix.length,
            "execution_horizon": int(execution_horizon),
        }

    def clear_rtc(self):
        self._prefix.set(None)

    def generate_actions(
        self,
        *,
        images,
        instruction,
        state,
        num_steps=10,
        enable_cuda_graph=False,
        seed=None,
        noise=None,
    ):
        if seed is not None and noise is not None:
            self.clear_rtc()
            raise ValueError("supply either seed or explicit noise, not both")
        if enable_cuda_graph and self.name == "abcvla-torch":
            self.clear_rtc()
            raise ValueError(
                "select backend=cuda-graph for the full native ABC CUDA graph"
            )
        return self._predict(
            images, instruction, state, num_steps, seed=seed, noise=noise
        )

    def _predict(
        self, images, instruction, state, num_steps=10, *, seed=None, noise=None
    ):
        prefix = self._prefix.get()
        self.clear_rtc()  # one-shot, including validation failures
        if isinstance(num_steps, bool) or num_steps != 10:
            raise ValueError(
                "This ABC checkpoint/engine set is qualified for exactly 10 Euler steps"
            )
        torch = self.torch
        with self._lock, torch.no_grad():
            if self._closed:
                raise AbcVlaError("ABC backend is closed")
            device_context = (
                torch.cuda.device(self.device)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with (
                device_context,
                torch.autocast(device_type=self.device.type, enabled=False),
            ):
                prepared = self.host.prepare(images, instruction, state, prefix)
                if noise is None:
                    noise = self.host.noise(seed)
                else:
                    noise = torch.as_tensor(
                        noise, device=self.device, dtype=torch.float32
                    )
                    if noise.shape == (30, 14):
                        noise = noise[None]
                    if tuple(noise.shape) != (1, 30, 14) or not bool(
                        torch.isfinite(noise).all()
                    ):
                        raise ValueError(
                            "ABC noise must be finite with shape (1,30,14)"
                        )
                return self.host.finish(
                    self._infer(prepared, instruction, noise),
                    restore_prefix=prefix if self.restore_output_prefix else None,
                )

    def warmup(self):
        images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        state = np.zeros(14, dtype=np.float32)
        for prompt in ("fold and stack the towels", "fold and stack the t-shirts"):
            self.generate_actions(
                images=images, instruction=prompt, state=state, seed=0
            )
            self.arm_rtc(np.zeros((5, 14), dtype=np.float32), prefix_length=5)
            self.generate_actions(
                images=images, instruction=prompt, state=state, seed=0
            )
        self.torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None

    def serving_metadata(self):
        return {
            "rtc_mode": self.rtc_mode,
            "max_prefix_length": self.max_prefix_length,
            "image_preprocessing": "native ABC Torch antialiased bilinear + black pad to 224; RGB",
            "client_resize": False,
            "step_options": [10],
        }

    def mount_routes(self, app, pipeline):
        from ...serving.abc_websocket import mount_routes

        mount_routes(app, pipeline)


class AbcVlaNativeBackend(_BaseBackend):
    def __init__(self, desc, device, dtype, *, compiled=False, evidence_dir=None):
        super().__init__(desc, device, dtype)
        torch = self.torch
        from abc_minimal.config import GemmaVLAConfig, VLADiTConfig, VLAModelConfig
        from abc_minimal.vla import VLAPolicy, default_dtype, inference_model_config
        from safetensors.torch import load_file

        raw = desc["config"]["model_config"]
        cfg = VLAModelConfig(
            camera_keys=tuple(raw["camera_keys"]),
            backbone=GemmaVLAConfig(**raw["backbone"]),
            dit=VLADiTConfig(**raw["dit"]),
        )
        weights = verify_file(
            desc["root"], desc["checkpoint"]["weights"], "ABC checkpoint weights"
        )
        with default_dtype(torch.float32):
            self.model = VLAPolicy(
                inference_model_config(cfg),
                backbone_dtype=torch.bfloat16,
                backbone_autocast=False,
            )
        tensors = load_file(str(weights), device="cpu")
        state = {
            k.removesuffix("::complex_as_real"): torch.view_as_complex(v.contiguous())
            if k.endswith("::complex_as_real")
            else v
            for k, v in tensors.items()
        }
        self.model.load_state_dict(state, strict=True)
        del tensors, state
        self.model = self.model.to(self.device).eval()
        self._native_graph = self._sampler = None
        self.compiled = compiled
        self.name = "abcvla-cuda-graph" if compiled else "abcvla-torch"
        if compiled:
            if self.device.type != "cuda":
                raise AbcVlaError("native ABC CUDA graphs require a CUDA device")
            from .native_graph import NativeCudaGraphBackend
            from .native_sampler import TensorVLASampler

            self._native_graph = NativeCudaGraphBackend(self.model, evidence_dir)
            self._sampler = torch.compile(
                TensorVLASampler(self.model, 10).eval(),
                fullgraph=True,
                dynamic=False,
                backend=self._native_graph,
            )

    def _infer(self, p, instruction, noise):
        if self.compiled:
            actions = self._sampler(
                p["images"],
                p["state"],
                p["padded_ids"],
                p["valid"],
                noise,
                p["action_prefix"],
                p["prefix_length"],
            )
            if not self._native_graph.graphs:
                raise AbcVlaError(
                    "ABC CUDA graph was not captured; ensure TORCH_COMPILE_DISABLE=0"
                )
            return actions
        return self.model.sample_actions(
            {"images": p["images"], "state": p["state"], "prompt": [instruction]},
            num_steps=10,
            noise=noise,
            action_prefix=p["action_prefix"],
            prefix_length=p["prefix_length"],
        )

    def serving_metadata(self):
        return {
            **super().serving_metadata(),
            "execution": self.name,
            "graph_breaks": 0 if self.compiled else None,
            "cuda_graph_count": len(self._native_graph.graphs) if self.compiled else 0,
        }

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.clear_rtc()
            if self.device.type == "cuda":
                self.torch.cuda.synchronize(self.device)
            self._sampler = None
            self._native_graph = None
            self.model = None
            self.host = None
            self._closed = True


class AbcVlaTensorRTBackend(_BaseBackend):
    restore_output_prefix = True

    def __init__(self, desc, device, dtype):
        super().__init__(desc, device, dtype)
        if self.device.type != "cuda":
            raise AbcVlaError("ABC TensorRT requires a CUDA device")
        self.name = "abcvla-tensorrt"
        self.runtimes = {}
        self._checked_lengths = set()
        self.host.load_host_weights()
        self._vision = self._flow = None
        # Validate all declarations before admitting any request; engine loads
        # are lazy by prefix length to avoid allocating unused model contexts.
        self.engines = {
            "vision": engine_declaration(desc, desc["engines"]["vision"], "vision"),
            "flow": engine_declaration(desc, desc["engines"]["flow"], "flow"),
            "prefill": {
                k: engine_declaration(desc, v, f"prefill {k}")
                for k, v in desc["engines"]["prefill"].items()
            },
        }

    def _runtime(self, length):
        from .trt_runtime import PipelineTensorRTRuntime, runtime_from_declaration

        key = str(length) if str(length) in self.engines["prefill"] else "dynamic"
        if key not in self.engines["prefill"]:
            raise ValueError(
                f"No ABC prefill engine supports the prompt's {length} tokens"
            )
        if key not in self.runtimes:
            if self._vision is None:
                self._vision = runtime_from_declaration(
                    "vision", self.engines["vision"]
                )
                self._flow = runtime_from_declaration(
                    "flow", self.engines["flow"], steps=10
                )
            prefill = runtime_from_declaration("prefill", self.engines["prefill"][key])
            shape = prefill.input_shape("inputs_embeds")
            if key != "dynamic" and tuple(shape) != (1, int(key), 2560):
                raise AbcVlaError(f"prefill {key} plan has incompatible shape {shape}")
            self.runtimes[key] = PipelineTensorRTRuntime(
                self._vision, prefill, self._flow
            )
        self.last_prefill_key = key
        return self.runtimes[key]

    def _infer(self, p, instruction, noise):
        length = p["ids"].shape[1]
        runtime = self._runtime(length)
        inputs = self.host.trt_inputs(p, instruction, noise)
        if length not in self._checked_lengths:
            runtime.prefill_runtime.prepare_inputs("prefill", inputs)
            self._checked_lengths.add(length)
        # The host supplies fresh images, state, prompt and noise on each call.
        vision = runtime.encode_vision(inputs["pixel_values"])
        prefix = runtime.assemble_prefix(vision, inputs["inputs_embeds"])
        cond, hidden = runtime.prefill(
            prefix, inputs["attention_bias"], inputs["position_ids"]
        )
        actions = runtime.denoise(cond, hidden, inputs["cross_mask"], noise, 10)
        runtime.execution_count += 1
        return actions

    def serving_metadata(self):
        return {
            **super().serving_metadata(),
            "execution": self.name,
            "prefill_variants": sorted(self.engines["prefill"]),
            "last_prefill": getattr(self, "last_prefill_key", None),
            "all_prompt_lengths_warmed": getattr(self, "_all_lengths_warmed", False),
        }

    def warmup(self):
        super().warmup()
        if "dynamic" not in self.engines["prefill"]:
            return
        from .trt_runtime import canonical_attention_bias

        torch = self.torch
        # cuDNN/TensorRT may materialize per-shape resources on first enqueue.
        # Pay that cost before accepting clients, not while a robot is waiting.
        with self._lock, torch.no_grad(), torch.cuda.device(self.device):
            for length in range(207, 257):
                runtime = self._runtime(length)
                embeds = torch.zeros(
                    (1, length, 2560), device=self.device, dtype=torch.bfloat16
                )
                bias = canonical_attention_bias(
                    torch, length, dtype=torch.bfloat16, device=self.device
                )
                positions = torch.arange(length, device=self.device, dtype=torch.int64)[
                    None
                ]
                runtime.prefill(embeds, bias, positions)
            torch.cuda.synchronize(self.device)
            self._all_lengths_warmed = True

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.clear_rtc()
            self.torch.cuda.synchronize(self.device)
            # Runtimes share vision/flow; close each engine only once.
            for runtime in self.runtimes.values():
                runtime.prefill_runtime.close()
            for runtime in (self._vision, self._flow):
                if runtime is not None:
                    runtime.close()
            self.runtimes.clear()
            self._vision = self._flow = None
            self.host = None
            self._closed = True


def load_backend(
    *,
    backend,
    policy,
    embodiment,
    device="cuda:0",
    dtype="bfloat16",
    engine_dir=None,
    checkpoint=None,
    pad_multiple=None,
    fast_vision=False,
    evidence_dir=None,
    **kwargs,
):
    if backend not in ("torch", "cuda-graph", "tensorrt"):
        raise ValueError(f"unsupported ABC backend: {backend!r}")
    if pad_multiple is not None or fast_vision:
        raise ValueError("ABC uses its native prompt lengths and packaged vision plan")
    if engine_dir is None:
        raise ValueError(
            "ABC serving requires --engine-dir pointing to an abcvla-serving.json bundle"
        )
    desc = load_bundle(engine_dir, policy, embodiment)
    if (
        checkpoint is not None
        and Path(checkpoint).expanduser().resolve()
        != (desc["root"] / desc["checkpoint"]["config"]["path"]).parent
    ):
        raise ValueError(
            "checkpoint override must identify this bundle's pinned checkpoint directory"
        )
    if backend == "tensorrt":
        return AbcVlaTensorRTBackend(desc, device, dtype)
    return AbcVlaNativeBackend(
        desc, device, dtype, compiled=backend == "cuda-graph", evidence_dir=evidence_dir
    )
