"""TensorRT backend: the checkpoint's own host path, compiled hot stages.

Like the ``torch`` reference, this backend exposes ``generate_actions`` and
runs the checkpoint's *unmodified* tokenization, normalization, and glue code.
What changes is where the three hot stages execute: the model instance's
``generate_actions_from_inputs`` is shadowed with an implementation that runs

    vision.plan       crops -> pooled image features
    llm_prefill.plan  merged embeddings -> projected cross-attn KV context
    action_flow.plan  noise + KV context -> action chunk (all steps in-graph)

Cheap tensor work (embedding lookup + feature scatter, attention bias, masks,
RNG) stays in PyTorch. That is the part where a reimplementation could
silently diverge from the checkpoint, so it is deliberately not reimplemented.
Anything the engines were not built for (non-default step counts without a
matching plan, precomputed KV, batch > 1) falls back to the original PyTorch
path in-band.

Plans are loaded from ``engine_dir``. When a ``MANIFEST.json`` is present in
that directory or its parent (the layout of the prebuilt artifact bundle),
the environment is validated against it first. A mismatched plan does not
fail cleanly on its own (see ``artifacts.py``).
"""

from __future__ import annotations

import ctypes
import logging
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..base import REGISTRY
from . import flow
from .artifacts import (
    MANIFEST_NAME,
    check_compatible,
    effective_token_limit,
    exact_device_match,
    load_serving_config,
)
from .engine import TrtEngine

log = logging.getLogger(__name__)

# fp16-safe stand-in for finfo.min in additive attention masks.
MASK_VALUE = -30000.0

# --------------------------------------------------------------------- plugins
#
# Custom TensorRT plugins live in ``<engine_dir>/plugins``. Any plan built
# against one CANNOT deserialize until its library is loaded, so load them
# before touching a plan. Deserialization otherwise fails with a bare None.
# RTLD_GLOBAL is required: TensorRT resolves the creator through the
# process-global registry. Handles are kept alive for the process.
#
# A plugin that loads its CUDA module lazily stays inert until its runtime
# initializer runs: every enqueue returns -1, reported only as a bare
# "Failed to enqueue" from deep inside the runtime. The convention is an
# extern "C" ``int <namespace>_initialize(void)`` returning 0 on success; the
# ELF dynamic symbol table is the source of truth, so scan it and run every
# exported initializer rather than trusting a fixed name list.

_PLUGIN_INIT_RE = re.compile(r"(?:^|_)initialize$")
_LOADED_PLUGINS: list[Any] = []


def _plugin_init_symbols(path: Path) -> tuple[str, ...]:
    try:
        out = subprocess.run(
            ["nm", "-D", "--defined-only", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return ()
    return tuple(
        parts[2]
        for parts in (line.split() for line in out.splitlines())
        if len(parts) == 3 and parts[1] in ("T", "W")
        and _PLUGIN_INIT_RE.search(parts[2])
    )


def load_plugin_library(path: Path) -> ctypes.CDLL:
    """CDLL a TensorRT plugin RTLD_GLOBAL and run its initializer(s)."""
    handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    for symbol in _plugin_init_symbols(path):
        try:
            initialize = getattr(handle, symbol)
        except AttributeError:
            continue
        initialize.restype = ctypes.c_int
        if initialize():
            raise RuntimeError(f"{path}: {symbol}() failed")
    return handle


def _load_engine_plugins(engine_dir: Path) -> None:
    plugin_dir = engine_dir / "plugins"
    if not plugin_dir.is_dir():
        return
    for library in sorted(plugin_dir.glob("*.so")):
        _LOADED_PLUGINS.append(load_plugin_library(library))
        log.info("loaded TensorRT plugin %s", library.name)


# --------------------------------------------------------------------- backend


class TensorRTBackend:
    """Whole-pass backend with engine-substituted internals.

    ``pad_multiple`` pads the prompt to a multiple of that length before the
    prefill (padded tokens are fully masked, so valid-token math is
    unchanged). Leave it unset to use the engine set's own default: a
    ``serving.json`` shipped beside the plans (the artifact bundle's fixed-
    bracket sets need 704 and declare it there), else 1 for dynamic-shape
    plans. A prompt that pads past what the engine accepts raises a loud,
    actionable error before anything executes rather than silently
    truncating. See ``TrtEngine._bind`` for the in-engine backstop.
    """

    name = "tensorrt"
    staged = False

    def __init__(
        self,
        model: Any,
        processor: Any,
        embodiment: Any,
        engine_dir: str | Path,
        pad_multiple: int | None = None,
        use_graph: bool = True,
        validate_artifacts: bool = True,
        fast_vision: bool = False,
    ) -> None:
        import torch

        self.model = model
        self.processor = processor
        self.embodiment = embodiment
        self._lock = threading.Lock()

        engine_dir = Path(engine_dir).resolve()
        if not engine_dir.is_dir():
            raise FileNotFoundError(f"engine dir not found: {engine_dir}")

        if pad_multiple is None:
            serving = load_serving_config(engine_dir)
            pad_multiple = serving.get("pad_multiple", 1)
            if "pad_multiple" in serving:
                log.info(
                    "%s: pad_multiple=%d from serving.json",
                    engine_dir.name, pad_multiple,
                )
        self.pad_multiple = max(1, int(pad_multiple))

        suppress_device_warning = False
        if validate_artifacts:
            for candidate in (engine_dir, engine_dir.parent):
                if (candidate / MANIFEST_NAME).exists():
                    check_compatible(candidate)
                    suppress_device_warning = exact_device_match(candidate)
                    if suppress_device_warning:
                        log.info(
                            "artifact build device exactly matches this "
                            "runtime; suppressing TensorRT's generic "
                            "cross-device warning"
                        )
                    break

        _load_engine_plugins(engine_dir)  # must precede any deserialize

        # Most parameters intentionally remain on ``meta`` in the compact
        # host model. The token embedding is always materialized and is the
        # authoritative execution device for both full and compact hosts.
        device = next(model.model.transformer.wte.parameters()).device
        self.device = torch.device(device)
        stream = torch.cuda.Stream(device=self.device)
        # A bundle may ship a second vision engine (vision_fp8.plan): the
        # faster FP8 build, behaviorally validated on the same 400-episode
        # evaluation as the default set (no measured loss; see the bundle
        # README for its numbers). Serving it is an explicit opt-in; the
        # default engine remains the parity-preserving fp16 build.
        vision_plan = engine_dir / "vision.plan"
        if fast_vision:
            fast_plan = engine_dir / "vision_fp8.plan"
            if not fast_plan.is_file():
                raise FileNotFoundError(
                    f"--fast-vision requested but {fast_plan} is not in this "
                    f"engine set; serve without the flag, or use a bundle "
                    f"that ships it"
                )
            vision_plan = fast_plan
            log.info("%s: serving fast vision engine %s",
                     engine_dir.name, fast_plan.name)
        self.vision_engine = TrtEngine(
            vision_plan, device, stream=stream,
            use_graph=use_graph,
            suppress_device_mismatch_warning=suppress_device_warning,
        )
        self.llm_engine = TrtEngine(
            engine_dir / "llm_prefill.plan", device, stream=stream,
            use_graph=use_graph,
            suppress_device_mismatch_warning=suppress_device_warning,
        )
        # The flow step count is baked into each action plan at export time.
        self.action_engines: dict[int, TrtEngine] = {}
        for steps, name in [(10, "action_flow.plan"), (5, "action_flow_5.plan")]:
            if (engine_dir / name).exists():
                self.action_engines[steps] = TrtEngine(
                    engine_dir / name, device, stream=stream,
                    use_graph=use_graph,
                    suppress_device_mismatch_warning=suppress_device_warning,
                )
        if not self.action_engines:
            raise FileNotFoundError(f"no action_flow*.plan in {engine_dir}")

        core = model.model
        self._core = core

        # Optional compiled flow stage shipped with the bundle. Absent or
        # unusable, the action engine above serves the stage unchanged.
        self._flow = flow.load(engine_dir, core.action_expert, self.device)

        # Real-Time Chunking guidance rides on the compiled flow stage: the
        # bundle's runner consumes one staged request per inference from this
        # context (see rtc_guidance). Without the flow package there is no
        # in-loop guidance path, so arm_rtc refuses rather than letting a
        # client believe its prefix was honored.
        self.rtc_schedule = "linear"
        self.rtc_max_guidance = 10.0
        self._rtc_context = None
        if self._flow is not None and hasattr(self._flow, "rtc_context"):
            from . import rtc_guidance

            self._rtc_context = rtc_guidance.RTCContext()
            self._flow.rtc_context = self._rtc_context

        self._orig_generate = (
            core.generate_actions_from_inputs
            if getattr(model, "_vla_edge_torch_fallback", True)
            else None
        )
        core.generate_actions_from_inputs = self._generate_actions_trt

    @property
    def rtc_available(self) -> bool:
        return self._rtc_context is not None

    def arm_rtc(
        self,
        prefix_actions: Any,
        inference_delay: int = 0,
        execution_horizon: int = 10,
        rtc_schedule: str | None = None,
        rtc_max_guidance: float | None = None,
    ) -> dict:
        """Stage one RTC prefix for the next inference. Returns telemetry.

        Mirrors the reference server's contract: the prefix is normalized with
        the checkpoint's own action statistics, weighted by the requested
        schedule over [inference_delay, execution_horizon), and consumed by
        the flow runner exactly once.
        """
        if self._rtc_context is None:
            raise ValueError(
                "RTC guidance is unavailable: this engine set has no usable "
                "flow package (it is what implements in-loop guidance)"
            )
        from . import rtc_guidance

        delay = int(inference_delay)
        horizon = int(execution_horizon)
        if delay < 0:
            raise ValueError(f"inference_delay must be >= 0, got {delay}")
        if horizon < 1:
            raise ValueError(f"execution_horizon must be >= 1, got {horizon}")
        schedule = self.rtc_schedule if rtc_schedule is None else rtc_schedule
        max_guidance = (
            self.rtc_max_guidance
            if rtc_max_guidance is None
            else float(rtc_max_guidance)
        )
        if not np.isfinite(max_guidance) or max_guidance < 0.0:
            raise ValueError(
                f"rtc_max_guidance must be finite and >= 0, got {max_guidance}"
            )

        stats = self.model._get_robot_stats()
        normalizer = stats.action_normalizers.get(self.embodiment.norm_tag)
        if normalizer is None:
            raise ValueError(
                "checkpoint has no action normalizer for norm tag "
                f"{self.embodiment.norm_tag!r}"
            )
        chunk = int(self._core._resolve_action_horizon())
        adim = int(self._core.config.max_action_dim)
        model_prefix, prefix_len = rtc_guidance.build_model_prefix(
            prefix_actions, normalizer, chunk=chunk, adim=adim,
        )
        if prefix_len < 1:
            raise ValueError("prefix_actions must contain at least one row")
        horizon_eff = min(horizon, prefix_len)
        weights = rtc_guidance.get_prefix_weights(
            start=delay, end=horizon_eff, total=chunk, schedule=schedule,
        )
        self._rtc_context.set(rtc_guidance.RTCRequest(
            prefix=model_prefix,
            weights=weights,
            max_guidance_weight=max_guidance,
            mode="hybrid",
        ))
        return {
            "rtc_armed": True,
            "prefix_len": prefix_len,
            "inference_delay": delay,
            "execution_horizon": horizon,
            "execution_horizon_eff": horizon_eff,
            "schedule": str(schedule),
            "max_guidance": max_guidance,
        }

    # ------------------------------------------------------------- public API

    def generate_actions(
        self,
        images: list[Any],
        instruction: str,
        state: np.ndarray,
        num_steps: int,
        enable_cuda_graph: bool = False,  # torch-path knob; unused here
    ) -> np.ndarray:
        import torch

        if getattr(self, "_closed", False):
            raise RuntimeError("TensorRT backend is closed")
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        expected = (self.embodiment.state_dim,)
        if state_f32.shape != expected:
            raise ValueError(
                f"state must be shape {expected}, got {state_f32.shape}"
            )

        with self._lock, torch.inference_mode():
            out = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=instruction,
                state=state_f32,
                norm_tag=self.embodiment.norm_tag,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=num_steps,
                normalize_language=True,
            )

        raw = out.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return actions

    def close(self) -> None:
        """Synchronize and destroy graph-owning objects in a safe order."""
        if getattr(self, "_closed", False):
            return
        import gc

        import torch

        with self._lock:
            torch.cuda.synchronize(self.device)
            runner = self._flow
            if runner is not None:
                close = getattr(runner, "close", None)
                if callable(close):
                    close()
                for name in ("_graphs", "_mods", "_rtc_wstep_cache"):
                    value = getattr(runner, name, None)
                    if hasattr(value, "clear"):
                        value.clear()
                for name in ("ext", "ae", "_wt_source_parameters", "_wt_bank"):
                    if hasattr(runner, name):
                        setattr(runner, name, None)
                self._flow = None
                del runner
            for engine in (
                self.vision_engine,
                self.llm_engine,
                *self.action_engines.values(),
            ):
                engine.close()
            self.action_engines.clear()
            self._rtc_context = None
            if "generate_actions_from_inputs" in self._core.__dict__:
                delattr(self._core, "generate_actions_from_inputs")
            self._orig_generate = None
            self._core = None
            self.model = None
            self.processor = None
            gc.collect()
            torch.cuda.synchronize(self.device)
            self._closed = True

    def warmup(self) -> None:
        """Two dummy passes: the first captures CUDA graphs, the second
        replays them (a graph captured but never replayed returns stale
        buffers, which looks like one garbage inference)."""
        from PIL import Image

        dummy = Image.new("RGB", (320, 180))
        for _ in range(2):
            self.generate_actions(
                images=[dummy] * self.embodiment.num_cameras,
                instruction="warmup",
                state=np.zeros(self.embodiment.state_dim, dtype=np.float32),
                num_steps=self.embodiment.default_num_steps,
            )

    # --------------------------------------------------------------- TRT path

    def _generate_actions_trt(
        self,
        *,
        input_ids: Any,
        pixel_values: Any = None,
        image_token_pooling: Any = None,
        image_grids: Any = None,
        image_num_crops: Any = None,
        attention_mask: Any = None,
        token_type_ids: Any = None,
        states: Any = None,
        action_dim_is_pad: Any = None,
        action_horizon: Any = None,
        num_steps: Any = None,
        generator: Any = None,
        encoder_kv_states: Any = None,
        encoder_attention_mask: Any = None,
        **extra: Any,
    ) -> Any:
        import torch

        core = self._core
        steps = int(num_steps or core.config.flow_matching_num_steps)
        needs_fallback = (
            (steps not in self.action_engines and self._flow is None)
            or encoder_kv_states is not None
            or input_ids.shape[0] != 1
        )
        if needs_fallback:
            # In-band fallback: the engines were not built for this call.
            if self._orig_generate is None:
                reasons = []
                if steps not in self.action_engines and self._flow is None:
                    reasons.append(f"no {steps}-step flow engine")
                if encoder_kv_states is not None:
                    reasons.append("precomputed encoder KV was supplied")
                if input_ids.shape[0] != 1:
                    reasons.append(f"batch size is {input_ids.shape[0]}, not 1")
                raise ValueError(
                    "this local TensorRT bundle cannot execute the request "
                    f"({'; '.join(reasons)}). The lightweight serving path "
                    "does not load 22 GB of unused PyTorch fallback weights; "
                    "use an engine set built for this request or the torch "
                    "backend."
                )
            return self._orig_generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_pooling=image_token_pooling,
                image_grids=image_grids,
                image_num_crops=image_num_crops,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                states=states,
                action_dim_is_pad=action_dim_is_pad,
                action_horizon=action_horizon,
                num_steps=num_steps,
                generator=generator,
                encoder_kv_states=encoder_kv_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        # 0) pad the prompt; padded tokens are masked everywhere downstream,
        # so valid-token outputs are unchanged.
        native_s = int(input_ids.shape[1])
        pad = (-native_s) % self.pad_multiple
        padded_s = native_s + pad
        try:
            max_s = int(self.llm_engine.profile_max("inputs_embeds")[1])
        except Exception:  # noqa: BLE001  # engines may not expose profiles
            max_s = None
        if max_s is not None and padded_s > max_s:
            limit = effective_token_limit(max_s, self.pad_multiple)
            advice = (
                "Shorten the instruction."
                if self.pad_multiple <= 1
                else "Shorten the instruction, or serve the dynamic-shape "
                     "set, which accepts any prompt up to its profile."
            )
            raise ValueError(
                f"prompt is {native_s} tokens, over this engine set's "
                f"{limit}-token limit (pad_multiple={self.pad_multiple} pads "
                f"it to {padded_s}; the engine profile ends at {max_s}). "
                f"Refusing before execution. {advice}"
            )
        if pad:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            pad_id = int(core.config.pad_token_id or 0)
            input_ids = torch.cat(
                [input_ids, input_ids.new_full((1, pad), pad_id)], dim=1
            )
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_zeros(1, pad)], dim=1
            )
            if token_type_ids is not None:
                token_type_ids = torch.cat(
                    [token_type_ids, token_type_ids.new_zeros(1, pad)], dim=1
                )

        # 1) vision: batch crops + TRT encoder + valid-token gather
        images, token_pooling = core.merge_visual_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
        )
        input_ids_clean = input_ids * (input_ids != -1).to(input_ids.dtype)
        x = core.transformer.wte(input_ids_clean)
        if images is not None:
            pooled = self.vision_engine(
                images=images, pooled_patches_idx=token_pooling
            )["pooled_features"]
            valid_token = (token_pooling >= 0).any(-1).flatten()
            feats = pooled.reshape(-1, pooled.shape[-1])[valid_token]
            is_image_patch = input_ids.view(-1) == core.config.image_patch_id
            x.view(-1, x.shape[-1])[is_image_patch] += feats.to(x.dtype)

        # 2) LLM prefill -> projected KV context
        bias = core._build_native_attention_bias(
            inputs_embeds=x,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=None,
        )
        bias = bias.to(torch.float32).clamp_(min=MASK_VALUE)
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        kv = self.llm_engine(
            inputs_embeds=x, attention_bias=bias, position_ids=position_ids
        )

        # 3) flow-matching action generation
        enc_mask = core._get_encoder_attention_mask(input_ids, attention_mask)
        cross_bias = (
            1.0 - enc_mask.to(torch.float32)[:, None, None, :]
        ) * MASK_VALUE
        traj_dtype = getattr(self.model, "_vla_edge_dtype", None)
        if traj_dtype is None:
            traj_dtype = core.action_expert.action_embed.weight.dtype
        horizon = core._resolve_action_horizon(action_horizon)
        noise = torch.randn(
            (1, horizon, core.config.max_action_dim),
            device=x.device,
            dtype=traj_dtype,
            generator=generator,
        )
        noise = core._mask_action_dim_tensor(
            noise,
            action_dim_is_pad=action_dim_is_pad,
            enabled=core.config.mask_action_dim_padding,
        )
        if action_dim_is_pad is not None:
            action_dim_mask = (
                (~action_dim_is_pad.bool()).to(torch.float32).view(1, 1, -1)
            )
        else:
            action_dim_mask = torch.ones(
                (1, 1, core.config.max_action_dim), device=x.device
            )
        if self._flow is not None:
            trajectory = self._flow(
                noise=noise,
                k_ctx=kv["k_ctx"],
                v_ctx=kv["v_ctx"],
                cross_bias=cross_bias,
                action_dim_mask=action_dim_mask,
                steps=steps,
            )
        else:
            trajectory = self.action_engines[steps](
                noise=noise,
                k_ctx=kv["k_ctx"],
                v_ctx=kv["v_ctx"],
                cross_bias=cross_bias,
                action_dim_mask=action_dim_mask,
            )["trajectory"]
        return trajectory.to(traj_dtype)


def _factory(
    model: Any, processor: Any, embodiment: Any, **kwargs: Any
) -> TensorRTBackend:
    if "engine_dir" not in kwargs:
        raise TypeError(
            "the tensorrt backend needs engine_dir=<path to built or "
            "prebuilt plans> (e.g. the yam/ directory of the artifact bundle)"
        )
    return TensorRTBackend(model, processor, embodiment, **kwargs)


REGISTRY.register("tensorrt", _factory)
