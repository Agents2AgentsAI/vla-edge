"""Shared Real-Time Chunking guidance utilities for YAM serving.

The implementation mirrors LeRobot's RTC processor, with the sign convention
adapted to MolmoAct2's forward flow loop.  It intentionally has no dependency
on TensorRT or the model package so the numerical pieces remain CPU-testable.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np
import torch

_SCHEDULES = frozenset({"zeros", "ones", "linear", "exp"})


def _schedule_name(schedule) -> str:
    value = getattr(schedule, "value", schedule)
    name = str(value).strip().lower()
    if name not in _SCHEDULES:
        raise ValueError(
            f"RTC schedule must be one of {sorted(_SCHEDULES)}, got {schedule!r}"
        )
    return name


def get_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: str = "linear",
) -> torch.Tensor:
    """Port of ``RTCProcessor.get_prefix_weights``.

    The linear/exp ramps use the open interval between one and zero, exactly as
    the reference does.  In particular, ``start`` is clamped to ``end`` rather
    than the other way around when inference takes longer than the requested
    execution horizon.
    """

    start = int(start)
    end = int(end)
    total = int(total)
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    start = min(start, end)
    name = _schedule_name(schedule)

    if name == "zeros":
        weights = torch.zeros(total, dtype=torch.float32)
        weights[:start] = 1.0
        return weights
    if name == "ones":
        weights = torch.ones(total, dtype=torch.float32)
        weights[end:] = 0.0
        return weights

    skip_steps_at_end = max(total - end, 0)
    linspace_steps = total - skip_steps_at_end - start
    if end <= start or linspace_steps <= 0:
        ramp = torch.empty(0, dtype=torch.float32)
    else:
        ramp = torch.linspace(
            1.0, 0.0, linspace_steps + 2, dtype=torch.float32
        )[1:-1]
    if name == "exp":
        ramp = ramp * torch.expm1(ramp) / (math.e - 1.0)

    zeros_len = total - end
    if zeros_len > 0:
        ramp = torch.cat((ramp, torch.zeros(zeros_len, dtype=torch.float32)))
    ones_len = min(start, total)
    if ones_len > 0:
        ramp = torch.cat((torch.ones(ones_len, dtype=torch.float32), ramp))
    return ramp


def guidance_weight(tau: float, max_w: float = 10.0) -> float:
    """Return the reference RTC scalar guidance strength at flow time ``tau``."""

    tau_value = float(torch.as_tensor(tau).item())
    max_value = float(max_w)
    if not 0.0 <= tau_value <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau_value}")
    if not math.isfinite(max_value) or max_value < 0.0:
        raise ValueError(
            f"max guidance weight must be finite and >= 0, got {max_value}"
        )

    # Keep the reference's operation and nan_to_num ordering.  This matters at
    # both endpoints: tau=0 maps to max_w, while the 0*inf at tau=1 maps to 0.
    tau_tensor = torch.tensor(tau_value, dtype=torch.float32)
    squared_one_minus_tau = (1.0 - tau_tensor) ** 2
    inv_r2 = (
        squared_one_minus_tau + tau_tensor**2
    ) / squared_one_minus_tau
    c = torch.nan_to_num(
        (1.0 - tau_tensor) / tau_tensor,
        posinf=max_value,
    )
    weight = torch.nan_to_num(c * inv_r2, posinf=max_value)
    weight = torch.minimum(weight, torch.tensor(max_value, dtype=torch.float32))
    return float(weight.item())


def step_guidance_weights(steps: int, max_w: float = 10.0) -> torch.Tensor:
    """Precompute one RTC guidance scalar for each Euler step."""

    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    return torch.tensor(
        [guidance_weight(step / steps, max_w) for step in range(steps)],
        dtype=torch.float32,
    )


def _as_float_tensor(value, *, device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _padded_prefix(prefix, target: torch.Tensor) -> torch.Tensor:
    prefix_tensor = _as_float_tensor(prefix, device=target.device)
    if target.ndim == 2:
        if prefix_tensor.ndim == 3 and prefix_tensor.shape[0] == 1:
            prefix_tensor = prefix_tensor.squeeze(0)
        if prefix_tensor.ndim != 2:
            raise ValueError(
                "prefix must be 2D for an unbatched trajectory, got "
                f"{tuple(prefix_tensor.shape)}"
            )
    elif target.ndim == 3:
        if prefix_tensor.ndim == 2:
            prefix_tensor = prefix_tensor.unsqueeze(0)
        if prefix_tensor.ndim != 3:
            raise ValueError(
                "prefix must be 2D or 3D for a batched trajectory, got "
                f"{tuple(prefix_tensor.shape)}"
            )
        if prefix_tensor.shape[0] == 1 and target.shape[0] != 1:
            prefix_tensor = prefix_tensor.expand(target.shape[0], -1, -1)
    else:
        raise ValueError(
            f"RTC trajectories must be 2D or 3D, got {tuple(target.shape)}"
        )

    if any(
        prefix_size > target_size
        for prefix_size, target_size in zip(prefix_tensor.shape, target.shape)
    ):
        raise ValueError(
            f"prefix shape {tuple(prefix_tensor.shape)} exceeds trajectory "
            f"shape {tuple(target.shape)}"
        )
    if tuple(prefix_tensor.shape) == tuple(target.shape):
        return prefix_tensor
    padded = torch.zeros_like(target, dtype=torch.float32)
    slices = tuple(slice(0, size) for size in prefix_tensor.shape)
    padded[slices] = prefix_tensor
    return padded


def apply_guidance(
    velocity: torch.Tensor,
    x_t: torch.Tensor,
    tau: float,
    prefix,
    weights,
    max_w: float = 10.0,
) -> torch.Tensor:
    """Apply closed-form RTC guidance using MolmoAct2's flow convention.

    ``x_hat_1 = x_t + (1 - tau) * velocity`` and the returned fp32 velocity is
    pulled toward ``prefix`` according to the per-token attention weights.
    Passing ``prefix=None`` is an exact no-op and returns ``velocity`` itself.
    """

    if prefix is None:
        return velocity
    if not torch.is_tensor(velocity) or not torch.is_tensor(x_t):
        raise TypeError("velocity and x_t must be torch tensors")
    if tuple(velocity.shape) != tuple(x_t.shape):
        raise ValueError(
            f"velocity shape {tuple(velocity.shape)} does not match x_t "
            f"shape {tuple(x_t.shape)}"
        )
    if velocity.ndim not in (2, 3):
        raise ValueError(
            f"RTC trajectories must be 2D or 3D, got {tuple(velocity.shape)}"
        )

    velocity_f32 = velocity.float()
    x_t_f32 = x_t.float()
    prefix_f32 = _padded_prefix(prefix, x_t_f32)
    token_count = x_t_f32.shape[-2]
    weights_f32 = _as_float_tensor(weights, device=x_t.device).reshape(-1)
    if weights_f32.numel() != token_count:
        raise ValueError(
            f"weights must contain {token_count} values, got "
            f"{weights_f32.numel()}"
        )
    weights_shape = (1, token_count, 1) if velocity.ndim == 3 else (token_count, 1)
    weights_f32 = weights_f32.view(weights_shape)

    tau_value = float(torch.as_tensor(tau).item())
    x_hat_1 = torch.add(
        x_t_f32,
        velocity_f32,
        alpha=1.0 - tau_value,
    )
    correction = (prefix_f32 - x_hat_1) * weights_f32
    return velocity_f32 + guidance_weight(tau_value, max_w) * correction


def build_model_prefix(
    prefix_actions,
    normalizer,
    chunk: int = 30,
    adim: int = 32,
) -> tuple[torch.Tensor, int]:
    """Normalize raw ``(K, 14)`` YAM actions and pad to model space."""

    chunk = int(chunk)
    adim = int(adim)
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    if adim < 14:
        raise ValueError(f"adim must be >= 14, got {adim}")
    if torch.is_tensor(prefix_actions):
        raw = (
            prefix_actions.detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
    else:
        raw = np.asarray(prefix_actions, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 14:
        raise ValueError(
            f"prefix_actions must have shape (K, 14), got {raw.shape}"
        )
    prefix_len = int(raw.shape[0])
    if prefix_len > chunk:
        raise ValueError(
            f"prefix has {prefix_len} rows but model chunk length is {chunk}"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("prefix_actions contains non-finite values")
    normalize = getattr(normalizer, "normalize", None)
    if not callable(normalize):
        raise TypeError("normalizer must provide a callable normalize() method")
    normalized = normalize(raw)
    if torch.is_tensor(normalized):
        normalized = (
            normalized.detach().to(device="cpu", dtype=torch.float32).numpy()
        )
    else:
        normalized = np.asarray(normalized, dtype=np.float32)
    if normalized.shape != raw.shape:
        raise ValueError(
            f"normalizer changed prefix shape from {raw.shape} to "
            f"{normalized.shape}"
        )
    if not np.all(np.isfinite(normalized)):
        raise ValueError("normalized RTC prefix contains non-finite values")

    model_prefix = torch.zeros((chunk, adim), dtype=torch.float32)
    if prefix_len:
        model_prefix[:prefix_len, :14].copy_(torch.from_numpy(normalized))
    return model_prefix, prefix_len


@dataclass(frozen=True)
class RTCRequest:
    """One guided flow request staged through the serving boundary."""

    prefix: torch.Tensor
    weights: torch.Tensor
    max_guidance_weight: float = 10.0
    mode: str = "hybrid"


class RTCContext:
    """Lock-protected single-slot handoff between server and flow runner."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slot = None

    def set(self, request: RTCRequest) -> None:
        if not isinstance(request, RTCRequest):
            raise TypeError(f"request must be RTCRequest, got {type(request)!r}")
        with self._lock:
            self._slot = request

    def take(self):
        with self._lock:
            request = self._slot
            self._slot = None
            return request

    def clear(self) -> None:
        with self._lock:
            self._slot = None

