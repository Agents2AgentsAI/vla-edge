"""ABC's shared live-input contract, with only the small learned host modules for TRT."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bundle import AbcVlaError, verify_file
from .geometry import VISION_TOKENS, native_positions


@dataclass(frozen=True)
class Prefix:
    actions: np.ndarray
    length: int


def validate_state(state: Any) -> np.ndarray:
    value = np.asarray(state, dtype=np.float32)
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError("ABC state must be a finite (14,) vector in robot coordinates")
    return value.copy()


def validate_images(images: list[Any]) -> list[np.ndarray]:
    if len(images) != 3:
        raise ValueError("ABC requires RGB head, left wrist and right wrist images")
    result = []
    for image in images:
        a = np.asarray(image)
        if (
            a.ndim != 3
            or a.shape[2] != 3
            or min(a.shape[:2]) < 1
            or a.dtype != np.uint8
        ):
            raise ValueError("ABC images must be HWC RGB uint8 arrays")
        result.append(np.array(a.transpose(2, 0, 1), copy=True, order="C"))
    return result


def validate_prefix(actions: Any, length: int | None, maximum: int = 7) -> Prefix:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2 or a.shape[1] != 14 or not np.isfinite(a).all():
        raise ValueError("ABC prefix must be a finite (rows, 14) action array")
    if length is None:
        length = len(a)
    if (
        isinstance(length, bool)
        or not isinstance(length, (int, np.integer))
        or not 0 <= length <= maximum
    ):
        raise ValueError(f"ABC prefix_length must be an integer in [0, {maximum}]")
    if len(a) not in (length, 30):
        raise ValueError(
            "ABC prefix rows must equal prefix_length or the full 30-row chunk"
        )
    full = np.zeros((30, 14), dtype=np.float32)
    full[: len(a)] = a
    return Prefix(full, int(length))


class AbcHost:
    """Native Torch resize/normalization and cached input-independent token layouts."""

    def __init__(self, descriptor: dict, device: Any):
        import torch
        from abc_minimal.gemma.tokenizer import Tokenizer

        self.torch = torch
        self.descriptor = descriptor
        self.config = descriptor["config"]
        self.device = torch.device(device)
        self.stats = {}
        for name in ("state", "actions"):
            stats = self.config["norm_stats"][name]
            mean, std = (
                np.asarray(stats[k], dtype=np.float32) for k in ("mean", "std")
            )
            if (
                mean.shape != (14,)
                or std.shape != (14,)
                or not np.isfinite(mean).all()
                or not np.isfinite(std).all()
                or (std <= 0).any()
            ):
                raise AbcVlaError(f"invalid {name} normalization statistics")
            self.stats[name] = (mean, std + np.float32(1e-6))
        tokenizer = verify_file(
            descriptor["root"], descriptor["tokenizer"], "ABC tokenizer"
        )
        self.tokenizer = Tokenizer(str(tokenizer))
        self.tokenizer.register_state_token(self.tokenizer.get_unused_id(), "<state>")
        self.state_id = self.tokenizer.state_proj_id
        self.tokenizer.register_special_token(
            self.tokenizer.get_unused_id(), "<action_start>"
        )
        self.token_cache: OrderedDict[str, tuple] = OrderedDict()
        self.prefix_cache: OrderedDict[str, tuple] = OrderedDict()
        self.embedding = self.state_mlp = None

    def token_ids(self, instruction: str):
        if not isinstance(instruction, str):
            raise ValueError("ABC instruction must be a string")  # noqa: TRY004 - preserve the wire-validation ValueError contract.
        key = instruction.replace("_", " ")
        if key not in self.token_cache:
            t = self.tokenizer
            tokens = [t.bos_id]
            for _ in range(3):
                tokens.extend(t.encode("\n\n", bos=False, eos=False))
                tokens.append(t.boi_id)
                tokens.extend([t.image_token_placeholder_id] * 64)
                tokens.append(t.eoi_id)
                tokens.extend(t.encode("\n\n", bos=False, eos=False))
            text = (
                f"<state> {key} <action_start>"
                if self.config["model_config"]["backbone"]["proprio_token"]
                else f"{key} <action_start>"
            )
            tokens.extend(t.encode(text, bos=False, eos=False))
            length = len(tokens)
            if not 207 <= length <= 256:
                raise ValueError(
                    f"ABC prompt uses {length} tokens; supported valid length is 207..256"
                )
            image, other = native_positions(length)
            if [
                i for i, x in enumerate(tokens) if x == t.image_token_placeholder_id
            ] != image:
                raise AbcVlaError(
                    "ABC tokenizer changed the trained three-image layout"
                )
            ids = self.torch.tensor(
                [tokens], dtype=self.torch.int64, device=self.device
            )
            padded = self.torch.full(
                (1, 256), t.pad_id, dtype=self.torch.int64, device=self.device
            )
            padded[:, :length] = ids
            valid = self.torch.arange(256, device=self.device)[None] < length
            self.token_cache[key] = (ids, padded, valid, other)
            if len(self.token_cache) > 32:
                expired, _ = self.token_cache.popitem(last=False)
                self.prefix_cache.pop(expired, None)
        self.token_cache.move_to_end(key)
        return self.token_cache[key]

    def prepare(
        self, images: list[Any], instruction: str, state: Any, prefix: Prefix | None
    ):
        from abc_minimal.preprocess import resize_pad_normalize_batch

        torch = self.torch
        state = validate_state(state)
        arrays = validate_images(images)
        ids, padded, valid, _ = self.token_ids(instruction)
        m, s = self.stats["state"]
        normalized_state = torch.from_numpy(((state - m) / s)[None]).to(self.device)
        # Match the original inference policy exactly: RGB -> uint8 CHW -> GPU
        # float32 /255 -> antialiased bilinear resize -> centered black padding.
        views = [
            resize_pad_normalize_batch(
                torch.from_numpy(a).to(self.device)[None], 224, 224, preset=None
            )
            for a in arrays
        ]
        raw_images = torch.stack(views, dim=1)
        prefix_tensor = length_tensor = None
        if prefix is not None:
            m, s = self.stats["actions"]
            prefix_tensor = torch.from_numpy(((prefix.actions - m) / s)[None]).to(
                self.device
            )
            length_tensor = torch.tensor(
                [prefix.length], dtype=torch.int64, device=self.device
            )
        return {
            "images": raw_images,
            "state": normalized_state,
            "ids": ids,
            "padded_ids": padded,
            "valid": valid,
            "action_prefix": prefix_tensor,
            "prefix_length": length_tensor,
        }

    def noise(self, seed: int | None = None):
        torch = self.torch
        generator = None
        if seed is not None:
            if (
                isinstance(seed, bool)
                or not isinstance(seed, (int, np.integer))
                or not 0 <= seed < 2**63
            ):
                raise ValueError("seed must be an integer in [0, 2**63)")
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        return torch.randn(
            1, 30, 14, dtype=torch.float32, device=self.device, generator=generator
        )

    def finish(
        self, actions: Any, *, restore_prefix: Prefix | None = None
    ) -> np.ndarray:
        a = actions.detach().float().cpu().numpy()
        if a.shape != (1, 30, 14) or not np.isfinite(a).all():
            raise AbcVlaError("ABC sampler returned invalid actions")
        mean, scale = self.stats["actions"]
        result = (a[0] * scale + mean).astype(np.float32)
        if not np.isfinite(result).all():
            raise AbcVlaError("ABC denormalization returned non-finite actions")
        if restore_prefix is not None:
            # These rows are already committed by the controller. The FP16
            # engine rounds them internally, even though its I/O is FP32.
            # Preserve the original physical actions exactly at the wire
            # boundary; newly sampled rows are never changed here.
            result[: restore_prefix.length] = restore_prefix.actions[
                : restore_prefix.length
            ]
        return result.copy()

    def load_host_weights(self):
        from safetensors.torch import load_file

        torch = self.torch
        path = verify_file(
            self.descriptor["root"], self.descriptor["host_weights"], "ABC host weights"
        )
        tensors = load_file(str(path), device=str(self.device))
        embedding = tensors.pop("vla.gemma_model.text_token_embedder.weight")
        if embedding.dtype != torch.bfloat16 or tuple(embedding.shape) != (
            262144,
            2560,
        ):
            raise AbcVlaError("ABC text embedding has the wrong shape or precision")
        self.embedding = torch.nn.Embedding.from_pretrained(embedding, freeze=True)
        self.state_mlp = torch.nn.Sequential(
            torch.nn.Linear(14, 256, dtype=torch.bfloat16, device=self.device),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 2560, dtype=torch.bfloat16, device=self.device),
        ).eval()
        state = {k.removeprefix("vla.state_proj_mlp."): v for k, v in tensors.items()}
        self.state_mlp.load_state_dict(state, strict=True)

    def trt_inputs(self, prepared: dict, instruction: str, noise: Any) -> dict:
        torch = self.torch
        key = instruction.replace("_", " ")
        ids, _, _, other = self.token_ids(instruction)
        if key not in self.prefix_cache:
            from .trt_runtime import canonical_attention_bias

            selection = torch.tensor(other, dtype=torch.int64, device=self.device)
            non_image_ids = ids.index_select(1, selection)
            embeddings = self.embedding(non_image_ids)
            normalizer = torch.tensor(
                2560**0.5, device=self.device, dtype=embeddings.dtype
            )
            language = embeddings * normalizer
            state_mask = (non_image_ids == self.state_id).unsqueeze(-1)
            positions = torch.arange(
                ids.shape[1], device=self.device, dtype=torch.int64
            )[None]
            bias = canonical_attention_bias(
                torch, ids.shape[1], dtype=torch.bfloat16, device=self.device
            )
            self.prefix_cache[key] = (language, state_mask, bias, positions)
        language, state_mask, bias, positions = self.prefix_cache[key]
        # State is evaluated on EVERY request; only input-independent token
        # embeddings, positions and mask are cached. Vision is replaced live.
        state_embedding = self.state_mlp(prepared["state"].to(torch.bfloat16))
        tail = torch.where(state_mask, state_embedding.unsqueeze(1), language)
        template = torch.cat(
            (
                torch.zeros(
                    1, VISION_TOKENS, 2560, dtype=torch.bfloat16, device=self.device
                ),
                tail,
            ),
            dim=1,
        )
        return {
            "pixel_values": (prepared["images"] - 0.5) / 0.5,
            "inputs_embeds": template,
            "attention_bias": bias,
            "position_ids": positions,
            "noise": noise,
            "cross_mask": {
                "state": prepared["state"],
                "action_prefix": prepared["action_prefix"],
                "prefix_length": prepared["prefix_length"],
            },
            "steps": 10,
        }
