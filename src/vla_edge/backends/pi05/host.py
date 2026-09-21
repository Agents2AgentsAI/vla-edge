"""Checkpoint preprocessing with NumPy, Pillow and SentencePiece."""
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .bundle import Pi05Error, read_json

log = logging.getLogger(__name__)

def _resize_with_pad(image: np.ndarray | Image.Image, height: int, width: int) -> np.ndarray:
    """Match OpenPI client's PIL bilinear resize-with-zero-pad exactly.

    Takes an HxWx3 uint8 array or an RGB PIL image; a PIL image is resized as it is (the same pixels
    Image.fromarray would rebuild from its array, without the two full-frame copies)."""

    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            raise Pi05Error(f"Pi0.5 PIL images must be RGB, got mode={image.mode}")
        if image.size == (width, height):
            return np.array(image, dtype=np.uint8, copy=True)
        source = image
    else:
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
            raise Pi05Error(
                f"Pi0.5 images must be HxWx3 uint8, got shape={value.shape}, dtype={value.dtype}"
            )
        if value.shape[:2] == (height, width):
            return np.array(value, copy=True)
        source = Image.fromarray(value, mode="RGB")
    current_width, current_height = source.size
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = source.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    output = Image.new("RGB", (width, height), 0)
    output.paste(
        resized,
        (
            max(0, int((width - resized_width) / 2)),
            max(0, int((height - resized_height) / 2)),
        ),
    )
    return np.array(output, dtype=np.uint8, copy=True)



class Host:
    def __init__(self, desc, device):
        import sentencepiece
        import torch
        from safetensors.torch import load_file
        self.torch, self.device, self.desc = torch, torch.device(device), desc
        stats = read_json(desc["assets"]["norm_stats"])["norm_stats"]
        for name, width in (("state", desc["state_dim"]), ("actions", desc["action_dim"])):
            values = []
            for key in ("q01", "q99"):
                value = np.asarray(stats[name][key], dtype=np.float64)
                if value.shape != (width,) or not np.isfinite(value).all():
                    raise Pi05Error(f"invalid {name}.{key} normalization")
                values.append(value)
            if np.any(values[1] < values[0]):
                raise Pi05Error(f"invalid {name} quantile range")
            setattr(self, name + "_q", values)
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(desc["assets"]["tokenizer"]))
        if self.tokenizer.vocab_size() != 257152:
            raise Pi05Error("unexpected tokenizer vocabulary")
        self.embedding = load_file(str(desc["assets"]["embedding"]))["embedding"]
        if self.embedding.shape != (257152, 2048) or self.embedding.dtype != torch.bfloat16:
            raise Pi05Error("invalid language embedding")
        self.embedding = self.embedding.to(self.device)
        self.pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="pi05-images")

    def tokenize(self, prompt, state):
        if not isinstance(prompt, str) or not prompt.strip():
            raise Pi05Error("instruction must be a nonempty string")
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        if self.desc["discrete_state_input"]:
            q01, q99 = self.state_q
            normalized = (state - q01) / (q99 - q01 + 1e-6) * 2 - 1
            discrete = np.digitize(normalized, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            text = f"Task: {cleaned}, State: {' '.join(map(str, discrete))};\nAction: "
            tokens = self.tokenizer.encode(text, add_bos=True)
        else:
            tokens = self.tokenizer.encode(cleaned, add_bos=True) + self.tokenizer.encode("\n")
        if len(tokens) > 200:
            log.warning("prompt has %d tokens; truncating to 200", len(tokens))
        tokens = tokens[:200]
        valid = np.arange(200) < len(tokens)
        return np.asarray(tokens + [0] * (200 - len(tokens)), dtype=np.int64), valid

    def prepare(self, images, instruction, state):
        torch = self.torch
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.desc["state_dim"],) or not np.isfinite(state).all():
            raise Pi05Error("state has wrong width or non-finite values")
        if len(images) != len(self.desc["camera_names"]):
            raise Pi05Error("wrong number of cameras")
        tokens, language_mask = self.tokenize(instruction, state)
        futures = [self.pool.submit(_resize_with_pad, image, 224, 224) for image in images]
        ordered = [future.result() for future in futures]
        real_cameras = len(ordered)
        while len(ordered) < 3:
            ordered.append(np.zeros((224, 224, 3), dtype=np.uint8))
        pixels = torch.from_numpy(np.stack(ordered)).to(self.device).float()
        pixels = (pixels.permute(0, 3, 1, 2).unsqueeze(0) / 255.0 * 2.0 - 1.0).contiguous()
        ids = torch.from_numpy(tokens).to(self.device)[None]
        language = torch.nn.functional.embedding(ids, self.embedding)
        language = language * (2048 ** 0.5)
        mask = np.concatenate((np.repeat(np.arange(3) < real_cameras, 256), language_mask))
        return pixels, language, mask

    def postprocess(self, actions):
        value = actions.detach().float().cpu().numpy()[0, :, :self.desc["action_dim"]]
        q01, q99 = self.actions_q
        result = (value + 1) / 2 * (q99 - q01 + 1e-6) + q01
        if not np.isfinite(result).all():
            raise Pi05Error("model returned non-finite actions")
        return result.astype(np.float32)

    def close(self):
        self.pool.shutdown(wait=True)
        self.embedding = None
