"""Checkpoint loading choices that must remain stable across dependencies."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

from vla_edge import checkpoint


def test_processor_selects_exact_image_path_and_fast_tokenizer(tmp_path, monkeypatch):
    calls = []

    class Component:
        label = ""

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((cls.label, path, kwargs))
            return cls.label

    class Image(Component):
        label = "image"

    class Video(Component):
        label = "video"

    class Tokenizer(Component):
        label = "tokenizer"

    class Processor:
        @classmethod
        def get_processor_dict(cls, path, **kwargs):
            return {"saved": True}, {"resolved": True}

        @classmethod
        def from_args_and_dict(cls, args, config, **kwargs):
            return SimpleNamespace(args=args, config=config, kwargs=kwargs)

    transformers = ModuleType("transformers")
    transformers.AutoImageProcessor = Image
    transformers.AutoVideoProcessor = Video
    transformers.AutoTokenizer = Tokenizer
    dynamic = ModuleType("transformers.dynamic_module_utils")
    dynamic.get_class_from_dynamic_module = lambda ref, path: Processor
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", dynamic)
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"auto_map": {"AutoProcessor": "processing.ModelProcessor"}})
    )

    processor = checkpoint.load_processor(tmp_path)

    assert processor.args == ["image", "video", "tokenizer"]
    assert calls[0][2]["use_fast"] is False
    assert "use_fast" not in calls[1][2]
    assert calls[2][2]["use_fast"] is True
    assert calls[2][2]["extra_special_tokens"] == {}


def test_full_checkpoint_uses_transformers_dtype_keyword(tmp_path, monkeypatch):
    captured = {}
    bf16, fp16, fp32 = object(), object(), object()

    class FakeParameter:
        dtype = bf16

    class FakeModel:
        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter([FakeParameter()])

    class AutoModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            return FakeModel()

    torch = ModuleType("torch")
    torch.bfloat16 = bf16
    torch.float16 = fp16
    torch.float32 = fp32
    transformers = ModuleType("transformers")
    transformers.AutoModelForImageTextToText = AutoModel
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kwargs: str(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(checkpoint, "_patch_modeling_for_bf16", lambda path: None)
    monkeypatch.setattr(checkpoint, "load_processor", lambda path: "processor")

    model, processor, local_dir = checkpoint.load_checkpoint(
        str(tmp_path), device="cuda:7", dtype="bfloat16"
    )

    assert isinstance(model, FakeModel)
    assert processor == "processor"
    assert local_dir == str(tmp_path)
    assert captured["device"] == "cuda:7"
    assert captured["kwargs"]["dtype"] is bf16
    assert "torch_dtype" not in captured["kwargs"]
