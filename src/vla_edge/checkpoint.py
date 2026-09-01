"""Checkpoint loading, and the workarounds it requires.

This module exists so that every backend inherits the same set of fixes rather
than rediscovering them. All four of the workarounds below were found by
something failing in a way that did not point at the cause; none of them are
optional, and none of them are our preference about how loading should work.

They are also why `docs/spec.md` argues for keeping one shared host path: a
backend that reimplements loading reacquires these bugs independently.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("vla_edge.checkpoint")

#: Textual patches applied to the checkpoint's own modeling code.
#: Marked idempotent by a sentinel comment so re-running is safe.
_BF16_PATCHES = [
    (
        "device=device,\n            dtype=torch.float32,\n            generator=generator,",
        (
            "device=device,\n"
            "            dtype=source_tensor.dtype,  # patched_bf16_dtype\n"
            "            generator=generator,"
        ),
        "patched_bf16_dtype",
    ),
    (
        "return value.detach().cpu().numpy().astype(np.float32, copy=False)",
        (
            "return value.detach().cpu().float().numpy()"
            ".astype(np.float32, copy=False)  # patched_bf16_to_array"
        ),
        "patched_bf16_to_array",
    ),
]


def _patch_modeling_for_bf16(local_dir: str) -> None:
    """Make the checkpoint's modeling code bf16-safe.

    Two edits, both idempotent:

    1. The flow-matching trajectory is allocated as hardcoded float32; under
       bf16 weights this mismatches at the first matmul.
    2. ``_to_array`` calls ``.numpy()`` on a bf16 tensor, which numpy has no
       dtype for.

    Applied to the snapshot AND to the copy under
    ``~/.cache/huggingface/modules/transformers_modules/``, because the latter
    is what ``trust_remote_code`` actually imports. Patching only the snapshot
    looks like it worked and changes nothing.

    A "needle not found" warning is expected on revisions that already fixed
    the issue upstream; it is not an error.
    """
    candidates = [os.path.join(local_dir, "modeling_molmoact2.py")]
    modules_root = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules"
    )
    if os.path.isdir(modules_root):
        for sub in os.listdir(modules_root):
            p = os.path.join(modules_root, sub, "modeling_molmoact2.py")
            if os.path.isfile(p):
                candidates.append(p)

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            continue
        new_src = src
        applied: list[str] = []
        for needle, replacement, marker in _BF16_PATCHES:
            if marker in new_src:
                continue
            if needle not in new_src:
                log.debug("patch %s: needle not found in %s", marker, path)
                continue
            new_src = new_src.replace(needle, replacement, 1)
            applied.append(marker)
        if new_src != src:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(new_src)
            log.info("applied %s in %s", applied, path)


def load_processor(local_dir: str | os.PathLike[str]) -> Any:
    """Load the saved processor without changing its component choices.

    ``AutoProcessor.from_pretrained`` forwards one ``use_fast`` flag to every
    component. MolmoAct2 needs the saved Python image processor for exact
    parity, but its tokenizer should remain the fast Qwen tokenizer. Leaving
    the flag unset selects those two choices correctly but emits a warning;
    setting it either way changes one of them. Load the components explicitly
    so the choices are stable across Transformers releases and no warning has
    to be hidden.
    """
    from transformers import (
        AutoImageProcessor,
        AutoTokenizer,
        AutoVideoProcessor,
    )
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    local_dir = str(local_dir)
    common = {"trust_remote_code": True}
    image_processor = AutoImageProcessor.from_pretrained(
        local_dir, use_fast=False, **common
    )
    video_processor = AutoVideoProcessor.from_pretrained(local_dir, **common)
    tokenizer = AutoTokenizer.from_pretrained(
        local_dir,
        use_fast=True,
        extra_special_tokens={},
        **common,
    )

    processor_config = json.loads(
        (Path(local_dir) / "processor_config.json").read_text()
    )
    class_ref = processor_config.get("auto_map", {}).get("AutoProcessor")
    if not class_ref:
        raise ValueError(
            f"{local_dir}/processor_config.json has no AutoProcessor mapping"
        )
    processor_class = get_class_from_dynamic_module(class_ref, local_dir)
    processor_dict, processor_kwargs = processor_class.get_processor_dict(
        local_dir, trust_remote_code=True
    )
    return processor_class.from_args_and_dict(
        [image_processor, video_processor, tokenizer],
        processor_dict,
        **processor_kwargs,
    )


def load_checkpoint(
    repo_id: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
) -> tuple[Any, Any, str]:
    """Load model + processor with every known workaround applied.

    Returns ``(model, processor, local_dir)``.

    ``dtype`` must remain bf16 for any checkpoint trained in bf16. This is not
    a performance preference: fp16 does not have the exponent range for these
    activations, and the failure is silent. See ``docs/spec.md``.
    """
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    if torch_dtype is torch.float16:
        log.warning(
            "loading in float16. If this checkpoint was trained in bfloat16, "
            "the language backbone will emit corrupted key/value tensors "
            "WITHOUT raising. Verify against docs/gates.md before trusting "
            "any output."
        )

    # Workaround 1: `predict_action` resolves norm_stats.json from
    # `config._name_or_path`. Loading by repo id leaves that a non-path string
    # and it fails at inference time, not load time.
    local_dir = repo_id if os.path.isdir(repo_id) else snapshot_download(repo_id=repo_id)
    log.info("checkpoint dir: %s", local_dir)

    # Workaround 2: bf16 patches to the checkpoint's own modeling code.
    _patch_modeling_for_bf16(local_dir)

    # Workaround 3: select the saved image processor and fast tokenizer
    # explicitly. Besides avoiding a misleading Transformers warning, this
    # keeps a future default change from altering image preprocessing.
    processor = load_processor(local_dir)

    model = (
        AutoModelForImageTextToText.from_pretrained(
            local_dir, trust_remote_code=True, dtype=torch_dtype
        )
        .to(device)
        .eval()
    )

    # Workaround 4: upstream `_move_inputs_to_device` moves tensors but does
    # not cast them. The processor emits fp32 pixel_values; against bf16
    # weights that raises "mat1 and mat2 must have the same dtype" from deep
    # inside the vision tower, which points nowhere useful.
    target_dtype = next(model.parameters()).dtype

    def _move_and_cast(inputs: Any, dev: Any, _t: Any = target_dtype) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                value = value.to(dev)
                if value.is_floating_point() and value.dtype != _t:
                    value = value.to(_t)
            out[key] = value
        return out

    model._move_inputs_to_device = _move_and_cast

    return model, processor, local_dir
