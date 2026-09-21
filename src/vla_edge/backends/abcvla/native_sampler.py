"""Full numeric ABC-VLA graph. Reuses native modules and sampler unchanged.
CPU tokenization is cached separately; input pixels/state/noise remain dynamic.
"""

from __future__ import annotations

import torch
from torch import nn


class TensorVLASampler(nn.Module):
    def __init__(self, policy, num_steps=10):
        super().__init__()
        self.policy = policy
        self.num_steps = num_steps

    def forward(
        self,
        images,
        state,
        input_ids,
        valid,
        noise,
        action_prefix=None,
        prefix_length=None,
    ):
        model = self.policy
        backbone = model.vla
        normalized = backbone._normalize_images(images)
        context = backbone._embed_text(input_ids)
        context = backbone._inject_state(context, input_ids, state)
        context = backbone._encode_images(
            context, normalized.to(context.dtype), input_ids
        )
        # Native prepare embeds only the non-padded prefix, then fills the rest
        # with zero. Embedding the padded fixed-length IDs first is equivalent.
        embeddings = torch.where(valid.unsqueeze(-1), context, 0.0)
        attention, local_attention = backbone._attention_masks(
            input_ids, embeddings.dtype
        )
        positions = torch.arange(backbone.fixed_seq_len, device=images.device)
        hidden = backbone.gemma_model.model(
            hidden_states=embeddings,
            freqs_cis=backbone._freqs(positions),
            mask=attention,
            local_mask=local_attention,
            stop_layer_index=backbone.feature_layer,
            apply_final_norm=False,
        )
        with torch.autocast(device_type=images.device.type, enabled=False):
            condition = model.obs_pool(hidden.float(), key_padding_mask=~valid)
            return model.diffusion_head.sample(
                condition,
                state=state.float()
                if model.config.dit.direct_state_conditioning
                else None,
                num_steps=self.num_steps,
                noise=noise,
                action_prefix=action_prefix,
                prefix_length=prefix_length,
            )
