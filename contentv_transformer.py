## Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
## 
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
## 
##     http://www.apache.org/licenses/LICENSE-2.0
## 
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

## Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_sd3.py

from diffusers import SD3Transformer2DModel
from diffusers.models.embeddings import get_3d_sincos_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
import torch
from einops import rearrange
from typing import Any, Dict, Optional, Union
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
import os

from attention.causal_attention import build_causal_attention_mask
from attention.attention_processor import SD3AttnProcessorNPU

USE_ASCEND_NPU = int(os.getenv('USE_ASCEND_NPU', '0'))
if USE_ASCEND_NPU:
    import torch_npu
    import math

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SD3PatchEmbed(torch.nn.Module):

    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = torch.nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True)

    def forward(self, x):
        b, _, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w -> b (t h w) c", b=b)

        pos_embed = get_3d_sincos_pos_embed(
            embed_dim=self.embed_dim,
            spatial_size=(w // self.patch_size, h // self.patch_size),
            temporal_size=t,
            spatial_interpolation_scale=0.5,
            temporal_interpolation_scale=1.0,
            output_type="pt",
        )
        pos_embed = pos_embed.flatten(0, 1).unsqueeze(0).to(device=x.device, dtype=x.dtype)
        return x + pos_embed

class SD3Transformer3DModel(SD3Transformer2DModel):

    def __init__(
            self,
            in_channels=16,
            out_channels=16,
            patch_size=2,
            num_layers=38,
            num_attention_heads=38,
            attention_head_dim=64,
    ):
        self.inner_dim = num_attention_heads * attention_head_dim
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            caption_projection_dim=self.inner_dim,
            pos_embed_max_size=192,
            qk_norm='rms_norm',
        )
        self.pos_embed = SD3PatchEmbed(
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
        )
        if USE_ASCEND_NPU:
            self.set_attn_processor(SD3AttnProcessorNPU())

    def unpatchify(self, x, frame, height, width):
        _, s, c = x.shape
        p = self.config.patch_size
        t, h, w = frame, height // p, width // p
        x = rearrange(x, "b (t h w) (p q c) -> b c t (h p) (w q)", h=h, w=w, p=p, q=p)
        return x

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        ## Patchify and add position embedding
        _, _, frame, height, width = hidden_states.shape
        hidden_states = self.pos_embed(hidden_states)

        ## Add timestep and text embedding
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ## Compute Attention Mask
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        # Inferred values
        num_text_tokens = encoder_hidden_states.shape[1]  # if text tokens are pre-embedded
        tokens_per_frame = (height // self.config.patch_size) * (width // self.config.patch_size)
        num_frames = frame
        causal_window = joint_attention_kwargs.get("causal_window", 3)  # default window size

        attn_mask = build_causal_attention_mask(
            num_text_tokens=num_text_tokens,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            current_frame_idx=num_frames - 1,
            causal_window=causal_window,
            device=device,
        )  # shape: (total_seq_len, total_seq_len)

        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1)  # (B, S, S)

        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}
        joint_attention_kwargs["attention_mask"] = attn_mask

        ## Transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        # Unpatchify
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        output = self.unpatchify(hidden_states, frame, height, width)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
