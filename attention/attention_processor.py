from diffusers import SD3Transformer2DModel
from diffusers.models.embeddings import get_3d_sincos_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
import torch
from einops import rearrange
from typing import Any, Dict, Optional, Union
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
import os

USE_ASCEND_NPU = int(os.getenv('USE_ASCEND_NPU', '0'))
if USE_ASCEND_NPU:
    import torch_npu
    import math

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name




class SD3AttnProcessorNPU:

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
    ):
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        ## Latent projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        ## QK-Norm
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        ## Text projections
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        ## QK-Norm
        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        ## Merge QKV
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        ## SDPA
        if not USE_ASCEND_NPU:
            hidden_states = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0,
                                                                             is_causal=False)
        else:
            hidden_states = torch_npu.npu_fusion_attention(
                query, key, value,
                head_num=attn.heads,
                input_layout="BNSD",
                pse=None,
                atten_mask=attention_mask,
                scale=1.0 / math.sqrt(head_dim),
                pre_tockens=65536,
                next_tockens=65536,
                keep_prob=1.0,
                sync=False,
                inner_precise=0,
            )[0]

        ## Unmerge QKV
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]].contiguous(),
            hidden_states[:, residual.shape[1]:].contiguous(),
        )

        # Output projection
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states, encoder_hidden_states
