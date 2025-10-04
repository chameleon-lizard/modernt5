# coding=utf-8
# Copyright 2025 ModernBERT-T5Gemma team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    add_start_docstrings,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg

# Import ModernBERT for encoder
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertConfig,
    ModernBertModel,
    ModernBertPreTrainedModel,
)

# Import ModernBERT for encoder
from transformers.models.t5gemma.modeling_t5gemma import (
    T5GemmaRMSNorm,
    T5GemmaMLP,
    T5GemmaRotaryEmbedding,
    T5GemmaSelfAttention,
    T5GemmaCrossAttention,
    #ModernBertModel,
    #ModernBertPreTrainedModel,
)


# Import masking utilities
try:
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )
except ImportError:
    # Fallback for older transformers versions
    from transformers.models.gemma2.modeling_gemma2 import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

logger = logging.get_logger(__name__)


# ============================================================================
# Configuration Classes
# ============================================================================

class ModernBertT5GemmaDecoderConfig(PretrainedConfig):
    """Configuration for the Gemma-style decoder in ModernBERT-T5Gemma"""
    
    model_type = "modernbert_t5gemma_decoder"
    
    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=21,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        dropout_rate=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        layer_types=None,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        cross_attention_hidden_size=None,
        is_decoder=True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.cross_attention_hidden_size = cross_attention_hidden_size
        self.is_decoder = is_decoder
        self.layer_types = layer_types
        
        if self.layer_types is None:
            # Alternate between sliding and full attention
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]


class ModernBertT5GemmaConfig(PretrainedConfig):
    """Configuration for ModernBERT-T5Gemma hybrid model"""
    
    model_type = "modernbert_t5gemma"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        encoder: Optional[Union[ModernBertConfig, dict]] = None,
        decoder: Optional[Union[ModernBertT5GemmaDecoderConfig, dict]] = None,
        is_encoder_decoder: bool = True,
        dropout_rate: float = 0.0,
        classifier_dropout_rate: float = 0.0,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        vocab_size: int = 128256,
        **kwargs,
    ):
        # Initialize encoder config
        if isinstance(encoder, dict):
            encoder = ModernBertConfig(**encoder)
        elif encoder is None:
            encoder = ModernBertConfig(vocab_size=vocab_size)
        
        # Initialize decoder config  
        if isinstance(decoder, dict):
            decoder = ModernBertT5GemmaDecoderConfig(**decoder)
        elif decoder is None:
            # Create decoder config with matching dimensions
            decoder = ModernBertT5GemmaDecoderConfig(
                vocab_size=vocab_size,
                hidden_size=encoder.hidden_size,
                cross_attention_hidden_size=encoder.hidden_size,
            )
        
        # Set cross-attention hidden size to match encoder
        decoder.cross_attention_hidden_size = encoder.hidden_size
        decoder.is_decoder = True
        decoder.use_cache = True
        decoder.dropout_rate = dropout_rate
        decoder.attention_dropout = attention_dropout
        
        self.encoder = encoder
        self.decoder = decoder
        
        super().__init__(**kwargs)
        
        self.is_encoder_decoder = is_encoder_decoder
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.classifier_dropout_rate = classifier_dropout_rate
        self.use_cache = decoder.use_cache
        self.initializer_range = decoder.initializer_range


# ============================================================================
# Decoder Components (Gemma2-style)
# ============================================================================

# class ModernBertT5GemmaRMSNorm(nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.zeros(dim))
    
#     def _norm(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
#     def forward(self, x):
#         output = self._norm(x.float())
#         output = output * (1.0 + self.weight.float())
#         return output.type_as(x)

# class ModernBertT5GemmaMLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
#         self.act_fn = ACT2FN[config.hidden_activation]
#         self.dropout = nn.Dropout(config.dropout_rate)
    
#     def forward(self, x):
#         hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
#         hidden_states = self.dropout(hidden_states)
#         return self.down_proj(hidden_states)


# class ModernBertT5GemmaRotaryEmbedding(nn.Module):
#     def __init__(self, config, device=None):
#         super().__init__()
#         self.rope_type = "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.config = config
        
#         # Initialize inv_freq
#         head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
#         base = config.rope_theta
#         inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.attention_scaling = 1.0
    
#     @torch.no_grad()
#     def forward(self, x, position_ids):
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
#         position_ids_expanded = position_ids[:, None, :].float()
        
#         device_type = x.device.type if x.device.type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos() * self.attention_scaling
#             sin = emb.sin() * self.attention_scaling
        
#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
#     """Apply Rotary Position Embedding to query and key tensors."""
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """Multi-Query Attention: Repeat KV heads to match Q heads"""
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# def eager_attention_forward(
#     module: nn.Module,
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     attention_mask: Optional[torch.Tensor],
#     dropout: float = 0.0,
#     scaling: Optional[float] = None,
#     softcap: Optional[float] = None,
#     **kwargs,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     if scaling is None:
#         scaling = module.head_dim ** -0.5
    
#     key_states = repeat_kv(key, module.num_key_value_groups)
#     value_states = repeat_kv(value, module.num_key_value_groups)
    
#     attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
#     if softcap is not None:
#         attn_weights = attn_weights / softcap
#         attn_weights = torch.tanh(attn_weights)
#         attn_weights = attn_weights * softcap
    
#     if attention_mask is not None:
#         causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#         attn_weights = attn_weights + causal_mask
    
#     attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
#     attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
#     attn_output = torch.matmul(attn_weights, value_states)
#     attn_output = attn_output.transpose(1, 2).contiguous()
    
#     return attn_output, attn_weights

# class ModernBertT5GemmaSelfAttention(nn.Module):
#     """Self-attention for decoder"""
    
#     def __init__(self, config: ModernBertT5GemmaDecoderConfig, layer_idx: int):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
#         self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
#         self.scaling = config.query_pre_attn_scalar ** -0.5
#         self.attention_dropout = config.attention_dropout
#         self.is_causal = config.is_decoder
        
#         self.q_proj = nn.Linear(
#             config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.k_proj = nn.Linear(
#             config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.v_proj = nn.Linear(
#             config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.o_proj = nn.Linear(
#             config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
#         )
        
#         self.attn_logit_softcapping = config.attn_logit_softcapping
#         self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
    
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         position_embeddings: tuple[torch.Tensor, torch.Tensor],
#         attention_mask: Optional[torch.Tensor],
#         past_key_values: Optional[Cache] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
#         input_shape = hidden_states.shape[:-1]
#         hidden_shape = (*input_shape, -1, self.head_dim)
        
#         query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#         key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#         value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
#         cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
#         if past_key_values is not None:
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
#         attention_interface: Callable = eager_attention_forward
#         if self.config._attn_implementation != "eager":
#             attention_interface = ALL_ATTENTION_FUNCTIONS.get(
#                 self.config._attn_implementation, eager_attention_forward
#             )
        
#         attn_output, attn_weights = attention_interface(
#             self,
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             dropout=self.attention_dropout if self.training else 0.0,
#             scaling=self.scaling,
#             sliding_window=self.sliding_window,
#             softcap=self.attn_logit_softcapping,
#             **kwargs,
#         )
        
#         attn_output = attn_output.reshape(*input_shape, -1).contiguous()
#         attn_output = self.o_proj(attn_output)
#         return attn_output, attn_weights


# class ModernBertT5GemmaCrossAttention(nn.Module):
#     """Cross-attention for attending to encoder outputs"""
    
#     def __init__(self, config: ModernBertT5GemmaDecoderConfig, layer_idx: int):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
#         self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
#         self.scaling = config.query_pre_attn_scalar ** -0.5
#         self.attention_dropout = config.attention_dropout
#         self.is_causal = False
        
#         if config.cross_attention_hidden_size is None:
#             raise ValueError("Cross-attention requires cross_attention_hidden_size to be specified")
        
#         self.q_proj = nn.Linear(
#             config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.k_proj = nn.Linear(
#             config.cross_attention_hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.v_proj = nn.Linear(
#             config.cross_attention_hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.o_proj = nn.Linear(
#             config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
#         )
        
#         self.attn_logit_softcapping = config.attn_logit_softcapping
    
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor],
#         encoder_hidden_states: Optional[torch.Tensor],
#         past_key_values: Optional[EncoderDecoderCache] = None,
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
#         if encoder_hidden_states is None:
#             raise ValueError("Encoder hidden states are required for cross-attention")
        
#         input_shape = hidden_states.shape[:-1]
#         hidden_shape = (*input_shape, -1, self.head_dim)
#         query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
#         if past_key_values is not None:
#             is_updated = past_key_values.is_updated.get(self.layer_idx, False)
#             curr_past_key_value = past_key_values.cross_attention_cache
        
#         if past_key_values is None or not is_updated:
#             encoder_input_shape = encoder_hidden_states.shape[:-1]
#             encoder_hidden_shape = (*encoder_input_shape, -1, self.head_dim)
#             key_states = self.k_proj(encoder_hidden_states).view(encoder_hidden_shape).transpose(1, 2)
#             value_states = self.v_proj(encoder_hidden_states).view(encoder_hidden_shape).transpose(1, 2)
            
#             if past_key_values is not None:
#                 key_states, value_states = curr_past_key_value.update(key_states, value_states, self.layer_idx)
#                 past_key_values.is_updated[self.layer_idx] = True
#         else:
#             key_states = curr_past_key_value.layers[self.layer_idx].keys
#             value_states = curr_past_key_value.layers[self.layer_idx].values
        
#         attention_interface: Callable = eager_attention_forward
#         if self.config._attn_implementation != "eager":
#             attention_interface = ALL_ATTENTION_FUNCTIONS.get(
#                 self.config._attn_implementation, eager_attention_forward
#             )
        
#         attn_output, attn_weights = attention_interface(
#             self,
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             dropout=self.attention_dropout if self.training else 0.0,
#             scaling=self.scaling,
#             sliding_window=None,
#             softcap=self.attn_logit_softcapping,
#             **kwargs,
#         )
        
#         attn_output = attn_output.reshape(*input_shape, -1).contiguous()
#         attn_output = self.o_proj(attn_output)
#         return attn_output, attn_weights

class ModernBertT5GemmaDecoderLayer(GradientCheckpointingLayer):
    """Decoder layer with self-attention and cross-attention"""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        
        self.self_attn = T5GemmaSelfAttention(config, layer_idx)
        self.cross_attn = T5GemmaCrossAttention(config, layer_idx)
        
        self.pre_self_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_cross_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_cross_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.mlp = T5GemmaMLP(config)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values.self_attention_cache if past_key_values is not None else None,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        
        # Cross-attention
        residual = hidden_states
        hidden_states = self.pre_cross_attn_layernorm(hidden_states)
        hidden_states, _ = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_cross_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        
        return hidden_states


# ============================================================================
# Model Classes
# ============================================================================

class ModernBertT5GemmaPreTrainedModel(PreTrainedModel):
    config_class = ModernBertT5GemmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ModernBertEncoderLayer", "T5GemmaDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, T5GemmaRMSNorm):
            module.weight.data.zero_()  # RMSNorm uses (1 + weight)
    
    def _shift_right(self, input_ids):
        """Shift input ids one token to the right for decoder inputs"""
        decoder_start_token_id = self.config.decoder.bos_token_id
        pad_token_id = self.config.decoder.pad_token_id
        
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        
        # Replace possible -100 values in labels by pad_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        
        return shifted_input_ids


class ModernBertT5GemmaDecoder(ModernBertT5GemmaPreTrainedModel):
    """Gemma-style decoder stack"""
    
    def __init__(self, config: ModernBertT5GemmaDecoderConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.norm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = T5GemmaRotaryEmbedding(config)
        
        self.layers = nn.ModuleList([
            ModernBertT5GemmaDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gradient_checkpointing = False
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided for decoder")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        
        # Initialize cache if needed
        if not self.training and use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(
                DynamicCache(config=self.config),
                DynamicCache(config=self.config)
            )
            # Initialize is_updated tracking
            past_key_values.is_updated = {i: False for i in range(len(self.layers))}
        
        # Setup position ids and cache position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_len, device=device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        # Prepare attention masks
        if attention_mask is None and past_key_values is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
        
        # Create causal masks
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values.self_attention_cache if past_key_values is not None else None,
                "position_ids": position_ids,
            }
            
            self_attn_mask_mapping = {}
            for layer_type in set(self.config.layer_types):
                if layer_type == "full_attention":
                    self_attn_mask_mapping[layer_type] = create_causal_mask(**mask_kwargs)
                elif layer_type == "sliding_attention":
                    self_attn_mask_mapping[layer_type] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            self_attn_mask_mapping = attention_mask
        
        # Process encoder attention mask
        if encoder_attention_mask is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_attention_mask = encoder_attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_len, encoder_seq_len
            ).to(inputs_embeds.dtype)
            encoder_attention_mask = (1.0 - encoder_attention_mask) * torch.finfo(inputs_embeds.dtype).min
        
        # Forward through layers
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Scale embeddings
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        hidden_states = self.dropout(hidden_states)
        
        for i, layer_module in enumerate(self.layers):
            layer_attention_mask = self_attn_mask_mapping.get(
                self.config.layer_types[i], self_attn_mask_mapping["full_attention"]
            )
            
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer_module,
                    hidden_states,
                    position_embeddings,
                    layer_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_values,
                    use_cache,
                    cache_position,
                )
            else:
                hidden_states = layer_module(
                    hidden_states,
                    position_embeddings,
                    layer_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_values,
                    use_cache,
                    cache_position,
                    **kwargs,
                )
        
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@add_start_docstrings(
    "ModernBERT-T5Gemma encoder-decoder model",
)
class ModernBertT5GemmaModel(ModernBertT5GemmaPreTrainedModel):
    
    def __init__(self, config: ModernBertT5GemmaConfig):
        super().__init__(config)
        
        if not config.is_encoder_decoder:
            raise ValueError("This model only supports encoder-decoder mode")
        
        # Use ModernBERT as encoder
        self.encoder = ModernBertModel(config.encoder)
        
        # Use Gemma-style decoder
        self.decoder = ModernBertT5GemmaDecoder(config.decoder)
        
        self.post_init()
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()
    
    def set_input_embeddings(self, new_embeddings):
        self.encoder.set_input_embeddings(new_embeddings)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Seq2SeqModelOutput:
        # Encode if needed
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            encoder_last_hidden_state=encoder_hidden_states,
        )


@add_start_docstrings(
    "ModernBERT-T5Gemma for conditional generation",
)
class ModernBertT5GemmaForConditionalGeneration(ModernBertT5GemmaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight", "model.decoder.embed_tokens.weight"]
    
    def __init__(self, config: ModernBertT5GemmaConfig):
        config.is_encoder_decoder = True
        super().__init__(config)
        
        self.model = ModernBertT5GemmaModel(config)
        self.lm_head = nn.Linear(config.decoder.hidden_size, config.vocab_size, bias=False)
        
        self.post_init()
    
    def get_encoder(self):
        return self.model.encoder
    
    def get_decoder(self):
        return self.model.decoder
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.lm_head, self.model.decoder.get_input_embeddings())
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        # Prepare decoder input ids from labels if needed
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)
        
        # Forward through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        
        # Language modeling head
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        # Apply logit softcapping if configured
        if self.config.decoder.final_logit_softcapping is not None:
            cap = self.config.decoder.final_logit_softcapping
            logits = logits / cap
            logits = torch.tanh(logits)
            logits = logits * cap
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        )
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # Cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            if past_length > 0:
                decoder_input_ids = decoder_input_ids[:, -1:]
        
        return {
            "input_ids": None,  # Encoder inputs not needed during generation
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "use_cache": True,
        }


__all__ = [
    "ModernBertT5GemmaConfig",
    "ModernBertT5GemmaDecoderConfig", 
    "ModernBertT5GemmaModel",
    "ModernBertT5GemmaForConditionalGeneration",
    "ModernBertT5GemmaPreTrainedModel",
]