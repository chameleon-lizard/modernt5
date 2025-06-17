import math
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.generation import GenerationMixin # Import GenerationMixin

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    # add_start_docstrings_to_model_forward,
    logging,
    # replace_return_docstrings,
)
# Import from the provided modeling_modernbert.py
# Ensure this import path is correct based on your file structure.
# For example, if modeling_modernbert.py is in the same directory:
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertConfig,
    ModernBertModel, # Encoder
    ModernBertPreTrainedModel, # Base for our ModernT5PreTrainedModel
    ModernBertEmbeddings, # Used as a reference for embedding structure
    ModernBertMLP, # Used in decoder layers
    ModernBertRotaryEmbedding, # RoPE implementation for padded attention path
    apply_rotary_pos_emb,
    # _prepare_4d_attention_mask, # We'll use T5-style mask helpers
)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "your-org/ModernT5-base" # Placeholder
_CONFIG_FOR_DOC = "ModernT5Config"
MODERNBERT_START_DOCSTRING = ModernBertModel.__doc__.split("Parameters:")[0] # Reuse ModernBert's intro


# Helper functions for masks (inspired by T5's implementation)
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for decoder self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None, src_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, L = mask.size()
    tgt_len = tgt_len if tgt_len is not None else L
    src_len = src_len if src_len is not None else L

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class ModernT5Config(ModernBertConfig):
    model_type = "modernt5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_decoder_layers: Optional[int] = None,
        decoder_intermediate_size: Optional[int] = None,
        decoder_hidden_activation: Optional[str] = None,
        decoder_num_attention_heads: Optional[int] = None, # Renamed for clarity
        decoder_attention_bias: Optional[bool] = None,
        decoder_mlp_bias: Optional[bool] = None,
        decoder_norm_eps: Optional[float] = None,
        decoder_norm_bias: Optional[bool] = None,
        decoder_rope_theta: Optional[float] = None,
        # ModernBERT local attention params for decoder (if desired, defaults to global)
        decoder_local_attention: Optional[int] = -1, # -1 means global
        decoder_global_attn_every_n_layers: Optional[int] = 1,
        decoder_local_rope_theta: Optional[float] = None,
        tie_word_embeddings: bool = True,
        decoder_start_token_id: int = 0, # Typically same as pad_token_id for T5
        # Add use_cache and return_dict to the signature with default values
        use_cache: bool = True,
        return_dict: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs) # Pass remaining kwargs to ModernBertConfig
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else self.num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size if decoder_intermediate_size is not None else self.intermediate_size
        self.decoder_hidden_activation = decoder_hidden_activation if decoder_hidden_activation is not None else self.hidden_activation
        self.decoder_num_attention_heads = decoder_num_attention_heads if decoder_num_attention_heads is not None else self.num_attention_heads
        self.decoder_attention_bias = decoder_attention_bias if decoder_attention_bias is not None else self.attention_bias
        self.decoder_mlp_bias = decoder_mlp_bias if decoder_mlp_bias is not None else self.mlp_bias
        self.decoder_norm_eps = decoder_norm_eps if decoder_norm_eps is not None else self.norm_eps
        self.decoder_norm_bias = decoder_norm_bias if decoder_norm_bias is not None else self.norm_bias
        self.decoder_rope_theta = decoder_rope_theta if decoder_rope_theta is not None else self.global_rope_theta

        self.decoder_local_attention = decoder_local_attention
        self.decoder_global_attn_every_n_layers = decoder_global_attn_every_n_layers
        self.decoder_local_rope_theta = decoder_local_rope_theta

        self.tie_word_embeddings = tie_word_embeddings
        self.decoder_start_token_id = decoder_start_token_id
        # Ensure pad_token_id is set for _shift_right, often same as decoder_start_token_id
        if self.pad_token_id is None and self.decoder_start_token_id == 0:
             self.pad_token_id = 0 # Common T5 default

        self.use_cache = use_cache
        self.return_dict = return_dict


class ModernT5Attention(nn.Module):
    def __init__(self, config: ModernT5Config, is_decoder: bool = True, layer_id: Optional[int] = None, is_self_attn: bool = True):
        super().__init__()
        self.config = config
        self.is_decoder = is_decoder # Always True for this module in ModernT5 context
        self.layer_id = layer_id
        self.is_self_attn = is_self_attn

        self.hidden_size = config.hidden_size
        self.num_heads = config.decoder_num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_bias = config.decoder_attention_bias
        self.attention_dropout = config.attention_dropout # Assuming same dropout as encoder

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention heads ({self.num_heads})"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.attention_bias)

        if self.is_self_attn: # RoPE only for self-attention in decoder
            rope_theta = config.decoder_rope_theta
            max_position_embeddings = config.max_position_embeddings # Use global max for RoPE

            # Decoder local attention RoPE config (if applicable, similar to ModernBertAttention)
            # Not implementing sliding window masking here, just RoPE param adjustment. Causal mask is primary.
            if config.decoder_local_attention > 0 and \
               (layer_id % config.decoder_global_attn_every_n_layers != 0):
                if config.decoder_local_rope_theta is not None:
                    rope_theta = config.decoder_local_rope_theta
                # max_position_embeddings = config.decoder_local_attention # RoPE cache length

            # For decoder, using ModernBertRotaryEmbedding (padded attention path)
            self.rotary_emb = ModernBertRotaryEmbedding(
                config=config, # Pass full config for rope_scaling type etc.
                dim=self.head_dim,
                base=rope_theta,
            )
            self.rotary_emb.max_seq_len_cached = max_position_embeddings
            self.rotary_emb.original_max_seq_len = max_position_embeddings


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # For RoPE in self-attention
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        is_cross_attention = key_value_states is not None

        if is_cross_attention:
            query_states = self.q_proj(hidden_states)
            # K and V are projected from encoder_hidden_states (key_value_states)
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
            kv_seq_len = key_value_states.size(1)
        else: # Self-attention
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            kv_seq_len = q_len # Length of K/V before considering cache

        query_states = self._shape(query_states, q_len, bsz) # (bsz, n_heads, q_len, head_dim)
        # For K/V, seq_len might be different due to cache or cross-attention
        key_states = self._shape(key_states, kv_seq_len if not (self.is_self_attn and past_key_value) else q_len, bsz)
        value_states = self._shape(value_states, kv_seq_len if not (self.is_self_attn and past_key_value) else q_len, bsz)


        if self.is_self_attn:
            if position_ids is None: # Should be provided by the caller stack
                past_kv_len = past_key_value[0].shape[2] if past_key_value is not None else 0
                position_ids = torch.arange(
                    past_kv_len, past_kv_len + q_len, dtype=torch.long, device=hidden_states.device
                ).unsqueeze(0)

            cos, sin = self.rotary_emb(query_states, position_ids=position_ids)
            # Apply RoPE to query and *new* keys
            query_states, key_states_rotated = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # Concatenate past K/V with new K/V
                key_states = torch.cat([past_key_value[0], key_states_rotated], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2) # value_states are new, not rotated
            else:
                key_states = key_states_rotated
        # Else (cross-attention), K/V are from encoder, no RoPE here.

        if use_cache and self.is_self_attn: # Cache only for self-attention
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        # SDPA handles causal masking if attention_mask is None and is_causal=True.
        # Here, attention_mask is expected to be pre-computed (causal + padding).
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attn_weights = None # F.sdp doesn't return weights directly by default.

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, present_key_value


class ModernT5Layer(nn.Module):
    def __init__(self, config: ModernT5Config, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.self_attn_layer_norm = nn.RMSNorm(
            config.hidden_size, eps=config.decoder_norm_eps,
        )
        self.self_attention = ModernT5Attention(config, is_decoder=True, layer_id=layer_id, is_self_attn=True)

        self.cross_attn_layer_norm = nn.RMSNorm(
            config.hidden_size, eps=config.decoder_norm_eps, 
        )
        self.cross_attention = ModernT5Attention(config, is_decoder=True, layer_id=layer_id, is_self_attn=False)

        self.mlp_layer_norm = nn.RMSNorm(
            config.hidden_size, eps=config.decoder_norm_eps, 
        )
        # Create a temporary ModernBertConfig for the MLP, using decoder_intermediate_size
        mlp_config = ModernBertConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.decoder_intermediate_size,
            hidden_activation=config.decoder_hidden_activation,
            mlp_dropout=config.mlp_dropout, # Assuming same dropout
            mlp_bias=config.decoder_mlp_bias,
        )
        self.mlp = ModernBertMLP(mlp_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,         # Causal mask for self-attn
        position_ids: Optional[torch.LongTensor] = None,       # For RoPE in self-attn
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None, # Mask for cross-attn
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # For self-attn cache
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        residual = hidden_states
        normed_hidden_states = self.self_attn_layer_norm(hidden_states)

        self_attn_outputs = self.self_attention(
            hidden_states=normed_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value, # Only self-attn KV cache
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_output = self_attn_outputs[0]
        hidden_states = residual + attn_output

        present_key_value = self_attn_outputs[2] if use_cache else None
        all_self_attentions = (self_attn_outputs[1],) if output_attentions else ()

        # Cross-Attention
        residual = hidden_states
        normed_hidden_states = self.cross_attn_layer_norm(hidden_states)
        cross_attn_outputs = self.cross_attention(
            hidden_states=normed_hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            # past_key_value for cross-attn is not typically cached per step, KVs are static from encoder
            output_attentions=output_attentions,
            use_cache=False, # No KV cache for cross-attention that changes per step
        )
        attn_output = cross_attn_outputs[0]
        hidden_states = residual + attn_output
        all_cross_attentions = (cross_attn_outputs[1],) if output_attentions else ()

        # MLP
        residual = hidden_states
        normed_hidden_states = self.mlp_layer_norm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = residual + mlp_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (all_self_attentions + all_cross_attentions,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class ModernT5PreTrainedModel(ModernBertPreTrainedModel, GenerationMixin): # Inherit from ModernBert's base
    config_class = ModernT5Config
    base_model_prefix = "model" # Standard for encoder-decoder
    supports_gradient_checkpointing = True # Set to True if layers support it
    _no_split_modules = ["ModernBertEncoderLayer", "ModernT5Layer"]

    def _init_weights(self, module: nn.Module):
        super()._init_weights(module) # Use ModernBert's initialization logic

    # _autoset_attn_implementation from ModernBertPreTrainedModel is inherited.
    # It will set config._attn_implementation which ModernBertModel (encoder) uses.
    # Decoder (ModernT5Attention) uses F.sdpa, which has its own internal dispatch.


class ModernT5Stack(ModernT5PreTrainedModel): # Decoder stack
    def __init__(self, config: ModernT5Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.config = config

        self.embed_tokens = embed_tokens
        if self.embed_tokens is None:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.dropout = nn.Dropout(config.embedding_dropout if hasattr(config, 'embedding_dropout') else config.hidden_dropout_prob)
        self.layers = nn.ModuleList(
            [ModernT5Layer(config, layer_id=i) for i in range(config.num_decoder_layers)]
        )
        self.final_layer_norm = nn.RMSNorm(
            config.hidden_size, eps=config.decoder_norm_eps, 
        )
        self.gradient_checkpointing = False # Default, can be enabled
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        if input_shape[-1] > 1 or past_key_values_length > 0 : # Only need causal mask if seq_len > 1 or generating
            combined_attention_mask = _make_causal_mask(
                input_shape, dtype, device, past_key_values_length=past_key_values_length
            )
        if attention_mask is not None: # Provided padding mask
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1])
            if combined_attention_mask is not None:
                combined_attention_mask = expanded_attn_mask + combined_attention_mask
            else:
                combined_attention_mask = expanded_attn_mask
        return combined_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # Decoder input padding mask
        position_ids: Optional[torch.LongTensor] = None, # Decoder RoPE position_ids
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None, # Encoder output padding mask
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify input_ids or inputs_embeds, not both.")
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape[:-1]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.dropout(inputs_embeds)

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if position_ids is None:
            seq_length = input_shape[-1]
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape[0], seq_length)

        self_attn_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        
        cross_attn_mask = None
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            q_len = input_shape[-1]
            kv_seq_len = encoder_hidden_states.shape[1]
            cross_attn_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=q_len, src_len=kv_seq_len)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions and encoder_hidden_states is not None else None
        next_decoder_cache = [] if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=self_attn_mask,
                position_ids=position_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=cross_attn_mask,
                past_key_value=layer_past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache.append(layer_outputs[-1])
            if output_attentions:
                all_self_attns += (layer_outputs[1][0],) # (self_attn_weights, cross_attn_weights)
                if encoder_hidden_states is not None:
                    all_cross_attns += (layer_outputs[1][1],)
        
        next_cache = tuple(next_decoder_cache) if use_cache else None
        hidden_states = self.final_layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attns] if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns or None, # Ensure None if empty
            cross_attentions=all_cross_attns or None, # Ensure None if empty
        )


@add_start_docstrings(
    "The ModernT5 Model transformer which combines a ModernBERT encoder and a ModernT5 decoder.",
    MODERNBERT_START_DOCSTRING,
)
class ModernT5Model(ModernT5PreTrainedModel, GenerationMixin):
    def __init__(self, config: ModernT5Config):
        super().__init__(config)
        self.config = config

        if config.tie_word_embeddings:
            self.shared = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        else:
            self.shared = None # No globally shared embedding module

        encoder_config = ModernBertConfig(**config.to_dict())
        encoder_config._attn_implementation = config._attn_implementation
        self.encoder = ModernBertModel(encoder_config)

        if self.shared is not None:
            self.encoder.set_input_embeddings(self.shared)

        self.decoder = ModernT5Stack(config, embed_tokens=self.shared)

    def get_input_embeddings(self):
        return self.shared if self.shared is not None else self.encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        if self.shared is not None:
            self.shared = new_embeddings
            self.encoder.set_input_embeddings(self.shared)
            self.decoder.set_input_embeddings(self.shared) # ModernT5Stack also needs its reference updated
        else:
            self.encoder.set_input_embeddings(new_embeddings)
            if self.decoder.embed_tokens is None: # If decoder has its own separate embedding
                 self.decoder.set_input_embeddings(new_embeddings) # Or decide if it should stay separate

    def get_encoder(self): return self.encoder
    def get_decoder(self): return self.decoder

    # @add_start_docstrings_to_model_forward(MODERNBERT_INPUTS_DOCSTRING.replace("ModernBert", "ModernT5"))
    # @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # Encoder attention mask
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None, # Decoder padding mask
        decoder_position_ids: Optional[torch.LongTensor] = None, # Decoder RoPE position_ids
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # Encoder position_ids
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_args = {
                "input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids,
                "inputs_embeds": inputs_embeds, "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states, "return_dict": return_dict,
            }
            encoder_outputs = self.encoder(**{k:v for k,v in encoder_args.items() if v is not None})
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        encoder_hidden_states = encoder_outputs[0]

        decoder_args = {
            "input_ids": decoder_input_ids, "attention_mask": decoder_attention_mask,
            "position_ids": decoder_position_ids, "inputs_embeds": decoder_inputs_embeds,
            "encoder_hidden_states": encoder_hidden_states, "encoder_attention_mask": attention_mask,
            "past_key_values": past_key_values, "use_cache": use_cache,
            "output_attentions": output_attentions, "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        decoder_outputs = self.decoder(**{k:v for k,v in decoder_args.items() if v is not None})

        if not return_dict:
            return decoder_outputs + encoder_outputs
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    "ModernT5 Model with a `language modeling` head on top for conditional generation.",
    MODERNBERT_START_DOCSTRING,
)
class ModernT5ForConditionalGeneration(ModernT5PreTrainedModel, GenerationMixin):
    def __init__(self, config: ModernT5Config):
        super().__init__(config) # This calls ModernT5PreTrainedModel.__init__

        self.model = ModernT5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            # Register all tied weights paths
            self._tied_weights_keys = [
                "lm_head.weight",
                "model.shared.weight",
                "model.encoder.embeddings.tok_embeddings.weight",
                "model.decoder.embed_tokens.weight"
            ]
        else:
            self._tied_weights_keys = []

        self.post_init()

    def get_input_embeddings(self): return self.model.get_input_embeddings()
    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)
        if self.lm_head is not None and self.config.tie_word_embeddings: # or other conditions for tying
            self.tie_weights()

    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings
    def get_encoder(self): return self.model.get_encoder()
    def get_decoder(self): return self.model.get_decoder()

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None, **kwargs):
        # From T5 HuggingFace implementation
        if "encoder_outputs" not in model_kwargs:
            encoder_kwargs = {
                k: v for k, v in model_kwargs.items()
                if k not in ["decoder_input_ids", "decoder_attention_mask", "decoder_position_ids", "labels", "use_cache", "output_attentions", "output_hidden_states", "return_dict"]
            }
            # Encoder specific kwargs
            # ModernBertModel needs attention_mask, optionally position_ids if not FA2
            if "attention_mask" not in encoder_kwargs: # `generate` passes attention_mask for input_ids
                 encoder_kwargs["attention_mask"] = torch.ones_like(inputs_tensor)

            # If encoder uses RoPE and not FA2, it needs position_ids.
            # `generate` does not typically provide `position_ids` for the encoder input.
            if self.config._attn_implementation != "flash_attention_2":
                 if "position_ids" not in encoder_kwargs:
                    seq_len = inputs_tensor.shape[1]
                    encoder_kwargs["position_ids"] = torch.arange(seq_len, device=inputs_tensor.device).unsqueeze(0).expand(inputs_tensor.shape[0], -1)

            model_kwargs["encoder_outputs"] = self.model.encoder(input_ids=inputs_tensor, **encoder_kwargs)
        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self, batch_size: int, decoder_start_token_id: int = None, bos_token_id: int = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None, device: torch.device = None
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        # From T5 HF
        if model_kwargs is None: model_kwargs = {}
        if device is None: device = self.device

        _decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        _bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id # T5 uses pad as BOS
        if _bos_token_id is None: _bos_token_id = _decoder_start_token_id # Fallback

        if "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        else:
            decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * _bos_token_id
        
        # For RoPE, decoder_position_ids must be initialized for the first token (usually 0)
        if "decoder_position_ids" not in model_kwargs:
             model_kwargs["decoder_position_ids"] = torch.zeros_like(decoder_input_ids, dtype=torch.long)

        return decoder_input_ids, model_kwargs

    def _shift_right(self, input_ids): # T5 utility
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        assert decoder_start_token_id is not None, "decoder_start_token_id must be defined."
        assert pad_token_id is not None, "pad_token_id must be defined for label shifting."

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id) # For ignored labels
        return shifted_input_ids

    # @add_start_docstrings_to_model_forward(MODERNBERT_INPUTS_DOCSTRING.replace("ModernBert", "ModernT5"))
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # In class ModernT5ForConditionalGeneration:
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # Encoder position_ids
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[bool] = None
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = self._shift_right(labels)
            if decoder_position_ids is None and decoder_input_ids is not None: # Create for shifted labels if not provided
                 decoder_position_ids = torch.arange(decoder_input_ids.shape[-1], device=decoder_input_ids.device).expand(decoder_input_ids.shape[0], -1)
        elif decoder_input_ids is None and decoder_inputs_embeds is None:
            # If no labels and no explicit decoder_input_ids/embeds, create default ones for inference.
            # This mimics the first step of generation.
            current_bsz, current_device = -1, None
            if input_ids is not None:
                current_bsz, current_device = input_ids.shape[0], input_ids.device
            elif inputs_embeds is not None:
                current_bsz, current_device = inputs_embeds.shape[0], inputs_embeds.device
            elif encoder_outputs is not None:
                _encoder_last_hidden_state = encoder_outputs.last_hidden_state if isinstance(encoder_outputs, BaseModelOutput) else encoder_outputs[0]
                current_bsz, current_device = _encoder_last_hidden_state.shape[0], _encoder_last_hidden_state.device
            elif past_key_values is not None:
                 # past_key_values[layer_idx][key_or_value_idx (0 for key, 1 for value)]
                 # Shape: (batch_size, num_heads, sequence_length, head_dim)
                current_bsz, current_device = past_key_values[0][0].shape[0], past_key_values[0][0].device
            else:
                # This condition should ideally not be met if the model is used correctly (e.g. encoder needs input)
                # However, to prevent failure here, we might default or raise a more specific error.
                # For now, let's assume one of the above will provide batch_size and device.
                # If GenerationMixin is used for `generate`, it always provides batch_size.
                # If this is a direct forward call, user must provide enough info.
                raise ValueError(
                    "Cannot determine batch_size and device for decoder_input_ids. "
                    "Provide input_ids, inputs_embeds, encoder_outputs, or past_key_values."
                )

            if self.config.decoder_start_token_id is None:
                raise ValueError("config.decoder_start_token_id must be set for default decoder_input_ids generation.")

            decoder_input_ids = torch.full(
                (current_bsz, 1), self.config.decoder_start_token_id, dtype=torch.long, device=current_device
            )
            if decoder_position_ids is None: # For the first token, position is 0
                decoder_position_ids = torch.zeros_like(decoder_input_ids, dtype=torch.long, device=current_device)
        
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids, decoder_inputs_embeds=decoder_inputs_embeds,
            encoder_outputs=encoder_outputs, past_key_values=past_key_values,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
        )
        sequence_output = outputs[0] # Decoder last hidden state
        
        if self.config.tie_word_embeddings and hasattr(self.lm_head, 'weight') and self.lm_head.weight.data.device != sequence_output.device:
            self.lm_head = self.lm_head.to(sequence_output.device)
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device) # Ensure labels are on the same device as logits
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return Seq2SeqLMOutput(
            loss=loss, logits=lm_logits, past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past_key_values=None, attention_mask=None, # encoder AM
        decoder_attention_mask=None, decoder_position_ids=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # From T5 HF, adapted for decoder_position_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_position_ids is not None: # If provided, take the last one
                 decoder_position_ids = decoder_position_ids[:, -1:]
            else: # Else, create for current token
                 decoder_position_ids = torch.full_like(decoder_input_ids, past_length, dtype=torch.long)
        elif decoder_position_ids is None: # First step, if not provided
            decoder_position_ids = torch.zeros_like(decoder_input_ids, dtype=torch.long)

        model_inputs = {
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask, # Encoder attention mask
            "decoder_attention_mask": decoder_attention_mask, # Decoder padding mask
            "decoder_position_ids": decoder_position_ids, # For RoPE
            "use_cache": use_cache,
        }
        model_inputs.update(kwargs) # Pass along other relevant kwargs
        return {k: v for k, v in model_inputs.items() if v is not None}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx): # Standard for beam search
        reordered_past = ()
        for layer_past in past_key_values: # layer_past is (self_attn_key, self_attn_value)
            reordered_layer_past = ()
            for past_state in layer_past: # key or value
                reordered_layer_past += (past_state.index_select(0, beam_idx.to(past_state.device)),)
            reordered_past += (reordered_layer_past,)
        return reordered_past


__all__ = [
    "ModernT5Model",
    "ModernT5ForConditionalGeneration",
    "ModernT5Config",
    "ModernT5PreTrainedModel",
    "ModernT5Stack",
]

if __name__ == '__main__':
    import torch
    from transformers import AutoTokenizer
    # ModernBertModel is imported at the top of the file, so it's available here.

    # --- 1. Load Pretrained Encoder and Tokenizer ---
    encoder_name = "deepvk/RuModernBERT-small"
    print(f"Loading pretrained encoder and tokenizer from: {encoder_name}")
    
    # Load the encoder model. This requires the ModernBertModel class to be available.
    encoder = ModernBertModel.from_pretrained(encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # Set BOS token if not present. For many BERT-like models, CLS serves this role.
    if tokenizer.bos_token is None:
        print("Tokenizer does not have a BOS token. Setting bos_token to cls_token.")
        tokenizer.bos_token = tokenizer.cls_token
    
    # --- 2. Configure the Encoder-Decoder Model ---
    # We will create a ModernT5 model. The encoder part will be replaced by our loaded one.
    # The decoder can be configured symmetrically to the encoder.
    print("Configuring ModernT5 model...")
    encoder_config = encoder.config
    
    # Create a ModernT5Config using the encoder's config. It will automatically
    # copy encoder properties to the decoder if decoder-specific ones are not provided.
    encoder_config_dict = encoder_config.to_dict()
    # Correctly assign values to the config dictionary
    encoder_config_dict['decoder_start_token_id'] = tokenizer.bos_token_id
    encoder_config_dict['tie_word_embeddings'] = True
    encoder_config_dict['is_encoder_decoder'] = True
    config = ModernT5Config(
        **encoder_config_dict,
    )
    
    # --- 3. Initialize the Model and Set Pretrained Encoder ---
    print("Initializing ModernT5ForConditionalGeneration with random weights...")
    model = ModernT5ForConditionalGeneration(config)
    
    print(f"Replacing encoder with pretrained '{encoder_name}'...")
    # The encoder is part of the `model` attribute of `ModernT5ForConditionalGeneration`.
    model.model.encoder = encoder
    
    print("Tying word embeddings between encoder, decoder, and LM head...")
    # Ensure the embeddings are shared across the new encoder, the decoder, and the LM head.
    # We get the embeddings from the new encoder and set them for the whole model.
    model.set_input_embeddings(model.get_encoder().get_input_embeddings())
    
    # --- 4. Final Steps (Verification, Saving) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model moved to {device.upper()}")
    
    # Print parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params / 1e6:.2f} M")
    
    # Save the composed model and the tokenizer
    save_directory = "./modernt5_from_rumodernbert"
    print(f"Saving model and tokenizer to {save_directory}...")
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print("Save complete.")
    
    # --- 5. Load and Test ---
    print(f"\nLoading model from {save_directory}...")
    loaded_model = ModernT5ForConditionalGeneration.from_pretrained(save_directory)
    loaded_model.to(device)
    print("Model loaded successfully from checkpoint.")
    
    # Test with a dummy forward pass
    print("Performing a dummy forward pass...")
    dummy_input_ids = torch.randint(0, config.vocab_size, (2, 16)).to(device)
    dummy_labels = torch.randint(0, config.vocab_size, (2, 10)).to(device)
    dummy_output = loaded_model(input_ids=dummy_input_ids, labels=dummy_labels)
    print(f"Dummy forward pass successful. Loss: {dummy_output.loss.item():.4f}")
    
    # Test generation
    print("\nPerforming a dummy generation...")
    # Use the tokenizer to prepare input
    prompt = "Перевод с русского на английский: как дела?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate output
    generated_ids = loaded_model.generate(**inputs, max_length=50)
    decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"Input: '{prompt}'")
    print(f"Generated output: '{decoded_text}'")
