import torch
import types
from typing import Optional, Tuple

from transformers import EncoderDecoderModel, AutoTokenizer, AutoConfig, AutoModel
from transformers.models.bart.modeling_bart import BartAttention
from transformers.models.bert.modeling_bert import RotaryPositionEmbedding # Assuming this is the RoPE from deepvk/RuModernBERT-small

# 1. Load Tokenizer and Encoder
tokenizer = AutoTokenizer.from_pretrained('deepvk/RuModernBERT-small')
encoder = AutoModel.from_pretrained("deepvk/RuModernBERT-small")

# 2. Configure and Load BART Decoder
bart_config = AutoConfig.from_pretrained('facebook/bart-base')
bart_config.max_position_embeddings = 8192  # For BartLearnedPositionalEmbedding, though we'll nullify its effect
bart_config.vocab_size = encoder.config.vocab_size # Match encoder vocab for LM head compatibility

decoder = AutoModel.from_pretrained('facebook/bart-base', config=bart_config)
decoder.resize_token_embeddings(len(tokenizer)) # Resize decoder's embeddings matrix

# 3. Disable Absolute Positional Embeddings in Decoder's BartLearnedPositionalEmbedding
def new_learned_pos_embed_forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
    """ Returns zeros instead of learned positional embeddings. """
    bsz, seq_len = input_ids_shape[:2] # input_ids_shape can be (bsz, seq_len) or (bsz, seq_len, hidden_dim)
    zero_embeds = torch.zeros(
        1, seq_len, self.embedding_dim, # BART pos embeds are (1, max_pos, dim) sliced
        device=self.weight.device, dtype=self.weight.dtype
    )
    return zero_embeds

decoder.decoder.embed_positions.forward = types.MethodType(new_learned_pos_embed_forward, decoder.decoder.embed_positions)

# 4. Implement RoPE in Decoder's Attention Layers
head_dim = decoder.config.d_model // decoder.config.decoder_attention_heads
rope_emb = RotaryPositionEmbedding(
    dim=head_dim, 
    max_position_embeddings=8192, 
    device=decoder.device, # Will be moved to correct device by model.to(device) later if necessary
    seq_len_dim=2 # for (batch_size, num_heads, seq_len, head_dim)
)

original_bart_attention_forward = BartAttention.forward # Keep a reference if needed, though not strictly used in the new_forward below

def new_bart_attention_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()

    # Initial projections
    query_states_proj = self.q_proj(hidden_states)
    
    if is_cross_attention:
        # Cross-attention: RoPE is not applied. K,V come from encoder_hidden_states.
        key_states_proj = self.k_proj(key_value_states)
        value_states_proj = self.v_proj(key_value_states)
    else:
        # Self-attention: K,V come from current hidden_states.
        key_states_proj = self.k_proj(hidden_states)
        value_states_proj = self.v_proj(hidden_states)

    # Apply scaling (BART does it on Q)
    query_states_scaled = query_states_proj * self.scaling

    # Reshape Q, K_current_proj, V_current_proj to (bsz, num_heads, tgt_len, head_dim) for RoPE
    # Note: self._shape returns (bsz * num_heads, tgt_len, head_dim)
    # So we view it after.
    q_B_H_T_D = self._shape(query_states_scaled, tgt_len, bsz).view(bsz, self.num_heads, tgt_len, self.head_dim)
    
    if not is_cross_attention:
        k_curr_B_H_T_D = self._shape(key_states_proj, tgt_len, bsz).view(bsz, self.num_heads, tgt_len, self.head_dim)
        
        offset = 0
        if past_key_value is not None: # past_key_value stores (K_past_rotated, V_past)
            # K_past_rotated shape is (bsz, num_heads, past_seq_len, head_dim)
            offset = past_key_value[0].size(2) 

        # Apply RoPE to current Q and K
        # rope_emb is from the outer scope (closure)
        q_rotated, k_curr_rotated = rope_emb(q_B_H_T_D, k_curr_B_H_T_D, seq_len=tgt_len, offset=offset)
    else:
        # For cross-attention, no RoPE. K needs to be shaped.
        # key_value_states (encoder output) has seq_len src_len.
        src_len = key_value_states.size(1)
        k_curr_rotated = self._shape(key_states_proj, src_len, bsz).view(bsz, self.num_heads, src_len, self.head_dim) # "rotated" is a misnomer here
        q_rotated = q_B_H_T_D # Q for cross-attn is not rotated by RoPE

    # Prepare V (not rotated by RoPE)
    # For self-attention, V is from current hidden_states, tgt_len
    # For cross-attention, V is from key_value_states, src_len
    v_len = tgt_len if not is_cross_attention else key_value_states.size(1)
    v_B_H_S_D = self._shape(value_states_proj, v_len, bsz).view(bsz, self.num_heads, v_len, self.head_dim)

    # Concatenate with past K, V if available (only for self-attention)
    if past_key_value is not None and not is_cross_attention:
        # past_key_value[0] is K_past_rotated (B, H, S_past, D_head)
        # past_key_value[1] is V_past (B, H, S_past, D_head)
        k_final = torch.cat([past_key_value[0], k_curr_rotated], dim=2) 
        v_final = torch.cat([past_key_value[1], v_B_H_S_D], dim=2)
    else:
        k_final = k_curr_rotated
        v_final = v_B_H_S_D
    
    # Reshape Q, K_final, V_final for BMM: (bsz * num_heads, seq_len, head_dim)
    query_states_for_bmm = q_rotated.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
    
    # k_final could have different seq_len for cross-attn vs self-attn with cache
    final_key_seq_len = k_final.size(2)
    key_states_for_bmm = k_final.reshape(bsz * self.num_heads, final_key_seq_len, self.head_dim)
    value_states_for_bmm = v_final.reshape(bsz * self.num_heads, final_key_seq_len, self.head_dim)

    # Standard Attention Calculation (copied from BartAttention.forward)
    attn_weights = torch.bmm(query_states_for_bmm, key_states_for_bmm.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, final_key_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, final_key_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, final_key_seq_len):
             # If a static mask of shape (tgt_len, src_len) is used, broadcast it to (bsz, 1, tgt_len, src_len)
            if attention_mask.ndim == 2 and attention_mask.shape == (tgt_len, final_key_seq_len):
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0) # (1,1,T,S)
            elif attention_mask.ndim == 3 and attention_mask.shape == (bsz, tgt_len, final_key_seq_len): # (B,T,S)
                attention_mask = attention_mask.unsqueeze(1) # (B,1,T,S)
            else:
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, final_key_seq_len)} or broadcastable, but is {attention_mask.size()}"
                )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, final_key_seq_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, final_key_seq_len)

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        if layer_head_mask.size() != (self.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                f" {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, final_key_seq_len)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, final_key_seq_len)

    if output_attentions:
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, final_key_seq_len)
    else:
        attn_weights_reshaped = None

    attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.bmm(attn_probs, value_states_for_bmm)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)
    
    # For self-attention, the new past_key_value is (k_final, v_final)
    # k_final and v_final are already in (B, H, S, D_head) shape.
    # For cross-attention, past_key_value is not returned/updated by this layer.
    new_past_key_value = (k_final, v_final) if not is_cross_attention else None

    return attn_output, attn_weights_reshaped, new_past_key_value


for layer in decoder.decoder.layers:
    # The rope_emb instance will be part of the closure of the new_bart_attention_forward method
    layer.self_attn.forward = types.MethodType(new_bart_attention_forward, layer.self_attn)
    # Cross attention layers should not be patched with RoPE logic for K,V from encoder.
    # The new_bart_attention_forward handles is_cross_attention correctly.
    if hasattr(layer, 'encoder_attn'):
        layer.encoder_attn.forward = types.MethodType(new_bart_attention_forward, layer.encoder_attn)


# 5. Create EncoderDecoderModel
bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

# 6. Configure bert2bert
# Tie token embeddings if share_vocab is True or if decoder was from pretrained string
# Here, we manually ensured vocab sizes match and resized decoder embeddings.
# EncoderDecoderModel will use the individual configs unless overridden.
bert2bert.config.encoder = encoder.config
bert2bert.config.decoder = decoder.config # This decoder.config already has updated vocab_size and max_pos_embeds

bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id # As per original script
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id

# This ensures the top-level config.vocab_size (used by generate) matches.
bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size

bert2bert.config.max_length = 8192
bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4

# Print number of parameters after full model construction
print(f"Model parameters: {bert2bert.num_parameters()}")

# 7. Save model and tokenizer
bert2bert.save_pretrained('encdec')
tokenizer.save_pretrained('encdec')

