from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, EncoderDecoderModel
from rotary_embedding_torch import RotaryEmbedding
import torch
import types

# 1. Load encoder and decoder
encoder_model_name = "deepvk/RuModernBERT-small"
tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
encoder = AutoModel.from_pretrained(encoder_model_name)

decoder_model_name = "ai-forever/ruT5-base"
decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
decoder = AutoModelForSeq2SeqLM.from_pretrained(decoder_model_name)

# 2. Monkey-patch: Apply RoPE to ruT5 decoder self-attention

def add_rope_to_t5_decoder(decoder, max_position_embeddings=8192):
    for layer in decoder.decoder.block:
        sa = layer.layer[0].SelfAttention
        d = sa.q.shape[-1] // sa.num_heads

        # Initialize RoPE
        sa.rotary_emb = RotaryEmbedding(dim=d, use_xpos=True, base=10000, interleaved=False, max_seq_len=max_position_embeddings)

        # Save the original forward
        orig_forward = sa.forward

        def rope_forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None, past_key_value=None, layer_head_mask=None, query_length=None, use_cache=False, output_attentions=False):
            # This matches T5Attention.forward signature (HF v4.38+)
            import torch.nn.functional as F

            # The following is mostly copied from HuggingFace, with RoPE added for self-attention (cross-attention is left unchanged)
            is_cross_attention = key_value_states is not None

            bsz, seq_len, _ = hidden_states.size()
            real_seq_length = seq_len if not is_cross_attention else key_value_states.size(1)

            # Project
            query_states = self.q(hidden_states)
            key_states = self.k(key_value_states if is_cross_attention else hidden_states)
            value_states = self.v(key_value_states if is_cross_attention else hidden_states)

            # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
            def shape(states):
                return states.view(bsz, -1, self.num_heads, d).transpose(1, 2)

            query_states = shape(query_states)
            key_states = shape(key_states)
            value_states = shape(value_states)

            # Apply RoPE ONLY if not cross-attention
            if not is_cross_attention:
                # RoPE expects [batch, num_heads, seq_len, head_dim]
                pos = torch.arange(seq_len, device=hidden_states.device)
                query_states, key_states = self.rotary_emb(query_states, key_states, seq_len=seq_len)

            # Compute scores
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / (d ** 0.5)

            # Masking
            if mask is not None:
                attn_weights += mask

            # Attention
            attn_probs = F.softmax(attn_weights, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

            # Attention output
            attn_output = torch.matmul(attn_probs, value_states)

            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

            attn_output = self.o(attn_output)

            outputs = (attn_output,)

            if output_attentions:
                outputs += (attn_probs,)

            return outputs if len(outputs) > 1 else outputs[0]

        # Patch the method
        sa.forward = types.MethodType(rope_forward, sa)

# Apply the monkey-patch
add_rope_to_t5_decoder(decoder, max_position_embeddings=8192)

# 3. Set decoder max length & n_positions
decoder.config.max_length = 8192
decoder.config.n_positions = 8192
decoder.model.decoder.config.n_positions = 8192

# 4. Create EncoderDecoderModel
bert2rut5 = EncoderDecoderModel(encoder=encoder, decoder=decoder)

# 5. Adjust configuration
bert2rut5.config.decoder_start_token_id = decoder_tokenizer.cls_token_id or decoder_tokenizer.bos_token_id
bert2rut5.config.eos_token_id = decoder_tokenizer.sep_token_id or decoder_tokenizer.eos_token_id
bert2rut5.config.pad_token_id = decoder_tokenizer.pad_token_id
bert2rut5.config.vocab_size = bert2rut5.config.decoder.vocab_size
bert2rut5.config.max_length = 8192
bert2rut5.config.no_repeat_ngram_size = 3
bert2rut5.config.early_stopping = True
bert2rut5.config.length_penalty = 2.0
bert2rut5.config.num_beams = 4

# 6. Save everything
bert2rut5.save_pretrained('encdec')
tokenizer.save_pretrained('encdec')

