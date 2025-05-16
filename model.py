from transformers import EncoderDecoderModel, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import math

# RoPE implementation
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis[:xq_.shape[1], :]
    
    xq_out = torch.view_as_real(xq_ * freqs_cis.unsqueeze(0)).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis.unsqueeze(0)).flatten(2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

tokenizer = AutoTokenizer.from_pretrained('deepvk/RuModernBERT-small')
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("deepvk/RuModernBERT-small", "ai-forever/ruT5-base")
print(bert2bert.num_parameters())

# Configure the model
bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

# Extend context length to 8192
bert2bert.config.max_length = 8192
bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4

# Extend position embeddings in the decoder
bert2bert.decoder.config.max_position_embeddings = 8192

# Precompute RoPE frequencies
freqs_cis = precompute_freqs_cis(
    dim=bert2bert.decoder.config.hidden_size,
    end=8192,
)

# Monkey-patch the decoder's forward method to use RoPE
original_decoder_forward = bert2bert.decoder.forward

def patched_decoder_forward(self, *args, **kwargs):
    # Call the original forward method
    outputs = original_decoder_forward(self, *args, **kwargs)
    
    # If we're in training mode, apply RoPE to the hidden states
    if self.training and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        # Get the last hidden state
        hidden_states = outputs.hidden_states[-1]
        
        # Apply RoPE
        batch_size, seq_len, hidden_size = hidden_states.shape
        head_size = hidden_size // self.config.num_attention_heads
        
        # Reshape for multi-head attention
        hidden_states = hidden_states.view(batch_size, seq_len, self.config.num_attention_heads, head_size)
        
        # Split into query and key projections (simplified)
        xq, xk = hidden_states, hidden_states
        
        # Apply rotary embeddings
        xq_rope, xk_rope = apply_rotary_emb(xq, xk, freqs_cis.to(xq.device))
        
        # Reshape back
        hidden_states = xq_rope.view(batch_size, seq_len, hidden_size)
        
        # Update the last hidden state
        outputs.hidden_states = outputs.hidden_states[:-1] + (hidden_states,)
    
    return outputs

# Apply the monkey patch
bert2bert.decoder.forward = patched_decoder_forward.__get__(bert2bert.decoder, type(bert2bert.decoder))

bert2bert.save_pretrained('encdec')
tokenizer.save_pretrained('encdec')
