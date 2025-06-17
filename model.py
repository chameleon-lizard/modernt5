import torch
import torch.nn as nn
import types
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    AutoConfig,
    GPT2Config,
    GPT2LMHeadModel,
    AutoModel,
)
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertUnpaddedRotaryEmbedding,
    apply_rotary_pos_emb,
    ModernBertModel,
)

ENCODER_NAME = "deepvk/RuModernBERT-small"
DECODER_NAME = "gpt2"
CONTEXT_LENGTH = 8192

# ========== 1. Load encoder, tokenizer, get embedding =============
tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
encoder = AutoModel.from_pretrained(ENCODER_NAME)
encoder_emb = encoder.embeddings.tok_embeddings  # nn.Embedding

# ========== 2. Build decoder config to match embedding/vocab =============
dec_config = GPT2Config.from_pretrained(
    DECODER_NAME,
    n_embd=384,
    n_positions=CONTEXT_LENGTH,
    n_ctx=CONTEXT_LENGTH,
    n_layer=12,
    n_head=6,  # 384/6 = 64, matches RuModernBERT
    vocab_size=encoder_emb.num_embeddings,
    bos_token_id=tokenizer.cls_token_id,
    eos_token_id=tokenizer.sep_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# ========== 3. Build new GPT2 decoder =============
decoder = GPT2LMHeadModel(dec_config)

# ========== 4. Clone encoder embeddings to decoder (NO sharing!) ==========
decoder.transformer.wte = nn.Embedding.from_pretrained(
    encoder_emb.weight.clone(),
    freeze=True,
    padding_idx=encoder_emb.padding_idx,
)

# Set lm_head and clone weights
decoder.lm_head = nn.Linear(
    encoder_emb.embedding_dim, encoder_emb.num_embeddings, bias=False
)
decoder.lm_head.weight.data = encoder_emb.weight.clone()

# ========== 5. Expand positional emb and bias mask =============
old_len, dim = decoder.transformer.wpe.weight.size()
if CONTEXT_LENGTH > old_len:
    new_emb = nn.Embedding(CONTEXT_LENGTH, dim)
    with torch.no_grad():
        new_emb.weight[:old_len] = decoder.transformer.wpe.weight
        new_emb.weight[old_len:] = 0.0
    new_emb.requires_grad_(False)
    decoder.transformer.wpe = new_emb
else:
    decoder.transformer.wpe.weight.data.zero_()
    decoder.transformer.wpe.requires_grad_(False)

# Expand attention bias for long context
for block in decoder.transformer.h:
    bias_dtype = block.attn.bias.dtype
    new_bias = torch.tril(torch.ones((CONTEXT_LENGTH, CONTEXT_LENGTH), dtype=bias_dtype)).view(1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH)
    block.attn.register_buffer("bias", new_bias, persistent=False)

# ========== 6. Patch self-attn to use RoPE ============
def build_rope_attn_forward(head_dim, rope_base=160_000.0):
    rotary = ModernBertUnpaddedRotaryEmbedding(dim=head_dim, base=rope_base)
    def rope_forward(self, hidden_states, layer_past=None, attention_mask=None, position_ids=None,
                     head_mask=None, use_cache=False, output_attentions=False):
        query, key, value = self._split_heads(self.c_attn(hidden_states))
        batch, heads, seq_len, _ = query.size()
        cos, sin = rotary(seq_len, dtype=query.dtype, device=query.device)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        attn_output, present = self._attn(
            query, key, value,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (None,)
        return outputs
    return rope_forward

n_heads = decoder.config.n_head
head_dim = decoder.config.n_embd // n_heads
rope_base = 160_000.0

for block in decoder.transformer.h:
    attn = block.attn
    attn.forward = types.MethodType(build_rope_attn_forward(head_dim, rope_base), attn)

# ========== 7. Assemble EncoderDecoderModel ===========
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id           = tokenizer.sep_token_id
model.config.pad_token_id           = tokenizer.pad_token_id
model.config.vocab_size             = encoder_emb.num_embeddings
model.config.max_length             = CONTEXT_LENGTH
model.config.no_repeat_ngram_size   = 3
model.config.early_stopping         = True
model.config.length_penalty         = 2.0
model.config.num_beams              = 4

# ========== 8. Move model to CUDA ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Total parameters: {model.num_parameters():,}")

# ========== 9. Save (now works with safetensors!) ==========
model.save_pretrained("encdec")
tokenizer.save_pretrained("encdec")

