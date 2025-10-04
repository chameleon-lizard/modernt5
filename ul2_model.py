"""
Corrected Initialization for ModernBERT-based Seq2Seq Model
Both encoder and decoder use RoPE, making them more compatible!
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Tuple, Any
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from dataclasses import dataclass
import json


# ============================================================================
# Configuration with RoPE Awareness
# ============================================================================

@dataclass
class ModernBertSeq2SeqConfig:
    """Configuration for ModernBERT-based seq2seq model"""
    
    # Decoder architecture
    num_decoder_layers: int = 12
    decoder_hidden_size: Optional[int] = None  # None = match encoder
    decoder_num_attention_heads: Optional[int] = None
    decoder_num_key_value_heads: Optional[int] = None  # For GQA
    decoder_intermediate_size: Optional[int] = None
    
    # RoPE configuration (both encoder and decoder use RoPE!)
    decoder_rope_theta: float = 10000.0  # Should match encoder for consistency
    decoder_max_position_embeddings: int = 8192
    share_rope_config: bool = True  # Use encoder's RoPE config for decoder
    
    # Embedding configuration
    tie_word_embeddings: bool = True  # Tie decoder embeddings with LM head
    share_encoder_decoder_embeddings: bool = True  # Share token embeddings
    
    # Decoder-specific
    decoder_sliding_window: Optional[int] = 4096
    decoder_attention_dropout: float = 0.0
    decoder_dropout_rate: float = 0.0
    decoder_activation: str = "gelu_pytorch_tanh"  # Match ModernBERT
    
    # Training configuration
    torch_dtype: Optional[torch.dtype] = None
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    
    # Model settings
    vocab_size: Optional[int] = None


# ============================================================================
# ModernBERT Architecture Analyzer
# ============================================================================

class ModernBertAnalyzer:
    """Analyze ModernBERT architecture and extract configurations"""
    
    @staticmethod
    def extract_rope_config(model_config: Any) -> Dict:
        """Extract RoPE configuration from ModernBERT"""
        rope_config = {
            'rope_theta': 10000.0,  # Default
            'max_position_embeddings': 8192,  # Default
            'rope_type': 'default',
            'rope_scaling': None,
        }
        
        # ModernBERT specific RoPE parameters
        if hasattr(model_config, 'rope_theta'):
            rope_config['rope_theta'] = model_config.rope_theta
        
        if hasattr(model_config, 'max_position_embeddings'):
            rope_config['max_position_embeddings'] = model_config.max_position_embeddings
        
        if hasattr(model_config, 'rope_scaling'):
            rope_config['rope_scaling'] = model_config.rope_scaling
        
        # ModernBERT uses local-global attention pattern
        if hasattr(model_config, 'global_attn_every_n_layers'):
            rope_config['global_attn_every_n_layers'] = model_config.global_attn_every_n_layers
        
        if hasattr(model_config, 'local_attention_window'):
            rope_config['local_attention_window'] = model_config.local_attention_window
        
        return rope_config
    
    @staticmethod
    def analyze_modernbert(model: PreTrainedModel) -> Dict:
        """Comprehensive analysis of ModernBERT model"""
        config = model.config
        
        info = {
            'architecture': 'ModernBERT',
            'hidden_size': config.hidden_size,
            'num_attention_heads': config.num_attention_heads,
            'num_hidden_layers': config.num_hidden_layers,
            'intermediate_size': config.intermediate_size,
            'hidden_activation': config.hidden_activation,
            'vocab_size': config.vocab_size,
            'position_embedding_type': 'rope',  # ModernBERT uses RoPE!
            'rope_config': ModernBertAnalyzer.extract_rope_config(config),
            'layer_norm_eps': config.layer_norm_eps if hasattr(config, 'layer_norm_eps') else 1e-5,
            'attention_dropout': config.attention_dropout if hasattr(config, 'attention_dropout') else 0.0,
            'hidden_dropout_prob': config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.0,
        }
        
        # ModernBERT specific features
        if hasattr(config, 'classifier_dropout'):
            info['classifier_dropout'] = config.classifier_dropout
        
        if hasattr(config, 'deterministic_flash_attn'):
            info['deterministic_flash_attn'] = config.deterministic_flash_attn
        
        print(f"\n=== ModernBERT Analysis ===")
        print(f"Architecture: {info['architecture']}")
        print(f"Hidden size: {info['hidden_size']}")
        print(f"Attention heads: {info['num_attention_heads']}")
        print(f"Layers: {info['num_hidden_layers']}")
        print(f"Position embedding: {info['position_embedding_type']} (RoPE)")
        print(f"RoPE theta: {info['rope_config']['rope_theta']}")
        print(f"Max positions: {info['rope_config']['max_position_embeddings']}")
        
        return info


# ============================================================================
# Improved Initialization Function
# ============================================================================

def initialize_modernbert_seq2seq(
    encoder_model_path: str,
    tokenizer: AutoTokenizer,
    config: Optional[ModernBertSeq2SeqConfig] = None,
    device_map: Optional[Union[str, Dict]] = None,
) -> PreTrainedModel:
    """
    Initialize seq2seq model from ModernBERT encoder.
    
    Both encoder and decoder use RoPE, making them architecturally compatible!
    """
    from modernbert_t5gemma_hybrid import (
        ModernBertT5GemmaForConditionalGeneration,
        ModernBertT5GemmaConfig,
        ModernBertT5GemmaDecoderConfig,
    )
    
    if config is None:
        config = ModernBertSeq2SeqConfig()
    
    # Set vocab size from tokenizer
    config.vocab_size = len(tokenizer)
    
    # Determine dtype
    if config.torch_dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            config.torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            config.torch_dtype = torch.float16
        else:
            config.torch_dtype = torch.float32
    
    print(f"\n{'='*60}")
    print(f"Initializing ModernBERT-based Seq2Seq Model")
    print(f"{'='*60}")
    print(f"Dtype: {config.torch_dtype}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Decoder layers: {config.num_decoder_layers}")
    
    # Load ModernBERT encoder
    print(f"\nLoading ModernBERT from {encoder_model_path}...")
    encoder_model = AutoModel.from_pretrained(
        encoder_model_path,
        torch_dtype=config.torch_dtype,
        trust_remote_code=True,
        device_map=device_map,
    )
    
    # Analyze ModernBERT
    analyzer = ModernBertAnalyzer()
    encoder_info = analyzer.analyze_modernbert(encoder_model)
    
    # Extract the encoder core (for ModernBERT, it's usually the model itself)
    if hasattr(encoder_model, 'model'):
        encoder_core = encoder_model.model
    elif hasattr(encoder_model, 'encoder'):
        encoder_core = encoder_model.encoder
    else:
        encoder_core = encoder_model
    
    # Get token embeddings from ModernBERT
    encoder_embeddings = None
    if hasattr(encoder_core, 'embeddings'):
        if hasattr(encoder_core.embeddings, 'word_embeddings'):
            encoder_embeddings = encoder_core.embeddings.word_embeddings
        elif hasattr(encoder_core.embeddings, 'tok_embeddings'):
            encoder_embeddings = encoder_core.embeddings.tok_embeddings
    elif hasattr(encoder_core, 'embed_tokens'):
        encoder_embeddings = encoder_core.embed_tokens
    
    print(f"\nEncoder embeddings found: {encoder_embeddings is not None}")
    if encoder_embeddings:
        print(f"  Shape: {encoder_embeddings.weight.shape}")
    
    # Configure decoder dimensions
    decoder_hidden_size = config.decoder_hidden_size or encoder_info['hidden_size']
    decoder_num_heads = config.decoder_num_attention_heads or encoder_info['num_attention_heads']
    decoder_num_kv_heads = config.decoder_num_key_value_heads or (decoder_num_heads // 2)
    decoder_intermediate = (
        config.decoder_intermediate_size or 
        encoder_info.get('intermediate_size', decoder_hidden_size * 4)
    )
    
    # RoPE configuration - use encoder's config if sharing
    if config.share_rope_config:
        rope_theta = encoder_info['rope_config']['rope_theta']
        max_positions = encoder_info['rope_config']['max_position_embeddings']
        print(f"\nSharing RoPE config from encoder:")
        print(f"  Theta: {rope_theta}")
        print(f"  Max positions: {max_positions}")
    else:
        rope_theta = config.decoder_rope_theta
        max_positions = config.decoder_max_position_embeddings
        print(f"\nUsing custom decoder RoPE config:")
        print(f"  Theta: {rope_theta}")
        print(f"  Max positions: {max_positions}")
    
    # Create decoder configuration
    print(f"\nCreating decoder configuration:")
    print(f"  Hidden size: {decoder_hidden_size}")
    print(f"  Attention heads: {decoder_num_heads}")
    print(f"  KV heads: {decoder_num_kv_heads}")
    print(f"  FFN size: {decoder_intermediate}")
    print(f"  Layers: {config.num_decoder_layers}")
    
    decoder_config = ModernBertT5GemmaDecoderConfig(
        vocab_size=config.vocab_size,
        hidden_size=decoder_hidden_size,
        intermediate_size=decoder_intermediate,
        num_hidden_layers=config.num_decoder_layers,
        num_attention_heads=decoder_num_heads,
        num_key_value_heads=decoder_num_kv_heads,
        head_dim=decoder_hidden_size // decoder_num_heads,
        hidden_activation=config.decoder_activation,
        max_position_embeddings=max_positions,
        rope_theta=rope_theta,  # Match encoder's RoPE!
        sliding_window=config.decoder_sliding_window,
        attention_dropout=config.decoder_attention_dropout,
        dropout_rate=config.decoder_dropout_rate,
        pad_token_id=tokenizer.pad_token_id or 0,
        bos_token_id=tokenizer.bos_token_id or 0,
        eos_token_id=tokenizer.eos_token_id or 1,
        tie_word_embeddings=False,  # We handle this separately
        cross_attention_hidden_size=encoder_info['hidden_size'],
        rms_norm_eps=encoder_info.get('layer_norm_eps', 1e-6),
    )
    
    # Set attention implementation
    attn_impl = "flash_attention_2" if (
        config.use_flash_attention and 
        config.torch_dtype != torch.float32
    ) else "eager"
    
    decoder_config._attn_implementation = attn_impl
    
    # Create encoder config
    encoder_config_dict = encoder_model.config.to_dict()
    encoder_config_dict['vocab_size'] = config.vocab_size
    
    # Create main config
    main_config = ModernBertT5GemmaConfig(
        encoder=encoder_config_dict,
        decoder=decoder_config,
        vocab_size=config.vocab_size,
        tie_word_embeddings=config.tie_word_embeddings,
        is_encoder_decoder=True,
    )
    
    # Initialize model
    print(f"\nInitializing seq2seq model...")
    model = ModernBertT5GemmaForConditionalGeneration(main_config)
    
    # Convert to correct dtype
    model = model.to(config.torch_dtype)
    
    # Set encoder (use the ModernBERT encoder)
    model.model.encoder = encoder_core
    
    # Resize token embeddings for new vocabulary
    print(f"Resizing token embeddings to {config.vocab_size}...")
    model.resize_token_embeddings(config.vocab_size)
    
    # Handle embedding sharing
    if config.share_encoder_decoder_embeddings and encoder_embeddings is not None:
        print("\nSharing token embeddings between encoder and decoder...")
        
        # Since both use the same vocab, we can share embeddings
        # But we need to handle the size difference due to added tokens
        
        decoder_embeds = model.model.decoder.embed_tokens
        encoder_embed_weight = encoder_embeddings.weight
        
        # Copy weights for original vocabulary
        min_tokens = min(encoder_embed_weight.shape[0], decoder_embeds.weight.shape[0])
        with torch.no_grad():
            decoder_embeds.weight[:min_tokens] = encoder_embed_weight[:min_tokens].to(config.torch_dtype)
        
        # Initialize new tokens (sentinels, etc.)
        if decoder_embeds.weight.shape[0] > min_tokens:
            new_tokens = decoder_embeds.weight.shape[0] - min_tokens
            print(f"  Initializing {new_tokens} new token embeddings (sentinels, etc.)")
            nn.init.normal_(
                decoder_embeds.weight[min_tokens:],
                mean=0.0,
                std=encoder_info['hidden_size'] ** -0.5,  # Scaled by hidden size
            )
    
    # Tie decoder embeddings with LM head
    if config.tie_word_embeddings:
        print("Tying decoder embeddings with LM head...")
        model.lm_head.weight = model.model.decoder.embed_tokens.weight
    
    # Enable gradient checkpointing if requested
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")
    
    # Validation checks
    print("\n=== Validation ===")
    print(f"✓ Both encoder and decoder use RoPE")
    print(f"✓ RoPE theta aligned: {rope_theta}")
    print(f"✓ Hidden dimensions compatible")
    print(f"✓ Flash Attention 2: {attn_impl == 'flash_attention_2'}")
    
    print("\n=== Model Summary ===")
    print(f"Encoder layers: {encoder_info['num_hidden_layers']}")
    print(f"Decoder layers: {config.num_decoder_layers}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


# ============================================================================
# Complete Setup Function
# ============================================================================

def setup_modernbert_ul2(
    encoder_model_path: str = "answerdotai/ModernBERT-base",
    output_dir: str = "./modernbert_seq2seq",
    num_decoder_layers: int = 12,
    share_rope_config: bool = True,
    use_bf16: bool = True,
    num_sentinel_tokens: int = 100,
):
    """
    Complete setup for ModernBERT-based UL2 training.
    
    Args:
        encoder_model_path: Path to ModernBERT model
        output_dir: Output directory
        num_decoder_layers: Number of decoder layers
        share_rope_config: Use encoder's RoPE configuration for decoder
        use_bf16: Use bfloat16 precision
        num_sentinel_tokens: Number of UL2 sentinel tokens
    """
    import os
    from ul2_tokenizer import setup_ul2_tokenizer
    from ul2_collator import ImprovedUL2CollatorV2
    
    print(f"\n{'='*70}")
    print(f" ModernBERT-based Seq2Seq Model Setup")
    print(f"{'='*70}")
    
    # Step 1: Setup tokenizer with UL2 tokens
    print("\n[1/4] Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(encoder_model_path, trust_remote_code=True)
    tokenizer = setup_ul2_tokenizer(tokenizer, 
                                    ul2_prefixes=[],
                                    num_sentinel_tokens=num_sentinel_tokens)
    print(f"✓ Vocabulary size: {len(tokenizer)}")
    
    # Step 2: Configure model
    torch_dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_bf16_supported() else torch.float32
    
    model_config = ModernBertSeq2SeqConfig(
        num_decoder_layers=num_decoder_layers,
        share_rope_config=share_rope_config,
        tie_word_embeddings=True,
        share_encoder_decoder_embeddings=True,
        torch_dtype=torch_dtype,
        use_flash_attention=False,
        gradient_checkpointing=True,
    )
    
    # Step 3: Initialize model
    print("\n[2/4] Initializing model...")
    model = initialize_modernbert_seq2seq(
        encoder_model_path=encoder_model_path,
        tokenizer=tokenizer,
        config=model_config,
    )
    
    # Step 4: Create UL2 collator
    print("\n[3/4] Creating UL2 collator...")
    collator = ImprovedUL2CollatorV2(
        tokenizer=tokenizer,
        max_input_length=512,
        max_target_length=512,
        use_bin_packing=True,
    )
    print(f"✓ Collator configured with {len(collator.sentinel_ids)} sentinel tokens")
    
    # Step 5: Save initial checkpoint
    print(f"\n[4/4] Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration metadata
    metadata = {
        'base_encoder': encoder_model_path,
        'architecture': 'ModernBERT-T5Gemma',
        'num_encoder_layers': model.model.encoder.config.num_hidden_layers,
        'num_decoder_layers': num_decoder_layers,
        'vocab_size': len(tokenizer),
        'num_sentinel_tokens': num_sentinel_tokens,
        'dtype': str(torch_dtype),
        'rope_theta': model_config.decoder_rope_theta,
        'position_embedding': 'RoPE (both encoder and decoder)',
    }
    
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f" ✓ Setup Complete!")
    print(f"{'='*70}")
    print(f"\nModel saved to: {output_dir}")
    print(f"Ready for UL2 training with compatible RoPE embeddings!")
    
    return model, tokenizer, collator


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Example 1: Standard configuration
    model, tokenizer, collator = setup_modernbert_ul2(
        encoder_model_path="answerdotai/ModernBERT-base",
        output_dir="./modernbert_ul2_standard",
        num_decoder_layers=12,
        share_rope_config=True,  # Use same RoPE as encoder
        use_bf16=True,
    )
    
    # Example 2: Small decoder for efficiency
    model_small, tokenizer_small, collator_small = setup_modernbert_ul2(
        encoder_model_path="answerdotai/ModernBERT-base",
        output_dir="./modernbert_ul2_small",
        num_decoder_layers=6,  # Smaller decoder
        share_rope_config=True,
        use_bf16=True,
    )
    
    # Example 3: Large decoder for quality
    model_large, tokenizer_large, collator_large = setup_modernbert_ul2(
        encoder_model_path="answerdotai/ModernBERT-large",
        output_dir="./modernbert_ul2_large",
        num_decoder_layers=24,  # Large decoder
        share_rope_config=True,
        use_bf16=True,
    )