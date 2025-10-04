# ============================================================================
# Fixed UL2 Collator
# ============================================================================
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class UL2DenoiserConfig:
    """Configuration for a single UL2 denoiser"""
    name: str
    prefix: str
    noise_density: float
    mean_span_length: float
    max_spans: int = 100


class ImprovedUL2CollatorV2:
    """Fixed UL2 collator with proper encoder-decoder separation"""
    
    def __init__(
        self,
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 512,
        denoiser_configs: Optional[List[UL2DenoiserConfig]] = None,
        denoiser_probs: Optional[List[float]] = None,
        use_bin_packing: bool = False,  # DISABLED by default - breaks UL2
        min_sequence_length: int = 10,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.use_bin_packing = use_bin_packing
        self.min_sequence_length = min_sequence_length
        
        # Default UL2 denoisers
        if denoiser_configs is None:
            denoiser_configs = [
                UL2DenoiserConfig("R", "[NLU]", 0.0, 0.25, max_spans=1),
                UL2DenoiserConfig("S", "[S2S]", 0.15, 3.0, max_spans=100),
                UL2DenoiserConfig("X", "[NLG]", 0.5, 3.0, max_spans=100),
            ]
        
        self.denoisers = denoiser_configs
        self.denoiser_probs = denoiser_probs or [0.5, 0.25, 0.25]
        
        # Normalize probabilities
        total = sum(self.denoiser_probs)
        self.denoiser_probs = [p/total for p in self.denoiser_probs]
        
        # Setup sentinel tokens
        self._setup_sentinel_ids()
        
        # Pre-compute prefix token IDs
        self.prefix_ids = {}
        for denoiser in self.denoisers:
            ids = tokenizer.encode(denoiser.prefix, add_special_tokens=False)
            self.prefix_ids[denoiser.name] = ids
    
    def _setup_sentinel_ids(self):
        """Cache sentinel token IDs"""
        self.sentinel_ids = []
        for i in range(100):
            token = f"<unused{i}>"
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                self.sentinel_ids.append(token_id)
        
        if not self.sentinel_ids:
            raise ValueError("No sentinel tokens found in tokenizer!")
    
    def _apply_prefix_lm(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        """Apply prefix LM: encoder sees prefix, decoder predicts suffix"""
        if len(tokens) < 2:
            return tokens, []
        
        # Split at 75-25 ratio
        split_idx = max(1, int(len(tokens) * 0.75))
        split_idx = min(split_idx, len(tokens) - 1)
        
        # FIX: Encoder sees only the PREFIX (not full text)
        encoder_input = tokens[:split_idx]
        # Decoder predicts the SUFFIX
        decoder_target = tokens[split_idx:]
        
        return encoder_input, decoder_target
    
    def _apply_span_corruption(self, tokens: List[int], denoiser: UL2DenoiserConfig) -> Tuple[List[int], List[int]]:
        """Apply span corruption with sentinels"""
        if len(tokens) < 2:
            return tokens, []
        
        length = len(tokens)
        noise_density = denoiser.noise_density
        mean_span_length = denoiser.mean_span_length
        
        # Generate noise mask
        num_noise_tokens = int(round(length * noise_density))
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        
        num_noise_spans = min(
            denoiser.max_spans,
            int(round(num_noise_tokens / mean_span_length))
        )
        num_noise_spans = max(num_noise_spans, 1)
        
        # Random segmentation
        def random_segmentation(num_items, num_segments):
            if num_segments == 1:
                return np.array([num_items])
            boundaries = np.sort(np.random.choice(num_items - 1, num_segments - 1, replace=False))
            boundaries = np.concatenate([[0], boundaries + 1, [num_items]])
            return np.diff(boundaries)
        
        noise_span_lengths = random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = random_segmentation(length - num_noise_tokens, num_noise_spans)
        
        # Interleave spans
        interleaved = np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1).flatten()
        
        # Create mask
        mask = np.zeros(length, dtype=bool)
        position = 0
        for i, span_length in enumerate(interleaved):
            if i % 2 == 1:
                mask[position:position + span_length] = True
            position += span_length
        
        # Build encoder input and decoder targets
        encoder_input = []
        decoder_target = []
        sentinel_idx = 0
        
        prev_masked = False
        for i, is_masked in enumerate(mask):
            if is_masked:
                if not prev_masked:  # Start of masked span
                    sentinel = self.sentinel_ids[sentinel_idx % len(self.sentinel_ids)]
                    encoder_input.append(sentinel)  # Encoder sees sentinel
                    decoder_target.append(sentinel)  # Decoder outputs sentinel
                    sentinel_idx += 1
                decoder_target.append(tokens[i])  # Decoder outputs masked tokens
            else:
                encoder_input.append(tokens[i])  # Encoder sees unmasked tokens
            prev_masked = is_masked
        
        return encoder_input, decoder_target
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process batch with proper encoder-decoder separation"""
        
        # Filter short sequences
        examples = [ex for ex in examples if len(ex.get('text', '')) >= self.min_sequence_length]
        
        if not examples:
            return self._dummy_batch()
        
        # Process each example WITHOUT bin packing
        batch_encoder_inputs = []
        batch_decoder_targets = []
        
        for example in examples:
            text = example.get('text', '')
            tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            
            if len(tokens) < self.min_sequence_length:
                continue
            
            # Sample denoiser
            denoiser_idx = np.random.choice(len(self.denoisers), p=self.denoiser_probs)
            denoiser = self.denoisers[denoiser_idx]
            
            # Truncate if needed
            if len(tokens) > self.max_input_length:
                start_idx = np.random.randint(0, len(tokens) - self.max_input_length + 1)
                tokens = tokens[start_idx:start_idx + self.max_input_length]
            
            # Apply denoising objective
            if denoiser.noise_density == 0.0:
                encoder_input, decoder_target = self._apply_prefix_lm(tokens)
            else:
                encoder_input, decoder_target = self._apply_span_corruption(tokens, denoiser)
            
            # FIX: Add prefix to ENCODER input (it conditions the corruption pattern)
            prefix_ids = self.prefix_ids[denoiser.name]
            encoder_input = prefix_ids + encoder_input
            
            #Дополнительно, возможно не стоит делать
            #Давайте обернем encoder_input тэгами начала и конца предложения 
            # Add BOS/EOS to match ModernBERT's pretraining
            if self.tokenizer.bos_token_id is not None:
                encoder_input = [self.tokenizer.bos_token_id] + encoder_input
            encoder_input = encoder_input[:self.max_input_length-1] #на один меньше макс длины, чтобы влез eos_token_id
            if self.tokenizer.eos_token_id is not None:
                encoder_input = encoder_input + [self.tokenizer.eos_token_id]
            
            #оригинальная версия, без дополнительных токенов
            # Truncate to max lengths
            #encoder_input = encoder_input[:self.max_input_length]
            #decoder_target = decoder_target[:self.max_target_length]
            
            # Add EOS to targets
            if self.tokenizer.eos_token_id is not None and decoder_target:
                decoder_target = decoder_target + [self.tokenizer.eos_token_id]
            
            batch_encoder_inputs.append(encoder_input)
            batch_decoder_targets.append(decoder_target)
        
        if not batch_encoder_inputs:
            return self._dummy_batch()
        
        # Pad sequences
        max_encoder_len = max(len(seq) for seq in batch_encoder_inputs)
        max_decoder_len = max(len(seq) for seq in batch_decoder_targets)
        
        pad_id = self.tokenizer.pad_token_id or 0
        
        input_ids = []
        attention_mask = []
        decoder_input_ids = []
        labels = []
        
        for i in range(len(batch_encoder_inputs)):
            # Encoder input
            enc_seq = batch_encoder_inputs[i] + [pad_id] * (max_encoder_len - len(batch_encoder_inputs[i]))
            input_ids.append(enc_seq)
            
            # Encoder attention mask
            attn_mask = [1] * len(batch_encoder_inputs[i]) + [0] * (max_encoder_len - len(batch_encoder_inputs[i]))
            attention_mask.append(attn_mask)
            
            # FIX: Create decoder_input_ids by shifting targets right
            # The model's _shift_right will add BOS, so we just pass the targets
            dec_input = batch_decoder_targets[i] + [pad_id] * (max_decoder_len - len(batch_decoder_targets[i]))
            decoder_input_ids.append(dec_input)
            
            # Labels (with -100 for padding)
            label_seq = batch_decoder_targets[i] + [-100] * (max_decoder_len - len(batch_decoder_targets[i]))
            labels.append(label_seq)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            #'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),  # FIX: Added!
            'labels': torch.tensor(labels, dtype=torch.long),
        }
    
    def _dummy_batch(self):
        """Return dummy batch for empty inputs"""
        return {
            'input_ids': torch.zeros((1, 1), dtype=torch.long),
            'attention_mask': torch.zeros((1, 1), dtype=torch.long),
            'labels': torch.full((1, 1), -100, dtype=torch.long),
        }