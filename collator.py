import random
import torch
from typing import Dict, List, Tuple, Union
import numpy as np

class UL2MoDCollator:
    """
    A collator that implements the UL2 Mixture of Denoisers (MoD) strategy.
    - R-denoiser: Prefix Language Modeling (causal prediction of a suffix).
    - S-denoiser: Standard span corruption.
    - X-denoiser: Extreme span corruption (higher corruption rate).
    """

    def __init__(
        self,
        tokenizer,
        ul2_prefixes=["[NLU]", "[S2S]", "[NLG]"],  # Prefixes for R, S, X denoisers respectively
        ul2_denoiser_probs=[1/3, 1/3, 1/3],    # Probabilities for R, S, X denoisers
        r_denoiser_suffix_ratio=0.15,  # R-denoiser: ratio of sequence length to be used as suffix (target)
        s_denoiser_corrupt_prob=0.15,  # Corruption rate for S denoiser
        x_denoiser_corrupt_prob=0.5,   # Corruption rate for X denoiser
        mean_span_length=3,            # Mean span length for S and X denoiser corruption
        sentinel_token_ids: List[int] = None, # List of sentinel token IDs for span corruption
        max_sentinels_per_sequence=20, # Max sentinels to use for S/X to avoid overly long targets
        ignore_index=-100,             # Value to ignore in loss calculation
        max_seq_length: int = None,    # Max sequence length for input and labels
    ):
        self.tokenizer = tokenizer
        self.prefixes = ul2_prefixes
        self.denoiser_probs = ul2_denoiser_probs
        self.r_denoiser_suffix_ratio = r_denoiser_suffix_ratio
        self.s_denoiser_corrupt_prob = s_denoiser_corrupt_prob
        self.x_denoiser_corrupt_prob = x_denoiser_corrupt_prob
        self.mean_span_length = mean_span_length
        self.ignore_index = ignore_index
        self.max_sentinels_per_sequence = max_sentinels_per_sequence
        self.max_seq_length = max_seq_length

        # Validate tokenizer
        if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id.")
        if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an eos_token_id for UL2-style processing.")
        
        self.eos_token_id = tokenizer.eos_token_id

        # Set up sentinel tokens
        if sentinel_token_ids is None:
            _sentinels = []
            if hasattr(tokenizer, 'additional_special_tokens_ids') and tokenizer.additional_special_tokens_ids:
                for token_id, token_str in zip(tokenizer.additional_special_tokens_ids, tokenizer.additional_special_tokens):
                    if "extra_id" in token_str or "sentinel" in token_str.lower():
                        _sentinels.append(token_id)
            
            if _sentinels:
                self.sentinel_token_ids = sorted(list(set(_sentinels)), reverse=True)
                print(f"Using {len(self.sentinel_token_ids)} sentinel tokens from tokenizer: {self.sentinel_token_ids[:5]}...")
            else:
                print("Warning: Sentinel tokens not found in tokenizer. Generating new ones. "
                      "Ensure tokenizer and model are resized if using this fallback.")
                num_sentinels = 100
                vocab_size = tokenizer.vocab_size
                self.sentinel_token_ids = list(range(vocab_size, vocab_size + num_sentinels))
        else:
            self.sentinel_token_ids = sentinel_token_ids

        if not self.sentinel_token_ids:
            raise ValueError("No sentinel token IDs available. Provide them or ensure tokenizer has them.")

    def __call__(self, examples: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        # Track sentinel usage across the batch to avoid conflicts
        batch_sentinel_offset = 0

        for example in examples:
            input_ids = example["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()

            # Choose denoiser
            denoiser_choice_idx = random.choices(range(len(self.prefixes)), weights=self.denoiser_probs, k=1)[0]
            prefix_token_ids = self.tokenizer.encode(self.prefixes[denoiser_choice_idx], add_special_tokens=False)
            
            # Truncate input if necessary
            if self.max_seq_length:
                max_len = self.max_seq_length - len(prefix_token_ids) - 5  # Reserve space for prefix and EOS
                if len(input_ids) > max_len:
                    input_ids = input_ids[:max_len]

            # Apply denoising based on chosen strategy
            if denoiser_choice_idx == 0:  # R-Denoiser (Prefix LM)
                processed_input_ids, denoised_labels = self._r_denoise(input_ids, self.r_denoiser_suffix_ratio)
                sentinels_used = 0
            elif denoiser_choice_idx == 1:  # S-Denoiser (Span Corruption)
                processed_input_ids, denoised_labels, sentinels_used = self._span_corruption_denoise(
                    input_ids, self.s_denoiser_corrupt_prob, batch_sentinel_offset
                )
            elif denoiser_choice_idx == 2:  # X-Denoiser (Extreme Span Corruption)
                processed_input_ids, denoised_labels, sentinels_used = self._span_corruption_denoise(
                    input_ids, self.x_denoiser_corrupt_prob, batch_sentinel_offset
                )
            else:
                raise ValueError(f"Invalid denoiser choice index: {denoiser_choice_idx}")

            # Update batch sentinel offset
            batch_sentinel_offset += sentinels_used

            # Combine prefix with processed input
            current_input_ids = prefix_token_ids + processed_input_ids
            current_labels = list(denoised_labels)

            # Append EOS to input and labels
            final_input_ids = current_input_ids + [self.eos_token_id]
            
            if current_labels:
                final_labels = current_labels + [self.eos_token_id]
            else:
                final_labels = [self.eos_token_id]

            attention_mask = [1] * len(final_input_ids)

            batch_input_ids.append(final_input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(final_labels)

        # Pad sequences
        padded_batch = self._pad_sequences(batch_input_ids, batch_attention_mask, batch_labels)
        return padded_batch

    def _r_denoise(self, tokens: List[int], suffix_ratio: float) -> Tuple[List[int], List[int]]:
        """R-denoiser: Prefix Language Modeling"""
        if len(tokens) < 2:
            return tokens, []
        
        # Calculate suffix length (target part)
        num_suffix_tokens = max(1, int(len(tokens) * suffix_ratio))
        num_suffix_tokens = min(num_suffix_tokens, len(tokens) - 1)  # Leave at least 1 token in prefix
        
        split_point = len(tokens) - num_suffix_tokens
        input_part = tokens[:split_point]
        target_part = tokens[split_point:]
        
        return input_part, target_part

    def _span_corruption_denoise(self, tokens: List[int], mask_prob: float, sentinel_offset: int = 0) -> Tuple[List[int], List[int], int]:
        """S/X-denoiser: Span Corruption"""
        if not tokens:
            return [], [], 0

        num_tokens_to_mask_ideal = int(len(tokens) * mask_prob)
        if num_tokens_to_mask_ideal == 0 and mask_prob > 0 and len(tokens) > 0:
            num_tokens_to_mask_ideal = 1
        
        if num_tokens_to_mask_ideal == 0:
            return tokens, [], 0

        token_mask_bool = np.zeros(len(tokens), dtype=bool)
        num_masked_total = 0
        spans_to_mask_details = []

        # Generate spans to mask
        max_attempts = len(tokens) * 2
        for attempt in range(max_attempts):
            if (num_masked_total >= num_tokens_to_mask_ideal or 
                len(spans_to_mask_details) >= self.max_sentinels_per_sequence):
                break
            
            # Sample span length from geometric distribution
            span_length = max(1, np.random.geometric(1.0 / self.mean_span_length))
            
            # Find available positions
            unmasked_indices = [i for i, masked in enumerate(token_mask_bool) if not masked]
            if not unmasked_indices:
                break

            start_index = random.choice(unmasked_indices)
            current_actual_span_tokens = []
            end_index = start_index
            
            # Mask tokens in the span
            for i in range(start_index, min(len(tokens), start_index + span_length)):
                if not token_mask_bool[i]:
                    token_mask_bool[i] = True
                    current_actual_span_tokens.append(tokens[i])
                    num_masked_total += 1
                end_index = i + 1

            if current_actual_span_tokens:
                spans_to_mask_details.append({
                    'start': start_index,
                    'end': end_index,
                    'tokens': current_actual_span_tokens
                })

        if not spans_to_mask_details:
            return tokens, [], 0

        # Sort spans by start index and merge overlapping/adjacent spans
        spans_to_mask_details.sort(key=lambda x: x['start'])
        merged_spans_with_sentinels = []

        if spans_to_mask_details:
            current_merged_span_tokens = list(spans_to_mask_details[0]['tokens'])
            current_merged_start_idx = spans_to_mask_details[0]['start']
            current_merged_mask_region_end_idx = spans_to_mask_details[0]['end']

            for i in range(1, len(spans_to_mask_details)):
                next_span = spans_to_mask_details[i]
                
                # Check if spans should be merged (overlapping or adjacent)
                if next_span['start'] <= current_merged_mask_region_end_idx:
                    # Merge spans
                    current_merged_span_tokens.extend(next_span['tokens'])
                    current_merged_mask_region_end_idx = max(current_merged_mask_region_end_idx, next_span['end'])
                else:
                    # Finalize current merged span
                    sentinel_id = self.sentinel_token_ids[
                        (sentinel_offset + len(merged_spans_with_sentinels)) % len(self.sentinel_token_ids)
                    ]
                    merged_spans_with_sentinels.append({
                        'start_original': current_merged_start_idx,
                        'end_original': current_merged_mask_region_end_idx,
                        'sentinel': sentinel_id,
                        'target_span_tokens': list(current_merged_span_tokens)
                    })
                    
                    if len(merged_spans_with_sentinels) >= self.max_sentinels_per_sequence:
                        break
                    
                    # Start new merged span
                    current_merged_span_tokens = list(next_span['tokens'])
                    current_merged_start_idx = next_span['start']
                    current_merged_mask_region_end_idx = next_span['end']

            # Add the last span if within limits
            if len(merged_spans_with_sentinels) < self.max_sentinels_per_sequence:
                sentinel_id = self.sentinel_token_ids[
                    (sentinel_offset + len(merged_spans_with_sentinels)) % len(self.sentinel_token_ids)
                ]
                merged_spans_with_sentinels.append({
                    'start_original': current_merged_start_idx,
                    'end_original': current_merged_mask_region_end_idx,
                    'sentinel': sentinel_id,
                    'target_span_tokens': list(current_merged_span_tokens)
                })

        if not merged_spans_with_sentinels:
            return tokens, [], 0

        # Build corrupted input and target sequences
        corrupted_tokens = []
        target_tokens = []
        current_pos_in_original = 0

        for merged_span_info in merged_spans_with_sentinels:
            # Add unmasked tokens before the current span
            corrupted_tokens.extend(tokens[current_pos_in_original:merged_span_info['start_original']])
            
            # Add sentinel to corrupted input
            corrupted_tokens.append(merged_span_info['sentinel'])
            
            # Add sentinel and original span to target
            target_tokens.append(merged_span_info['sentinel'])
            target_tokens.extend(merged_span_info['target_span_tokens'])
            
            current_pos_in_original = merged_span_info['end_original']

        # Add any remaining unmasked tokens at the end
        corrupted_tokens.extend(tokens[current_pos_in_original:])
        
        return corrupted_tokens, target_tokens, len(merged_spans_with_sentinels)

    def _pad_sequences(self, batch_input_ids: List[List[int]], 
                      batch_attention_mask: List[List[int]], 
                      batch_labels: List[List[int]]) -> Dict[str, torch.Tensor]:
        """Pad sequences to the same length within the batch"""
        max_input_len = max(len(x) for x in batch_input_ids) if batch_input_ids else 0
        max_target_len = max(len(x) for x in batch_labels) if batch_labels else 0

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for i in range(len(batch_input_ids)):
            # Pad input_ids and attention_mask
            input_padding_len = max_input_len - len(batch_input_ids[i])
            padded_input_ids.append(batch_input_ids[i] + [self.tokenizer.pad_token_id] * input_padding_len)
            padded_attention_mask.append(batch_attention_mask[i] + [0] * input_padding_len)

            # Pad labels
            label_padding_len = max_target_len - len(batch_labels[i])
            padded_labels.append(batch_labels[i] + [self.ignore_index] * label_padding_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
