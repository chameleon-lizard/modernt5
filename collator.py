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
        # R-denoiser: ratio of sequence length to be used as suffix (target)
        r_denoiser_suffix_ratio=0.15,
        s_denoiser_corrupt_prob=0.15,  # Corruption rate for S denoiser
        x_denoiser_corrupt_prob=0.5,   # Corruption rate for X denoiser
        mean_span_length=3,            # Mean span length for S and X denoiser corruption
        sentinel_token_ids: List[int] = None, # List of sentinel token IDs for span corruption
        max_sentinels_per_sequence=20, # Max sentinels to use for S/X to avoid overly long targets
        ignore_index=-100,             # Value to ignore in loss calculation
        max_seq_length: int = None,    # New parameter: Max sequence length for input and labels
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
        self.max_seq_length = max_seq_length # Store max_seq_length

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
        
        self.sentinel_assignment_idx = 0

    def __call__(self, examples: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for example in examples:
            input_ids = example["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()

            denoiser_choice_idx = random.choices(range(len(self.prefixes)), weights=self.denoiser_probs, k=1)[0]
            paradigm_prefix_str = self.prefixes[denoiser_choice_idx]
            prefix_token_ids = self.tokenizer.encode(paradigm_prefix_str, add_special_tokens=False)

            processed_input_ids = []
            denoised_labels = [] # Renamed from 'labels' to avoid confusion with final 'labels'

            if denoiser_choice_idx == 0: # R-Denoiser (Prefix LM)
                processed_input_ids, denoised_labels = self._r_denoise(input_ids, self.r_denoiser_suffix_ratio)
            elif denoiser_choice_idx == 1: # S-Denoiser (Span Corruption)
                processed_input_ids, denoised_labels = self._span_corruption_denoise(
                    input_ids, self.s_denoiser_corrupt_prob
                )
            elif denoiser_choice_idx == 2: # X-Denoiser (Extreme Span Corruption)
                processed_input_ids, denoised_labels = self._span_corruption_denoise(
                    input_ids, self.x_denoiser_corrupt_prob
                )
            else:
                raise ValueError(f"Invalid denoiser choice index: {denoiser_choice_idx}")

            # Combine prefix with processed input
            current_input_ids = prefix_token_ids + processed_input_ids
            current_labels = list(denoised_labels) # Make a mutable copy

            # Truncate if max_seq_length is set, reserving space for EOS
            if self.max_seq_length is not None:
                max_len_for_content = self.max_seq_length - 1 # -1 for EOS

                if len(current_input_ids) > max_len_for_content:
                    current_input_ids = current_input_ids[:max_len_for_content]
                
                if len(current_labels) > max_len_for_content:
                    current_labels = current_labels[:max_len_for_content]

            # Append EOS to input and labels
            final_input_ids = current_input_ids + [self.eos_token_id]
            
            if current_labels: # If there are any label tokens after potential truncation
                 final_labels = current_labels + [self.eos_token_id]
            else: # If labels became empty (either from R-denoiser or truncation of all content)
                 final_labels = [self.eos_token_id] # Labels are just EOS

            attention_mask = [1] * len(final_input_ids)

            batch_input_ids.append(final_input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(final_labels)

        # Pad sequences
        padded_batch = self._pad_sequences(batch_input_ids, batch_attention_mask, batch_labels)
        return padded_batch

    def _r_denoise(self, tokens: List[int], suffix_ratio: float) -> Tuple[List[int], List[int]]:
        if not tokens:
            return [], []
        
        n_tokens = len(tokens)
        if n_tokens < 2 : 
            return tokens, [] 

        min_suffix_len = 1
        max_suffix_len = n_tokens if suffix_ratio == 1.0 else n_tokens -1
        if max_suffix_len < min_suffix_len: # handles n_tokens=1 if prefix must be non-empty
            return tokens, []


        num_suffix_tokens = int(n_tokens * suffix_ratio)
        num_suffix_tokens = max(min_suffix_len, min(num_suffix_tokens, max_suffix_len))
        
        split_point = n_tokens - num_suffix_tokens

        input_part = tokens[:split_point]
        target_part = tokens[split_point:]
        
        return input_part, target_part

    def _span_corruption_denoise(self, tokens: List[int], mask_prob: float) -> Tuple[List[int], List[int]]:
        if not tokens:
            return [], []

        corrupted_tokens = []
        target_tokens = []
        
        num_tokens_to_mask_ideal = int(len(tokens) * mask_prob)
        if num_tokens_to_mask_ideal == 0 and mask_prob > 0 and len(tokens) > 0:
             num_tokens_to_mask_ideal = 1
        
        if num_tokens_to_mask_ideal == 0: # No tokens to mask
            return tokens, []

        token_mask_bool = np.zeros(len(tokens), dtype=bool)
        num_masked_total = 0
        spans_to_mask_details = [] # Stores (start, end, original_length_before_merge)

        current_sentinel_id_idx_offset = self.sentinel_assignment_idx # Per-sequence offset
        self.sentinel_assignment_idx = (self.sentinel_assignment_idx + 1) % len(self.sentinel_token_ids)
        
        # Limit the number of generated spans to avoid excessive sentinel usage or overly long targets
        # even before max_sentinels_per_sequence truncation, also to avoid infinite loop if tokens can't be masked
        max_attempts_to_find_span = len(tokens) * 2 # Heuristic

        for attempt in range(max_attempts_to_find_span):
            if num_masked_total >= num_tokens_to_mask_ideal or \
               len(spans_to_mask_details) >= self.max_sentinels_per_sequence:
                break
            
            span_length = np.random.geometric(1.0 / self.mean_span_length)
            span_length = max(1, span_length) # ensure span length is at least 1

            unmasked_indices = [i for i, masked in enumerate(token_mask_bool) if not masked]
            if not unmasked_indices:
                break # All tokens are already masked

            start_index = random.choice(unmasked_indices)
            
            current_actual_span_tokens = []
            end_index = start_index
            for i in range(start_index, min(len(tokens), start_index + span_length)):
                if not token_mask_bool[i]:
                    token_mask_bool[i] = True
                    current_actual_span_tokens.append(tokens[i])
                    num_masked_total += 1
                end_index = i + 1 # exclusive end
            
            if current_actual_span_tokens: # if any token was actually masked in this attempt
                # Store with original start and the actual end index for this segment
                spans_to_mask_details.append({'start': start_index, 'end': end_index, 'tokens': current_actual_span_tokens})

        if not spans_to_mask_details:
            return tokens, []

        # Sort spans by start index to process them in order for merging
        spans_to_mask_details.sort(key=lambda x: x['start'])
        
        # Merge overlapping/adjacent spans and assign sentinels
        merged_spans_with_sentinels = []
        if spans_to_mask_details:
            # Initialize with the first span
            current_merged_span_tokens = list(spans_to_mask_details[0]['tokens'])
            current_merged_start_idx = spans_to_mask_details[0]['start']
            # The end of the *masked region* for the current merged span
            current_merged_mask_region_end_idx = spans_to_mask_details[0]['end'] 

            for i in range(1, len(spans_to_mask_details)):
                next_span = spans_to_mask_details[i]
                # Check if next_span's start is within or adjacent to current_merged_mask_region_end_idx
                if next_span['start'] <= current_merged_mask_region_end_idx: 
                    # Merge: extend current span's tokens and update its effective end
                    current_merged_span_tokens.extend(next_span['tokens'])
                    current_merged_mask_region_end_idx = max(current_merged_mask_region_end_idx, next_span['end'])
                else:
                    # Finalize previous merged span
                    sentinel_id = self.sentinel_token_ids[
                        (current_sentinel_id_idx_offset + len(merged_spans_with_sentinels)) % len(self.sentinel_token_ids)
                    ]
                    merged_spans_with_sentinels.append({
                        'start_original': current_merged_start_idx, 
                        'end_original': current_merged_mask_region_end_idx, # This is the end of the masked region
                        'sentinel': sentinel_id,
                        'target_span_tokens': list(current_merged_span_tokens)
                    })
                    if len(merged_spans_with_sentinels) >= self.max_sentinels_per_sequence:
                        break
                    # Start a new merged span
                    current_merged_span_tokens = list(next_span['tokens'])
                    current_merged_start_idx = next_span['start']
                    current_merged_mask_region_end_idx = next_span['end']
            
            # Add the last processed or only span, if not over sentinel limit
            if len(merged_spans_with_sentinels) < self.max_sentinels_per_sequence:
                sentinel_id = self.sentinel_token_ids[
                    (current_sentinel_id_idx_offset + len(merged_spans_with_sentinels)) % len(self.sentinel_token_ids)
                ]
                merged_spans_with_sentinels.append({
                    'start_original': current_merged_start_idx,
                    'end_original': current_merged_mask_region_end_idx,
                    'sentinel': sentinel_id,
                    'target_span_tokens': list(current_merged_span_tokens)
                })

        if not merged_spans_with_sentinels:
             return tokens, []

        # Construct corrupted_tokens and target_tokens
        current_pos_in_original = 0
        for merged_span_info in merged_spans_with_sentinels:
            # Add unmasked tokens before the current span
            corrupted_tokens.extend(tokens[current_pos_in_original : merged_span_info['start_original']])
            
            # Add sentinel to corrupted input
            corrupted_tokens.append(merged_span_info['sentinel'])
            
            # Add sentinel and original span to target
            target_tokens.append(merged_span_info['sentinel'])
            target_tokens.extend(merged_span_info['target_span_tokens'])
            
            current_pos_in_original = merged_span_info['end_original']

        # Add any remaining unmasked tokens at the end
        corrupted_tokens.extend(tokens[current_pos_in_original:])
        
        return corrupted_tokens, target_tokens


    def _pad_sequences(self, batch_input_ids: List[List[int]], 
                       batch_attention_mask: List[List[int]], 
                       batch_labels: List[List[int]]) -> Dict[str, torch.Tensor]:
        max_input_len = max(len(x) for x in batch_input_ids) if batch_input_ids else 0
        max_target_len = max(len(x) for x in batch_labels) if batch_labels else 0

        # If self.max_seq_length is set, these lengths should already be capped at self.max_seq_length
        # due to the truncation logic in __call__.
        # No explicit use of self.max_seq_length needed here for padding length calculation,
        # as it pads to the current batch's max (which is already truncated).

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for i in range(len(batch_input_ids)):
            input_padding_len = max_input_len - len(batch_input_ids[i])
            padded_input_ids.append(batch_input_ids[i] + [self.tokenizer.pad_token_id] * input_padding_len)
            padded_attention_mask.append(batch_attention_mask[i] + [0] * input_padding_len)

            label_padding_len = max_target_len - len(batch_labels[i])
            padded_labels.append(batch_labels[i] + [self.ignore_index] * label_padding_len)
        
        return_dict = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        return return_dict
