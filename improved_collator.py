import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from datasets import load_from_disk
import argparse
from tqdm import tqdm
import os
import json
import wandb
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path

class ImprovedUL2Collator:
    """Improved UL2 collator with better error handling and T5 compatibility"""
    
    def __init__(
        self,
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 512,
        ul2_prefixes: List[str] = ["[NLU]", "[S2S]", "[NLG]"],
        ul2_denoiser_probs: List[float] = [0.5, 0.25, 0.25],
        r_denoiser_suffix_ratio: float = 0.25,
        s_denoiser_corrupt_prob: float = 0.15,
        x_denoiser_corrupt_prob: float = 0.5,
        mean_span_length: int = 3,
        num_sentinel_tokens: int = 100,
        ignore_index: int = -100,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prefixes = ul2_prefixes
        self.denoiser_probs = ul2_denoiser_probs
        self.r_suffix_ratio = r_denoiser_suffix_ratio
        self.s_corrupt_prob = s_denoiser_corrupt_prob
        self.x_corrupt_prob = x_denoiser_corrupt_prob
        self.mean_span_length = mean_span_length
        self.ignore_index = ignore_index
        
        # Setup special tokens properly
        self._setup_special_tokens(num_sentinel_tokens)
        
    def _setup_special_tokens(self, num_sentinel_tokens: int):
        """Setup sentinel tokens and task prefixes"""
        
        # Create sentinel tokens
        sentinel_tokens = [f"<extra_id_{i}>" for i in range(num_sentinel_tokens)]
        
        # Collect all special tokens to add
        special_tokens_to_add = []
        existing_vocab = self.tokenizer.get_vocab()
        
        # Add sentinel tokens if not present
        for token in sentinel_tokens:
            if token not in existing_vocab:
                special_tokens_to_add.append(token)
        
        # Add prefix tokens if not present
        for prefix in self.prefixes:
            if prefix not in existing_vocab:
                special_tokens_to_add.append(prefix)
        
        if special_tokens_to_add:
            # Add all at once
            num_added = self.tokenizer.add_special_tokens({
                'additional_special_tokens': self.tokenizer.additional_special_tokens + special_tokens_to_add
            })
            print(f"Added {num_added} special tokens to tokenizer")
        
        # Store sentinel token IDs (verify they exist)
        self.sentinel_token_ids = []
        for i in range(num_sentinel_tokens):
            token = f"<extra_id_{i}>"
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Failed to add sentinel token {token}")
            self.sentinel_token_ids.append(token_id)
        
        # Get prefix token IDs
        self.prefix_token_ids = {}
        for prefix in self.prefixes:
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            if not prefix_ids:
                raise ValueError(f"Failed to encode prefix {prefix}")
            self.prefix_token_ids[prefix] = prefix_ids
        
        # Ensure required special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "</s>"
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = "<s>"
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process batch with UL2 objectives"""
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for example in examples:
            # Get text from example
            text = example.get("text", "")
            if not text or len(text.strip()) == 0:
                continue  # Skip empty examples
            
            # Tokenize text (limit length to leave room for corruption)
            max_length = self.max_input_length + self.max_target_length - 20  # Leave room for sentinels
            tokens = self.tokenizer.encode(
                text, 
                add_special_tokens=False, 
                truncation=True, 
                max_length=max_length
            )
            
            if len(tokens) < 10:  # Skip very short sequences
                continue
            
            # Choose denoising objective
            objective_idx = np.random.choice(len(self.prefixes), p=self.denoiser_probs)
            prefix_ids = self.prefix_token_ids[self.prefixes[objective_idx]]
            
            # Apply denoising
            if objective_idx == 0:  # R-denoiser (Prefix LM)
                input_ids, target_ids = self._apply_prefix_lm(tokens)
            elif objective_idx == 1:  # S-denoiser (Standard corruption)
                input_ids, target_ids = self._apply_span_corruption(tokens, self.s_corrupt_prob)
            else:  # X-denoiser (Extreme corruption)
                input_ids, target_ids = self._apply_span_corruption(tokens, self.x_corrupt_prob)
            
            # Add prefix to input
            input_ids = prefix_ids + input_ids
            
            # Add EOS token to target if needed
            if target_ids and self.tokenizer.eos_token_id is not None:
                target_ids = target_ids + [self.tokenizer.eos_token_id]
            
            # Truncate if needed
            input_ids = input_ids[:self.max_input_length]
            target_ids = target_ids[:self.max_target_length]
            
            # Create attention mask
            attention_mask = [1] * len(input_ids)
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(target_ids)
        
        # Handle empty batch
        if not batch_input_ids:
            # Return a dummy batch with single padding token
            return {
                "input_ids": torch.tensor([[self.tokenizer.pad_token_id]], dtype=torch.long),
                "attention_mask": torch.tensor([[0]], dtype=torch.long),
                "labels": torch.tensor([[self.ignore_index]], dtype=torch.long),
            }
        
        # Pad sequences
        batch = self._pad_sequences(batch_input_ids, batch_attention_mask, batch_labels)
        return batch
    
    def _apply_prefix_lm(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        """Apply prefix language modeling"""
        
        if len(tokens) < 2:
            return tokens, [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []
        
        # Split at ratio point
        split_point = max(1, int(len(tokens) * (1 - self.r_suffix_ratio)))
        split_point = min(split_point, len(tokens) - 1)  # Ensure we have targets
        
        input_ids = tokens[:split_point]
        target_ids = tokens[split_point:]
        
        return input_ids, target_ids
    
    def _apply_span_corruption(self, tokens: List[int], corrupt_prob: float) -> Tuple[List[int], List[int]]:
        """Apply span corruption with sentinels"""
        
        if not tokens or corrupt_prob == 0 or len(tokens) < 2:
            return tokens, []
        
        num_to_mask = max(1, int(len(tokens) * corrupt_prob))
        num_to_mask = min(num_to_mask, len(tokens) - 1)  # Leave at least one token
        
        # Create mask for corruption
        mask = np.zeros(len(tokens), dtype=bool)
        
        # Generate spans to mask
        spans = []
        current_masked = 0
        max_spans = min(len(self.sentinel_token_ids), 20)  # Limit number of spans
        
        while current_masked < num_to_mask and len(spans) < max_spans:
            # Sample span length with upper bound
            span_length = min(
                np.random.poisson(self.mean_span_length) + 1,  # Use Poisson instead of geometric
                num_to_mask - current_masked,
                10  # Max span length
            )
            
            if span_length <= 0:
                continue
            
            # Find valid start position
            valid_starts = []
            for i in range(len(tokens) - span_length + 1):
                if not mask[i:i+span_length].any():
                    valid_starts.append(i)
            
            if not valid_starts:
                break
            
            start = np.random.choice(valid_starts)
            mask[start:start + span_length] = True
            spans.append((start, start + span_length))
            current_masked += span_length
        
        if not spans:  # No spans created, fallback
            spans = [(0, min(3, len(tokens)))]
        
        # Sort spans by position
        spans.sort(key=lambda x: x[0])
        
        # Build corrupted input and targets
        input_ids = []
        target_ids = []
        last_pos = 0
        
        for i, (start, end) in enumerate(spans):
            # Add uncorrupted tokens to input
            input_ids.extend(tokens[last_pos:start])
            
            # Add sentinel to input
            sentinel_id = self.sentinel_token_ids[i % len(self.sentinel_token_ids)]
            input_ids.append(sentinel_id)
            
            # Add to targets: sentinel followed by masked tokens
            target_ids.append(sentinel_id)
            target_ids.extend(tokens[start:end])
            
            last_pos = end
        
        # Add remaining tokens to input
        input_ids.extend(tokens[last_pos:])
        
        return input_ids, target_ids
    
    def _pad_sequences(
        self,
        input_ids: List[List[int]],
        attention_mask: List[List[int]],
        labels: List[List[int]]
    ) -> Dict[str, torch.Tensor]:
        """Pad sequences to same length"""
        
        # Find max lengths
        max_input_len = max(len(ids) for ids in input_ids)
        max_label_len = max(len(ids) for ids in labels) if labels[0] else 1
        
        # Pad sequences
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        pad_token_id = self.tokenizer.pad_token_id or 0
        
        for i in range(len(input_ids)):
            # Pad input
            input_padding_length = max_input_len - len(input_ids[i])
            padded_input_ids.append(input_ids[i] + [pad_token_id] * input_padding_length)
            
            # Pad attention mask
            attn_padding_length = max_input_len - len(attention_mask[i])
            padded_attention_mask.append(attention_mask[i] + [0] * attn_padding_length)
            
            # Pad labels (use ignore_index for padding)
            if labels[i]:
                label_padding_length = max_label_len - len(labels[i])
                padded_labels.append(labels[i] + [self.ignore_index] * label_padding_length)
            else:
                padded_labels.append([self.ignore_index] * max_label_len)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }