import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass

def setup_ul2_tokenizer(tokenizer, num_sentinel_tokens=100, ul2_prefixes=None):
    """
    Setup tokenizer with UL2 special tokens BEFORE model initialization.
    This ensures the model's embedding layers are properly sized.
    """
    if ul2_prefixes is None:
        ul2_prefixes = ["[NLU]", "[S2S]", "[NLG]"]
    
    # Collect all special tokens to add
    special_tokens_to_add = []
    existing_vocab = tokenizer.get_vocab()
    
    # Add sentinel tokens (T5-style)
    #sentinel_tokens = [f"<extra_id_{i}>" for i in range(num_sentinel_tokens)]
    sentinel_tokens = [f"<unused{i}>" for i in range(num_sentinel_tokens)]
    for token in sentinel_tokens:
        if token not in existing_vocab:
            special_tokens_to_add.append(token)
    
    # Add UL2 prefix tokens
    for prefix in ul2_prefixes:
        if prefix not in existing_vocab:
            special_tokens_to_add.append(prefix)
    
    # Add all special tokens at once
    if special_tokens_to_add:
        # Preserve existing special tokens
        existing_special = tokenizer.additional_special_tokens or []
        all_special = existing_special + special_tokens_to_add
        
        num_added = tokenizer.add_special_tokens({
            'additional_special_tokens': all_special
        })
        print(f"Added {num_added} special tokens to tokenizer")
        print(f"New vocabulary size: {len(tokenizer)}")
    
    # Ensure required special tokens exist
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
    
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '</s>'})
    
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '<s>'})
    
    return tokenizer