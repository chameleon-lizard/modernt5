import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
import random
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from datasets import load_from_disk
import argparse
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path
from accelerate import Accelerator

# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optional imports for optimizers
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

try:
    #from torch.optim import Muon
    from dion import Muon
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False

try:
    from dion import Dion
    DION_AVAILABLE = True
except ImportError:
    DION_AVAILABLE = False

# Optional logging imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import trackio
    TRACKIO_AVAILABLE = True
except ImportError:
    TRACKIO_AVAILABLE = False

#from fixed_modernt5 import ModernT5ForConditionalGeneration
from modernbert_t5gemma_hybrid import ModernBertT5GemmaForConditionalGeneration
from ul2_collator import ImprovedUL2CollatorV2


def worker_init_fn(worker_id):
    """Initialize worker process for DataLoader"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def halflife_to_decay(halflife_tokens: float, tokens_per_step: int) -> float:
    """Convert token half-life to decay rate (β).
    
    Args:
        halflife_tokens: Number of tokens for contribution to decay by half
        tokens_per_step: batch_size * sequence_length
    
    Returns:
        Decay rate β where β^(halflife_tokens/tokens_per_step) = 0.5
    """
    halflife_steps = halflife_tokens / tokens_per_step
    return 0.5 ** (1.0 / halflife_steps)

def decay_to_halflife(decay: float, tokens_per_step: int) -> float:
    """Convert decay rate (β) to token half-life."""
    import math
    halflife_steps = math.log(0.5) / math.log(decay)
    return halflife_steps * tokens_per_step

def scale_beta2_for_batch_size(beta2: float, old_batch_size: int, new_batch_size: int) -> float:
    """Scale β2 when changing batch size to maintain constant token half-life."""
    return beta2 ** (new_batch_size / old_batch_size)


class Seq2SeqTrainer:
    """Simplified trainer for seq2seq models with multiple optimizer support"""
    
    def __init__(
        self,
        accelerator: Accelerator,
        model: ModernBertT5GemmaForConditionalGeneration,
        tokenizer,
        train_dataset,
        val_dataset=None,
        collator=None,
        args=None,
    ):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collator = collator
        self.args = args
        
        # Setup gradient checkpointing if requested
        if args.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Setup parameter groups for special optimizers
        self._setup_parameter_groups()
        
        # Setup data loaders
        self._setup_data_loaders()
        
        # Setup optimizer and scheduler
        self._setup_optimization()
        
        # Prepare with Accelerate
        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        
        if self.val_dataset:
            self.val_loader = self.accelerator.prepare(self.val_loader)
        
        # Setup logging (only on main process)
        if self.accelerator.is_main_process:
            self._setup_logging()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        model = self.accelerator.unwrap_model(self.model)
        
        # Try to enable for encoder
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'gradient_checkpointing_enable'):
            model.encoder.gradient_checkpointing_enable()
            if self.accelerator.is_local_main_process:
                print("Enabled gradient checkpointing for encoder")
        
        # Try to enable for decoder
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'gradient_checkpointing_enable'):
            model.decoder.gradient_checkpointing_enable()
            if self.accelerator.is_local_main_process:
                print("Enabled gradient checkpointing for decoder")

    def _setup_parameter_groups(self):
        """Setup parameter groups for optimizer-specific handling"""
        model = self.accelerator.unwrap_model(self.model)
        
        if self.args.optimizer in ['dion','muon']:
            self.matrix_params = []
            self.vector_params = []
            self.norm_params = []
            self.embed_params = []
            self.lm_head_params = []
            
            # FIX: Track seen tensor IDs to avoid duplicates from weight tying
            seen_param_ids = set()
            
            # First pass: identify tied parameters
            lm_head_weight_id = None
            decoder_embed_weight_id = None
            
            if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
                lm_head_weight_id = id(model.lm_head.weight)
            
            if hasattr(model, 'model') and hasattr(model.model, 'decoder'):
                decoder = model.model.decoder
                if hasattr(decoder, 'embed_tokens'):
                    decoder_embed_weight_id = id(decoder.embed_tokens.weight)
            
            # Check if they're tied
            weights_are_tied = (lm_head_weight_id is not None and 
                            lm_head_weight_id == decoder_embed_weight_id)
            
            if self.accelerator.is_local_main_process and weights_are_tied:
                print("Detected tied embeddings between decoder and lm_head. Matching lm_head as embedding layer (no special lr)")
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                param_id = id(param)
                
                # Skip if already seen (handles weight tying)
                if param_id in seen_param_ids:
                    if self.accelerator.is_local_main_process:
                        print(f"Skipping duplicate parameter: {name}")
                    continue
                
                seen_param_ids.add(param_id)
                
                # Categorize parameter
                # FIX: For tied weights, categorize by the decoder embedding name
                if param_id == lm_head_weight_id and weights_are_tied:
                    # It's the tied weight - treat as embedding
                    self.embed_params.append(param)
                elif 'lm_head' in name and 'weight' not in name:
                    self.lm_head_params.append(param)
                elif ('embed' in name or 'wte' in name or 'wpe' in name):
                    self.embed_params.append(param)
                elif 'norm' in name or 'ln' in name or 'layer_norm' in name:
                    self.norm_params.append(param)
                elif 'bias' in name or param.ndim == 1:
                    self.vector_params.append(param)
                elif param.ndim == 2:
                    if ('layers' in name or 'block' in name or 'attention' in name or 
                        'mlp' in name or 'dense' in name or 'proj' in name):
                        self.matrix_params.append(param)
                    else:
                        self.vector_params.append(param)
                else:
                    self.vector_params.append(param)
            
            if self.accelerator.is_local_main_process:
                print(f"Dion parameter groups (no duplicates):")
                print(f"  Matrix params (Dion): {len(self.matrix_params)}")
                print(f"  Vector params (AdamW): {len(self.vector_params)}")
                print(f"  Norm params (AdamW): {len(self.norm_params)}")
                print(f"  Embedding params (AdamW): {len(self.embed_params)}")
                print(f"  LM Head params (AdamW, scaled): {len(self.lm_head_params)}")

    # def _setup_parameter_groups(self):
    #     """Setup parameter groups for optimizer-specific handling"""
    #     model = self.accelerator.unwrap_model(self.model)
        
    #     if self.args.optimizer == 'dion':
    #         # Categorize parameters according to Dion paper recommendations
    #         self.matrix_params = []      # 2D weight matrices -> Dion/Muon
    #         self.vector_params = []      # Biases and 1D params -> Lion
    #         self.norm_params = []        # LayerNorm/RMSNorm -> Lion
    #         self.embed_params = []       # Embeddings -> Lion
    #         self.lm_head_params = []     # Output projection -> Lion with scaled LR
            
    #         # First, identify tied parameters by checking tensor identity
    #         lm_head_weight_id = None
    #         if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
    #             lm_head_weight_id = id(model.lm_head.weight)
            
    #         # Track which parameters we've already categorized
    #         categorized_ids = set()
            
    #         for name, param in model.named_parameters():
    #             if not param.requires_grad:
    #                 continue
                
    #             param_id = id(param)
                
    #             # Check if this is the lm_head weight (even if named differently due to weight tying)
    #             if lm_head_weight_id and param_id == lm_head_weight_id:
    #                 if param_id not in categorized_ids:
    #                     self.lm_head_params.append(param)
    #                     categorized_ids.add(param_id)
    #                 continue
                
    #             # Skip if already categorized (for tied weights)
    #             if param_id in categorized_ids:
    #                 continue
                
    #             # LM Head parameters (by name, in case bias exists)
    #             if 'lm_head' in name and 'weight' not in name:  # bias if exists
    #                 self.lm_head_params.append(param)
    #                 categorized_ids.add(param_id)
    #             # Embeddings (but not if it's the tied lm_head weight)
    #             elif ('embed' in name or 'wte' in name or 'wpe' in name or 
    #                 ('shared' in name and param_id != lm_head_weight_id)):
    #                 self.embed_params.append(param)
    #                 categorized_ids.add(param_id)
    #             # Normalization layers
    #             elif 'norm' in name or 'ln' in name or 'layer_norm' in name:
    #                 self.norm_params.append(param)
    #                 categorized_ids.add(param_id)
    #             # Bias vectors and other 1D parameters
    #             elif 'bias' in name or param.ndim == 1:
    #                 self.vector_params.append(param)
    #                 categorized_ids.add(param_id)
    #             # 2D weight matrices in transformer blocks (for Dion)
    #             elif param.ndim == 2:
    #                 if ('layers' in name or 'block' in name or 'attention' in name or 
    #                     'mlp' in name or 'dense' in name or 'proj' in name or 
    #                     'wi' in name or 'wo' in name):
    #                     self.matrix_params.append(param)
    #                     categorized_ids.add(param_id)
    #                 else:
    #                     self.vector_params.append(param)
    #                     categorized_ids.add(param_id)
    #             else:
    #                 self.vector_params.append(param)
    #                 categorized_ids.add(param_id)
            
    #         if self.accelerator.is_local_main_process:
    #             print(f"Dion parameter groups:")
    #             print(f"  Matrix params (Dion): {len(self.matrix_params)}")
    #             print(f"  Vector params (Lion): {len(self.vector_params)}")
    #             print(f"  Norm params (Lion): {len(self.norm_params)}")
    #             print(f"  Embedding params (Lion): {len(self.embed_params)}")
    #             print(f"  LM Head params (Lion, scaled): {len(self.lm_head_params)}")
                
    #             # Debug: Check if lm_head was properly identified
    #             if lm_head_weight_id:
    #                 found_lm_head = any(id(p) == lm_head_weight_id for p in self.lm_head_params)
    #                 print(f"  LM head weight correctly identified: {found_lm_head}")
                    
    # def _setup_parameter_groups(self):
    #     """Setup parameter groups for optimizer-specific handling"""
    #     model = self.accelerator.unwrap_model(self.model)
        
    #     if self.args.optimizer == 'dion':
    #         # Categorize parameters according to Dion paper recommendations
    #         self.matrix_params = []      # 2D weight matrices -> Dion/Muon
    #         self.vector_params = []      # Biases and 1D params -> Lion
    #         self.norm_params = []        # LayerNorm/RMSNorm -> Lion
    #         self.embed_params = []       # Embeddings -> Lion
    #         self.lm_head_params = []     # Output projection -> Lion with scaled LR
            
    #         for name, param in model.named_parameters():
    #             if not param.requires_grad:
    #                 continue
                
    #             # LM Head (unembedding) - needs special LR scaling
    #             if 'lm_head' in name or 'output_projection' in name:
    #                 self.lm_head_params.append(param)
    #             # Embeddings
    #             elif 'embed' in name or 'wte' in name or 'wpe' in name or 'shared' in name:
    #                 self.embed_params.append(param)
    #             # Normalization layers
    #             elif 'norm' in name or 'ln' in name or 'layer_norm' in name:
    #                 self.norm_params.append(param)
    #             # Bias vectors and other 1D parameters
    #             elif 'bias' in name or param.ndim == 1:
    #                 self.vector_params.append(param)
    #             # 2D weight matrices in transformer blocks (for Dion)
    #             elif param.ndim == 2:
    #                 if ('layers' in name or 'block' in name or 'attention' in name or 
    #                     'mlp' in name or 'dense' in name or 'proj' in name or 
    #                     'wi' in name or 'wo' in name):
    #                     self.matrix_params.append(param)
    #                 else:
    #                     self.vector_params.append(param)
    #             else:
    #                 self.vector_params.append(param)
            
    #         if self.accelerator.is_local_main_process:
    #             print(f"Dion parameter groups:")
    #             print(f"  Matrix params (Dion): {len(self.matrix_params)}")
    #             print(f"  Vector params (Lion): {len(self.vector_params)}")
    #             print(f"  Norm params (Lion): {len(self.norm_params)}")
    #             print(f"  Embedding params (Lion): {len(self.embed_params)}")
    #             print(f"  LM Head params (Lion, scaled): {len(self.lm_head_params)}")
        
    #     elif self.args.optimizer == 'muon':
    #         # MUON-specific parameter grouping
    #         self.embedding_params = []
    #         self.layer_params = []
    #         self.head_params = []
            
    #         for name, param in model.named_parameters():
    #             if not param.requires_grad:
    #                 continue
                    
    #             if 'embed' in name or 'wte' in name or 'wpe' in name:
    #                 self.embedding_params.append(param)
    #             elif 'lm_head' in name or 'classifier' in name:
    #                 self.head_params.append(param)
    #             else:
    #                 self.layer_params.append(param)
            
    #         if self.accelerator.is_local_main_process:
    #             print(f"MUON parameter groups:")
    #             print(f"  Embedding parameters: {len(self.embedding_params)}")
    #             print(f"  Layer parameters: {len(self.layer_params)}")
    #             print(f"  Head parameters: {len(self.head_params)}")
    
    def _setup_data_loaders(self):
        """Setup data loaders with proper worker initialization"""
        num_workers = self.args.num_workers
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                collate_fn=self.collator,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False,
            )
    
    def _setup_optimization(self):
        """Setup optimizer and learning rate scheduler"""
        model = self.accelerator.unwrap_model(self.model)

        #Based on paper about small batch size training https://arxiv.org/abs/2507.07101
        # Calculate tokens per optimizer step
        tokens_per_step = self.args.batch_size * self.args.max_target_length
        # Option 1: Specify token half-life directly (recommended)
        if hasattr(self.args, 'adam_beta2_halflife_tokens'):
            # User specified half-life in tokens (e.g., 10M tokens)
            self.args.adam_beta2 = halflife_to_decay(
                self.args.adam_beta2_halflife_tokens,
                tokens_per_step
            )
            if self.accelerator.is_local_main_process:
                print(f"Using β2 token half-life: {self.args.adam_beta2_halflife_tokens:,} tokens")
                print(f"Computed β2: {self.args.adam_beta2:.6f}")

        
        # Check optimizer availability
        if self.args.optimizer == 'dion' and not DION_AVAILABLE:
            raise ValueError("Dion not installed. Install with: pip install git+https://github.com/microsoft/dion.git")
        if self.args.optimizer in ['adam8bit', 'adamw8bit', 'lion8bit'] and not BNB_AVAILABLE:
            raise ValueError("8-bit optimizers require bitsandbytes. Install with: pip install bitsandbytes")
        if self.args.optimizer == 'muon' and not MUON_AVAILABLE:
            raise ValueError("MUON optimizer requires PyTorch >= 2.5.0")
        
        # Create optimizer based on type
        if self.args.optimizer == 'dion':
            # Dion with proper parameter grouping
            model_dim = model.config.encoder.hidden_size
            lm_head_lr_scale = 1.0 / math.sqrt(model_dim)
            
            param_groups = []
            
            if self.matrix_params:
                param_groups.append({
                    #'params': self.matrix_params,
                    'params': [p.data if isinstance(p, nn.Parameter) else p for p in self.matrix_params],
                    'algorithm': 'dion',
                    'lr': self.args.learning_rate,
                    'rank_fraction':0.25,
                })
            
            if self.vector_params:
                param_groups.append({
                    'params': self.vector_params,
                    'algorithm': 'adamw',
                    'lr': self.args.learning_rate,
                })
            
            if self.norm_params:
                param_groups.append({
                    'params': self.norm_params,
                    'algorithm': 'adamw',
                    'lr': self.args.learning_rate,
                })
            
            if self.embed_params:
                param_groups.append({
                    'params': self.embed_params,
                    'algorithm': 'adamw',
                    'lr': self.args.learning_rate,
                })
            
            if self.lm_head_params:
                param_groups.append({
                    'params': self.lm_head_params,
                    'algorithm': 'adamw',
                    'lr': self.args.learning_rate * lm_head_lr_scale,
                })
            
            self.optimizer = Dion(
                param_groups,
                lr=self.args.learning_rate,
            )
            
            if self.accelerator.is_local_main_process:
                print(f"Using Dion optimizer")
                print(f"  Learning rate: {self.args.learning_rate}")
                print(f"  LM Head LR: {self.args.learning_rate * lm_head_lr_scale:.2e}")
        
        elif self.args.optimizer == 'muon':
            # MUON with parameter groups
            optimizer_params = [
                {'params': self.embedding_params, 'lr': self.args.learning_rate * 0.1, 'momentum': 0.95},
                {'params': self.layer_params, 'lr': self.args.learning_rate, 'momentum': self.args.muon_momentum},
                {'params': self.head_params, 'lr': self.args.learning_rate * 0.1, 'momentum': 0.95}
            ]
            
            self.optimizer = Muon(
                optimizer_params,
                lr=self.args.learning_rate,
                momentum=self.args.muon_momentum,
                nesterov=self.args.muon_nesterov,
                ns_steps=self.args.muon_ns_steps,
                update_type=self.args.muon_update_type,
            )
        
        elif self.args.optimizer == 'adamw':
            from torch.optim import AdamW
            self.optimizer = AdamW(
                model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        
        elif self.args.optimizer == 'adam8bit':
            self.optimizer = bnb.optim.Adam8bit(
                model.parameters(),
                lr=self.args.learning_rate,
            )
        
        elif self.args.optimizer == 'adamw8bit':
            self.optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        
        elif self.args.optimizer == 'lion8bit':
            self.optimizer = bnb.optim.Lion8bit(
                model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.weight_decay,
            )
        
        elif self.args.optimizer == 'adafactor':
            from transformers import Adafactor
            self.optimizer = Adafactor(
                model.parameters(),
                lr=self.args.learning_rate,
                scale_parameter=True,
                relative_step=False,
                warmup_init=False,
            )
        
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
        
        # Setup scheduler
        total_steps = len(self.train_loader) * self.args.num_epochs // self.accelerator.num_processes
        warmup_steps = min(self.args.warmup_steps, total_steps // 10)
        
        if self.args.scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.args.scheduler == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.args.scheduler == 'constant':
            from transformers import get_constant_schedule
            self.scheduler = get_constant_schedule(self.optimizer)
        elif self.args.scheduler == 'constant_with_warmup':
            from transformers import get_constant_schedule_with_warmup
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)
        
        # Log model info
        self.logger.info(f"Using optimizer: {self.args.optimizer}")
        self.logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # Initialize TrackIO if requested
        if self.args.use_trackio and TRACKIO_AVAILABLE:
            trackio.init(
                project=self.args.trackio_project,
                config=vars(self.args),
                name=self.args.trackio_run_name,
            )
            self.logger.info("Initialized TrackIO logging")
        
        # Initialize wandb if requested
        if self.args.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.wandb_run_name,
                config=vars(self.args),
            )
            self.logger.info("Initialized wandb logging")
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """Log metrics to tracking services"""
        if not self.accelerator.is_main_process:
            return
        
        log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
        log_dict.update({
            f"{prefix}/global_step": self.global_step,
            f"{prefix}/epoch": self.current_epoch,
        })
        
        if self.args.use_trackio and TRACKIO_AVAILABLE:
            trackio.log(log_dict)
        
        if self.args.use_wandb and WANDB_AVAILABLE:
            wandb.log(log_dict)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with validation"""
        self.model.train()
        
        # FIX: Validate batch structure
        required_keys = ['input_ids', 'attention_mask', 'labels']
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Batch missing required key: {key}")
        
        # FIX: Check for decoder_input_ids
        if 'decoder_input_ids' not in batch:
            if self.accelerator.is_local_main_process and self.global_step == 0:
                print("⚠️  WARNING: No decoder_input_ids in batch!")
                print("   The model will auto-generate them from labels")
        
        with self.accelerator.autocast():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch.get("decoder_input_ids"),  # Optional
                labels=batch["labels"],
            )
            loss = outputs.loss
        
        # FIX: Check for NaN loss
        if torch.isnan(loss):
            if self.accelerator.is_local_main_process:
                print(f"\n❌ NaN loss detected at step {self.global_step}!")
                print(f"   Batch shapes: input={batch['input_ids'].shape}, labels={batch['labels'].shape}")
            return {'loss': float('inf'), 'learning_rate': 0.0}
        
        # Scale loss for gradient accumulation
        loss = loss / self.args.gradient_accumulation_steps
        
        # Backward pass
        self.accelerator.backward(loss)
        
        # Gradient accumulation
        if (self.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            if self.args.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm
                )
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        loss = self.accelerator.reduce(loss.detach(), reduction="mean").item() * self.args.gradient_accumulation_steps
        
        return {
            'loss': loss,
            'learning_rate': self.scheduler.get_last_lr()[0],
        }
    
    def validate(self) -> float:
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", disable=not self.accelerator.is_local_main_process):
                with self.accelerator.autocast():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                
                batch_size = batch["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Reduce across processes
        total_loss = self.accelerator.reduce(torch.tensor(total_loss, device=self.accelerator.device), reduction="sum").item()
        total_samples = self.accelerator.reduce(torch.tensor(total_samples, device=self.accelerator.device), reduction="sum").item()
        
        return total_loss / total_samples if total_samples > 0 else float('inf')
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        if path is None:
            path = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        
        os.makedirs(path, exist_ok=True)
        
        # Unwrap and save model
        model_to_save = self.accelerator.unwrap_model(self.model)
        model_to_save.save_pretrained(path, 
                                    #safe_serialization=False
                                    )
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save Accelerate state
        self.accelerator.save_state(path)
        
        self.logger.info(f"Saved checkpoint to {path}")
    
    def train(self):
        """Main training loop"""
        if self.accelerator.is_main_process:
            self.logger.info(f"Starting training for {self.args.num_epochs} epochs")
            self.logger.info(f"Total training steps: {len(self.train_loader) * self.args.num_epochs}")
        
        for epoch in range(self.args.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0
            
            # Training loop
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.accelerator.is_local_main_process)
            
            for batch_idx, batch in enumerate(progress_bar):
                # Training step
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                
                # Update progress bar
                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix({
                        'loss': metrics['loss'],
                        'lr': metrics['learning_rate'],
                    })
                
                # Logging
                if self.accelerator.is_main_process and self.global_step % self.args.logging_steps == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    self.logger.info(
                        f"Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {metrics['learning_rate']:.2e}"
                    )
                    self._log_metrics(metrics, prefix="train")
                
                # Validation
                if self.global_step > 0:
                    if self.val_loader and self.global_step % self.args.eval_steps == 0:
                        val_loss = self.validate()
                        if self.accelerator.is_main_process:
                            self.logger.info(f"Validation loss: {val_loss:.4f}")
                            self._log_metrics({'loss': val_loss}, prefix="val")
                            
                            # Save best model
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self.save_checkpoint(os.path.join(self.args.output_dir, 'best_model'))
                
                # Save checkpoint
                if self.accelerator.is_main_process and self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
                self.accelerator.wait_for_everyone()
            
            # End of epoch validation
            if self.val_loader:
                val_loss = self.validate()
                if self.accelerator.is_main_process:
                    self.logger.info(f"End of epoch {epoch} - Validation loss: {val_loss:.4f}")
                    self._log_metrics({'loss': val_loss}, prefix="val")
        
        # Save final model
        if self.accelerator.is_main_process:
            self.save_checkpoint(os.path.join(self.args.output_dir, 'final_model'))
            
            # Finish logging
            if self.args.use_trackio and TRACKIO_AVAILABLE:
                trackio.finish()
            if self.args.use_wandb and WANDB_AVAILABLE:
                wandb.finish()
            
            self.logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train ModernT5 with various optimizers")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, help="Path to model")
    
    # Data arguments
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, default=None)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=512)
    
    # UL2 collator arguments
    parser.add_argument("--ul2_denoiser_probs", type=float, nargs=3, default=[0.5, 0.25, 0.25])
    parser.add_argument("--ul2_r_ratio", type=float, default=0.25)
    parser.add_argument("--ul2_s_corrupt", type=float, default=0.15)
    parser.add_argument("--ul2_x_corrupt", type=float, default=0.5)
    parser.add_argument("--ul2_mean_span", type=int, default=3)
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Optimizer selection
    parser.add_argument("--optimizer", type=str, 
                       choices=["adamw", "adam8bit", "adamw8bit", "lion8bit", "muon", "dion", "adafactor"], 
                       default="adamw")
    parser.add_argument("--scheduler", type=str, 
                       choices=["linear", "cosine", "constant", "constant_with_warmup"], 
                       default="linear")
    
    # Adam optimizer arguments
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--adam_beta2_halflife_tokens", type=int, default=None,
                   help="Token half-life for β2 (e.g., 10000000 for 10M tokens)")
    
    # MUON optimizer arguments
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_nesterov", action="store_true")
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_update_type", type=str, default="newton", choices=["newton", "fisher"])
    
    # Efficiency arguments
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # Logging arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_trackio", action="store_true", default=False)
    parser.add_argument("--trackio_project", type=str, default="modernt5-training")
    parser.add_argument("--trackio_run_name", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="modernt5-training")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision='bf16' if args.bf16 else 'no',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    if accelerator.is_local_main_process:
        print(f"Loading model from {args.model_name}")
        print(f"Using dtype: {torch_dtype}")
        print(f"Using optimizer: {args.optimizer}")
    
    model = ModernBertT5GemmaForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        #attn_implementation="eager",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create collator
    # collator = ImprovedUL2CollatorV2(
    #     tokenizer=tokenizer,
    #     max_input_length=args.max_input_length,
    #     max_target_length=args.max_target_length,
    #     ul2_denoiser_probs=args.ul2_denoiser_probs,
    #     r_denoiser_suffix_ratio=args.ul2_r_ratio,
    #     s_denoiser_corrupt_prob=args.ul2_s_corrupt,
    #     x_denoiser_corrupt_prob=args.ul2_x_corrupt,
    #     mean_span_length=args.ul2_mean_span,
    # )
    collator = ImprovedUL2CollatorV2(
        tokenizer=tokenizer,
        max_input_length=512,
        max_target_length=512,
        use_bin_packing=False,
    )
    
    # Resize model embeddings if needed
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        print('Model was resized!')
        model.resize_token_embeddings(len(tokenizer))
    
    # Load datasets
    if accelerator.is_local_main_process:
        print("Loading datasets...")
    
    train_dataset = load_from_disk(args.train_dataset)
    val_dataset = None
    
    if args.val_dataset:
        val_dataset = load_from_disk(args.val_dataset)
    else:
        # Split dataset for validation
        dataset = train_dataset.train_test_split(test_size=0.10, seed=args.seed)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        args=args,
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()