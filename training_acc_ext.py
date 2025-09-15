import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
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
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path
from accelerate import Accelerator

# Optional imports for optimizers
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

# Check for MUON optimizer availability
try:
    from torch.optim import Muon
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False

# Optional imports for logging
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

from fixed_modernt5 import ModernT5ForConditionalGeneration
from improved_collator import ImprovedUL2Collator


class AdvancedSeq2SeqTrainer:
    """Advanced trainer with cross-attention warmup, mixed precision, and Accelerate support"""
    
    def __init__(
        self,
        accelerator: Accelerator,
        model: ModernT5ForConditionalGeneration,
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
        
        # Setup gradient checkpointing
        if args.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Track training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Setup cross-attention warmup
        self.warmup_phase = args.cross_attention_warmup_steps > 0
        self.cross_attn_params = []
        self.other_params = []
        if self.warmup_phase:
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
        if hasattr(model.encoder, 'gradient_checkpointing_enable'):
            model.encoder.gradient_checkpointing_enable()
            if self.accelerator.is_local_main_process:
                print("Enabled gradient checkpointing for encoder")
        
        if hasattr(model.decoder, 'gradient_checkpointing_enable'):
            model.decoder.gradient_checkpointing_enable()
            if self.accelerator.is_local_main_process:
                print("Enabled gradient checkpointing for decoder")
    
    def _setup_parameter_groups(self):
        """Setup parameter groups for cross-attention warmup and optimizer-specific handling"""
        model = self.accelerator.unwrap_model(self.model)
        
        # Separate parameters for MUON if needed
        if self.args.optimizer == 'muon':
            # MUON works best with specific parameter grouping
            self.embedding_params = []
            self.layer_params = []
            self.head_params = []
            
            for name, param in model.named_parameters():
                if 'embed' in name or 'wte' in name or 'wpe' in name:
                    self.embedding_params.append(param)
                elif 'lm_head' in name or 'classifier' in name:
                    self.head_params.append(param)
                else:
                    self.layer_params.append(param)
            
            if self.accelerator.is_local_main_process:
                print(f"MUON parameter groups:")
                print(f"  Embedding parameters: {len(self.embedding_params)}")
                print(f"  Layer parameters: {len(self.layer_params)}")
                print(f"  Head parameters: {len(self.head_params)}")
        else:
            # Standard cross-attention warmup grouping
            for name, param in model.named_parameters():
                if 'cross_attention' in name or 'encoder_attn' in name:
                    self.cross_attn_params.append(param)
                else:
                    self.other_params.append(param)
            
            if self.accelerator.is_local_main_process:
                print(f"Cross-attention parameters: {len(self.cross_attn_params)}")
                print(f"Other parameters: {len(self.other_params)}")
    
    def _setup_data_loaders(self):
        """Setup data loaders"""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                collate_fn=self.collator,
                num_workers=self.args.num_workers,
                pin_memory=True,
            )
    
    def _setup_optimization(self):
        """Setup optimizer and learning rate scheduler with extended optimizer support"""
        model = self.accelerator.unwrap_model(self.model)
        
        # Check optimizer availability
        if self.args.optimizer == 'adam8bit' and not BNB_AVAILABLE:
            raise ValueError("8-bit Adam requested but bitsandbytes not installed. Install with: pip install bitsandbytes")
        
        if self.args.optimizer == 'muon' and not MUON_AVAILABLE:
            raise ValueError("MUON optimizer requested but not available. Ensure you have PyTorch >= 2.5.0")
        
        # Setup optimizer parameters based on type
        if self.args.optimizer == 'muon':
            # MUON uses different learning rates for different parameter groups
            optimizer_params = [
                {
                    'params': self.embedding_params,
                    'lr': self.args.learning_rate * 0.1,  # Lower LR for embeddings
                    'momentum': 0.95,
                },
                {
                    'params': self.layer_params,
                    'lr': self.args.learning_rate,  # Full LR for hidden layers
                    'momentum': self.args.muon_momentum,
                },
                {
                    'params': self.head_params,
                    'lr': self.args.learning_rate * 0.1,  # Lower LR for output head
                    'momentum': 0.95,
                }
            ]
        elif self.warmup_phase:
            # During warmup, freeze non-cross-attention parameters
            for param in self.other_params:
                param.requires_grad = False
            
            optimizer_params = [
                {
                    'params': self.cross_attn_params,
                    'lr': self.args.cross_attention_lr,
                    'weight_decay': self.args.weight_decay,
                }
            ]
        else:
            # Standard parameter grouping
            optimizer_params = [
                {
                    'params': model.parameters(),
                    'lr': self.args.learning_rate,
                    'weight_decay': self.args.weight_decay,
                }
            ]
        
        # Create optimizer based on type
        if self.args.optimizer == 'adamw':
            from torch.optim import AdamW
            self.optimizer = AdamW(
                optimizer_params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        
        elif self.args.optimizer == 'adam8bit':
            # 8-bit Adam from bitsandbytes
            self.optimizer = bnb.optim.Adam8bit(
                optimizer_params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                min_8bit_size=self.args.min_8bit_size,
                percentile_clipping=self.args.percentile_clipping,
                block_wise=self.args.block_wise,
            )
            if self.accelerator.is_local_main_process:
                print(f"Using 8-bit Adam with min_8bit_size={self.args.min_8bit_size}")
        
        elif self.args.optimizer == 'adamw8bit':
            # 8-bit AdamW from bitsandbytes
            self.optimizer = bnb.optim.AdamW8bit(
                optimizer_params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                min_8bit_size=self.args.min_8bit_size,
                percentile_clipping=self.args.percentile_clipping,
                block_wise=self.args.block_wise,
            )
        
        elif self.args.optimizer == 'lion8bit':
            # 8-bit Lion optimizer (if you want to try it)
            self.optimizer = bnb.optim.Lion8bit(
                optimizer_params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.weight_decay,
                min_8bit_size=self.args.min_8bit_size,
            )
        
        elif self.args.optimizer == 'muon':
            # MUON optimizer for hidden layers
            self.optimizer = Muon(
                optimizer_params,
                lr=self.args.learning_rate,
                momentum=self.args.muon_momentum,
                nesterov=self.args.muon_nesterov,
                ns_steps=self.args.muon_ns_steps,
                update_type=self.args.muon_update_type,
            )
            if self.accelerator.is_local_main_process:
                print(f"Using MUON optimizer with momentum={self.args.muon_momentum}, ns_steps={self.args.muon_ns_steps}")
        
        elif self.args.optimizer == 'adafactor':
            from transformers import Adafactor
            self.optimizer = Adafactor(
                optimizer_params,
                lr=self.args.learning_rate,
                scale_parameter=True,
                relative_step=False,
                warmup_init=False,
            )
        
        elif self.args.optimizer == 'sgd':
            # Standard SGD for comparison
            from torch.optim import SGD
            self.optimizer = SGD(
                optimizer_params,
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=self.args.weight_decay,
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
            # Constant LR (useful for MUON)
            from transformers import get_constant_schedule
            self.scheduler = get_constant_schedule(self.optimizer)
        elif self.args.scheduler == 'constant_with_warmup':
            from transformers import get_constant_schedule_with_warmup
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
    
    def _setup_logging(self):
        """Setup logging with TrackIO and optional wandb"""
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)
        
        # Log optimizer info
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
        """Log metrics to TrackIO and/or wandb"""
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
    
    def _end_warmup_phase(self):
        """End cross-attention warmup and unfreeze all parameters"""
        # ALL processes must execute this
        if self.accelerator.is_main_process:
            self.logger.info("Ending cross-attention warmup phase")
        
        # Unfreeze all parameters on ALL processes
        model = self.accelerator.unwrap_model(self.model)
        for param in model.parameters():
            param.requires_grad = True
        
        # ALL processes recreate optimizer with all parameters
        self._setup_optimization()
        
        # ALL processes re-prepare (critical for DDP sync)
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )
        self.warmup_phase = False
        
        # Synchronize all processes
        self.accelerator.wait_for_everyone()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        with self.accelerator.autocast():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
        
        # Scale loss for gradient accumulation
        loss = loss / self.args.gradient_accumulation_steps
        
        # Backward pass
        self.accelerator.backward(loss)
        
        # Gradient accumulation
        if (self.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.args.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        # Gather loss for logging
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
            path = os.path.join(
                self.args.output_dir,
                f"checkpoint-{self.global_step}"
            )
        
        os.makedirs(path, exist_ok=True)
        
        # Unwrap and save model
        model_to_save = self.accelerator.unwrap_model(self.model)
        model_to_save.save_pretrained(path)
        
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
                # Check if we should end warmup
                if self.warmup_phase and self.global_step >= self.args.cross_attention_warmup_steps:
                    self._end_warmup_phase()
                
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
                #if self.val_loader and self.global_step % self.args.eval_steps == 0:

                if False:
                    val_loss = self.validate()
                    if self.accelerator.is_main_process:
                        self.logger.info(f"Validation loss: {val_loss:.4f}")
                        
                        self._log_metrics({'loss': val_loss}, prefix="val")
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(
                                os.path.join(self.args.output_dir, 'best_model')
                            )
                
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
    parser = argparse.ArgumentParser(description="Train ModernT5 with extended optimizer support")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="./modernt5_from_mmBERT-base_e21_d8")
    
    # Data arguments
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, default=None)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=512)
    
    # UL2 arguments
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
    parser.add_argument("--cross_attention_lr", type=float, default=1e-4)
    parser.add_argument("--cross_attention_warmup_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Optimizer arguments
    parser.add_argument("--optimizer", type=str, 
                       choices=["adamw", "adam8bit", "adamw8bit", "lion8bit", "muon", "adafactor", "sgd"], 
                       default="adamw",
                       help="Optimizer to use")
    parser.add_argument("--scheduler", type=str, 
                       choices=["linear", "cosine", "constant", "constant_with_warmup"], 
                       default="linear")
    
    # Adam-based optimizer arguments
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    
    # 8-bit optimizer arguments (bitsandbytes)
    parser.add_argument("--min_8bit_size", type=int, default=4096,
                       help="Minimum parameter size for 8-bit optimization")
    parser.add_argument("--percentile_clipping", type=int, default=100,
                       help="Percentile for gradient clipping in 8-bit optimizers")
    parser.add_argument("--block_wise", action="store_true",
                       help="Use block-wise quantization for 8-bit optimizers")
    
    # MUON optimizer arguments
    parser.add_argument("--muon_momentum", type=float, default=0.95,
                       help="Momentum for MUON optimizer")
    parser.add_argument("--muon_nesterov", action="store_true",
                       help="Use Nesterov momentum in MUON")
    parser.add_argument("--muon_ns_steps", type=int, default=5,
                       help="Number of Newton steps for MUON")
    parser.add_argument("--muon_update_type", type=str, default="newton",
                       choices=["newton", "fisher"],
                       help="Update type for MUON optimizer")
    
    # Efficiency arguments
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # Logging arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_trackio", action="store_true", default=True)
    parser.add_argument("--trackio_project", type=str, default="modernt5-training")
    parser.add_argument("--trackio_run_name", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="modernt5-training")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)
    
    # Parse args
    args = parser.parse_args()
    
    # Initialize Accelerator with correct settings
    accelerator = Accelerator(
        mixed_precision='bf16' if args.bf16 else 'no',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check optimizer availability
    if args.optimizer in ['adam8bit', 'adamw8bit', 'lion8bit'] and not BNB_AVAILABLE:
        if accelerator.is_main_process:
            print(f"WARNING: {args.optimizer} requested but bitsandbytes not installed.")
            print("Install with: pip install bitsandbytes")
            print("Falling back to adamw")
        args.optimizer = 'adamw'
    
    if args.optimizer == 'muon' and not MUON_AVAILABLE:
        if accelerator.is_main_process:
            print("WARNING: MUON optimizer requested but not available in your PyTorch version.")
            print("MUON requires PyTorch >= 2.5.0")
            print("Falling back to adamw")
        args.optimizer = 'adamw'
    
    # Load model
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    if accelerator.is_local_main_process:
        print(f"Loading model from {args.model_name}")
        print(f"Using dtype: {torch_dtype}")
        print(f"Using optimizer: {args.optimizer}")
    
    model = ModernT5ForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=torch_dtype,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create collator
    collator = ImprovedUL2Collator(
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        ul2_denoiser_probs=args.ul2_denoiser_probs,
        r_denoiser_suffix_ratio=args.ul2_r_ratio,
        s_denoiser_corrupt_prob=args.ul2_s_corrupt,
        x_denoiser_corrupt_prob=args.ul2_x_corrupt,
        mean_span_length=args.ul2_mean_span,
    )
    
    # Resize model embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    # Load datasets
    if accelerator.is_local_main_process:
        print("Loading datasets...")
    train_dataset = load_from_disk(args.train_dataset)
    val_dataset = None
    
    if args.val_dataset:
        val_dataset = load_from_disk(args.val_dataset)
    else:
        # Split dataset
        dataset = train_dataset.train_test_split(test_size=0.05, seed=args.seed)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    
    # Create trainer
    trainer = AdvancedSeq2SeqTrainer(
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