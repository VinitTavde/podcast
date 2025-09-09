"""
Custom trainer for VibeVoice fine-tuning.
"""

import os
import json
import logging
from typing import Dict, Optional, Union, Any, List
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Trainer, 
    TrainingArguments,
    EvalPrediction,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from transformers.trainer_utils import speed_metrics
from transformers.utils import logging as transformers_logging

from modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from processor.vibevoice_processor import VibeVoiceProcessor
from training.config import TrainingConfig
from training.dataset import VibeVoiceDataset, CollateFunction

logger = logging.getLogger(__name__)


class VibeVoiceTrainer(Trainer):
    """
    Custom trainer for VibeVoice fine-tuning with specialized loss computation.
    """
    
    def __init__(
        self,
        model: VibeVoiceForConditionalGeneration,
        training_config: TrainingConfig,
        train_dataset: VibeVoiceDataset,
        eval_dataset: Optional[VibeVoiceDataset] = None,
        processor: Optional[VibeVoiceProcessor] = None,
        **kwargs
    ):
        self.training_config = training_config
        self.processor = processor
        
        # Create training arguments
        training_args = self._create_training_arguments()
        
        # Create data collator
        data_collator = CollateFunction(processor) if processor else None
        
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            **kwargs
        )
        
        # Setup parameter groups with different learning rates
        self._setup_parameter_groups()
        
        # Setup logging
        self._setup_logging()
    
    def _create_training_arguments(self) -> TrainingArguments:
        """Create TrainingArguments from config."""
        return TrainingArguments(
            output_dir=self.training_config.model_output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_steps=self.training_config.warmup_steps,
            max_grad_norm=self.training_config.max_grad_norm,
            
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            logging_steps=self.training_config.logging_steps,
            save_total_limit=self.training_config.save_total_limit,
            
            eval_strategy="steps" if self.training_config.eval_steps > 0 else "no",
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            fp16=self.training_config.fp16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            ddp_find_unused_parameters=self.training_config.ddp_find_unused_parameters,
            
            report_to=["tensorboard"],
            run_name=f"vibevoice-{self.training_config.finetune_strategy}",
            
            # Additional settings
            remove_unused_columns=False,  # Keep all columns for custom processing
            prediction_loss_only=self.training_config.prediction_loss_only,
            
            # Save and load settings
            save_safetensors=True,
            save_on_each_node=False,
        )
    
    def _setup_parameter_groups(self):
        """Setup parameter groups with different learning rates."""
        # Freeze parameters based on strategy
        frozen_params = self.training_config.get_frozen_parameters()
        
        for name, param in self.model.named_parameters():
            should_freeze = any(frozen_pattern in name for frozen_pattern in frozen_params)
            if should_freeze:
                param.requires_grad = False
                logger.info(f"Frozen parameter: {name}")
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
        logger.info(f"Trainable ratio: {trainable_params / total_params:.2%}")
    
    def _setup_logging(self):
        """Setup training logging."""
        # Create logs directory
        os.makedirs(os.path.join(self.args.output_dir, "logs"), exist_ok=True)
        
        # Save training config
        config_path = os.path.join(self.args.output_dir, "training_config.json")
        self.training_config.to_json(config_path)
        
        logger.info(f"Training configuration saved to: {config_path}")
        logger.info(f"Fine-tuning strategy: {self.training_config.finetune_strategy}")
        logger.info(f"Model output directory: {self.args.output_dir}")
    
    def create_optimizer(self):
        """Create optimizer with parameter-specific learning rates."""
        if self.optimizer is None:
            # Get parameter groups
            param_groups_config = self.training_config.get_parameter_groups()
            
            # Create parameter groups
            param_groups = []
            assigned_params = set()
            
            for group_config in param_groups_config:
                pattern = group_config["params_pattern"]
                group_params = []
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad and pattern in name and id(param) not in assigned_params:
                        group_params.append(param)
                        assigned_params.add(id(param))
                
                if group_params:
                    param_groups.append({
                        "params": group_params,
                        "lr": group_config["lr"],
                        "weight_decay": group_config.get("weight_decay", self.args.weight_decay),
                        "name": pattern
                    })
                    logger.info(f"Parameter group '{pattern}': {len(group_params)} parameters, lr={group_config['lr']}")
            
            # Add remaining parameters to default group
            remaining_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and id(param) not in assigned_params:
                    remaining_params.append(param)
            
            if remaining_params:
                param_groups.append({
                    "params": remaining_params,
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                    "name": "default"
                })
                logger.info(f"Parameter group 'default': {len(remaining_params)} parameters, lr={self.args.learning_rate}")
            
            # Create optimizer
            self.optimizer = AdamW(
                param_groups,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.args.weight_decay,
            )
        
        return self.optimizer
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute custom loss combining text and speech generation losses.
        """
        # Forward pass
        outputs = model(**inputs)
        
        # Extract losses
        text_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=model.device)
        diffusion_loss = outputs.diffusion_loss if outputs.diffusion_loss is not None else torch.tensor(0.0, device=model.device)
        
        # Combine losses with weights
        total_loss = (
            self.training_config.text_loss_weight * text_loss +
            self.training_config.diffusion_loss_weight * diffusion_loss
        )
        
        # Log individual losses
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "train/text_loss": text_loss.item(),
                "train/diffusion_loss": diffusion_loss.item(),
                "train/total_loss": total_loss.item(),
                "train/text_loss_weight": self.training_config.text_loss_weight,
                "train/diffusion_loss_weight": self.training_config.diffusion_loss_weight,
            })
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """Custom evaluation loop with speech-specific metrics."""
        
        # Set model to eval mode
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()
        
        batch_size = dataloader.batch_size
        num_samples = self.num_examples(dataloader)
        
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_samples}")
        logger.info(f"  Batch size = {batch_size}")
        
        losses = []
        text_losses = []
        diffusion_losses = []
        
        for step, inputs in enumerate(dataloader):
            # Move inputs to device
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                # Forward pass
                outputs = model(**inputs)
                
                # Extract losses
                text_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0)
                diffusion_loss = outputs.diffusion_loss if outputs.diffusion_loss is not None else torch.tensor(0.0)
                
                # Compute total loss
                total_loss = (
                    self.training_config.text_loss_weight * text_loss +
                    self.training_config.diffusion_loss_weight * diffusion_loss
                )
                
                losses.append(total_loss.cpu())
                text_losses.append(text_loss.cpu())
                diffusion_losses.append(diffusion_loss.cpu())
        
        # Compute metrics
        avg_loss = torch.stack(losses).mean().item()
        avg_text_loss = torch.stack(text_losses).mean().item()
        avg_diffusion_loss = torch.stack(diffusion_losses).mean().item()
        
        # Compute perplexity from text loss
        perplexity = math.exp(avg_text_loss) if avg_text_loss < 100 else float('inf')
        
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_text_loss": avg_text_loss,
            f"{metric_key_prefix}_diffusion_loss": avg_diffusion_loss,
            f"{metric_key_prefix}_perplexity": perplexity,
        }
        
        # Add speed metrics
        metrics.update(speed_metrics(metric_key_prefix, start_time=None, num_samples=num_samples))
        
        return EvalPrediction(predictions=None, label_ids=None, metrics=metrics)
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """Save checkpoint with additional metadata."""
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        
        # Call parent save method
        output = super()._save_checkpoint(model, trial, metrics)
        
        # Save additional training metadata
        checkpoint_path = os.path.join(self.args.output_dir, checkpoint_folder)
        
        # Save training progress
        progress_info = {
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "training_config": self.training_config.__dict__,
            "metrics": metrics if metrics else {},
            "model_config": self.model.config.to_dict() if hasattr(self.model.config, 'to_dict') else {},
        }
        
        progress_path = os.path.join(checkpoint_path, "training_progress.json")
        with open(progress_path, 'w') as f:
            json.dump(progress_info, f, indent=2)
        
        logger.info(f"Saved training progress to: {progress_path}")
        
        return output
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model with processor configuration."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        
        # Save model
        super().save_model(output_dir, _internal_call)
        
        # Save processor if available
        if self.processor:
            self.processor.save_pretrained(output_dir)
            logger.info(f"Saved processor to: {output_dir}")
        
        # Save final training configuration
        config_path = os.path.join(output_dir, "training_config.json")
        self.training_config.to_json(config_path)
        
        logger.info(f"Model saved to: {output_dir}")


def create_trainer(
    model: VibeVoiceForConditionalGeneration,
    processor: VibeVoiceProcessor,
    training_config: TrainingConfig,
    train_dataset: VibeVoiceDataset,
    eval_dataset: Optional[VibeVoiceDataset] = None,
) -> VibeVoiceTrainer:
    """
    Create a configured VibeVoice trainer.
    
    Args:
        model: The VibeVoice model to train
        processor: The VibeVoice processor
        training_config: Training configuration
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
    
    Returns:
        Configured trainer instance
    """
    
    # Enable gradient checkpointing if requested
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Create trainer
    trainer = VibeVoiceTrainer(
        model=model,
        training_config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
    )
    
    return trainer


if __name__ == "__main__":
    # Example usage
    from training.config import TrainingConfig
    
    config = TrainingConfig()
    print("Created trainer with config:")
    print(f"  Strategy: {config.finetune_strategy}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Output dir: {config.model_output_dir}")
