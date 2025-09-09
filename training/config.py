"""
Training configuration for VibeVoice fine-tuning.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import os
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for VibeVoice fine-tuning."""
    
    # Model settings
    model_name_or_path: str = "microsoft/VibeVoice-1.5B"
    model_output_dir: str = "./vibevoice-finetuned"
    
    # Training data
    train_data_path: str = "./my_training_data/train.json"
    val_data_path: Optional[str] = "./my_training_data/val.json"
    voice_samples_dir: str = "./voices"
    
    # Training hyperparameters
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Component-specific learning rates
    language_model_lr: float = 1e-5
    diffusion_head_lr: float = 1e-4
    connector_lr: float = 1e-4
    
    # Loss weighting
    text_loss_weight: float = 1.0
    diffusion_loss_weight: float = 0.1
    
    # Fine-tuning strategy
    finetune_strategy: str = "speaker_adaptation"  # "speaker_adaptation", "domain_adaptation", "full_finetune"
    freeze_tokenizers: bool = True
    freeze_language_model: bool = False
    
    # Training settings
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # System settings
    dataloader_num_workers: int = 4
    fp16: bool = True
    gradient_checkpointing: bool = True
    ddp_find_unused_parameters: bool = False
    
    # Evaluation
    eval_accumulation_steps: Optional[int] = None
    prediction_loss_only: bool = False
    
    # Audio processing
    target_sample_rate: int = 24000
    max_audio_length: float = 30.0  # seconds
    min_audio_length: float = 1.0   # seconds
    
    # Generation settings for evaluation
    eval_generation_max_length: int = 2048
    eval_generation_cfg_scale: float = 1.3
    eval_generation_steps: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.finetune_strategy not in ["speaker_adaptation", "domain_adaptation", "full_finetune"]:
            raise ValueError(f"Invalid finetune_strategy: {self.finetune_strategy}")
        
        # Adjust settings based on strategy
        if self.finetune_strategy == "speaker_adaptation":
            self.freeze_language_model = True
            self.diffusion_loss_weight = 0.5
        elif self.finetune_strategy == "domain_adaptation":
            self.freeze_tokenizers = True
            self.text_loss_weight = 1.0
            self.diffusion_loss_weight = 0.1
        elif self.finetune_strategy == "full_finetune":
            self.freeze_tokenizers = True
            self.freeze_language_model = False
    
    @classmethod
    def from_json(cls, json_path: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    def get_frozen_parameters(self) -> List[str]:
        """Get list of parameter patterns to freeze."""
        frozen_params = []
        
        if self.freeze_tokenizers:
            frozen_params.extend([
                "model.acoustic_tokenizer",
                "model.semantic_tokenizer"
            ])
        
        if self.freeze_language_model:
            frozen_params.append("model.language_model")
        
        return frozen_params
    
    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups with different learning rates."""
        param_groups = []
        
        if not self.freeze_language_model:
            param_groups.append({
                "params_pattern": "model.language_model",
                "lr": self.language_model_lr,
                "weight_decay": self.weight_decay
            })
        
        param_groups.extend([
            {
                "params_pattern": "model.prediction_head",
                "lr": self.diffusion_head_lr,
                "weight_decay": self.weight_decay
            },
            {
                "params_pattern": "model.acoustic_connector",
                "lr": self.connector_lr,
                "weight_decay": self.weight_decay
            },
            {
                "params_pattern": "model.semantic_connector", 
                "lr": self.connector_lr,
                "weight_decay": self.weight_decay
            },
            {
                "params_pattern": "lm_head",
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay
            }
        ])
        
        return param_groups


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Dataset settings
    max_conversations_per_file: int = 1000
    shuffle_data: bool = True
    val_split_ratio: float = 0.1
    
    # Text processing
    max_sequence_length: int = 2048
    min_sequence_length: int = 10
    
    # Voice processing
    max_voices_per_speaker: int = 5
    voice_sample_duration: float = 10.0  # seconds
    
    # Augmentation (future feature)
    enable_audio_augmentation: bool = False
    enable_text_augmentation: bool = False
    
    def validate_data_paths(self, config: TrainingConfig):
        """Validate that required data paths exist."""
        if not os.path.exists(config.train_data_path):
            raise FileNotFoundError(f"Training data not found: {config.train_data_path}")
        
        if config.val_data_path and not os.path.exists(config.val_data_path):
            raise FileNotFoundError(f"Validation data not found: {config.val_data_path}")
        
        if not os.path.exists(config.voice_samples_dir):
            raise FileNotFoundError(f"Voice samples directory not found: {config.voice_samples_dir}")


def create_default_configs():
    """Create default configuration files for different fine-tuning strategies."""
    
    configs = {
        "speaker_adaptation": TrainingConfig(
            finetune_strategy="speaker_adaptation",
            num_train_epochs=5,
            learning_rate=5e-5,
            diffusion_loss_weight=0.5,
            model_output_dir="./models/vibevoice-speaker-adapted"
        ),
        
        "domain_adaptation": TrainingConfig(
            finetune_strategy="domain_adaptation", 
            num_train_epochs=10,
            learning_rate=1e-5,
            text_loss_weight=1.0,
            diffusion_loss_weight=0.1,
            model_output_dir="./models/vibevoice-domain-adapted"
        ),
        
        "full_finetune": TrainingConfig(
            finetune_strategy="full_finetune",
            num_train_epochs=15,
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            model_output_dir="./models/vibevoice-full-finetuned"
        )
    }
    
    # Create configs directory
    os.makedirs("./training_configs", exist_ok=True)
    
    for strategy, config in configs.items():
        config.to_json(f"./training_configs/{strategy}_config.json")
    
    print("Created default configuration files:")
    for strategy in configs.keys():
        print(f"  - ./training_configs/{strategy}_config.json")


if __name__ == "__main__":
    create_default_configs()
