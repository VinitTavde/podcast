#!/usr/bin/env python3
"""
Main training script for VibeVoice fine-tuning.

Usage:
    python train_vibevoice.py --config training_configs/speaker_adaptation_config.json
    python train_vibevoice.py --strategy speaker_adaptation --data ./my_data
    python train_vibevoice.py --help
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
from transformers import set_seed

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from processor.vibevoice_processor import VibeVoiceProcessor
from training.config import TrainingConfig, DataConfig
from training.dataset import VibeVoiceDataset
from training.trainer import create_trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup training environment."""
    # Set random seeds for reproducibility
    set_seed(42)
    
    # Setup torch settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Log environment info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {gpu_props.name} ({gpu_props.total_memory // 1024**3} GB)")


def load_model_and_processor(config: TrainingConfig):
    """Load model and processor."""
    logger.info(f"Loading model from: {config.model_name_or_path}")
    
    # Load processor first
    processor = VibeVoiceProcessor.from_pretrained(config.model_name_or_path)
    
    # Load model
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16 if config.fp16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, processor


def create_datasets(config: TrainingConfig, processor: VibeVoiceProcessor):
    """Create training and validation datasets."""
    data_config = DataConfig()
    data_config.validate_data_paths(config)
    
    logger.info(f"Creating training dataset from: {config.train_data_path}")
    train_dataset = VibeVoiceDataset(
        data_path=config.train_data_path,
        processor=processor,
        voice_samples_dir=config.voice_samples_dir,
        max_length=data_config.max_sequence_length,
        max_audio_length=config.max_audio_length,
        min_audio_length=config.min_audio_length,
        target_sample_rate=config.target_sample_rate,
        cache_audio=True,  # Cache for faster training
    )
    
    eval_dataset = None
    if config.val_data_path and os.path.exists(config.val_data_path):
        logger.info(f"Creating validation dataset from: {config.val_data_path}")
        eval_dataset = VibeVoiceDataset(
            data_path=config.val_data_path,
            processor=processor,
            voice_samples_dir=config.voice_samples_dir,
            max_length=data_config.max_sequence_length,
            max_audio_length=config.max_audio_length,
            min_audio_length=config.min_audio_length,
            target_sample_rate=config.target_sample_rate,
            cache_audio=True,
        )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(eval_dataset) if eval_dataset else 0}")
    
    return train_dataset, eval_dataset


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune VibeVoice model")
    
    # Configuration options
    parser.add_argument("--config", type=str, help="Path to training configuration JSON file")
    parser.add_argument("--strategy", type=str, choices=["speaker_adaptation", "domain_adaptation", "full_finetune"],
                       help="Fine-tuning strategy (creates default config)")
    
    # Quick setup options
    parser.add_argument("--model", type=str, default="microsoft/VibeVoice-1.5B", help="Model name or path")
    parser.add_argument("--data", type=str, help="Training data directory or file")
    parser.add_argument("--voices", type=str, help="Voice samples directory")
    parser.add_argument("--output", type=str, help="Output directory for fine-tuned model")
    
    # Training overrides
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size per device")
    
    # System options
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        config = TrainingConfig.from_json(args.config)
    elif args.strategy:
        logger.info(f"Using default configuration for strategy: {args.strategy}")
        config = TrainingConfig(finetune_strategy=args.strategy)
    else:
        logger.info("Using default configuration")
        config = TrainingConfig()
    
    # Apply command line overrides
    if args.model:
        config.model_name_or_path = args.model
    if args.data:
        if os.path.isdir(args.data):
            config.train_data_path = os.path.join(args.data, "train.json")
            config.val_data_path = os.path.join(args.data, "val.json")
        else:
            config.train_data_path = args.data
    if args.voices:
        config.voice_samples_dir = args.voices
    if args.output:
        config.model_output_dir = args.output
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    if args.no_cuda:
        config.fp16 = False
    
    # Log configuration
    logger.info("Training Configuration:")
    logger.info(f"  Strategy: {config.finetune_strategy}")
    logger.info(f"  Model: {config.model_name_or_path}")
    logger.info(f"  Training data: {config.train_data_path}")
    logger.info(f"  Voice samples: {config.voice_samples_dir}")
    logger.info(f"  Output directory: {config.model_output_dir}")
    logger.info(f"  Epochs: {config.num_train_epochs}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Batch size: {config.per_device_train_batch_size}")
    
    try:
        # Load model and processor
        model, processor = load_model_and_processor(config)
        
        # Create datasets
        train_dataset, eval_dataset = create_datasets(config, processor)
        
        # Create trainer
        trainer = create_trainer(
            model=model,
            processor=processor,
            training_config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming training from: {args.resume}")
            trainer.train(resume_from_checkpoint=args.resume)
        elif args.eval_only:
            logger.info("Running evaluation only")
            if eval_dataset:
                eval_results = trainer.evaluate()
                logger.info(f"Evaluation results: {eval_results}")
            else:
                logger.error("No evaluation dataset provided")
                return
        else:
            # Start training
            logger.info("Starting training...")
            trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        
        # Run final evaluation
        if eval_dataset and not args.eval_only:
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate()
            logger.info(f"Final evaluation results: {eval_results}")
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {config.model_output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_quick_setup():
    """Create a quick setup with sample data and configs."""
    logger.info("Creating quick setup for VibeVoice fine-tuning...")
    
    # Create directories
    os.makedirs("./training_data", exist_ok=True)
    os.makedirs("./training_configs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
    # Create sample data
    from training.dataset import create_sample_dataset
    create_sample_dataset()
    
    # Create default configs
    from training.config import create_default_configs
    create_default_configs()
    
    logger.info("Quick setup completed!")
    logger.info("You can now run:")
    logger.info("  python train_vibevoice.py --config training_configs/speaker_adaptation_config.json")


if __name__ == "__main__":
    # Check if this is a setup call
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        create_quick_setup()
    else:
        main()
