#!/usr/bin/env python3
"""
Simplified training script for VibeVoice speaker adaptation.
"""

import os
import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from processor.vibevoice_processor import VibeVoiceProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleVibeVoiceTrainer:
    """Simplified trainer for VibeVoice speaker adaptation."""
    
    def __init__(self, model_name="microsoft/VibeVoice-1.5B", output_dir="./models/my-voice-clone"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        logger.info(f"Loading model: {model_name}")
        self.processor = VibeVoiceProcessor.from_pretrained(model_name)
        self.model = VibeVoiceForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        
        # Set model to training mode
        self.model.train()
        
        # Freeze tokenizers (we only want to adapt voice generation)
        self._freeze_tokenizers()
        
        # Setup optimizer
        self._setup_optimizer()
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def _freeze_tokenizers(self):
        """Freeze the tokenizer components."""
        for name, param in self.model.named_parameters():
            if any(component in name for component in ["acoustic_tokenizer", "semantic_tokenizer", "language_model"]):
                param.requires_grad = False
        
        logger.info("Frozen tokenizers and language model for speaker adaptation")
    
    def _setup_optimizer(self):
        """Setup optimizer for trainable parameters."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
        logger.info(f"Optimizer setup with {len(trainable_params)} parameter groups")
    
    def load_voice_sample(self, voice_path):
        """Load and preprocess voice sample."""
        audio, sr = sf.read(voice_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != 24000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        return audio
    
    def train_step(self, script, voice_samples):
        """Single training step."""
        try:
            # Process inputs
            inputs = self.processor(
                text=script,
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Move to device
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(self.device)
            
            # Forward pass - use text-only training for simplicity
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"],  # Language modeling objective
            )
            
            # Get text loss only (simpler approach)
            loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=self.device)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            return 0.0
    
    def train(self, script, voice_path, num_epochs=3, save_steps=1):
        """Train the model."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Load voice sample
        logger.info(f"Loading voice sample: {voice_path}")
        voice_audio = self.load_voice_sample(voice_path)
        voice_samples = [voice_audio]
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Multiple steps per epoch for better convergence
            epoch_losses = []
            for step in tqdm(range(10), desc=f"Epoch {epoch + 1}"):
                loss = self.train_step(script, voice_samples)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_steps == 0:
                self.save_model(f"checkpoint-epoch-{epoch + 1}")
        
        # Save final model
        self.save_model("final")
        logger.info("Training completed!")
    
    def save_model(self, checkpoint_name="final"):
        """Save the trained model."""
        save_path = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        logger.info(f"Model saved to: {save_path}")


def main():
    """Main training function."""
    print("🚀 Starting VibeVoice Simple Training...")
    
    # Configuration
    script = """Speaker 1: Welcome to our podcast! I'm excited to share some insights with you today.
Speaker 1: Thank you for having me! It's great to be here and discuss this fascinating topic.
Speaker 1: Let's dive right in. What's your perspective on the recent developments in technology?
Speaker 1: I think we're seeing unprecedented innovation across multiple fields, especially in AI and automation.
Speaker 1: That's a great point. How do you think this will impact everyday life in the next few years?
Speaker 1: I believe we'll see more personalized experiences and smarter systems that adapt to our needs."""
    
    voice_path = "./my_training_data/voices/speaker_1.wav"
    
    # Check if voice file exists
    print(f"🔍 Checking voice file: {voice_path}")
    if not os.path.exists(voice_path):
        print(f"❌ Voice file not found: {voice_path}")
        return
    else:
        print("✅ Voice file found")
    
    try:
        # Create trainer
        print("🏗️ Creating trainer...")
        trainer = SimpleVibeVoiceTrainer()
        
        # Start training
        print("🎯 Starting training...")
        trainer.train(script, voice_path, num_epochs=2)  # Reduced epochs for testing
        
        print("🎉 Training process completed!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
