#!/usr/bin/env python3
"""
Quick start example for VibeVoice fine-tuning.

This script demonstrates how to quickly set up and run a speaker adaptation
fine-tuning session with minimal configuration.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.config import TrainingConfig
from training.data_utils import DataPreprocessor, prepare_speaker_adaptation_data


def create_sample_voice_data():
    """Create sample voice data for demonstration."""
    print("📁 Creating sample voice data...")
    
    # Create directories
    os.makedirs("./sample_data/voices", exist_ok=True)
    os.makedirs("./sample_data/scripts", exist_ok=True)
    
    # Create sample conversation script
    sample_script = """Speaker 1: Welcome to our AI podcast! I'm excited to discuss the future of technology with you.
Speaker 2: Thank you for having me! I've been working in AI research for over a decade, and I'm passionate about its potential.
Speaker 1: Let's start with a big question - where do you see AI heading in the next five years?
Speaker 2: I believe we'll see incredible advances in multimodal AI systems that can understand and generate text, images, and audio seamlessly.
Speaker 1: That sounds fascinating! Can you give us a specific example of how this might impact everyday life?
Speaker 2: Imagine having a conversation with your computer where you can speak naturally, show it images, and it responds with rich, contextual understanding."""
    
    script_path = "./sample_data/scripts/ai_conversation.txt"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(sample_script)
    
    print(f"✅ Created sample script: {script_path}")
    
    # Note about voice samples
    print("\n🎤 Voice Sample Instructions:")
    print("To complete this example, you need to:")
    print("1. Record or obtain voice samples for Speaker 1 and Speaker 2")
    print("2. Save them as:")
    print("   - ./sample_data/voices/speaker_1.wav")
    print("   - ./sample_data/voices/speaker_2.wav")
    print("3. Each sample should be:")
    print("   - Clean, high-quality audio")
    print("   - 10-30 seconds long")
    print("   - 24kHz sample rate (or will be converted)")
    print("   - Mono channel")
    
    return script_path


def setup_speaker_adaptation():
    """Set up a speaker adaptation training session."""
    print("\n🎯 Setting up Speaker Adaptation Training...")
    
    # Check if voice samples exist
    voice_dir = "./sample_data/voices"
    if not os.path.exists(voice_dir) or not any(f.endswith('.wav') for f in os.listdir(voice_dir)):
        print("❌ Voice samples not found. Please add voice samples to ./sample_data/voices/")
        print("Example files needed:")
        print("  - speaker_1.wav")
        print("  - speaker_2.wav")
        return None
    
    # Create speaker adaptation configuration
    config = TrainingConfig(
        # Fine-tuning strategy
        finetune_strategy="speaker_adaptation",
        
        # Model settings
        model_name_or_path="microsoft/VibeVoice-1.5B",
        model_output_dir="./models/vibevoice-speaker-adapted",
        
        # Data paths
        train_data_path="./training_data/train.json",
        voice_samples_dir="./training_data/voices",
        
        # Training settings (quick training for demo)
        num_train_epochs=3,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        
        # Evaluation
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        
        # Loss weighting for speaker adaptation
        text_loss_weight=1.0,
        diffusion_loss_weight=0.5,
    )
    
    # Save configuration
    config_path = "./configs/speaker_adaptation_example.json"
    os.makedirs("./configs", exist_ok=True)
    config.to_json(config_path)
    
    print(f"✅ Created training config: {config_path}")
    
    # Prepare training data
    try:
        voice_samples, training_data = prepare_speaker_adaptation_data(
            voice_dir="./sample_data/voices",
            script_file="./sample_data/scripts/ai_conversation.txt",
            output_dir="./training_data",
            target_speakers=["1", "2"]
        )
        
        print(f"✅ Prepared training data with {len(voice_samples)} voice samples")
        
        return config_path
        
    except Exception as e:
        print(f"❌ Error preparing training data: {e}")
        return None


def run_training_example(config_path):
    """Run a quick training example."""
    print(f"\n🚀 Starting Training Example...")
    print(f"Config: {config_path}")
    
    # Import training modules
    try:
        from modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
        from processor.vibevoice_processor import VibeVoiceProcessor
        from training.config import TrainingConfig
        from training.dataset import VibeVoiceDataset
        from training.trainer import create_trainer
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the podcast directory")
        return False
    
    try:
        # Load configuration
        config = TrainingConfig.from_json(config_path)
        
        # Load model and processor
        print("📥 Loading model and processor...")
        processor = VibeVoiceProcessor.from_pretrained(config.model_name_or_path)
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        
        # Create dataset
        print("📊 Creating training dataset...")
        train_dataset = VibeVoiceDataset(
            data_path=config.train_data_path,
            processor=processor,
            voice_samples_dir=config.voice_samples_dir,
            max_length=2048,
            cache_audio=True,
        )
        
        print(f"Dataset size: {len(train_dataset)} samples")
        
        # Create trainer
        print("🏋️ Creating trainer...")
        trainer = create_trainer(
            model=model,
            processor=processor,
            training_config=config,
            train_dataset=train_dataset,
            eval_dataset=None,
        )
        
        # Start training
        print("🎯 Starting training...")
        print("This is a quick demo with 3 epochs - real training may take longer")
        
        trainer.train()
        
        # Save model
        print("💾 Saving trained model...")
        trainer.save_model()
        
        print(f"✅ Training completed! Model saved to: {config.model_output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_finetuned_model():
    """Test the fine-tuned model."""
    print("\n🧪 Testing Fine-tuned Model...")
    
    model_path = "./models/vibevoice-speaker-adapted"
    
    if not os.path.exists(model_path):
        print(f"❌ Fine-tuned model not found at: {model_path}")
        return False
    
    try:
        from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from processor.vibevoice_processor import VibeVoiceProcessor
        import torch
        import soundfile as sf
        
        # Load fine-tuned model
        print("📥 Loading fine-tuned model...")
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(model_path)
        processor = VibeVoiceProcessor.from_pretrained(model_path)
        
        # Test script
        test_script = "Speaker 1: Hello, this is a test of our fine-tuned model.\nSpeaker 2: It sounds great! The voice quality is impressive."
        
        # Load voice samples
        voice_samples = []
        voices_dir = "./training_data/voices"
        for voice_file in os.listdir(voices_dir):
            if voice_file.endswith('.wav'):
                voice_path = os.path.join(voices_dir, voice_file)
                audio, sr = sf.read(voice_path)
                voice_samples.append(audio)
        
        if not voice_samples:
            print("❌ No voice samples found for testing")
            return False
        
        # Generate test audio
        print("🎵 Generating test audio...")
        inputs = processor(
            text=test_script,
            voice_samples=[voice_samples],
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                cfg_scale=1.3,
                tokenizer=processor.tokenizer,
            )
        
        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0].cpu().float().numpy()
            
            # Save test audio
            output_path = "./test_output.wav"
            sf.write(output_path, audio, 24000)
            
            print(f"✅ Test audio generated: {output_path}")
            print(f"Audio duration: {len(audio) / 24000:.2f} seconds")
            return True
        else:
            print("❌ No audio generated")
            return False
            
    except Exception as e:
        print(f"❌ Testing error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the complete example."""
    print("🎙️ VibeVoice Fine-tuning Quick Start Example")
    print("=" * 50)
    
    # Step 1: Create sample data
    script_path = create_sample_voice_data()
    
    # Step 2: Check if we can proceed
    voice_dir = "./sample_data/voices"
    if not os.path.exists(voice_dir) or not any(f.endswith('.wav') for f in os.listdir(voice_dir)):
        print("\n⏸️  Setup Complete - Manual Step Required")
        print("Please add voice samples to ./sample_data/voices/ then run this script again")
        return
    
    # Step 3: Setup training
    config_path = setup_speaker_adaptation()
    if not config_path:
        print("❌ Failed to setup training")
        return
    
    # Step 4: Run training
    print("\n" + "=" * 50)
    print("Ready to start training!")
    print("This will download the model and run fine-tuning.")
    print("Estimated time: 10-30 minutes (depending on hardware)")
    
    proceed = input("\nProceed with training? (y/n): ").lower().strip()
    if proceed != 'y':
        print("Training cancelled. You can run training later with:")
        print(f"python train_vibevoice.py --config {config_path}")
        return
    
    success = run_training_example(config_path)
    if not success:
        print("❌ Training failed")
        return
    
    # Step 5: Test the model
    print("\n" + "=" * 50)
    test_success = test_finetuned_model()
    
    if test_success:
        print("\n🎉 Quick Start Example Completed Successfully!")
        print("\nWhat's Next?")
        print("1. Listen to the generated test_output.wav")
        print("2. Try generating more audio with your fine-tuned model")
        print("3. Experiment with different scripts and voice samples")
        print("4. Check out the full documentation in README_FINETUNING.md")
    else:
        print("\n⚠️  Training completed but testing failed")
        print("The model was saved but there may be issues with inference")


if __name__ == "__main__":
    main()
