# VibeVoice Fine-tuning Guide 🎯

This guide explains how to fine-tune VibeVoice models for custom speakers, domains, or use cases.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create training directories and sample data
python train_vibevoice.py setup

# Install additional dependencies if needed
pip install librosa soundfile tqdm
```

### 2. Prepare Your Data
```bash
# For speaker adaptation
python training/data_utils.py speaker-adapt \
    --voices ./my_voice_samples \
    --script ./my_conversation.txt \
    --output ./my_training_data

# For domain adaptation
python training/data_utils.py conversations \
    --input ./text_conversations \
    --output ./training_data/conversations.json

python training/data_utils.py split \
    --input ./training_data/conversations.json \
    --output ./training_data
```

### 3. Start Fine-tuning
```bash
# Speaker adaptation (quick, 2-6 hours)
python train_vibevoice.py --strategy speaker_adaptation \
    --data ./my_training_data \
    --voices ./my_training_data/voices \
    --epochs 5

# Domain adaptation (6-24 hours)
python train_vibevoice.py --strategy domain_adaptation \
    --data ./training_data \
    --voices ./voices \
    --epochs 10

# Full fine-tuning (1-3 days)
python train_vibevoice.py --strategy full_finetune \
    --data ./training_data \
    --voices ./voices \
    --epochs 15
```

### 4. Evaluate Results
```bash
# Test your fine-tuned model
python evaluate_model.py \
    --model ./models/vibevoice-speaker-adapted \
    --data ./test_data.json \
    --voices ./voices \
    --output ./evaluation_results

# Compare with baseline
python evaluate_model.py \
    --model ./models/vibevoice-speaker-adapted \
    --baseline microsoft/VibeVoice-1.5B \
    --data ./test_data.json \
    --voices ./voices \
    --output ./comparison_results
```

## 📊 Fine-tuning Strategies

### 🎭 Speaker Adaptation
**Best for**: Adding new voices, voice cloning, speaker-specific improvements

**What it does**: 
- Keeps conversation abilities intact
- Adapts voice generation components
- Fast training (2-6 hours)

**Data needed**:
- 30 seconds - 5 minutes of clean audio per speaker
- Sample conversation scripts

**Configuration**:
```json
{
  "finetune_strategy": "speaker_adaptation",
  "freeze_language_model": true,
  "diffusion_loss_weight": 0.5,
  "num_train_epochs": 5
}
```

### 📚 Domain Adaptation  
**Best for**: Specific content types (technical, educational, storytelling)

**What it does**:
- Improves vocabulary and style for specific domains
- Maintains general conversation ability
- Medium training time (6-24 hours)

**Data needed**:
- 1000+ conversation examples in your domain
- Domain-specific vocabulary

**Configuration**:
```json
{
  "finetune_strategy": "domain_adaptation",
  "freeze_tokenizers": true,
  "text_loss_weight": 1.0,
  "num_train_epochs": 10
}
```

### 🔧 Full Fine-tuning
**Best for**: Complete adaptation to new requirements

**What it does**:
- Adapts all components except base tokenizers
- Maximum flexibility
- Long training time (1-3 days)

**Data needed**:
- Large dataset (5000+ examples)
- Diverse conversation types

**Configuration**:
```json
{
  "finetune_strategy": "full_finetune",
  "freeze_tokenizers": true,
  "freeze_language_model": false,
  "num_train_epochs": 15
}
```

## 📁 Data Formats

### Conversation Data (JSON)
```json
[
  {
    "id": "conv_001",
    "script": "Speaker 1: Welcome to our podcast!\nSpeaker 2: Thanks for having me!",
    "speakers": ["1", "2"]
  }
]
```

### Voice Samples Directory
```
voices/
├── alice.wav          # Speaker samples
├── bob.wav
├── speaker_1.wav
└── speaker_2.wav
```

### Text Files (Auto-parsed)
```
Speaker 1: Hello everyone, welcome to our show.
Speaker 2: Great to be here today.
Speaker 1: Let's dive into our topic.
```

## ⚙️ Advanced Configuration

### Custom Training Config
```python
from training.config import TrainingConfig

config = TrainingConfig(
    # Model settings
    model_name_or_path="microsoft/VibeVoice-1.5B",
    model_output_dir="./my_finetuned_model",
    
    # Training data
    train_data_path="./my_data/train.json",
    val_data_path="./my_data/val.json",
    voice_samples_dir="./my_voices",
    
    # Hyperparameters
    num_train_epochs=10,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    
    # Loss weighting
    text_loss_weight=1.0,
    diffusion_loss_weight=0.1,
    
    # Component-specific learning rates
    language_model_lr=1e-5,
    diffusion_head_lr=1e-4,
    connector_lr=1e-4,
)

# Save custom config
config.to_json("./my_training_config.json")
```

### Hardware Requirements

| Strategy | Min GPU | Recommended | RAM | Training Time |
|----------|---------|-------------|-----|---------------|
| Speaker Adaptation | RTX 3090 (24GB) | A100 40GB | 64GB | 2-6 hours |
| Domain Adaptation | RTX 4090 (24GB) | A100 80GB | 128GB | 6-24 hours |
| Full Fine-tuning | A100 40GB | H100 80GB | 256GB | 1-3 days |

### Memory Optimization
```python
# For limited GPU memory
config = TrainingConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Effective batch size = 16
    gradient_checkpointing=True,
    fp16=True,
)
```

## 🔍 Monitoring Training

### TensorBoard
```bash
# View training progress
tensorboard --logdir ./models/vibevoice-finetuned/runs
```

### Training Logs
```bash
# Monitor training progress
tail -f training.log
```

### Key Metrics to Watch
- **Total Loss**: Should decrease steadily
- **Text Loss**: Language modeling performance  
- **Diffusion Loss**: Speech generation quality
- **Learning Rate**: Should follow warmup schedule
- **GPU Memory**: Should be stable, not increasing

## 🎯 Evaluation Metrics

### Automatic Metrics
- **Success Rate**: Percentage of successful generations
- **Generation Speed**: Audio duration / generation time
- **Perplexity**: Text modeling quality

### Manual Evaluation
- **Voice Similarity**: How well does it match target speakers?
- **Naturalness**: Does the speech sound natural?
- **Conversation Flow**: Is the dialogue coherent?
- **Audio Quality**: Are there artifacts or distortions?

## 🛠️ Troubleshooting

### Common Issues

**Out of Memory Error**:
```python
# Reduce batch size and increase gradient accumulation
config.per_device_train_batch_size = 1
config.gradient_accumulation_steps = 16
config.gradient_checkpointing = True
```

**Poor Voice Quality**:
- Check voice sample quality (clean, 24kHz, mono)
- Increase `diffusion_loss_weight`
- Use more voice samples per speaker

**Slow Convergence**:
- Increase learning rate for specific components
- Reduce frozen parameters
- Check data quality and variety

**Generation Failures**:
- Reduce `max_new_tokens`
- Check CFG scale (try 1.0-2.0)
- Verify voice samples are properly processed

### Debug Mode
```bash
# Run with detailed logging
python train_vibevoice.py --strategy speaker_adaptation \
    --data ./training_data \
    --voices ./voices \
    --epochs 1 \
    --batch-size 1
```

## 📈 Advanced Techniques

### Learning Rate Scheduling
```python
# Different learning rates for different components
param_groups = [
    {"params": model.language_model.parameters(), "lr": 1e-5},
    {"params": model.diffusion_head.parameters(), "lr": 1e-4},
    {"params": model.connectors.parameters(), "lr": 5e-4},
]
```

### Data Augmentation
```python
# In future versions
config.enable_audio_augmentation = True  # Pitch, speed variations
config.enable_text_augmentation = True   # Paraphrasing, synonyms
```

### Multi-GPU Training
```bash
# Use accelerate for multi-GPU
accelerate config
accelerate launch train_vibevoice.py --config my_config.json
```

## 🎉 Using Your Fine-tuned Model

### In Your App
```python
from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from processor.vibevoice_processor import VibeVoiceProcessor

# Load your fine-tuned model
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    "./models/my-finetuned-vibevoice"
)
processor = VibeVoiceProcessor.from_pretrained(
    "./models/my-finetuned-vibevoice"
)

# Use just like the original model
audio = generate_podcast(script, voice_samples, model, processor)
```

### Share Your Model
```python
# Push to Hugging Face Hub
model.push_to_hub("your-username/vibevoice-custom")
processor.push_to_hub("your-username/vibevoice-custom")
```

## 💡 Tips for Success

1. **Start Small**: Begin with speaker adaptation before trying full fine-tuning
2. **Quality Over Quantity**: Clean, high-quality voice samples matter more than quantity
3. **Monitor Closely**: Watch training metrics and stop if loss stops improving
4. **Test Early**: Evaluate on small samples during training to catch issues
5. **Save Checkpoints**: Enable frequent saving in case training is interrupted
6. **Compare Results**: Always compare with baseline model performance

## 📚 Additional Resources

- [VibeVoice Paper](https://arxiv.org/abs/2405.02017)
- [Hugging Face Model Hub](https://huggingface.co/microsoft/VibeVoice-1.5B)
- [Training Best Practices](./docs/training_best_practices.md)
- [Voice Sample Preparation](./docs/voice_preparation.md)

---

Happy fine-tuning! 🎙️✨
