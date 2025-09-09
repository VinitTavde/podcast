"""
Dataset classes for VibeVoice fine-tuning.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import soundfile as sf

from processor.vibevoice_processor import VibeVoiceProcessor

logger = logging.getLogger(__name__)


class VibeVoiceDataset(Dataset):
    """
    Dataset for VibeVoice fine-tuning.
    
    Supports multiple data formats:
    - JSON with conversation scripts and voice samples
    - Directory with audio files and transcripts
    """
    
    def __init__(
        self,
        data_path: str,
        processor: VibeVoiceProcessor,
        voice_samples_dir: str,
        max_length: int = 2048,
        max_audio_length: float = 30.0,
        min_audio_length: float = 1.0,
        target_sample_rate: int = 24000,
        cache_audio: bool = False,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file (JSON format)
            processor: VibeVoice processor for text and audio processing
            voice_samples_dir: Directory containing voice sample files
            max_length: Maximum sequence length for text
            max_audio_length: Maximum audio length in seconds
            min_audio_length: Minimum audio length in seconds
            target_sample_rate: Target sample rate for audio
            cache_audio: Whether to cache audio in memory (for small datasets)
        """
        self.data_path = data_path
        self.processor = processor
        self.voice_samples_dir = Path(voice_samples_dir)
        self.max_length = max_length
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.target_sample_rate = target_sample_rate
        self.cache_audio = cache_audio
        
        # Load data
        self.data = self._load_data()
        
        # Load and cache voice samples
        self.voice_samples = self._load_voice_samples()
        
        # Audio cache for faster training
        self.audio_cache = {} if cache_audio else None
        
        logger.info(f"Loaded {len(self.data)} training examples")
        logger.info(f"Available voice samples: {list(self.voice_samples.keys())}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load training data from file."""
        if self.data_path.endswith('.json'):
            return self._load_json_data()
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")
    
    def _load_json_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'conversations' in data:
            return data['conversations']
        else:
            raise ValueError("Invalid JSON format. Expected list or dict with 'conversations' key")
    
    def _load_voice_samples(self) -> Dict[str, str]:
        """Load available voice samples from directory."""
        voice_samples = {}
        
        if not self.voice_samples_dir.exists():
            logger.warning(f"Voice samples directory not found: {self.voice_samples_dir}")
            return voice_samples
        
        # Supported audio formats
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        
        for audio_file in self.voice_samples_dir.iterdir():
            if audio_file.suffix.lower() in audio_extensions:
                # Use filename without extension as speaker name
                speaker_name = audio_file.stem
                voice_samples[speaker_name] = str(audio_file)
        
        return voice_samples
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load and process audio file."""
        if self.cache_audio and audio_path in self.audio_cache:
            return self.audio_cache[audio_path]
        
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if necessary
            if sr != self.target_sample_rate:
                audio = librosa.resample(
                    audio, 
                    orig_sr=sr, 
                    target_sr=self.target_sample_rate
                )
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Check duration constraints
            duration = len(audio) / self.target_sample_rate
            if duration < self.min_audio_length:
                # Pad short audio
                target_length = int(self.min_audio_length * self.target_sample_rate)
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            elif duration > self.max_audio_length:
                # Truncate long audio
                target_length = int(self.max_audio_length * self.target_sample_rate)
                audio = audio[:target_length]
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95
            
            # Cache if enabled
            if self.cache_audio:
                self.audio_cache[audio_path] = audio
            
            return audio
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            # Return silence if audio loading fails
            return np.zeros(int(self.min_audio_length * self.target_sample_rate), dtype=np.float32)
    
    def _get_voice_sample_for_speaker(self, speaker_name: str) -> Optional[str]:
        """Get voice sample path for a speaker."""
        # Direct match
        if speaker_name in self.voice_samples:
            return self.voice_samples[speaker_name]
        
        # Try case-insensitive match
        for voice_name, voice_path in self.voice_samples.items():
            if voice_name.lower() == speaker_name.lower():
                return voice_path
        
        # Try partial match
        for voice_name, voice_path in self.voice_samples.items():
            if speaker_name.lower() in voice_name.lower():
                return voice_path
        
        # Return random voice sample if no match found
        if self.voice_samples:
            return random.choice(list(self.voice_samples.values()))
        
        return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample."""
        item = self.data[idx]
        
        try:
            # Extract conversation script
            if 'script' in item:
                script = item['script']
            elif 'conversation' in item:
                script = item['conversation']
            elif 'text' in item:
                script = item['text']
            else:
                raise KeyError("No script/conversation/text found in data item")
            
            # Get speakers from script or item
            if 'speakers' in item:
                speakers = item['speakers']
            else:
                # Extract speakers from script
                speakers = self._extract_speakers_from_script(script)
            
            # Get voice samples for speakers
            voice_samples = []
            # Check if item has voice_samples mapping
            if 'voice_samples' in item and isinstance(item['voice_samples'], dict):
                # Use explicit voice mapping from JSON
                for speaker in speakers:
                    if speaker in item['voice_samples']:
                        voice_filename = item['voice_samples'][speaker]
                        voice_path = os.path.join(self.voice_samples_dir, voice_filename)
                        if os.path.exists(voice_path):
                            voice_audio = self._load_audio(voice_path)
                            voice_samples.append(voice_audio)
            else:
                # Use automatic voice matching
                for speaker in speakers:
                    voice_path = self._get_voice_sample_for_speaker(speaker)
                    if voice_path:
                        voice_audio = self._load_audio(voice_path)
                        voice_samples.append(voice_audio)
            
            # Ensure we have at least one voice sample
            if not voice_samples and self.voice_samples:
                # Use random voice if no specific match
                random_voice_path = random.choice(list(self.voice_samples.values()))
                voice_audio = self._load_audio(random_voice_path)
                voice_samples = [voice_audio]
            
            # Process with VibeVoice processor
            processed = self.processor(
                text=script,
                voice_samples=[voice_samples] if voice_samples else None,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Prepare labels for training
            input_ids = processed['input_ids'].squeeze(0)
            attention_mask = processed['attention_mask'].squeeze(0)
            
            # Create labels (copy of input_ids for language modeling)
            labels = input_ids.clone()
            
            # Mask padding tokens in labels (-100 ignores these tokens in loss computation)
            labels[attention_mask == 0] = -100
            
            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
            
            # Add speech-related inputs if available
            if processed.get('speech_tensors') is not None:
                result['speech_tensors'] = processed['speech_tensors'].squeeze(0)
                result['speech_masks'] = processed['speech_masks'].squeeze(0)
                result['speech_input_mask'] = processed['speech_input_mask'].squeeze(0)
            
            # Add metadata
            result['conversation_id'] = item.get('id', idx)
            result['speakers'] = speakers
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            # Return a dummy sample to avoid training interruption
            return self._get_dummy_sample()
    
    def _extract_speakers_from_script(self, script: str) -> List[str]:
        """Extract speaker names from conversation script."""
        import re
        
        # Find all "Speaker X:" patterns
        speaker_matches = re.findall(r'Speaker\s+(\w+):', script, re.IGNORECASE)
        
        # Get unique speakers while preserving order
        speakers = []
        seen = set()
        for speaker in speaker_matches:
            if speaker not in seen:
                speakers.append(speaker)
                seen.add(speaker)
        
        return speakers
    
    def _get_dummy_sample(self) -> Dict[str, Any]:
        """Get a dummy sample for error recovery."""
        dummy_text = "Speaker 1: Hello, this is a test conversation."
        
        processed = self.processor(
            text=dummy_text,
            voice_samples=None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        input_ids = processed['input_ids'].squeeze(0)
        attention_mask = processed['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'conversation_id': -1,
            'speakers': ['1'],
        }


class CollateFunction:
    """Collate function for batching VibeVoice samples."""
    
    def __init__(self, processor: VibeVoiceProcessor, pad_to_multiple_of: int = 8):
        self.processor = processor
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples."""
        batch = {}
        
        # Get sequence lengths
        seq_lengths = [f['input_ids'].shape[0] for f in features]
        max_length = max(seq_lengths)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of > 1:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        batch_size = len(features)
        
        # Initialize tensors
        batch['input_ids'] = torch.full((batch_size, max_length), 
                                       self.processor.tokenizer.pad_id, 
                                       dtype=torch.long)
        batch['attention_mask'] = torch.zeros((batch_size, max_length), dtype=torch.long)
        batch['labels'] = torch.full((batch_size, max_length), -100, dtype=torch.long)
        
        # Fill tensors
        for i, feature in enumerate(features):
            seq_len = feature['input_ids'].shape[0]
            batch['input_ids'][i, :seq_len] = feature['input_ids']
            batch['attention_mask'][i, :seq_len] = feature['attention_mask']
            batch['labels'][i, :seq_len] = feature['labels']
        
        # Handle speech inputs if present
        speech_features = [f for f in features if 'speech_tensors' in f]
        if speech_features:
            # Get speech tensor info
            speech_tensors = [f['speech_tensors'] for f in speech_features]
            speech_masks = [f['speech_masks'] for f in speech_features]
            speech_input_masks = [f['speech_input_mask'] for f in speech_features]
            
            if speech_tensors and speech_tensors[0] is not None:
                # Batch speech tensors
                max_speech_len = max(t.shape[-1] for t in speech_tensors)
                max_speech_tokens = max(m.shape[-1] for m in speech_masks)
                
                batched_speech = torch.zeros((len(speech_tensors), max_speech_len))
                batched_speech_masks = torch.zeros((len(speech_masks), max_speech_tokens), dtype=torch.bool)
                batched_speech_input_masks = torch.zeros((batch_size, max_length), dtype=torch.bool)
                
                for i, (tensor, mask, input_mask) in enumerate(zip(speech_tensors, speech_masks, speech_input_masks)):
                    if tensor is not None:
                        batched_speech[i, :tensor.shape[-1]] = tensor
                        batched_speech_masks[i, :mask.shape[-1]] = mask
                        batched_speech_input_masks[i, :input_mask.shape[0]] = input_mask
                
                batch['speech_tensors'] = batched_speech
                batch['speech_masks'] = batched_speech_masks
                batch['speech_input_mask'] = batched_speech_input_masks
        
        # Add metadata
        batch['conversation_ids'] = [f.get('conversation_id', i) for i, f in enumerate(features)]
        
        return batch


def create_sample_dataset():
    """Create a sample dataset for testing."""
    sample_data = [
        {
            "id": "conv_001",
            "script": "Speaker 1: Hello everyone, welcome to our podcast!\nSpeaker 2: Thanks for having me on the show!",
            "speakers": ["1", "2"]
        },
        {
            "id": "conv_002", 
            "script": "Speaker 1: Today we're discussing AI and the future.\nSpeaker 2: It's a fascinating topic with many implications.",
            "speakers": ["1", "2"]
        },
        {
            "id": "conv_003",
            "script": "Speaker 1: What do you think about the latest developments?\nSpeaker 2: I believe we're seeing rapid progress in multiple areas.",
            "speakers": ["1", "2"]
        }
    ]
    
    os.makedirs("./training_data", exist_ok=True)
    
    # Split into train/val
    train_data = sample_data[:2]
    val_data = sample_data[2:]
    
    with open("./training_data/train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open("./training_data/val.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print("Created sample dataset:")
    print("  - ./training_data/train.json (2 conversations)")
    print("  - ./training_data/val.json (1 conversation)")


if __name__ == "__main__":
    create_sample_dataset()
