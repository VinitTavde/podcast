"""
Data preparation utilities for VibeVoice fine-tuning.
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import argparse

import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Utility class for preparing training data."""
    
    def __init__(
        self,
        target_sample_rate: int = 24000,
        max_audio_length: float = 30.0,
        min_audio_length: float = 1.0,
        target_db: float = -25.0,
    ):
        self.target_sample_rate = target_sample_rate
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.target_db = target_db
    
    def process_audio_file(self, input_path: str, output_path: str) -> bool:
        """
        Process a single audio file for training.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            # Load audio
            audio, sr = sf.read(input_path)
            
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
            
            # Check duration
            duration = len(audio) / self.target_sample_rate
            if duration < self.min_audio_length:
                logger.warning(f"Audio too short ({duration:.2f}s): {input_path}")
                return False
            elif duration > self.max_audio_length:
                # Truncate to max length
                max_samples = int(self.max_audio_length * self.target_sample_rate)
                audio = audio[:max_samples]
                logger.info(f"Truncated audio from {duration:.2f}s to {self.max_audio_length:.2f}s: {input_path}")
            
            # Normalize audio level
            if np.max(np.abs(audio)) > 0:
                # RMS normalization
                rms = np.sqrt(np.mean(audio**2))
                target_rms = 10**(self.target_db/20)
                audio = audio * (target_rms / rms)
                
                # Prevent clipping
                peak = np.max(np.abs(audio))
                if peak > 0.95:
                    audio = audio * (0.95 / peak)
            
            # Save processed audio
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, self.target_sample_rate)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing audio {input_path}: {e}")
            return False
    
    def create_voice_samples(
        self, 
        input_dir: str, 
        output_dir: str,
        speakers: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Process voice samples for fine-tuning.
        
        Args:
            input_dir: Directory containing raw voice samples
            output_dir: Directory to save processed voice samples
            speakers: Optional list of speaker names to process
            
        Returns:
            Dictionary mapping speaker names to processed audio paths
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        voice_samples = {}
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        
        # Find all audio files
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f"*{ext}"))
            audio_files.extend(input_path.glob(f"**/*{ext}"))
        
        logger.info(f"Found {len(audio_files)} audio files in {input_dir}")
        
        for audio_file in tqdm(audio_files, desc="Processing voice samples"):
            # Extract speaker name from filename or directory
            speaker_name = self._extract_speaker_name(audio_file, input_path)
            
            # Skip if specific speakers requested and this isn't one
            if speakers and speaker_name not in speakers:
                continue
            
            # Process audio
            output_file = output_path / f"{speaker_name}.wav"
            success = self.process_audio_file(str(audio_file), str(output_file))
            
            if success:
                voice_samples[speaker_name] = str(output_file)
                logger.info(f"Processed voice sample for speaker: {speaker_name}")
        
        logger.info(f"Created {len(voice_samples)} voice samples in {output_dir}")
        return voice_samples
    
    def _extract_speaker_name(self, audio_file: Path, base_path: Path) -> str:
        """Extract speaker name from audio file path."""
        # Try to get speaker from parent directory name
        relative_path = audio_file.relative_to(base_path)
        
        if len(relative_path.parts) > 1:
            # Use parent directory as speaker name
            speaker_name = relative_path.parts[-2]
        else:
            # Use filename as speaker name
            speaker_name = audio_file.stem
        
        # Clean speaker name
        speaker_name = re.sub(r'[^a-zA-Z0-9_-]', '_', speaker_name)
        return speaker_name


class ConversationDataProcessor:
    """Process conversation data for training."""
    
    def __init__(self):
        self.speaker_pattern = re.compile(r'^Speaker\s+(\w+):\s*(.*)$', re.IGNORECASE)
    
    def process_text_files(
        self, 
        input_dir: str, 
        output_file: str,
        min_speakers: int = 2,
        max_speakers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Process text files containing conversations.
        
        Args:
            input_dir: Directory containing text files
            output_file: Output JSON file path
            min_speakers: Minimum number of speakers per conversation
            max_speakers: Maximum number of speakers per conversation
            
        Returns:
            List of processed conversation data
        """
        input_path = Path(input_dir)
        conversations = []
        
        # Find all text files
        text_files = list(input_path.glob("*.txt"))
        text_files.extend(input_path.glob("**/*.txt"))
        
        logger.info(f"Found {len(text_files)} text files in {input_dir}")
        
        for text_file in tqdm(text_files, desc="Processing text files"):
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    continue
                
                # Parse conversation
                conversation = self._parse_conversation(content)
                
                if not conversation:
                    logger.warning(f"Could not parse conversation: {text_file}")
                    continue
                
                # Check speaker count
                speakers = list(set(line['speaker'] for line in conversation))
                if len(speakers) < min_speakers or len(speakers) > max_speakers:
                    logger.info(f"Skipping conversation with {len(speakers)} speakers: {text_file}")
                    continue
                
                # Create conversation data
                conv_data = {
                    "id": text_file.stem,
                    "source_file": str(text_file),
                    "script": self._format_script(conversation),
                    "speakers": speakers,
                    "num_turns": len(conversation),
                }
                
                conversations.append(conv_data)
                
            except Exception as e:
                logger.error(f"Error processing {text_file}: {e}")
        
        # Save processed data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed {len(conversations)} conversations to {output_file}")
        return conversations
    
    def _parse_conversation(self, content: str) -> List[Dict[str, str]]:
        """Parse conversation text into structured format."""
        lines = content.strip().split('\n')
        conversation = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match speaker pattern
            match = self.speaker_pattern.match(line)
            if match:
                speaker_id = match.group(1)
                text = match.group(2).strip()
                
                if text:  # Only add non-empty text
                    conversation.append({
                        "speaker": speaker_id,
                        "text": text
                    })
        
        return conversation
    
    def _format_script(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation as script text."""
        script_lines = []
        for turn in conversation:
            script_lines.append(f"Speaker {turn['speaker']}: {turn['text']}")
        return '\n'.join(script_lines)
    
    def split_dataset(
        self,
        data: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        output_dir: str = "./training_data"
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train/val/test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Shuffle data
        import random
        random.seed(42)
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        # Calculate split indices
        n_total = len(shuffled_data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split data
        train_data = shuffled_data[:n_train]
        val_data = shuffled_data[n_train:n_train + n_val]
        test_data = shuffled_data[n_train + n_val:]
        
        # Save splits
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            if split_data:
                output_file = os.path.join(output_dir, f"{split_name}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(split_data)} {split_name} samples to {output_file}")
        
        return train_data, val_data, test_data


def prepare_speaker_adaptation_data(
    voice_dir: str,
    script_file: str,
    output_dir: str,
    target_speakers: Optional[List[str]] = None
):
    """
    Prepare data for speaker adaptation fine-tuning.
    
    Args:
        voice_dir: Directory containing voice samples
        script_file: Text file with conversation script
        output_dir: Output directory for processed data
        target_speakers: List of target speaker names
    """
    logger.info("Preparing speaker adaptation data...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    voices_output = os.path.join(output_dir, "voices")
    
    # Process voice samples
    preprocessor = DataPreprocessor()
    voice_samples = preprocessor.create_voice_samples(
        input_dir=voice_dir,
        output_dir=voices_output,
        speakers=target_speakers
    )
    
    # Process conversation script
    conv_processor = ConversationDataProcessor()
    
    # Read script file
    with open(script_file, 'r', encoding='utf-8') as f:
        script_content = f.read().strip()
    
    # Parse conversation
    conversation = conv_processor._parse_conversation(script_content)
    
    if not conversation:
        raise ValueError(f"Could not parse conversation from {script_file}")
    
    # Create training data
    speakers = list(set(line['speaker'] for line in conversation))
    script_formatted = conv_processor._format_script(conversation)
    
    training_data = [{
        "id": "speaker_adaptation_conv",
        "script": script_formatted,
        "speakers": speakers,
        "voice_samples": voice_samples
    }]
    
    # Save training data
    train_file = os.path.join(output_dir, "train.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Speaker adaptation data prepared:")
    logger.info(f"  Voice samples: {len(voice_samples)} speakers")
    logger.info(f"  Conversation: {len(conversation)} turns")
    logger.info(f"  Output directory: {output_dir}")
    
    return voice_samples, training_data


def main():
    """Command line interface for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare data for VibeVoice fine-tuning")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Voice samples command
    voice_parser = subparsers.add_parser("voices", help="Process voice samples")
    voice_parser.add_argument("--input", required=True, help="Input directory with voice samples")
    voice_parser.add_argument("--output", required=True, help="Output directory for processed voices")
    voice_parser.add_argument("--speakers", nargs="+", help="Specific speakers to process")
    
    # Conversations command
    conv_parser = subparsers.add_parser("conversations", help="Process conversation data")
    conv_parser.add_argument("--input", required=True, help="Input directory with text files")
    conv_parser.add_argument("--output", required=True, help="Output JSON file")
    conv_parser.add_argument("--min-speakers", type=int, default=2, help="Minimum speakers per conversation")
    conv_parser.add_argument("--max-speakers", type=int, default=4, help="Maximum speakers per conversation")
    
    # Speaker adaptation command
    adapt_parser = subparsers.add_parser("speaker-adapt", help="Prepare speaker adaptation data")
    adapt_parser.add_argument("--voices", required=True, help="Voice samples directory")
    adapt_parser.add_argument("--script", required=True, help="Conversation script file")
    adapt_parser.add_argument("--output", required=True, help="Output directory")
    adapt_parser.add_argument("--speakers", nargs="+", help="Target speakers")
    
    # Split dataset command
    split_parser = subparsers.add_parser("split", help="Split dataset into train/val/test")
    split_parser.add_argument("--input", required=True, help="Input JSON file with conversations")
    split_parser.add_argument("--output", required=True, help="Output directory")
    split_parser.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio")
    split_parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio")
    split_parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    
    args = parser.parse_args()
    
    if args.command == "voices":
        preprocessor = DataPreprocessor()
        voice_samples = preprocessor.create_voice_samples(
            input_dir=args.input,
            output_dir=args.output,
            speakers=args.speakers
        )
        print(f"Processed {len(voice_samples)} voice samples")
        
    elif args.command == "conversations":
        processor = ConversationDataProcessor()
        conversations = processor.process_text_files(
            input_dir=args.input,
            output_file=args.output,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )
        print(f"Processed {len(conversations)} conversations")
        
    elif args.command == "speaker-adapt":
        voice_samples, training_data = prepare_speaker_adaptation_data(
            voice_dir=args.voices,
            script_file=args.script,
            output_dir=args.output,
            target_speakers=args.speakers
        )
        print("Speaker adaptation data prepared successfully")
        
    elif args.command == "split":
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processor = ConversationDataProcessor()
        train_data, val_data, test_data = processor.split_dataset(
            data=data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            output_dir=args.output
        )
        print(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
