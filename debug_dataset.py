#!/usr/bin/env python3
"""
Debug script to test dataset functionality.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from training.dataset import VibeVoiceDataset
from processor.vibevoice_processor import VibeVoiceProcessor

def main():
    print("🔍 Debugging dataset...")
    
    try:
        # Load processor
        print("Loading processor...")
        processor = VibeVoiceProcessor.from_pretrained('microsoft/VibeVoice-1.5B')
        print("✅ Processor loaded")
        
        # Test dataset
        print("Creating dataset...")
        dataset = VibeVoiceDataset(
            data_path='./my_training_data/train.json',
            processor=processor,
            voice_samples_dir='./my_training_data/voices',
            max_length=2048,
            cache_audio=True,
        )
        
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Get sample
        print("Getting sample...")
        sample = dataset[0]
        
        print(f"Sample keys: {list(sample.keys())}")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: {type(value)} with {len(value)} items")
            else:
                print(f"  {key}: {value}")
        
        # Check specifically for speech data
        print("\n🎵 Speech data check:")
        if 'speech_tensors' in sample and sample['speech_tensors'] is not None:
            print(f"✅ speech_tensors: {sample['speech_tensors'].shape}")
        else:
            print("❌ speech_tensors: None or missing")
            
        if 'speech_masks' in sample and sample['speech_masks'] is not None:
            print(f"✅ speech_masks: {sample['speech_masks'].shape}")
        else:
            print("❌ speech_masks: None or missing")
            
        if 'speech_input_mask' in sample and sample['speech_input_mask'] is not None:
            print(f"✅ speech_input_mask: {sample['speech_input_mask'].shape}")
        else:
            print("❌ speech_input_mask: None or missing")
        
        print("\n✅ Debug completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
