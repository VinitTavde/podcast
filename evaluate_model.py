#!/usr/bin/env python3
"""
Evaluation script for fine-tuned VibeVoice models.

Usage:
    python evaluate_model.py --model ./models/vibevoice-finetuned --data ./test_data.json
    python evaluate_model.py --model microsoft/VibeVoice-1.5B --data ./test_data.json --baseline
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

import torch
import numpy as np
from tqdm import tqdm
import soundfile as sf

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from processor.vibevoice_processor import VibeVoiceProcessor
from training.dataset import VibeVoiceDataset
from training.config import TrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VibeVoiceEvaluator:
    """Evaluator for VibeVoice models."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        inference_steps: int = 5,
        cfg_scale: float = 1.3
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to model or model name
            device: Device to use for inference
            inference_steps: Number of diffusion steps
            cfg_scale: CFG scale for generation
        """
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_steps = inference_steps
        self.cfg_scale = cfg_scale
        
        # Load model and processor
        self.model, self.processor = self._load_model_and_processor()
        
        logger.info(f"Evaluator initialized with model: {model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Inference steps: {inference_steps}")
        logger.info(f"CFG scale: {cfg_scale}")
    
    def _load_model_and_processor(self):
        """Load model and processor."""
        logger.info(f"Loading model from: {self.model_path}")
        
        # Load processor
        processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        
        # Load model for inference
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device if self.device != "cpu" else None,
        )
        
        # Set inference parameters
        model.set_ddpm_inference_steps(self.inference_steps)
        model.eval()
        
        return model, processor
    
    def generate_audio(
        self,
        script: str,
        voice_samples: List[np.ndarray],
        max_new_tokens: int = 2048,
        verbose: bool = False
    ) -> Optional[np.ndarray]:
        """
        Generate audio from script and voice samples.
        
        Args:
            script: Conversation script
            voice_samples: List of voice sample arrays
            max_new_tokens: Maximum tokens to generate
            verbose: Whether to show generation progress
            
        Returns:
            Generated audio array or None if generation failed
        """
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
            
            # Generate
            with torch.no_grad():
                start_time = time.time()
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    cfg_scale=self.cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={'do_sample': False},
                    verbose=verbose,
                )
                
                generation_time = time.time() - start_time
            
            # Extract audio
            if hasattr(outputs, 'speech_outputs') and outputs.speech_outputs[0] is not None:
                audio_tensor = outputs.speech_outputs[0]
                audio = audio_tensor.cpu().float().numpy()
                
                if audio.ndim > 1:
                    audio = audio.squeeze()
                
                logger.info(f"Generated {len(audio) / 24000:.2f}s audio in {generation_time:.2f}s")
                return audio
            else:
                logger.error("No audio generated")
                return None
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None
    
    def evaluate_dataset(
        self,
        test_data: List[Dict[str, Any]],
        voice_samples_dir: str,
        output_dir: str,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on a test dataset.
        
        Args:
            test_data: List of test conversation data
            voice_samples_dir: Directory containing voice samples
            output_dir: Directory to save generated audio
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Evaluation metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load voice samples
        voice_samples = self._load_voice_samples(voice_samples_dir)
        
        results = {
            "total_samples": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "average_generation_time": 0.0,
            "total_audio_duration": 0.0,
            "samples": []
        }
        
        # Limit samples if requested
        eval_data = test_data[:max_samples] if max_samples else test_data
        
        logger.info(f"Evaluating {len(eval_data)} samples...")
        
        generation_times = []
        
        for i, item in enumerate(tqdm(eval_data, desc="Evaluating")):
            try:
                # Get script and speakers
                script = item.get('script', item.get('conversation', item.get('text', '')))
                speakers = item.get('speakers', [])
                
                if not script:
                    logger.warning(f"No script found for sample {i}")
                    continue
                
                # Get voice samples for speakers
                sample_voices = []
                for speaker in speakers:
                    if speaker in voice_samples:
                        sample_voices.append(voice_samples[speaker])
                    elif voice_samples:
                        # Use random voice if specific not found
                        sample_voices.append(list(voice_samples.values())[0])
                
                if not sample_voices:
                    logger.warning(f"No voice samples available for sample {i}")
                    continue
                
                # Generate audio
                start_time = time.time()
                audio = self.generate_audio(
                    script=script,
                    voice_samples=sample_voices,
                    verbose=False
                )
                generation_time = time.time() - start_time
                
                sample_result = {
                    "sample_id": item.get('id', f"sample_{i}"),
                    "speakers": speakers,
                    "script_length": len(script),
                    "generation_time": generation_time,
                }
                
                if audio is not None:
                    # Save generated audio
                    audio_filename = f"generated_{i:04d}.wav"
                    audio_path = os.path.join(output_dir, audio_filename)
                    sf.write(audio_path, audio, 24000)
                    
                    # Update results
                    results["successful_generations"] += 1
                    generation_times.append(generation_time)
                    
                    audio_duration = len(audio) / 24000
                    results["total_audio_duration"] += audio_duration
                    
                    sample_result.update({
                        "status": "success",
                        "audio_path": audio_path,
                        "audio_duration": audio_duration,
                        "generation_speed": audio_duration / generation_time if generation_time > 0 else 0
                    })
                    
                    logger.info(f"Sample {i}: Generated {audio_duration:.2f}s audio in {generation_time:.2f}s")
                else:
                    results["failed_generations"] += 1
                    sample_result["status"] = "failed"
                    logger.error(f"Sample {i}: Generation failed")
                
                results["samples"].append(sample_result)
                results["total_samples"] += 1
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                results["failed_generations"] += 1
                results["total_samples"] += 1
        
        # Calculate summary metrics
        if generation_times:
            results["average_generation_time"] = np.mean(generation_times)
            results["median_generation_time"] = np.median(generation_times)
            results["success_rate"] = results["successful_generations"] / results["total_samples"]
            
            # Calculate average generation speed (audio_duration / generation_time)
            speeds = [s["generation_speed"] for s in results["samples"] if s.get("generation_speed")]
            if speeds:
                results["average_generation_speed"] = np.mean(speeds)
        
        # Save results
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation completed:")
        logger.info(f"  Total samples: {results['total_samples']}")
        logger.info(f"  Successful: {results['successful_generations']}")
        logger.info(f"  Failed: {results['failed_generations']}")
        logger.info(f"  Success rate: {results.get('success_rate', 0):.2%}")
        logger.info(f"  Average generation time: {results.get('average_generation_time', 0):.2f}s")
        logger.info(f"  Total audio generated: {results['total_audio_duration']:.2f}s")
        logger.info(f"  Results saved to: {results_file}")
        
        return results
    
    def _load_voice_samples(self, voice_samples_dir: str) -> Dict[str, np.ndarray]:
        """Load voice samples from directory."""
        voice_samples = {}
        voice_dir = Path(voice_samples_dir)
        
        if not voice_dir.exists():
            logger.warning(f"Voice samples directory not found: {voice_samples_dir}")
            return voice_samples
        
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        
        for audio_file in voice_dir.iterdir():
            if audio_file.suffix.lower() in audio_extensions:
                try:
                    audio, sr = sf.read(str(audio_file))
                    
                    # Convert to mono
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    
                    # Resample if needed
                    if sr != 24000:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
                    
                    speaker_name = audio_file.stem
                    voice_samples[speaker_name] = audio
                    
                except Exception as e:
                    logger.error(f"Error loading voice sample {audio_file}: {e}")
        
        logger.info(f"Loaded {len(voice_samples)} voice samples")
        return voice_samples
    
    def compare_models(
        self,
        baseline_model_path: str,
        test_data: List[Dict[str, Any]],
        voice_samples_dir: str,
        output_dir: str,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare current model with baseline model.
        
        Args:
            baseline_model_path: Path to baseline model
            test_data: Test data
            voice_samples_dir: Voice samples directory
            output_dir: Output directory
            max_samples: Maximum samples to compare
            
        Returns:
            Comparison results
        """
        logger.info("Starting model comparison...")
        
        # Evaluate current model
        current_output = os.path.join(output_dir, "current_model")
        current_results = self.evaluate_dataset(
            test_data, voice_samples_dir, current_output, max_samples
        )
        
        # Evaluate baseline model
        baseline_evaluator = VibeVoiceEvaluator(
            baseline_model_path, self.device, self.inference_steps, self.cfg_scale
        )
        baseline_output = os.path.join(output_dir, "baseline_model")
        baseline_results = baseline_evaluator.evaluate_dataset(
            test_data, voice_samples_dir, baseline_output, max_samples
        )
        
        # Compare results
        comparison = {
            "current_model": {
                "path": self.model_path,
                "results": current_results
            },
            "baseline_model": {
                "path": baseline_model_path,
                "results": baseline_results
            },
            "comparison": {
                "success_rate_diff": (
                    current_results.get("success_rate", 0) - 
                    baseline_results.get("success_rate", 0)
                ),
                "generation_time_diff": (
                    current_results.get("average_generation_time", 0) - 
                    baseline_results.get("average_generation_time", 0)
                ),
                "generation_speed_diff": (
                    current_results.get("average_generation_speed", 0) - 
                    baseline_results.get("average_generation_speed", 0)
                ),
            }
        }
        
        # Save comparison
        comparison_file = os.path.join(output_dir, "model_comparison.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info("Model comparison completed:")
        logger.info(f"  Current model success rate: {current_results.get('success_rate', 0):.2%}")
        logger.info(f"  Baseline model success rate: {baseline_results.get('success_rate', 0):.2%}")
        logger.info(f"  Success rate difference: {comparison['comparison']['success_rate_diff']:.2%}")
        logger.info(f"  Comparison saved to: {comparison_file}")
        
        return comparison


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate VibeVoice model")
    
    parser.add_argument("--model", required=True, help="Path to model or model name")
    parser.add_argument("--data", required=True, help="Path to test data JSON file")
    parser.add_argument("--voices", required=True, help="Path to voice samples directory")
    parser.add_argument("--output", default="./evaluation_results", help="Output directory")
    
    parser.add_argument("--baseline", help="Path to baseline model for comparison")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--inference-steps", type=int, default=5, help="Number of diffusion steps")
    parser.add_argument("--cfg-scale", type=float, default=1.3, help="CFG scale")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load test data
    logger.info(f"Loading test data from: {args.data}")
    with open(args.data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if isinstance(test_data, dict) and 'conversations' in test_data:
        test_data = test_data['conversations']
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Create evaluator
    evaluator = VibeVoiceEvaluator(
        model_path=args.model,
        device=args.device,
        inference_steps=args.inference_steps,
        cfg_scale=args.cfg_scale
    )
    
    # Run evaluation
    if args.baseline:
        # Model comparison
        results = evaluator.compare_models(
            baseline_model_path=args.baseline,
            test_data=test_data,
            voice_samples_dir=args.voices,
            output_dir=args.output,
            max_samples=args.max_samples
        )
    else:
        # Single model evaluation
        results = evaluator.evaluate_dataset(
            test_data=test_data,
            voice_samples_dir=args.voices,
            output_dir=args.output,
            max_samples=args.max_samples
        )
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
