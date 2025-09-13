#!/usr/bin/env python3
"""
train_finetune_vibevoice.py

Drop-in finetune script for VibeVoice-style models and the yasserrmd/VibeVoice space layout.

Key design:
- Use repo processor if available (processor.VibeVoiceProcessor)
- Load model from HF (microsoft/VibeVoice-1.5B) or local checkpoint
- Freeze acoustic/semantic tokenizers (they're used only for encoding)
- Train diffusion head (ε prediction) conditioned on LLM hidden states
- Optional QLoRA / PEFT for efficient LLM adaptation
- Curriculum support for growing context length
- Uses accelerate to prepare everything for distributed training

Notes:
- You must adapt `ACOUSTIC_FRAME_RATE`, `LATENT_DIM`, and `processor` method names if your repo differs.
- The diffusion U-Net here is a reasonably small conditional U-Net; you can replace it with the repo's head by setting --load_diffusion_from_model True
"""

import os
import argparse
import json
import math
import random
import signal
import sys
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from accelerate import Accelerator, DeepSpeedPlugin

# Optional imports for PEFT and quantization
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import bitsandbytes as bnb
    HAS_PEFT = True
except ImportError:
    print("Warning: PEFT and bitsandbytes not available. QLoRA features will be disabled.")
    HAS_PEFT = False

# Try to import repo processor and model (safe fallback)
try:
    # repo likely provides a processor with encode_audio / encode_text methods
    from processor.vibevoice_processor import VibeVoiceProcessor  # type: ignore
    HAS_REPO_PROCESSOR = True
except Exception:
    VibeVoiceProcessor = None
    HAS_REPO_PROCESSOR = False

# Try to import local VibeVoice model classes
try:
    from modular.modeling_vibevoice import VibeVoiceForConditionalGeneration, VibeVoiceConfig
    from modular.configuration_vibevoice import VibeVoiceConfig
    HAS_LOCAL_MODEL = True
except Exception:
    VibeVoiceForConditionalGeneration = None
    VibeVoiceConfig = None
    HAS_LOCAL_MODEL = False

# -------------------------
# ----- utils & config ----
# -------------------------
@dataclass
class HParams:
    manifest: str
    model_name_or_path: str
    out_dir: str
    batch_size: int = 2
    epochs: int = 10
    lr: float = 5e-5
    latent_dim: int = 768
    acoustic_frame_rate: float = 7.5   # frames per second (VibeVoice used ~7.5 Hz)
    context_curriculum: str = "4k,16k,32k,64k"  # comma-separated token lengths
    use_qlora: bool = True
    peft_r: int = 8
    peft_alpha: int = 16
    max_text_len: int = 2048
    gradient_accumulation_steps: int = 1
    save_every_n_epochs: int = 1
    load_diffusion_from_model: bool = True  # if model has diffusion_head attr
    device: str = None

# -------------------------
# ----- Dataset class -----
# -------------------------
class VibeVoiceFinetuneDataset(Dataset):
    def __init__(self, manifest_path, processor, acoustic_frame_rate):
        self.items = [json.loads(l) for l in open(manifest_path, "r", encoding="utf-8")]
        self.processor = processor
        self.acoustic_frame_rate = acoustic_frame_rate

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        wav = it["wav"]
        text = it.get("transcript", "")
        speaker = it.get("speaker", "spk0")
        # Encode acoustic latents (continuous) from processor; expected shape [T_frames, latent_dim]
        # Processor API assumptions (try multiple common names)
        z = None
        if self.processor is not None:
            # try common method names
            if hasattr(self.processor, "encode_audio"):
                z = self.processor.encode_audio(wav)
            elif hasattr(self.processor, "acoustic_encode"):
                z = self.processor.acoustic_encode(wav)
            elif hasattr(self.processor, "encode_wave"):
                z = self.processor.encode_wave(wav)
            else:
                raise RuntimeError("Processor found but does not implement known encode_audio methods.")
        else:
            raise RuntimeError("No processor available; you must provide an acoustic tokenizer/encoder.")

        # Ensure tensor
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32)

        # Optional: downsample/clip to fixed number of frames for batch stability if needed (handled in collate)
        return {"wav": wav, "text": text, "speaker": speaker, "z": z}

def collate_fn(batch, tokenizer, max_text_len, device):
    texts = [b["text"] for b in batch]
    speakers = [b["speaker"] for b in batch]
    zs = [b["z"] for b in batch]

    # pad z sequences to same length
    max_z_len = max(z.shape[0] for z in zs)
    latent_dim = zs[0].shape[1]
    z_padded = torch.zeros(len(zs), max_z_len, latent_dim, dtype=torch.float32)
    z_mask = torch.zeros(len(zs), max_z_len, dtype=torch.bool)
    for i, z in enumerate(zs):
        L = z.shape[0]
        z_padded[i, :L] = z
        z_mask[i, :L] = 1

    # tokenize texts
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_text_len)
    batch_out = {
        "input_ids": enc.input_ids.to(device),
        "attention_mask": enc.attention_mask.to(device),
        "texts": texts,
        "speakers": speakers,
        "z": z_padded.to(device),
        "z_mask": z_mask.to(device),
    }
    return batch_out

# -------------------------
# ---- small U-Net denoiser
# -------------------------
# This is a compact conditional UNet for continuous latent denoising.
# Replace with the repo's diffusion head if available.
class SmallConditionalUNet(nn.Module):
    def __init__(self, latent_dim, cond_dim, base_ch=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_fc = nn.Linear(cond_dim, base_ch)
        # encoder
        self.enc1 = nn.Sequential(nn.Conv1d(latent_dim, base_ch, 3, padding=1), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv1d(base_ch, base_ch*2, 3, stride=2, padding=1), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv1d(base_ch*2, base_ch*4, 3, stride=2, padding=1), nn.GELU())
        # bottleneck
        self.mid = nn.Sequential(nn.Conv1d(base_ch*4, base_ch*4, 3, padding=1), nn.GELU())
        # decoder
        self.dec3 = nn.Sequential(nn.ConvTranspose1d(base_ch*4, base_ch*2, 4, stride=2, padding=1), nn.GELU())
        self.dec2 = nn.Sequential(nn.ConvTranspose1d(base_ch*2, base_ch, 4, stride=2, padding=1), nn.GELU())
        self.out_conv = nn.Conv1d(base_ch, latent_dim, 3, padding=1)

    def forward(self, noisy_z, t_embed, cond):
        # noisy_z: [B, T, latent_dim] -> conv expects [B, latent_dim, T]
        x = noisy_z.permute(0,2,1)
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        m = self.mid(h3)
        d3 = self.dec3(m)
        d2 = self.dec2(d3 + h2)  # skip
        out = self.out_conv(d2 + h1)
        # incorporate condition by adding projected cond to time dimension (broadcast)
        # simpler conditioning: add cond vector to channel dimension via linear
        # (we assume cond shape [B, cond_dim])
        cond_proj = self.cond_fc(cond).unsqueeze(-1)  # [B, base_ch, 1]
        # match channel size if needed: broadcast-add to out
        # If out channels != cond_proj channels, we broadcast to out channels by simple linear -> but skip for brevity
        return out.permute(0,2,1)  # [B, T, latent_dim]

# -------------------------
# --- diffusion utilities ---
# -------------------------
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def q_sample(z0, betas, t, device):
    # standard DDPM forward noising: z_t = sqrt(alpha_cum) z0 + sqrt(1-alpha_cum) * noise
    alphas = 1.0 - betas
    alphas_cum = torch.cumprod(alphas, dim=0).to(device)
    sqrt_alpha_cum = torch.sqrt(alphas_cum[t])[:, None, None]  # batch indexing done by caller
    sqrt_one_minus_alpha_cum = torch.sqrt(1.0 - alphas_cum[t])[:, None, None]
    noise = torch.randn_like(z0, device=device)
    z_t = sqrt_alpha_cum * z0 + sqrt_one_minus_alpha_cum * noise
    return z_t, noise

# -------------------------
# ------- timeout handler --
# -------------------------
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Model initialization timed out")

# -------------------------
# ------- main -------------
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/VibeVoice-1.5B")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--latent_dim", type=int, default=768)
    parser.add_argument("--acoustic_frame_rate", type=float, default=7.5)
    parser.add_argument("--context_curriculum", type=str, default="4k,16k,32k,64k")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--peft_r", type=int, default=8)
    parser.add_argument("--peft_alpha", type=int, default=16)
    parser.add_argument("--max_text_len", type=int, default=2048)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--load_diffusion_from_model", action="store_true")
    parser.add_argument("--use_simple_model", action="store_true", help="Use a simple language model instead of full VibeVoice for faster initialization")
    args = parser.parse_args()

    hp = HParams(
        manifest=args.manifest,
        model_name_or_path=args.model_name_or_path,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        latent_dim=args.latent_dim,
        acoustic_frame_rate=args.acoustic_frame_rate,
        context_curriculum=args.context_curriculum,
        use_qlora=args.use_qlora,
        peft_r=args.peft_r,
        peft_alpha=args.peft_alpha,
        max_text_len=args.max_text_len,
        device=args.device
    )

    os.makedirs(hp.out_dir, exist_ok=True)

    # Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # --- Processor (acoustic + semantic tokenizers) ---
    processor = None
    if HAS_REPO_PROCESSOR:
        try:
            # Try to load processor from local path first
            if os.path.exists("processor"):
                processor = VibeVoiceProcessor.from_pretrained(".")
                print("Loaded local VibeVoice processor.")
            else:
                processor = VibeVoiceProcessor()  # many processors have zero-arg constructor
        except Exception as e:
            print(f"Failed to load local processor: {e}")
            # try other ways to instantiate: load from HF path if present
            try:
                processor = VibeVoiceProcessor.from_pretrained(hp.model_name_or_path)
            except Exception:
                processor = None

    if processor is None:
        # try to load a HF tokenizer for the tokenization part (text). Acoustic encoder must still be available!
        print("Warning: repo processor not found or failed to init. You must provide an acoustic encoder accessible by this script.")
        # fallthrough; later dataset will fail if no processor

    # --- Load LLM (decoder-only) ---
    print("Loading LLM:", hp.model_name_or_path)
    
    # Check if we should use local model or HuggingFace model
    if args.use_simple_model:
        print("Using simple language model for faster initialization...")
        # Use a simple Qwen2 model instead of full VibeVoice
        simple_model_path = "Qwen/Qwen2.5-1.5B"  # or another lightweight model
        tokenizer = AutoTokenizer.from_pretrained(simple_model_path, use_fast=True)
        if HAS_PEFT:
            model = AutoModelForCausalLM.from_pretrained(simple_model_path, trust_remote_code=True, device_map="auto", load_in_8bit=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(simple_model_path, trust_remote_code=True, device_map="auto")
    elif HAS_LOCAL_MODEL and (hp.model_name_or_path == "microsoft/VibeVoice-1.5B" or "vibevoice" in hp.model_name_or_path.lower()):
        print("Using local VibeVoice implementation...")
        # For local model, we need to load the config and model differently
        try:
            # Try to load from a local checkpoint or create a default config
            if os.path.exists(hp.model_name_or_path):
                # Local path - load config and model
                config = VibeVoiceConfig.from_pretrained(hp.model_name_or_path)
                model = VibeVoiceForConditionalGeneration.from_pretrained(hp.model_name_or_path, config=config)
            else:
                # Create default config for local model
                print("Creating default VibeVoice config...")
                config = VibeVoiceConfig()
                print("Config created. Initializing model components...")
                print("This may take a few minutes as it initializes acoustic/semantic tokenizers and language model...")
                
                # Set up timeout for model initialization (5 minutes)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)  # 5 minutes timeout
                
                try:
                    model = VibeVoiceForConditionalGeneration(config)
                    signal.alarm(0)  # Cancel timeout
                    print("Model initialization completed!")
                except TimeoutError:
                    signal.alarm(0)  # Cancel timeout
                    print("Model initialization timed out after 5 minutes.")
                    print("This might be due to memory constraints or slow initialization.")
                    print("Try using --use_simple_model flag for faster initialization.")
                    raise
            
            # Load tokenizer from the decoder config (Qwen2)
            tokenizer = AutoTokenizer.from_pretrained(config.decoder_config.name_or_path, use_fast=True)
            
        except Exception as e:
            print(f"Failed to load local VibeVoice model: {e}")
            print("Falling back to HuggingFace model...")
            # Fallback to HuggingFace
            tokenizer = AutoTokenizer.from_pretrained(hp.model_name_or_path, use_fast=True)
            if HAS_PEFT:
                model = AutoModelForCausalLM.from_pretrained(hp.model_name_or_path, trust_remote_code=True, device_map="auto", load_in_8bit=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(hp.model_name_or_path, trust_remote_code=True, device_map="auto")
    else:
        # Standard HuggingFace model loading
        tokenizer = AutoTokenizer.from_pretrained(hp.model_name_or_path, use_fast=True)
        if HAS_PEFT:
            model = AutoModelForCausalLM.from_pretrained(hp.model_name_or_path, trust_remote_code=True, device_map="auto", load_in_8bit=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(hp.model_name_or_path, trust_remote_code=True, device_map="auto")
    
    # ensure special tokens for acoustic placeholders exist
    if "<|acoustic|>" not in tokenizer.get_vocab():
        try:
            tokenizer.add_tokens(["<|acoustic|>"])
        except Exception:
            pass

    print("Loading model (may be large)...")
    model.config.use_cache = False  # ensure gradients where needed

    # optionally prepare for QLoRA / k-bit + LoRA
    if hp.use_qlora and HAS_PEFT:
        print("Preparing model for k-bit and LoRA (QLoRA)...")
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=hp.peft_r,
            lora_alpha=hp.peft_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
    elif hp.use_qlora and not HAS_PEFT:
        print("Warning: QLoRA requested but PEFT not available. Continuing without QLoRA...")
        hp.use_qlora = False

    # freeze LLM except peft adapters
    for n, p in model.named_parameters():
        if "lora_" in n or "adapter_" in n or (hp.use_qlora and ("lora" in n or "peft" in n)):
            p.requires_grad = True
        else:
            p.requires_grad = False

    # --- diffusion head: try to load from model object if present ---
    diffusion = None
    if args.load_diffusion_from_model and hasattr(model, "diffusion_head"):
        try:
            diffusion = model.diffusion_head
            print("Loaded diffusion_head from model object.")
        except Exception:
            diffusion = None
    elif HAS_LOCAL_MODEL and hasattr(model, "model") and hasattr(model.model, "prediction_head"):
        try:
            diffusion = model.model.prediction_head
            print("Loaded prediction_head from local VibeVoice model.")
        except Exception:
            diffusion = None

    if diffusion is None:
        print("Diffusion head not found on model; creating local SmallConditionalUNet...")
        # Get hidden size from model config
        if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            hidden_size = model.config.hidden_size
        elif hasattr(model, "config") and hasattr(model.config, "decoder_config"):
            hidden_size = model.config.decoder_config.hidden_size
        else:
            hidden_size = 768  # default
        diffusion = SmallConditionalUNet(latent_dim=hp.latent_dim, cond_dim=hidden_size)

    # Move diffusion to float32/float16 appropriately (accelerate will handle device)
    # Create dataset & dataloader
    ds = VibeVoiceFinetuneDataset(hp.manifest, processor, hp.acoustic_frame_rate)
    dl = DataLoader(ds, batch_size=hp.batch_size, shuffle=True, num_workers=4, collate_fn=lambda b: collate_fn(b, tokenizer, hp.max_text_len, device))

    # optimizer: diffusion params + peft params (if any)
    trainable = [p for p in diffusion.parameters() if p.requires_grad]
    trainable += [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=hp.lr)

    # prepare with accelerator
    diffusion, model, optimizer, dl = accelerator.prepare(diffusion, model, optimizer, dl)

    # diffusion schedule
    T = 1000
    betas = get_beta_schedule(T).to(device)

    total_steps = len(dl) * hp.epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=total_steps)

    # curriculum token lengths
    curriculum = [int(x.strip()) for x in hp.context_curriculum.split(",")]

    print("Begin training loop. steps:", total_steps)
    global_step = 0
    for epoch in range(hp.epochs):
        # pick curriculum based on epoch proportion
        curriculum_idx = min(len(curriculum)-1, epoch * len(curriculum) // max(1, hp.epochs))
        current_ctx = curriculum[curriculum_idx]
        print(f"Epoch {epoch+1}/{hp.epochs} - context tokens: {current_ctx}")

        for batch in dl:
            model.train()  # some gradients exist only for adapters
            diffusion.train()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            z = batch["z"]          # [B, Tz, latent_dim]
            z_mask = batch["z_mask"]  # [B, Tz]

            B, Tz, D = z.shape
            # Build hybrid prompt: we will append placeholder tokens equal to number of frames (or representative mapping)
            # Simpler approach: place one placeholder token at end representing next-token acoustic frame to predict
            placeholder = "<|acoustic|>"
            prompt_texts = [f"<|speaker:{s}|> {t} {placeholder}" for s,t in zip(batch["speakers"], batch["texts"])]
            enc = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=current_ctx).to(device)

            # Get LLM hidden states (we keep the forward no-grad to avoid finetuning full LLM)
            # But if using LoRA, we need gradients for those adapter params: so use model(**enc, output_hidden_states=True)
            out = model(**enc, output_hidden_states=True, return_dict=True)
            # last hidden states shape [B, seq_len, hidden]
            h_last = out.hidden_states[-1]  # [B, L, H]

            # For simplicity, pick hidden state of last token (placeholder) as conditioning vector for the target acoustic frame.
            # Real VibeVoice maps multiple acoustic frames -> multiple token positions; this simplified training is a valid finetune step to align conditioning.
            cond_hi = h_last[:, -1, :]  # [B, H]

            # Build diffusion target z0: choose first acoustic frame OR aggregate across frames (here we use first non-zero frame)
            # You should replace this with proper alignment mapping of acoustic frames to token positions.
            z0 = []
            for i in range(B):
                # find first valid frame (non-zero mask)
                valid_len = z_mask[i].sum().item()
                if valid_len == 0:
                    z0.append(torch.zeros(hp.latent_dim, device=device))
                else:
                    z0.append(z[i, 0, :].to(device))
            z0 = torch.stack(z0, dim=0)  # [B, latent_dim]

            # Expand z0 to time dimension T=1 for our UNet implementation shape [B, 1, latent_dim]
            z0 = z0.unsqueeze(1)

            # sample a random timestep t for each sample
            t = torch.randint(0, T, (B,), device=device).long()

            # q_sample to produce z_t and noise
            # For vectorized q_sample we compute sqrt_alpha_cum and sqrt_one_minus for each t
            alphas = 1.0 - betas
            alphas_cum = torch.cumprod(alphas, dim=0)
            sqrt_alpha_cum = torch.sqrt(alphas_cum[t]).unsqueeze(-1).unsqueeze(-1)
            sqrt_one_minus_alpha_cum = torch.sqrt(1.0 - alphas_cum[t]).unsqueeze(-1).unsqueeze(-1)
            noise = torch.randn_like(z0, device=device)
            z_t = sqrt_alpha_cum * z0 + sqrt_one_minus_alpha_cum * noise

            # Build timestep embedding (simple scalar embedding -> can be improved)
            t_embed = (t.float() / float(T)).unsqueeze(-1).to(device)  # [B,1]

            # Predict noise with diffusion network conditioned on cond_hi
            pred_noise = diffusion(z_t, t_embed, cond_hi)  # expected shape [B, 1, latent_dim] -> returns same

            # loss = mse between predicted noise and true noise
            loss = nn.functional.mse_loss(pred_noise, noise)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 50 == 0 and accelerator.is_main_process:
                print(f"Step {global_step} loss {loss.item():.6f}")

        # end epoch
        if accelerator.is_main_process and ((epoch + 1) % hp.save_every_n_epochs == 0):
            accelerator.wait_for_everyone()
            save_path = os.path.join(hp.out_dir, f"checkpoint_epoch{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            # save diffusion and PEFT adapters if present
            try:
                # diffusion (if it is part of model: prefer to save separately)
                torch.save(accelerator.get_state_dict(diffusion), os.path.join(save_path, "diffusion.pt"))
            except Exception:
                # fallback
                if accelerator.unwrap_model(diffusion) is not None:
                    torch.save(accelerator.unwrap_model(diffusion).state_dict(), os.path.join(save_path, "diffusion_raw.pt"))
            try:
                if hp.use_qlora and HAS_PEFT:
                    model.save_pretrained(save_path)
                else:
                    accelerator.unwrap_model(model).save_pretrained(save_path)
            except Exception as e:
                print("Warning: failed to save model:", e)

    # final save
    if accelerator.is_main_process:
        final_path = os.path.join(hp.out_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        torch.save(accelerator.unwrap_model(diffusion).state_dict(), os.path.join(final_path, "diffusion_final.pt"))
        if hp.use_qlora and HAS_PEFT:
            model.save_pretrained(final_path)
        else:
            accelerator.unwrap_model(model).save_pretrained(final_path)

    print("Training finished.")

if __name__ == "__main__":
    main()
