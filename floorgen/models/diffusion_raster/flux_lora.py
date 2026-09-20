"""
FLUX.1-dev Architectural LoRA Fine-Tuning & Inference Pipeline.
Adapted from Asadyousaf03/floorgen for photorealistic architectural raster generation.
Includes 4GB VRAM safety (quantization / offloading / lightweight latent fallback).

Currently unused at inference time — see pipeline_sota.py. Enable only via explicit training CLI flag.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class RPLANFluxDataset(Dataset):
    """
    Dataset loader for RPLAN paired floorplans and architectural rasters.
    """
    def __init__(self, json_dir: str, image_dir: str, target_size: int = 256, max_samples: Optional[int] = None):
        self.samples = list(Path(json_dir).glob("*.json"))
        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.fallback_count = 0

    def reset_fallback_count(self) -> int:
        """Resets and returns the fallback count for the current epoch."""
        count = self.fallback_count
        self.fallback_count = 0
        return count

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        json_file = self.samples[idx]
        with open(json_file, "r", encoding="utf-8") as f:
            plan = json.load(f)

        # Build detailed architectural prompt
        rooms = ", ".join([r.get("category", "").replace("_", " ") for r in plan.get("rooms", [])])
        adj_count = len(plan.get("adjacency", []))
        prompt = (
            f"Professional architectural floorplan, {rooms}, "
            f"{adj_count} room connections, "
            f"clean CAD lines, labeled zones, orthographic view, 8k architectural drawing"
        )

        # Load raster image
        raster_path = plan.get("raster_path", f"images/{json_file.stem}.png")
        full_img_path = self.image_dir / Path(raster_path).name
        
        if full_img_path.exists():
            image = cv2.imread(str(full_img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[0] != self.target_size or image.shape[1] != self.target_size:
                image = cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        else:
            # Fallback synthetic canvas with explicit warning and tracking
            self.fallback_count += 1
            logger.warning(
                f"Missing raster image at '{full_img_path}' for plan '{json_file.stem}'. "
                f"Falling back to synthetic canvas (epoch fallback count: {self.fallback_count})."
            )
            image = np.full((self.target_size, self.target_size, 3), 245, dtype=np.uint8)

        # Normalize to [-1, 1] tensor: [3, H, W]
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1.0

        return {
            "id": json_file.stem,
            "prompt": prompt,
            "image": img_tensor,
            "rooms": len(plan.get("rooms", []))
        }


class LightweightArchitecturalUNet(nn.Module):
    """
    4GB-VRAM optimized architectural latent diffusion backbone.
    Operates on 32x32 latent codes with cross-attention to text / graph conditioning.
    """
    def __init__(self, in_channels: int = 64, cond_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross attention to prompt/condition
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )
        self.out_conv = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, C, H, W], t: [B, 1], cond: [B, cond_dim]
        h = self.in_conv(x)
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb
        
        if cond is not None:
            c_emb = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
            h = h + c_emb

        h = self.block1(h)
        h = self.block2(h)
        return self.out_conv(h)


class FloorplanFluxLoRA(nn.Module):
    """
    Unified architectural diffusion generator.
    Supports:
    1. FLUX.1-dev LoRA pipeline (NF4 quantized + sequential CPU offload for 4GB VRAM safety)
    2. Lightweight Architectural Diffusion (fully trainable on 4GB VRAM RTX 3050 via --allow-fallback)
    """
    def __init__(
        self,
        use_flux_pipeline: bool = True,
        allow_fallback: bool = False,
        clean_cache: bool = False,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        latent_channels: int = 64,
        device: str = "cuda"
    ):
        super().__init__()
        self.use_flux_pipeline = use_flux_pipeline
        self.allow_fallback = allow_fallback
        self.clean_cache = clean_cache
        self.model_id = model_id
        self.device = device
        self.latent_channels = latent_channels
        
        if use_flux_pipeline:
            self._init_flux()
        else:
            print("[FloorplanFluxLoRA] Notice: Explicit fallback mode enabled with local architectural UNet.")
            self.unet = LightweightArchitecturalUNet(in_channels=latent_channels)
            self.to(device)

    def _init_flux(self):
        import os
        import shutil
        import traceback
        from pathlib import Path
        
        # 1. Startup Authentication & License Check
        token = os.environ.get("HF_TOKEN")
        if not token:
            env_file = Path(".env")
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        os.environ["HF_TOKEN"] = token
                        break
        
        if not token:
            raise RuntimeError(
                "Missing HF_TOKEN for gated FLUX.1-dev model. "
                "Please configure HF_TOKEN in your .env file or environment and ensure your "
                "Hugging Face account has accepted the FLUX.1-dev license at: "
                "https://huggingface.co/black-forest-labs/FLUX.1-dev"
            )

        tmp_base = os.environ.get("TEMP", "/tmp")
        hf_cache_dir = os.path.join(tmp_base, "hf_cache")
        offload_dir = os.path.join(tmp_base, "flux_offload")
        
        # Only wipe scratch disk offload space
        if os.path.exists(offload_dir):
            shutil.rmtree(offload_dir, ignore_errors=True)
        os.makedirs(offload_dir, exist_ok=True)
            
        # Wipe hf_cache ONLY on explicit request (clean_cache=True)
        if self.clean_cache and os.path.exists(hf_cache_dir):
            shutil.rmtree(hf_cache_dir, ignore_errors=True)
        os.makedirs(hf_cache_dir, exist_ok=True)

        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        
        try:
            from diffusers import FluxPipeline
            from transformers import BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            # Step 5: Sequential CPU offloading for 4GB VRAM safety (Option a)
            self.pipe = FluxPipeline.from_pretrained(
                self.model_id,
                dtype=torch.bfloat16,
                quantization_config=quantization_config,
                offload_folder=offload_dir,
                low_cpu_mem_usage=True,
                token=token,
            )
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.enable_gradient_checkpointing()
            
            # Diffusion Transformer LoRA configuration
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=[
                    "transformer_blocks.*.attn.to_q", "transformer_blocks.*.attn.to_k",
                    "transformer_blocks.*.attn.to_v", "transformer_blocks.*.attn.to_out.0",
                    "single_transformer_blocks.*.attn.to_q", "single_transformer_blocks.*.attn.to_k",
                    "single_transformer_blocks.*.attn.to_v", "single_transformer_blocks.*.attn.to_out.0",
                ],
                lora_dropout=0.1,
                bias="none",
            )
            self.pipe.transformer = get_peft_model(self.pipe.transformer, lora_config)
            print(f"[FLUX] LoRA params: {sum(p.numel() for p in self.pipe.transformer.parameters() if p.requires_grad)/1e6:.1f}M trainable")
        except Exception as e:
            if not self.allow_fallback:
                print(f"\n[FATAL FLUX INITIALIZATION FAILURE] Could not load FLUX pipeline:")
                traceback.print_exc()
                raise RuntimeError(
                    f"FLUX pipeline initialization failed: {e}. Hard Constraint #6 prohibits silent fallbacks. "
                    f"To use local architectural diffusion proxy for testing, explicitly pass allow_fallback=True."
                ) from e
            
            print(f"[FLUX] Fallback enabled: Quantized online pipeline skipped ({e}); using 4GB VRAM local architectural diffusion core.")
            self.use_flux_pipeline = False
            self.unet = LightweightArchitecturalUNet(in_channels=self.latent_channels)
            self.to(self.device)
        finally:
            # Wipe transient offload scratch space only; hf_cache persists across runs
            if os.path.exists(offload_dir):
                shutil.rmtree(offload_dir, ignore_errors=True)

    def forward(self, latents: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.unet(latents, t, cond)

    @torch.no_grad()
    def generate_latents(
        self,
        batch_size: int = 1,
        cond: Optional[torch.Tensor] = None,
        steps: int = 20,
        size: int = 32
    ) -> torch.Tensor:
        """
        Denoises random normal latents conditioned on vector/graph embeddings.
        Returns: [B, latent_channels, size, size]
        """
        self.eval()
        x = torch.randn(batch_size, self.latent_channels, size, size, device=self.device)
        
        for step in reversed(range(steps)):
            t = torch.full((batch_size, 1), step / float(steps), device=self.device)
            predicted_noise = self.unet(x, t, cond)
            # Simple DDIM step
            alpha = 1.0 - (step / float(steps))
            x = x - (1.0 / steps) * predicted_noise
            
        return x


def train_flux_lora(
    data_dir: str = "data/processed/rplan_80k",
    epochs: int = 2,
    batch_size: int = 2,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "checkpoints/flux_lora.pt",
    max_samples: Optional[int] = None,
    use_flux_pipeline: bool = True,
    allow_fallback: bool = False
):
    """
    Trains architectural LoRA / latent diffusion on RPLAN canonical pairs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_dir = os.path.join(data_dir, "train")
    img_dir = os.path.join(data_dir, "images")

    dataset = RPLANFluxDataset(train_dir, img_dir, max_samples=max_samples)
    if len(dataset) == 0:
        print(f"[FloorplanFluxLoRA] No training data found at {train_dir}. Aborting training.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = FloorplanFluxLoRA(use_flux_pipeline=use_flux_pipeline, allow_fallback=allow_fallback, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"[FloorplanFluxLoRA] Training on {len(dataset)} samples across {epochs} epochs on {device} (use_flux={use_flux_pipeline}, allow_fallback={allow_fallback})...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            # Target image downscaled to 32x32 latent feature proxy
            img = batch["image"].to(device)
            latents = F.interpolate(img, size=(32, 32), mode="bilinear")
            
            # Tile channels if needed to match latent_channels exactly
            if latents.shape[1] < model.latent_channels:
                repeats = (model.latent_channels + latents.shape[1] - 1) // latents.shape[1]
                latents = latents.repeat(1, repeats, 1, 1)[:, :model.latent_channels]
                
            noise = torch.randn_like(latents)
            t = torch.rand(latents.shape[0], 1, device=device)
            noisy_latents = latents + noise * t.unsqueeze(-1).unsqueeze(-1)
            
            pred_noise = model(noisy_latents, t)
            loss = F.mse_loss(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(loader))
        print(f"  Architectural Diffusion Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    torch.save({"state_dict": model.state_dict()}, save_path)
    print(f"[FloorplanFluxLoRA] Checkpoint saved to {save_path} ({os.path.getsize(save_path)/1024/1024:.2f} MB)")
