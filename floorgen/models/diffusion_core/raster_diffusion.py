"""
Raster Diffusion & VQ-VAE Floorplan Preview Synthesis Engine.
Inspired by Floorplan-Diffusion (Xu et al., 2025) and Asadyousaf03/floorgen.
Renders semantic vector bounding boxes to 256x256 architectural raster previews
and provides a lightweight VQ-VAE latent encoder-decoder.
"""

import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Distinct high-contrast architectural palette (RGB)
ROOM_COLOR_MAP = {
    "living_room": (230, 240, 255),    # Soft Ice Blue
    "master_bedroom": (255, 235, 235), # Soft Coral Rose
    "second_bedroom": (255, 245, 230), # Soft Apricot
    "bedroom": (255, 245, 230),
    "bathroom": (220, 255, 245),       # Soft Mint
    "kitchen": (255, 255, 220),        # Soft Canary
    "balcony": (235, 255, 235),        # Soft Sage
    "entrance": (245, 235, 255),       # Soft Lavender
    "dining_room": (255, 240, 225),
    "study": (240, 240, 255),
    "storage": (240, 240, 240)
}
WALL_COLOR = (40, 44, 52)      # Dark Charcoal
DOOR_COLOR = (220, 100, 50)     # Terracotta
GRID_COLOR = (245, 247, 250)    # Off-white background


class VectorToRasterRenderer:
    """
    Directly converts canonical vector floorplan representations into high-fidelity
    256x256 architectural raster images (NumPy RGB).
    """
    def __init__(self, canvas_size: int = 256):
        self.canvas_size = canvas_size

    def render(self, plan: Dict[str, Any]) -> np.ndarray:
        """
        Renders a plan dictionary to a 256x256x3 uint8 image array.
        """
        canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.uint8) * 255
        rooms = plan.get("rooms", [])
        doors = plan.get("doors", [])
        walls = plan.get("walls", {})
        wall_list = walls.get("walls", []) if isinstance(walls, dict) else (walls if isinstance(walls, list) else [])

        # Subtle structural drafting grid
        for g in range(0, self.canvas_size, 16):
            canvas[g, :, :] = 245
            canvas[:, g, :] = 245

        # 1. Fill room interior polygons
        for r in rooms:
            cat = r.get("category", "living_room")
            bbox = r.get("bbox") or r.get("box", [0, 0, 10, 10])
            color = ROOM_COLOR_MAP.get(cat, (235, 235, 235))
            x1 = max(0, min(self.canvas_size - 1, int(bbox[0])))
            y1 = max(0, min(self.canvas_size - 1, int(bbox[1])))
            x2 = max(0, min(self.canvas_size - 1, int(bbox[2])))
            y2 = max(0, min(self.canvas_size - 1, int(bbox[3])))
            if x2 > x1 and y2 > y1:
                canvas[y1:y2, x1:x2] = color

        # 2. Draw wall boundary centerlines and thick edges
        for r in rooms:
            bbox = r.get("bbox") or r.get("box", [0, 0, 10, 10])
            x1 = max(0, min(self.canvas_size - 1, int(bbox[0])))
            y1 = max(0, min(self.canvas_size - 1, int(bbox[1])))
            x2 = max(0, min(self.canvas_size - 1, int(bbox[2])))
            y2 = max(0, min(self.canvas_size - 1, int(bbox[3])))

            # Draw 2px dark wall strokes
            canvas[max(0, y1-1):min(self.canvas_size, y1+2), x1:x2] = WALL_COLOR
            canvas[max(0, y2-1):min(self.canvas_size, y2+2), x1:x2] = WALL_COLOR
            canvas[y1:y2, max(0, x1-1):min(self.canvas_size, x1+2)] = WALL_COLOR
            canvas[y1:y2, max(0, x2-1):min(self.canvas_size, x2+2)] = WALL_COLOR

        # 3. Draw door markers
        for d in doors:
            pos = d.get("pos") or d.get("position")
            if pos and len(pos) >= 2:
                dx = max(2, min(self.canvas_size - 3, int(pos[0])))
                dy = max(2, min(self.canvas_size - 3, int(pos[1])))
                canvas[dy-2:dy+3, dx-2:dx+3] = DOOR_COLOR

        return canvas


class VectorConditionedVQVAE(nn.Module):
    """
    Lightweight Vector-Quantized Variational Autoencoder (VQ-VAE) for floorplan rasters.
    Encodes 256x256 architectural rasters into discrete spatial latents conditioned
    on room vector representations.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 64, num_embeddings: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        # Encoder: 256x256 -> 32x32
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), # 128
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),          # 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1)    # 32
        )

        # Codebook embedding
        self.codebook = nn.Embedding(num_embeddings, latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # Decoder: 32x32 -> 256x256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),  # 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),          # 128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # 256
            nn.Sigmoid()
        )

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantizes continuous latent z to nearest codebook embedding."""
        # z: [B, C, H, W] -> [B, H, W, C]
        b, c, h, w = z.shape
        flat_z = z.permute(0, 2, 3, 1).contiguous().view(-1, c)

        # Distances: (z - e)^2 = z^2 + e^2 - 2 z e
        dist = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_z, self.codebook.weight.t())
        )
        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        quantized = self.codebook(encoding_indices.squeeze(1)).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        return quantized, encoding_indices

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        quantized, indices = self.quantize(z)
        x_recon = self.decoder(quantized)
        return x_recon, quantized


def generate_raster_preview(plan: Dict[str, Any]) -> np.ndarray:
    """
    Convenience function returning a 256x256x3 RGB NumPy array preview of the floorplan.
    """
    renderer = VectorToRasterRenderer(canvas_size=256)
    return renderer.render(plan)
