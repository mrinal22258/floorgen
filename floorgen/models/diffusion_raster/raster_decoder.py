"""
FloorGen Super-Resolution Architectural Raster Decoder.
Upsamples latent codes (32x32 or 256x256) to 512x512 photorealistic architectural rasters
with architectural textures and crisp wall edge maps.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Dict, Any, Union, Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)


class ArchitecturalRasterDecoder(nn.Module):
    """
    Decodes VQ-VAE / latent codes -> 512x512 photorealistic architectural rasters.
    Returns:
      texture: [B, 3, 512, 512] RGB architectural render in [-1, 1]
      edges: [B, 1, 512, 512] binary wall edge map in [0, 1]
    """
    def __init__(self, in_channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.up1 = UpBlock(in_channels, 64)      # 32 -> 64
        self.up2 = UpBlock(64, 64)                # 64 -> 128
        self.up3 = UpBlock(64, 48)                # 128 -> 256
        self.up4 = UpBlock(48, 32)                # 256 -> 512

        # Direct super-resolution path (256x256 RGB -> 512x512 feature map)
        self.sr_conv = nn.Sequential(
            nn.Conv2d(out_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            UpBlock(32, 32)
        )

        # Architectural refinement
        self.wall_refiner = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
        )
        self.texture_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, out_channels, 3, padding=1), nn.Tanh(),
        )
        self.edge_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid(),  # Wall edges
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] == 256 and x.shape[1] == 3:
            # Learned 256x256 -> 512x512 super-resolution path
            feats = self.sr_conv(x)
        else:
            if x.shape[1] != self.in_channels:
                repeats = (self.in_channels + x.shape[1] - 1) // x.shape[1]
                x = x.repeat(1, repeats, 1, 1)[:, :self.in_channels]
            x1 = self.up1(x)
            x2 = self.up2(x1)
            x3 = self.up3(x2)
            feats = self.up4(x3)

        edges = self.edge_head(feats)
        texture = self.texture_head(feats + 0.1 * self.wall_refiner(feats) * edges)
        return texture, edges

    def loss_fn(
        self,
        pred_texture: torch.Tensor,
        target_texture: torch.Tensor,
        pred_edges: torch.Tensor,
        target_edges: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        recon_loss = F.l1_loss(pred_texture, target_texture)
        edge_loss = F.mse_loss(pred_edges.float(), target_edges.float())
        total = recon_loss + 0.5 * edge_loss
        return {
            "loss": total,
            "recon_loss": recon_loss,
            "edge_loss": edge_loss
        }

    @torch.no_grad()
    def decode_to_image(self, latents: torch.Tensor) -> np.ndarray:
        """
        Inference helper returning (512, 512, 3) BGR uint8 image.
        """
        self.eval()
        texture, _ = self.forward(latents)
        img = ((texture[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return bgr


# Backwards compatibility alias
RasterDecoder = ArchitecturalRasterDecoder
