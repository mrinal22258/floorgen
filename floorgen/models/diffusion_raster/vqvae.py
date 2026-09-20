"""
Floorplan VQ-VAE: Vector Quantized Variational AutoEncoder for Floorplans.
Compresses 256x256 RGB floorplan rasters into 32x32 discrete latent codes.
Adapted from Floorplan-Diffusion / Asadyousaf architecture for architectural raster representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with straight-through estimator and commitment loss.
    """
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # inputs shape: [B, C, H, W] -> permute to [B, H, W, C]
        inputs_perm = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs_perm.shape
        flat_input = inputs_perm.view(-1, self.embedding_dim)

        # Distances between input and codebook vectors: (x - e)^2 = x^2 + e^2 - 2 x e
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # Encoding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantized latents
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs_perm)
        q_latent_loss = F.mse_loss(quantized, inputs_perm.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through gradient estimator
        quantized = inputs_perm + (quantized - inputs_perm).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        indices = encoding_indices.view(input_shape[0], input_shape[1], input_shape[2])
        return quantized, loss, indices


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Encoder(nn.Module):
    """Encodes 256x256 RGB image into 32x32 feature map (8x downsampling)."""
    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, latent_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=4, stride=2, padding=1)   # 128x128
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1)   # 64x64
        self.conv3 = nn.Conv2d(hidden_dim, latent_dim, kernel_size=4, stride=2, padding=1)        # 32x32
        self.res1 = ResidualBlock(latent_dim)
        self.res2 = ResidualBlock(latent_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class Decoder(nn.Module):
    """Decodes 32x32 quantized feature map back into 256x256 RGB image."""
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 64, out_channels: int = 3):
        super().__init__()
        self.res1 = ResidualBlock(latent_dim)
        self.res2 = ResidualBlock(latent_dim)
        self.deconv1 = nn.ConvTranspose2d(latent_dim, hidden_dim, kernel_size=4, stride=2, padding=1)       # 64x64
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1) # 128x128
        self.deconv3 = nn.ConvTranspose2d(hidden_dim // 2, out_channels, kernel_size=4, stride=2, padding=1) # 256x256
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))
        return x


class FloorplanVQVAE(nn.Module):
    """
    Floorplan VQ-VAE pipeline for discrete latent spatial compression.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=embedding_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )
        self.decoder = Decoder(latent_dim=embedding_dim, out_channels=in_channels)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        quantized, _, indices = self.quantizer(z)
        return quantized, indices

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantized)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: [B, H, W]
        B, H, W = indices.shape
        flat_indices = indices.view(-1)
        quantized = self.quantizer.embedding(flat_indices).view(B, H, W, -1)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        quantized, vq_loss, indices = self.quantizer(z)
        reconstruction = self.decoder(quantized)
        
        recon_loss = F.l1_loss(reconstruction, x)
        total_loss = recon_loss + vq_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "reconstruction": reconstruction,
            "indices": indices
        }
