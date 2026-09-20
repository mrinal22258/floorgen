"""
HouseDiffusion: Vector Floorplan Coordinate Diffusion Model (Shabani et al., CVPR 2023).
Learns a continuous denoising process directly over 2D polygon vertices and room bounding box coordinates.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from floorgen.models.shared.representation import SinusoidalPositionalEmbedding, RelationalGraphConv


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule as proposed in Nichol & Dhariwal (2021).
    Prevents abrupt drop in signal near t=T and produces sharper, less noisy geometries.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.999)


class DDPMScheduler:
    """
    Gaussian Discrete Diffusion Scheduler supporting both standard DDPM
    and accelerated DDIM deterministic reverse sampling.
    Operates directly on normalized coordinates in [-1, 1].
    """
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine"
    ):
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        if beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(num_timesteps)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion process q(x_t | x_0)."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def step(self, model_output: torch.Tensor, t: int, sample: torch.Tensor) -> torch.Tensor:
        """Single reverse denoising step p(x_{t-1} | x_t) (DDPM stochastic)."""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        
        # Mean
        pred_prev_sample = (1.0 / torch.sqrt(alpha_t)) * (
            sample - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * model_output
        )

        if t > 0:
            noise = torch.randn_like(sample)
            variance = torch.sqrt(torch.clamp(self.posterior_variance[t], min=1e-20))
            pred_prev_sample = pred_prev_sample + variance * noise

        return pred_prev_sample

    def ddim_step(
        self,
        model_output: torch.Tensor,
        t: int,
        t_prev: int,
        sample: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Deterministic/stochastic DDIM sampling step (Song et al., 2020).
        Enables 10-30 step high-fidelity sampling.
        """
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_prev = (
            self.alphas_cumprod[t_prev] if t_prev >= 0
            else torch.tensor(1.0, device=sample.device)
        )

        # 1. Predict x_0 from eps_pred
        pred_x0 = (sample - torch.sqrt(1.0 - alpha_cumprod_t) * model_output) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # 2. Compute variance sigma_t
        if eta > 0.0 and t_prev >= 0:
            sigma_t = eta * torch.sqrt(
                (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t) * (1.0 - alpha_cumprod_t / alpha_cumprod_prev)
            )
        else:
            sigma_t = 0.0

        # 3. Direction pointing to x_t
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_cumprod_prev - (sigma_t ** 2 if isinstance(sigma_t, torch.Tensor) else sigma_t ** 2), min=0.0)) * model_output

        # 4. Compute x_{t_prev}
        x_prev = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt
        if eta > 0.0 and t_prev >= 0:
            noise = torch.randn_like(sample)
            x_prev = x_prev + sigma_t * noise

        return x_prev


class HouseDiffusionDenoiser(nn.Module):
    """
    Coordinate Denoising Network with Relational Graph Attention.
    Predicts the noise epsilon added to room box/polygon coordinates.
    """
    def __init__(
        self,
        coord_dim: int = 4, # 4 for bbox [xmin, ymin, xmax, ymax]
        num_room_types: int = 10,
        hidden_dim: int = 192,
        num_layers: int = 4
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.room_embed = nn.Embedding(num_room_types, hidden_dim)
        self.coord_proj = nn.Linear(coord_dim, hidden_dim)
        self.boundary_proj = nn.Linear(4, hidden_dim)

        self.in_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        self.convs = nn.ModuleList([
            RelationalGraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, coord_dim)
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        room_types: torch.Tensor,
        adj_matrix: torch.Tensor,
        room_mask: torch.Tensor,
        boundary: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x_t: (B, N, coord_dim) - noisy coordinates
        timesteps: (B,) - diffusion timestep
        room_types: (B, N) - category IDs
        adj_matrix: (B, N, N) - bubble graph adjacency
        room_mask: (B, N) - valid room mask
        boundary: (B, 4) - site boundary box
        """
        B, N, _ = x_t.shape

        t_feat = self.time_embed(timesteps).unsqueeze(1).expand(-1, N, -1) # (B, N, H)
        r_feat = self.room_embed(room_types)                                # (B, N, H)
        c_feat = self.coord_proj(x_t)                                      # (B, N, H)

        h = self.in_proj(torch.cat([t_feat + c_feat, r_feat, t_feat], dim=-1))

        if boundary is not None:
            b_feat = self.boundary_proj(boundary).unsqueeze(1).expand(-1, N, -1)
            h = h + b_feat

        for conv in self.convs:
            h = conv(h, adj_matrix, room_mask)

        eps_pred = self.out_head(h)
        return eps_pred * room_mask.unsqueeze(-1).float()
