"""
Geometry-Enhanced Wall Graph Diffusion Model for Architectural Synthesis.
Inspired by GSDiff (Hu et al., AAAI 2025).
Operates directly over structured wall segments and topological junction graphs,
learning reverse diffusion transitions conditioned on room adjacencies and site boundaries.
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WallSegmentEncoder(nn.Module):
    """
    Encodes wall segment geometry: [x1, y1, x2, y2, length, angle, thickness, is_exterior].
    """
    def __init__(self, in_features: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class JunctionNodeEncoder(nn.Module):
    """
    Encodes junction vertex nodes: [x, y, degree, junction_type_onehot (4)].
    """
    def __init__(self, in_features: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class WallGraphMessagePassing(nn.Module):
    """
    Bipartite relational graph message passing between wall segments and junction nodes.
    Propagates structural alignment constraints (perpendicularity, continuity).
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.wall_to_junc = nn.Linear(hidden_dim, hidden_dim)
        self.junc_to_wall = nn.Linear(hidden_dim, hidden_dim)
        self.norm_junc = nn.LayerNorm(hidden_dim)
        self.norm_wall = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

    def forward(
        self,
        h_walls: torch.Tensor,
        h_juncs: torch.Tensor,
        incidence_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h_walls: [B, N_walls, hidden_dim]
        h_juncs: [B, N_juncs, hidden_dim]
        incidence_matrix: [B, N_juncs, N_walls] (1 if wall connects to junction)
        """
        # Junction update from incident walls
        msg_to_junc = torch.bmm(incidence_matrix, self.wall_to_junc(h_walls))
        deg = torch.clamp(incidence_matrix.sum(dim=2, keepdim=True), min=1.0)
        h_juncs_new = self.norm_junc(h_juncs + self.act(msg_to_junc / deg))

        # Wall update from incident junctions
        msg_to_wall = torch.bmm(incidence_matrix.transpose(1, 2), self.junc_to_wall(h_juncs_new))
        wall_deg = torch.clamp(incidence_matrix.transpose(1, 2).sum(dim=2, keepdim=True), min=1.0)
        h_walls_new = self.norm_wall(h_walls + self.act(msg_to_wall / wall_deg))

        return h_walls_new, h_juncs_new


class WallGraphDiffusion(nn.Module):
    """
    Complete GSDiff-style reverse diffusion model over wall topologies.
    Predicts coordinate noise on wall segment endpoints [dx1, dy1, dx2, dy2].
    """
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.wall_encoder = WallSegmentEncoder(in_features=8, hidden_dim=hidden_dim)
        self.junc_encoder = JunctionNodeEncoder(in_features=7, hidden_dim=hidden_dim)

        self.gnn_layers = nn.ModuleList([
            WallGraphMessagePassing(hidden_dim) for _ in range(num_layers)
        ])

        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4) # Predicts [dx1, dy1, dx2, dy2]
        )

    def _get_sinusoidal_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.hidden_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb_scale)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def forward(
        self,
        noisy_walls: torch.Tensor,
        junc_features: torch.Tensor,
        incidence_matrix: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        noisy_walls: [B, N_walls, 8]
        junc_features: [B, N_juncs, 7]
        incidence_matrix: [B, N_juncs, N_walls]
        timesteps: [B]
        """
        t_emb = self.time_embed(self._get_sinusoidal_time_embedding(timesteps))
        
        h_w = self.wall_encoder(noisy_walls) + t_emb.unsqueeze(1)
        h_j = self.junc_encoder(junc_features)

        for layer in self.gnn_layers:
            h_w, h_j = layer(h_w, h_j, incidence_matrix)

        pred_noise = self.noise_head(h_w)
        return pred_noise


def compute_gsdiff_structural_loss(
    pred_walls: torch.Tensor,
    target_walls: torch.Tensor,
    alpha_ortho: float = 0.1
) -> torch.Tensor:
    """
    Computes GSDiff composite loss:
    MSE on wall endpoints + Manhattan orthogonality penalty on wall vectors.
    """
    # 1. Coordinate regression loss
    mse_loss = F.mse_loss(pred_walls, target_walls)

    # 2. Manhattan angle regularization: penalize non-axis-aligned walls
    dx = pred_walls[:, :, 2] - pred_walls[:, :, 0]
    dy = pred_walls[:, :, 3] - pred_walls[:, :, 1]
    # In Manhattan geometry, |dx * dy| should be 0 (either dx=0 or dy=0)
    ortho_loss = torch.mean(torch.abs(dx * dy))

    return mse_loss + alpha_ortho * ortho_loss
