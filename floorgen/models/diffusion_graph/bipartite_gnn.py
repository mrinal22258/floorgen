"""
Bipartite Graph Neural Network (GSDiff-style) for walls and junctions.
Performs alternating message-passing:
  1. Walls -> Junctions (aggregates incident wall geometry into junction nodes)
  2. Junctions -> Walls (updates wall segments with connected junction topologies)
Pure PyTorch implementation for seamless cross-platform execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BipartiteGNNLayer(nn.Module):
    """
    Single bipartite interaction layer between wall segment nodes and junction nodes.
    """
    def __init__(self, wall_dim: int = 128, junction_dim: int = 128):
        super().__init__()
        # Wall -> Junction message function
        self.w2j_msg = nn.Sequential(
            nn.Linear(wall_dim + junction_dim, junction_dim),
            nn.LayerNorm(junction_dim),
            nn.SiLU(),
            nn.Linear(junction_dim, junction_dim)
        )
        self.j_update = nn.Sequential(
            nn.Linear(junction_dim * 2, junction_dim),
            nn.LayerNorm(junction_dim),
            nn.SiLU()
        )

        # Junction -> Wall message function
        self.j2w_msg = nn.Sequential(
            nn.Linear(junction_dim + wall_dim, wall_dim),
            nn.LayerNorm(wall_dim),
            nn.SiLU(),
            nn.Linear(wall_dim, wall_dim)
        )
        self.w_update = nn.Sequential(
            nn.Linear(wall_dim * 2, wall_dim),
            nn.LayerNorm(wall_dim),
            nn.SiLU()
        )

    def forward(
        self,
        h_walls: torch.Tensor,
        h_junctions: torch.Tensor,
        incidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          h_walls: [N, wall_dim]
          h_junctions: [J, junction_dim]
          incidence: [2, E] where row 0 is wall indices and row 1 is junction indices
        """
        N = h_walls.size(0)
        J = h_junctions.size(0)
        E = incidence.size(1) if incidence.numel() > 0 else 0

        if E == 0 or N == 0 or J == 0:
            return h_walls, h_junctions

        w_idx = incidence[0]  # [E]
        j_idx = incidence[1]  # [E]

        # 1. Walls -> Junctions
        w_feats = h_walls[w_idx]       # [E, wall_dim]
        j_feats = h_junctions[j_idx]   # [E, junction_dim]
        w2j_inputs = torch.cat([w_feats, j_feats], dim=-1)
        w2j_messages = self.w2j_msg(w2j_inputs)  # [E, junction_dim]

        # Aggregate messages into junctions by index_add
        j_agg = torch.zeros(J, h_junctions.size(1), device=h_junctions.device)
        j_agg.index_add_(0, j_idx, w2j_messages)
        new_h_junctions = self.j_update(torch.cat([h_junctions, j_agg], dim=-1)) + h_junctions

        # 2. Junctions -> Walls
        new_j_feats = new_h_junctions[j_idx]  # [E, junction_dim]
        j2w_inputs = torch.cat([new_j_feats, w_feats], dim=-1)
        j2w_messages = self.j2w_msg(j2w_inputs)  # [E, wall_dim]

        # Aggregate messages into walls by index_add
        w_agg = torch.zeros(N, h_walls.size(1), device=h_walls.device)
        w_agg.index_add_(0, w_idx, j2w_messages)
        new_h_walls = self.w_update(torch.cat([h_walls, w_agg], dim=-1)) + h_walls

        return new_h_walls, new_h_junctions


class BipartiteWallJunctionGNN(nn.Module):
    """
    Multi-layer Bipartite GNN network for structural wall graph conditioning and denoising.
    """
    def __init__(self, wall_dim: int = 128, junction_dim: int = 128, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            BipartiteGNNLayer(wall_dim, junction_dim) for _ in range(num_layers)
        ])

    def forward(
        self,
        h_walls: torch.Tensor,
        h_junctions: torch.Tensor,
        incidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h_walls, h_junctions = layer(h_walls, h_junctions, incidence)
        return h_walls, h_junctions
