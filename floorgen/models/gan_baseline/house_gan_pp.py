"""
House-GAN++ Relational Generative Adversarial Network Baseline for Floorplan Synthesis.
Implements the relational layout generator and discriminator with iterative graph refinement
(Nauata et al., CVPR 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
from floorgen.models.shared.representation import RelationalGraphConv


class HouseGANppGenerator(nn.Module):
    """
    Relational Graph Generator for House-GAN++.
    Takes bubble diagram adjacency + room category embeddings + noise vector,
    and iteratively refines room bounding box coordinates [xmin, ymin, xmax, ymax].
    """
    def __init__(
        self,
        num_room_types: int = 10,
        noise_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 4
    ):
        super().__init__()
        self.room_embed = nn.Embedding(num_room_types, hidden_dim)
        self.noise_proj = nn.Linear(noise_dim, hidden_dim)
        self.boundary_proj = nn.Linear(4, hidden_dim)

        self.in_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        
        self.convs = nn.ModuleList([
            RelationalGraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Bounding box regressor [xmin, ymin, xmax, ymax]
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 4),
            nn.Tanh() # normalized coordinates in [-1, 1]
        )

    def forward(
        self,
        room_types: torch.Tensor,
        adj_matrix: torch.Tensor,
        room_mask: torch.Tensor,
        boundary: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        room_types: (B, N)
        adj_matrix: (B, N, N)
        room_mask: (B, N)
        boundary: (B, 4)
        z: (B, N, noise_dim)
        Returns: (B, N, 4) predicted normalized bounding boxes
        """
        B, N = room_types.shape
        device = room_types.device

        if z is None:
            z = torch.randn(B, N, 32, device=device)

        r_feat = self.room_embed(room_types) # (B, N, H)
        z_feat = self.noise_proj(z)          # (B, N, H)

        if boundary is not None:
            b_feat = self.boundary_proj(boundary).unsqueeze(1).expand(-1, N, -1) # (B, N, H)
        else:
            b_feat = torch.zeros_like(r_feat)

        x = torch.cat([r_feat, z_feat, b_feat], dim=-1)
        h = self.in_proj(x)

        for conv in self.convs:
            h = conv(h, adj_matrix, room_mask)

        boxes = self.box_head(h)
        # Ensure xmin < xmax and ymin < ymax
        xmin = torch.min(boxes[..., 0], boxes[..., 2])
        xmax = torch.max(boxes[..., 0], boxes[..., 2]) + 0.05
        ymin = torch.min(boxes[..., 1], boxes[..., 3])
        ymax = torch.max(boxes[..., 1], boxes[..., 3]) + 0.05
        
        refined_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
        refined_boxes = torch.clamp(refined_boxes, -1.0, 1.0)
        
        return refined_boxes * room_mask.unsqueeze(-1).float()


class HouseGANppDiscriminator(nn.Module):
    """
    Relational Graph Discriminator for House-GAN++.
    Judges realism of room boxes and their relational compatibility with the bubble graph.
    """
    def __init__(
        self,
        num_room_types: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()
        self.room_embed = nn.Embedding(num_room_types, hidden_dim)
        self.box_proj = nn.Linear(4, hidden_dim)
        self.in_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.convs = nn.ModuleList([
            RelationalGraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        boxes: torch.Tensor,
        room_types: torch.Tensor,
        adj_matrix: torch.Tensor,
        room_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        boxes: (B, N, 4)
        room_types: (B, N)
        adj_matrix: (B, N, N)
        room_mask: (B, N)
        Returns: (B, 1) real/fake validity logits
        """
        r_feat = self.room_embed(room_types)
        b_feat = self.box_proj(boxes)
        h = self.in_proj(torch.cat([r_feat, b_feat], dim=-1))

        for conv in self.convs:
            h = conv(h, adj_matrix, room_mask)

        # Global graph pool
        mask_expanded = room_mask.unsqueeze(-1).float()
        pooled = torch.sum(h * mask_expanded, dim=1) / (torch.sum(mask_expanded, dim=1) + 1e-6)
        
        logits = self.out_head(pooled)
        return logits
