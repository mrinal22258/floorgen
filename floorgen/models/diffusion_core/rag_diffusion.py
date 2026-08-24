"""
FloorGen RAG-Conditioned Vector Floorplan Diffusion Core.
Conditions the continuous diffusion denoising process on top-k retrieved real floorplan exemplars
via cross-attention over exemplar geometry and structural graph embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from floorgen.models.shared.representation import (
    SinusoidalPositionalEmbedding,
    RelationalGraphConv,
    CrossAttentionLayer
)
from floorgen.models.diffusion_core.house_diffusion import DDPMScheduler


class RAGFloorplanDiffusion(nn.Module):
    """
    RAG-Conditioned Floorplan Diffusion Model.
    Accepts:
      - Current noisy coordinates x_t
      - User bubble diagram & room types
      - Context embeddings from top-k retrieved exemplars
    """

    def __init__(
        self,
        coord_dim: int = 4,
        num_room_types: int = 10,
        hidden_dim: int = 192,
        context_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Node encoders
        self.room_embed = nn.Embedding(num_room_types, hidden_dim)
        self.coord_proj = nn.Linear(coord_dim, hidden_dim)
        self.boundary_proj = nn.Linear(4, hidden_dim)

        # Exemplar context encoder (processes retrieved k floorplan bounding boxes + graphs)
        self.exemplar_encoder = nn.Sequential(
            nn.Linear(4, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, hidden_dim)
        )

        self.in_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        # Interleaved Relational Graph Convolutions and Cross-Attention to retrieved context
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "rgcn": RelationalGraphConv(hidden_dim, hidden_dim),
                "cross_attn": CrossAttentionLayer(hidden_dim, hidden_dim, heads=num_heads),
                "norm": nn.LayerNorm(hidden_dim)
            }))

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
        boundary: Optional[torch.Tensor] = None,
        retrieved_exemplars: Optional[torch.Tensor] = None # (B, K, N_exemplar, 4) or (B, K_tokens, hidden_dim)
    ) -> torch.Tensor:
        """
        x_t: (B, N, coord_dim)
        timesteps: (B,)
        room_types: (B, N)
        adj_matrix: (B, N, N)
        room_mask: (B, N)
        retrieved_exemplars: Optional (B, K_tokens, 4) tensor of retrieved room boxes
        """
        B, N, _ = x_t.shape
        device = x_t.device

        t_feat = self.time_embed(timesteps).unsqueeze(1).expand(-1, N, -1)
        r_feat = self.room_embed(room_types)
        c_feat = self.coord_proj(x_t)

        h = self.in_proj(torch.cat([t_feat + c_feat, r_feat, t_feat], dim=-1))

        if boundary is not None:
            b_feat = self.boundary_proj(boundary).unsqueeze(1).expand(-1, N, -1)
            h = h + b_feat

        # Process retrieved context tokens
        if retrieved_exemplars is not None and retrieved_exemplars.shape[1] > 0:
            if retrieved_exemplars.dim() == 4:
                # (B, K, N_ex, 4) -> flatten to (B, K * N_ex, 4)
                B_e, K_e, Ne_e, C_e = retrieved_exemplars.shape
                context_flat = retrieved_exemplars.view(B_e, K_e * Ne_e, C_e)
                context_tokens = self.exemplar_encoder(context_flat)
            else:
                context_tokens = self.exemplar_encoder(retrieved_exemplars)
        else:
            # Fallback zero context (k=0 ablation mode)
            context_tokens = torch.zeros((B, 1, self.hidden_dim), device=device)

        # Iterative message passing & cross-attention
        for layer in self.layers:
            # 1. Structural graph message passing
            h_graph = layer["rgcn"](h, adj_matrix, room_mask)
            
            # 2. Cross-attention to retrieved floorplan exemplars
            h_cross = layer["cross_attn"](h_graph, context_tokens)
            
            # 3. Residual & norm
            h = layer["norm"](h + h_cross)

        eps_pred = self.out_head(h)
        return eps_pred * room_mask.unsqueeze(-1).float()

    @torch.no_grad()
    def sample(
        self,
        scheduler: DDPMScheduler,
        room_types: torch.Tensor,
        adj_matrix: torch.Tensor,
        room_mask: torch.Tensor,
        boundary: Optional[torch.Tensor] = None,
        retrieved_exemplars: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50
    ) -> torch.Tensor:
        """
        Runs reverse diffusion sampling starting from standard normal Gaussian noise.
        """
        B, N = room_types.shape
        device = room_types.device
        x = torch.randn((B, N, self.coord_dim), device=device)

        # Step stride for fast DDIM/DDPM inference
        step_stride = max(1, scheduler.num_timesteps // num_inference_steps)
        timesteps = list(range(0, scheduler.num_timesteps, step_stride))[::-1]

        for t_val in timesteps:
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
            eps_pred = self.forward(
                x_t=x,
                timesteps=t_tensor,
                room_types=room_types,
                adj_matrix=adj_matrix,
                room_mask=room_mask,
                boundary=boundary,
                retrieved_exemplars=retrieved_exemplars
            )
            x = scheduler.step(eps_pred, t_val, x)

        # Ensure canonical bounding box coordinates [xmin, ymin, xmax, ymax]
        xmin = torch.min(x[..., 0], x[..., 2])
        xmax = torch.max(x[..., 0], x[..., 2]) + 0.05
        ymin = torch.min(x[..., 1], x[..., 3])
        ymax = torch.max(x[..., 1], x[..., 3]) + 0.05

        out_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
        out_boxes = torch.clamp(out_boxes, -1.0, 1.0)
        return out_boxes * room_mask.unsqueeze(-1).float()
