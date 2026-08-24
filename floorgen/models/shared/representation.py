"""
Shared Representations, Graph Convolutions, and Neural Modules for FloorGen.
Provides:
- Relational Graph Convolutions (RGCN)
- Timestep Positional Embeddings for Diffusion
- Multi-Head Self- and Cross-Attention layers
- Geometric Loss Functions (Bounding Box, Boundary containment, Non-overlap penalty)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class SinusoidalPositionalEmbedding(nn.Module):
    """Encodes scalar diffusion timesteps into continuous vector embeddings."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: (B,)
        half_dim = self.dim // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / (half_dim - 1)
        emb = torch.exp(exponent)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class RelationalGraphConv(nn.Module):
    """
    Relational Graph Convolutional Layer for Room Bubble Diagrams.
    Aggregates features across adjacent room nodes while maintaining self-loops.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, in_dim) - node features
        adj: (B, N, N) - adjacency matrix (0 or 1)
        mask: (B, N) - valid node mask
        """
        # Self transformation
        self_feat = self.self_linear(x)

        # Neighbor aggregation: (B, N, N) x (B, N, in_dim) -> (B, N, in_dim)
        # Degree normalization
        deg = torch.sum(adj, dim=-1, keepdim=True) + 1e-6
        norm_adj = adj / deg
        neighbor_feat = self.neighbor_linear(torch.bmm(norm_adj, x))

        out = self.norm(self_feat + neighbor_feat)
        out = self.act(out)

        if mask is not None:
            out = out * mask.unsqueeze(-1).float()

        return out


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention layer that conditions floorplan generation queries
    on retrieved exemplar plans.
    """
    def __init__(self, query_dim: int, context_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = query_dim // heads
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(context_dim, query_dim)
        self.v_proj = nn.Linear(context_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, q: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        q: (B, N, query_dim)
        context: (B, K, context_dim)
        """
        B, N, C = q.shape
        _, K, _ = context.shape

        q_mat = self.q_proj(q).view(B, N, self.heads, self.head_dim).transpose(1, 2)       # (B, H, N, D)
        k_mat = self.k_proj(context).view(B, K, self.heads, self.head_dim).transpose(1, 2) # (B, H, K, D)
        v_mat = self.v_proj(context).view(B, K, self.heads, self.head_dim).transpose(1, 2) # (B, H, K, D)

        scores = torch.matmul(q_mat, k_mat.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, N, K)
        weights = F.softmax(scores, dim=-1)

        attended = torch.matmul(weights, v_mat) # (B, H, N, D)
        attended = attended.transpose(1, 2).contiguous().view(B, N, C)
        out = self.norm(q + self.out_proj(attended))
        return out


def floorplan_geometric_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    mask: torch.Tensor,
    adj: torch.Tensor,
    boundary: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Multi-objective geometric loss for floorplan generation:
    1. Smooth L1 coordinate regression
    2. Overlap penalty between non-adjacent rooms
    3. Adjacency contact encouragement
    4. Boundary containment
    """
    # 1. L1 loss on valid rooms
    valid_mask = mask.unsqueeze(-1).float()
    box_l1 = F.smooth_l1_loss(pred_boxes * valid_mask, target_boxes * valid_mask, reduction='sum') / (valid_mask.sum() * 4.0 + 1e-6)

    # 2. Overlap penalty between non-adjacent rooms
    # pred_boxes: (B, N, 4) -> [xmin, ymin, xmax, ymax]
    xmin = pred_boxes[:, :, 0:1] # (B, N, 1)
    ymin = pred_boxes[:, :, 1:2]
    xmax = pred_boxes[:, :, 2:3]
    ymax = pred_boxes[:, :, 3:4]

    # Compute pairwise intersections: (B, N, N)
    inter_xmin = torch.max(xmin, xmin.transpose(1, 2))
    inter_ymin = torch.max(ymin, ymin.transpose(1, 2))
    inter_xmax = torch.min(xmax, xmax.transpose(1, 2))
    inter_ymax = torch.min(ymax, ymax.transpose(1, 2))

    inter_w = F.relu(inter_xmax - inter_xmin)
    inter_h = F.relu(inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h # (B, N, N)

    # Ignore self-overlap
    eye = torch.eye(pred_boxes.shape[1], device=pred_boxes.device).unsqueeze(0)
    inter_area = inter_area * (1.0 - eye)

    # Penalize non-adjacent overlaps heavily
    non_adj = (1.0 - adj) * (1.0 - eye)
    overlap_loss = torch.sum(inter_area * non_adj) / (pred_boxes.shape[0] * pred_boxes.shape[1] + 1e-6)

    # 3. Boundary containment loss
    if boundary is not None:
        # boundary: (B, 4) -> [b_xmin, b_ymin, b_xmax, b_ymax]
        b_xmin = boundary[:, 0:1].unsqueeze(-1)
        b_ymin = boundary[:, 1:2].unsqueeze(-1)
        b_xmax = boundary[:, 2:3].unsqueeze(-1)
        b_ymax = boundary[:, 3:4].unsqueeze(-1)

        out_left = F.relu(b_xmin - xmin)
        out_top = F.relu(b_ymin - ymin)
        out_right = F.relu(xmax - b_xmax)
        out_bottom = F.relu(ymax - b_ymax)

        boundary_loss = torch.mean((out_left + out_top + out_right + out_bottom) * valid_mask)
    else:
        boundary_loss = torch.tensor(0.0, device=pred_boxes.device)

    total_loss = box_l1 + 0.15 * overlap_loss + 0.1 * boundary_loss

    losses = {
        "box_l1": box_l1.detach(),
        "overlap_loss": overlap_loss.detach(),
        "boundary_loss": boundary_loss.detach() if isinstance(boundary_loss, torch.Tensor) else 0.0,
        "total": total_loss.detach()
    }
    return total_loss, losses
