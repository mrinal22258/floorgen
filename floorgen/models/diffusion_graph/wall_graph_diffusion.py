"""
WallGraphDiffusion: End-to-end learned structural wall graph diffusion.
Replaces post-hoc heuristic line tracing with continuous structural diffusion over:
  - Wall segment endpoints & thicknesses
  - Junction positions and topologies (L / T / X / End)
  - Bipartite incidence connectivity
"""

import math
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bipartite_gnn import BipartiteWallJunctionGNN
from .gsdiff_loss import GSDiffStructuralLoss
from .junctions import JUNCTION_CLASSES


class WallGraphDiffusion(nn.Module):
    """
    Joint generative diffusion model over wall segments and topological junctions.
    """
    def __init__(
        self,
        wall_in_dim: int = 6,        # (x1, y1, x2, y2, thickness, is_exterior)
        junction_in_dim: int = 3,    # (x, y, type_or_degree)
        cond_dim: int = 256,
        hidden_dim: int = 128,
        num_gnn_layers: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input encoders
        self.wall_encoder = nn.Sequential(
            nn.Linear(wall_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.junction_encoder = nn.Sequential(
            nn.Linear(junction_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Time step embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Conditioning projection
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # Bipartite structural message passing
        self.gnn = BipartiteWallJunctionGNN(
            wall_dim=hidden_dim,
            junction_dim=hidden_dim,
            num_layers=num_gnn_layers
        )

        # Denoising prediction heads
        self.wall_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)   # Predicted delta for (x1, y1, x2, y2)
        )
        self.junction_pos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)   # Predicted delta for (x, y)
        )
        self.junction_cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(JUNCTION_CLASSES))  # Logits for end, L, T, X
        )

        self.loss_fn = GSDiffStructuralLoss()

    def forward(
        self,
        walls: torch.Tensor,         # [N, 6]
        junctions: torch.Tensor,     # [J, 3]
        incidence: torch.Tensor,     # [2, E]
        t: torch.Tensor,             # [1] or scalar
        cond: Optional[torch.Tensor] = None  # [1, cond_dim]
    ) -> Dict[str, torch.Tensor]:
        if walls.ndim == 1:
            walls = walls.unsqueeze(0)
        if junctions.ndim == 1:
            junctions = junctions.unsqueeze(0)
        if t.ndim == 0:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)

        t_emb = self.time_mlp(t)  # [1, hidden_dim]

        # Embed components
        h_w = self.wall_encoder(walls) + t_emb
        h_j = self.junction_encoder(junctions) + t_emb

        if cond is not None:
            c_emb = self.cond_proj(cond)
            h_w = h_w + c_emb
            h_j = h_j + c_emb

        # Message passing across walls and junctions
        h_w, h_j = self.gnn(h_w, h_j, incidence)

        # Predictions
        pred_wall_delta = self.wall_head(h_w)
        pred_junction_pos_delta = self.junction_pos_head(h_j)
        pred_junction_logits = self.junction_cls_head(h_j)

        return {
            "pred_wall_delta": pred_wall_delta,
            "pred_junction_pos_delta": pred_junction_pos_delta,
            "pred_junction_logits": pred_junction_logits,
            "h_walls": h_w,
            "h_junctions": h_j
        }

    @torch.no_grad()
    def sample(
        self,
        initial_walls: torch.Tensor,        # [N, 6]
        initial_junctions: torch.Tensor,    # [J, 3]
        incidence: torch.Tensor,            # [2, E]
        cond: Optional[torch.Tensor] = None,
        steps: int = 15
    ) -> Dict[str, Any]:
        """
        Denoising sample loop to generate refined, CAD-orthogonal wall coordinates.
        """
        self.eval()
        cur_walls = initial_walls.clone()
        cur_junctions = initial_junctions.clone()

        for step in reversed(range(steps)):
            t = torch.tensor([[step / float(steps)]], device=initial_walls.device)
            out = self.forward(cur_walls, cur_junctions, incidence, t, cond)

            dt = 1.0 / steps
            # Update endpoints
            cur_walls[:, :4] = cur_walls[:, :4] - dt * out["pred_wall_delta"]
            # Update junction positions
            cur_junctions[:, :2] = cur_junctions[:, :2] - dt * out["pred_junction_pos_delta"]

        # Manhattan orthogonal snapping on final output
        snapped_walls = cur_walls[:, :4].clone()
        for i in range(snapped_walls.size(0)):
            x1, y1, x2, y2 = snapped_walls[i].tolist()
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx > dy and dy < 15.0:
                snapped_walls[i, 3] = snapped_walls[i, 1]  # Horizontal
            elif dy > dx and dx < 15.0:
                snapped_walls[i, 2] = snapped_walls[i, 0]  # Vertical

        pred_j_types = torch.argmax(out["pred_junction_logits"], dim=-1)

        return {
            "walls": snapped_walls.cpu().numpy(),
            "wall_thicknesses": cur_walls[:, 4].cpu().numpy() * 200.0,
            "is_exterior": (cur_walls[:, 5] > 0.5).cpu().numpy(),
            "junction_positions": cur_junctions[:, :2].cpu().numpy(),
            "junction_types": [JUNCTION_CLASSES[idx] for idx in pred_j_types.tolist()]
        }
