"""
GSDiff Structural Losses for Wall Graph Diffusion.
Implements:
1. Endpoint MSE (wall line coordinate alignment)
2. Junction Classification Cross-Entropy (L / T / X / End)
3. Manhattan Orthogonality Loss (|dx * dy| = 0 enforcement)
4. Wall-Junction Incidence & Connectivity Consistency (zero floating walls)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class GSDiffStructuralLoss(nn.Module):
    def __init__(
        self,
        weight_endpoint: float = 1.0,
        weight_junction: float = 0.5,
        weight_manhattan: float = 0.8,
        weight_incidence: float = 0.6
    ):
        super().__init__()
        self.weight_endpoint = weight_endpoint
        self.weight_junction = weight_junction
        self.weight_manhattan = weight_manhattan
        self.weight_incidence = weight_incidence

    def forward(
        self,
        pred_walls: torch.Tensor,        # [N, 4] -> (x1, y1, x2, y2)
        target_walls: torch.Tensor,      # [N, 4]
        pred_junction_logits: torch.Tensor,  # [J, 4]
        target_junction_types: torch.Tensor, # [J]
        pred_junction_pos: torch.Tensor,     # [J, 2]
        incidence: torch.Tensor              # [2, E]
    ) -> Dict[str, torch.Tensor]:
        # 1. Endpoint MSE
        loss_endpoint = F.mse_loss(pred_walls, target_walls)

        # 2. Junction Classification Cross-Entropy
        if pred_junction_logits.size(0) > 0 and target_junction_types.size(0) > 0:
            loss_junction = F.cross_entropy(pred_junction_logits, target_junction_types.long())
        else:
            loss_junction = torch.tensor(0.0, device=pred_walls.device)

        # 3. Manhattan Orthogonality Loss: penalty if walls are non-axial (dx * dy != 0)
        dx = pred_walls[:, 2] - pred_walls[:, 0]
        dy = pred_walls[:, 3] - pred_walls[:, 1]
        length_sq = dx ** 2 + dy ** 2 + 1e-6
        ortho_penalty = (torch.abs(dx * dy) / length_sq).mean()
        loss_manhattan = ortho_penalty

        # 4. Incidence / Connectivity Consistency:
        # Each incident wall endpoint must coincide with its connected junction
        if incidence.numel() > 0 and incidence.size(1) > 0:
            w_idx = incidence[0]
            j_idx = incidence[1]
            
            w_pts1 = pred_walls[w_idx, :2]    # (x1, y1)
            w_pts2 = pred_walls[w_idx, 2:4]   # (x2, y2)
            j_pts = pred_junction_pos[j_idx]  # (jx, jy)

            dist1 = torch.sum((w_pts1 - j_pts) ** 2, dim=-1)
            dist2 = torch.sum((w_pts2 - j_pts) ** 2, dim=-1)
            min_dist = torch.min(dist1, dist2)
            loss_incidence = min_dist.mean()
        else:
            loss_incidence = torch.tensor(0.0, device=pred_walls.device)

        total_loss = (
            self.weight_endpoint * loss_endpoint
            + self.weight_junction * loss_junction
            + self.weight_manhattan * loss_manhattan
            + self.weight_incidence * loss_incidence
        )

        return {
            "loss": total_loss,
            "loss_endpoint": loss_endpoint,
            "loss_junction": loss_junction,
            "loss_manhattan": loss_manhattan,
            "loss_incidence": loss_incidence
        }
