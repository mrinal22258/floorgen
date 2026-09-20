"""
Unit tests for GSDiff Geometry-Enhanced Wall Graph Diffusion.
"""

import torch
import pytest
from floorgen.models.diffusion_core.wall_graph_diffusion import (
    WallGraphDiffusion,
    compute_gsdiff_structural_loss
)


def test_wall_graph_diffusion_forward():
    model = WallGraphDiffusion(hidden_dim=64, num_layers=2)
    batch_size = 2
    n_walls = 8
    n_juncs = 6

    # Random noisy wall features [B, N_walls, 8]
    noisy_walls = torch.randn(batch_size, n_walls, 8)
    # Random junction features [B, N_juncs, 7]
    junc_features = torch.randn(batch_size, n_juncs, 7)
    # Bipartite incidence matrix [B, N_juncs, N_walls]
    incidence = torch.randint(0, 2, (batch_size, n_juncs, n_walls)).float()
    timesteps = torch.randint(0, 1000, (batch_size,))

    pred_noise = model(noisy_walls, junc_features, incidence, timesteps)
    assert pred_noise.shape == (batch_size, n_walls, 4)
    assert not torch.isnan(pred_noise).any()


def test_gsdiff_loss():
    pred_walls = torch.tensor([[[0.0, 0.0, 50.0, 0.0], [50.0, 0.0, 50.0, 40.0]]], requires_grad=True)
    target_walls = torch.tensor([[[0.0, 0.0, 50.0, 0.0], [50.0, 0.0, 50.0, 40.0]]])

    loss = compute_gsdiff_structural_loss(pred_walls, target_walls, alpha_ortho=0.1)
    assert loss.item() >= 0.0
    loss.backward()
    assert pred_walls.grad is not None
