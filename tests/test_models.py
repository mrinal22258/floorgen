"""
Unit tests for FloorGen Generative Models (Stage 3).
"""

import torch
import pytest
from floorgen.models.gan_baseline.house_gan_pp import HouseGANppGenerator, HouseGANppDiscriminator
from floorgen.models.diffusion_core.house_diffusion import DDPMScheduler, HouseDiffusionDenoiser
from floorgen.models.diffusion_core.rag_diffusion import RAGFloorplanDiffusion
from floorgen.models.shared.representation import floorplan_geometric_loss


def test_house_gan_pp():
    B, N = 2, 8
    room_types = torch.randint(0, 10, (B, N))
    adj = torch.eye(N).unsqueeze(0).repeat(B, 1, 1)
    mask = torch.ones((B, N), dtype=torch.bool)
    boundary = torch.tensor([[-1.0, -1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]])

    netG = HouseGANppGenerator(hidden_dim=64, num_layers=2)
    netD = HouseGANppDiscriminator(hidden_dim=64, num_layers=2)

    fake_boxes = netG(room_types, adj, mask, boundary)
    assert fake_boxes.shape == (B, N, 4)

    logits = netD(fake_boxes, room_types, adj, mask)
    assert logits.shape == (B, 1)


def test_rag_diffusion_model():
    B, N = 2, 8
    room_types = torch.randint(0, 10, (B, N))
    adj = torch.eye(N).unsqueeze(0).repeat(B, 1, 1)
    mask = torch.ones((B, N), dtype=torch.bool)
    noisy_boxes = torch.randn((B, N, 4))
    timesteps = torch.tensor([100, 200])

    exemplars = torch.randn((B, 3, N, 4)) # K=3 exemplars

    model = RAGFloorplanDiffusion(hidden_dim=64, num_layers=2, num_heads=2)
    scheduler = DDPMScheduler(num_timesteps=100)

    # Forward prediction
    eps_pred = model(
        x_t=noisy_boxes,
        timesteps=timesteps,
        room_types=room_types,
        adj_matrix=adj,
        room_mask=mask,
        retrieved_exemplars=exemplars
    )
    assert eps_pred.shape == (B, N, 4)

    # Denoising sampling
    sampled_boxes = model.sample(
        scheduler=scheduler,
        room_types=room_types,
        adj_matrix=adj,
        room_mask=mask,
        retrieved_exemplars=exemplars,
        num_inference_steps=5
    )
    assert sampled_boxes.shape == (B, N, 4)
    assert torch.all(sampled_boxes >= -1.0) and torch.all(sampled_boxes <= 1.0)
