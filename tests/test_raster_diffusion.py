"""
Unit tests for Raster Diffusion, Renderer, and VQ-VAE Floorplan Previews.
"""

import numpy as np
import torch
import pytest
from floorgen.models.diffusion_core.raster_diffusion import (
    VectorToRasterRenderer,
    VectorConditionedVQVAE,
    generate_raster_preview
)


def test_vector_to_raster_renderer():
    sample_plan = {
        "rooms": [
            {"category": "living_room", "box": [20, 20, 120, 100]},
            {"category": "master_bedroom", "box": [120, 20, 200, 100]},
            {"category": "bathroom", "box": [120, 100, 180, 150]}
        ],
        "doors": [{"pos": [120, 50]}],
        "boundary": [20, 20, 200, 150]
    }
    renderer = VectorToRasterRenderer(canvas_size=256)
    img = renderer.render(sample_plan)

    assert isinstance(img, np.ndarray)
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8
    # Should not be a blank white canvas
    assert np.any(img != 255)

    # Test convenience function
    preview = generate_raster_preview(sample_plan)
    assert preview.shape == (256, 256, 3)


def test_vqvae_forward_pass():
    vqvae = VectorConditionedVQVAE(in_channels=3, latent_dim=32, num_embeddings=64)
    dummy_img = torch.rand(2, 3, 256, 256)

    recon, quantized = vqvae(dummy_img)
    assert recon.shape == (2, 3, 256, 256)
    assert quantized.shape == (2, 32, 32, 32)
    assert not torch.isnan(recon).any()
