"""
Unit tests for FloorplanVQVAE and RasterDecoder.
"""

import pytest
import torch
from floorgen.models.diffusion_raster.vqvae import FloorplanVQVAE, VectorQuantizer
from floorgen.models.diffusion_raster.raster_decoder import RasterDecoder


def test_vector_quantizer():
    vq = VectorQuantizer(num_embeddings=64, embedding_dim=16)
    inputs = torch.randn(2, 16, 8, 8)
    quantized, loss, indices = vq(inputs)
    assert quantized.shape == (2, 16, 8, 8)
    assert indices.shape == (2, 8, 8)
    assert loss.item() >= 0.0


def test_floorplan_vqvae_reconstruction():
    model = FloorplanVQVAE(embedding_dim=32, num_embeddings=128)
    img = torch.randn(1, 3, 256, 256)
    res = model(img)
    assert "loss" in res
    assert "reconstruction" in res
    assert res["reconstruction"].shape == (1, 3, 256, 256)
    assert res["indices"].shape == (1, 32, 32)


def test_raster_decoder_upsample_512():
    decoder = RasterDecoder(in_channels=3, out_channels=3)
    img_256 = torch.randn(1, 3, 256, 256)
    out = decoder(img_256)
    if isinstance(out, tuple):
        texture, edges = out
        assert texture.shape == (1, 3, 512, 512)
        assert edges.shape == (1, 1, 512, 512)
    else:
        assert out.shape == (1, 3, 512, 512)
