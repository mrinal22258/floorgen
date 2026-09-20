"""
Unit tests for FLUX LoRA adapter and RPLAN dataset loader.
"""

import os
import pytest
import torch
from floorgen.models.diffusion_raster.flux_lora import (
    RPLANFluxDataset,
    FloorplanFluxLoRA,
    LightweightArchitecturalUNet
)


def test_flux_lora_model_forward():
    model = FloorplanFluxLoRA(use_flux_pipeline=False, device="cpu")
    latents = torch.randn(2, 64, 32, 32)
    t = torch.rand(2, 1)
    cond = torch.randn(2, 256)

    out = model(latents, t, cond)
    assert out.shape == (2, 64, 32, 32)
    assert not torch.isnan(out).any()


def test_flux_lora_generate_latents():
    model = FloorplanFluxLoRA(use_flux_pipeline=False, device="cpu")
    sampled = model.generate_latents(batch_size=1, steps=3, size=32)
    assert sampled.shape == (1, 64, 32, 32)
    assert not torch.isnan(sampled).any()


def test_rplan_flux_dataset_loader(tmp_path):
    # Create mock plan JSON
    import json
    plan_dir = tmp_path / "train"
    img_dir = tmp_path / "images"
    plan_dir.mkdir()
    img_dir.mkdir()

    plan = {
        "id": "mock_001",
        "rooms": [
            {"category": "living_room", "bbox": [10, 10, 100, 100]},
            {"category": "master_bedroom", "bbox": [100, 10, 200, 100]}
        ],
        "adjacency": [[0, 1]],
        "raster_path": "images/mock_001.png"
    }
    with open(plan_dir / "mock_001.json", "w") as f:
        json.dump(plan, f)

    dataset = RPLANFluxDataset(str(plan_dir), str(img_dir), target_size=128)
    assert len(dataset) == 1
    sample = dataset[0]
    assert "prompt" in sample
    assert "living room" in sample["prompt"]
    assert sample["image"].shape == (3, 128, 128)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not (os.environ.get("HF_TOKEN") or os.path.exists(".env")),
    reason="Requires CUDA GPU and HF_TOKEN configured in environment or .env"
)
def test_real_flux_pipeline_load():
    """
    Validates the real FLUX.1-dev pipeline path with allow_fallback=False.
    Asserts either successful pipeline load or the expected loud descriptive error.
    """
    # Test missing HF_TOKEN error condition
    token_saved = os.environ.pop("HF_TOKEN", None)
    if os.path.exists(".env"):
        os.rename(".env", ".env.tmp")
    
    try:
        with pytest.raises(RuntimeError, match="Missing HF_TOKEN"):
            FloorplanFluxLoRA(use_flux_pipeline=True, allow_fallback=False)
    finally:
        if os.path.exists(".env.tmp"):
            os.rename(".env.tmp", ".env")
        if token_saved:
            os.environ["HF_TOKEN"] = token_saved

    # Test initialization with valid token present and allow_fallback=False
    try:
        model = FloorplanFluxLoRA(use_flux_pipeline=True, allow_fallback=False)
        assert hasattr(model, "pipe")
        assert model.use_flux_pipeline is True
    except RuntimeError as err:
        # If model download is deferred or network limited, assert loud non-fallback error
        assert "FLUX pipeline initialization failed" in str(err)

