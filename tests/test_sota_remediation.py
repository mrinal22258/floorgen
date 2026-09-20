import os
import pytest
import torch
from floorgen.models.diffusion_raster.raster_decoder import ArchitecturalRasterDecoder
from floorgen.models.diffusion_raster.flux_lora import FloorplanFluxLoRA
from floorgen.postprocess.vectorize import export_svg
from floorgen.postprocess.export_ifc import export_ifc


def test_raster_decoder_strict_load():
    """Verify that ArchitecturalRasterDecoder loads strictly without silently dropping keys."""
    ckpt_path = "checkpoints/sota/raster_decoder.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("Raster decoder checkpoint not present")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model = ArchitecturalRasterDecoder(in_channels=64, out_channels=3)
    # Must load strictly without MissingKey / UnexpectedKey errors
    incompatible = model.load_state_dict(state_dict, strict=True)
    assert len(incompatible.missing_keys) == 0
    assert len(incompatible.unexpected_keys) == 0


def test_flux_prohibits_silent_fallback():
    """Assert that requesting the real FLUX pipeline fails loudly when unconfigured, without silent fallback."""
    with pytest.raises(Exception):
        # Without allow_fallback=True, unauthenticated/missing deps must raise
        FloorplanFluxLoRA(use_flux_pipeline=True, allow_fallback=False)


def test_svg_contains_room_labels():
    """Verify that exported SVG contains <text> elements with room category names."""
    sample = {
        "rooms": [
            {"id": 0, "category": "living_room", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]], "centroid": [50, 50], "area": 100.0}
        ]
    }
    svg = export_svg(sample)
    assert "<text" in svg
    assert "Living Room" in svg


def test_ifc_contains_walls_and_doors():
    """Verify that exported IFC contains IFCWALL and IFCDOOR entities."""
    sample = {
        "rooms": [
            {"id": 0, "category": "living_room", "bbox": [0, 0, 100, 100]}
        ]
    }
    ifc = export_ifc(sample)
    assert "IFCWALL" in ifc
    assert "IFCDOOR" in ifc
