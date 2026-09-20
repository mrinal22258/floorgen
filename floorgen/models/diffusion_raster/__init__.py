"""
FloorGen v2.0 Diffusion Raster Package:
- FloorplanVQVAE: Discrete latent compression for floorplan rasters
- FloorplanFluxLoRA: Text/geometry conditioned LoRA generation
- RasterDecoder: Super-resolution architectural raster decoding (512x512)
"""

from .vqvae import FloorplanVQVAE, VectorQuantizer
from .flux_lora import FloorplanFluxLoRA, RPLANFluxDataset
from .raster_decoder import RasterDecoder

__all__ = [
    "FloorplanVQVAE",
    "VectorQuantizer",
    "FloorplanFluxLoRA",
    "RPLANFluxDataset",
    "RasterDecoder"
]
