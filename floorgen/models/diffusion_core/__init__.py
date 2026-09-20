"""
Diffusion Core Module for FloorGen.
Includes Raster Diffusion (VQ-VAE preview) and Wall Graph Diffusion (GSDiff).
"""

from floorgen.models.diffusion_core.raster_diffusion import (
    VectorToRasterRenderer,
    VectorConditionedVQVAE,
    generate_raster_preview
)
from floorgen.models.diffusion_core.wall_graph_diffusion import (
    WallGraphDiffusion,
    compute_gsdiff_structural_loss
)

__all__ = [
    "VectorToRasterRenderer",
    "VectorConditionedVQVAE",
    "generate_raster_preview",
    "WallGraphDiffusion",
    "compute_gsdiff_structural_loss"
]
