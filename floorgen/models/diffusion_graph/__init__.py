"""
FloorGen v2.0 Wall Graph Diffusion Package (GSDiff-inspired architecture):
- WallGraph: Geometric & topological graph representation of walls and junctions
- JunctionClassifier: L / T / X / End junction classification
- BipartiteGNN: Message passing between wall segments and junction nodes
- GSDiffStructuralLoss: Endpoint MSE, junction cross-entropy, Manhattan orthogonality
- WallGraphDiffusion: Joint generative diffusion over wall networks
"""

from .wall_graph import WallGraph, WallSegment, JunctionNode
from .junctions import JunctionClassifier, classify_junction
from .bipartite_gnn import BipartiteGNNLayer, BipartiteWallJunctionGNN
from .gsdiff_loss import GSDiffStructuralLoss
from .wall_graph_diffusion import WallGraphDiffusion

__all__ = [
    "WallGraph",
    "WallSegment",
    "JunctionNode",
    "JunctionClassifier",
    "classify_junction",
    "BipartiteGNNLayer",
    "BipartiteWallJunctionGNN",
    "GSDiffStructuralLoss",
    "WallGraphDiffusion"
]
