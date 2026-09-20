"""
Junction classification and geometric analysis for architectural wall graphs.
Classifies junctions into End (degree 1), L-corner (degree 2, ~90°), T-junction (degree 3), and X-cross (degree 4).
"""

import math
from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


JUNCTION_CLASSES = ["end", "L", "T", "X"]
JUNCTION_TO_ID = {c: i for i, c in enumerate(JUNCTION_CLASSES)}
ID_TO_JUNCTION = {i: c for i, c in enumerate(JUNCTION_CLASSES)}


def classify_junction_from_angles(angles_rad: List[float]) -> str:
    """
    Geometric junction classification based on incident wall ray angles.
    """
    deg = len(angles_rad)
    if deg <= 1:
        return "end"
    if deg == 2:
        # Check angle between two walls
        diff = abs(angles_rad[0] - angles_rad[1])
        diff = min(diff, 2 * math.pi - diff)
        diff_deg = math.degrees(diff)
        if 70.0 <= diff_deg <= 110.0:
            return "L"
        elif diff_deg >= 160.0:
            return "L"  # Collinear joint, treated as continuous/L
        return "L"
    if deg == 3:
        return "T"
    return "X"


def classify_junction(incident_wall_segments: List[Tuple[float, float, float, float]], junction_pos: Tuple[float, float]) -> str:
    """
    Classifies a junction given its position (jx, jy) and a list of incident wall endpoints (x1, y1, x2, y2).
    """
    jx, jy = junction_pos
    angles = []
    for (x1, y1, x2, y2) in incident_wall_segments:
        # Determine which endpoint is away from junction
        d1 = math.hypot(x1 - jx, y1 - jy)
        d2 = math.hypot(x2 - jx, y2 - jy)
        other_x, other_y = (x2, y2) if d1 < d2 else (x1, y1)
        angle = math.atan2(other_y - jy, other_x - jx)
        angles.append(angle)
    return classify_junction_from_angles(angles)


class JunctionClassifier(nn.Module):
    """
    Neural classifier predicting junction type logits [end, L, T, X] from junction embeddings and neighbor features.
    """
    def __init__(self, in_features: int = 128, num_classes: int = 4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features] -> [B, 4]
        return self.classifier(x)
