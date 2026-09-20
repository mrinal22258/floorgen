"""
Geometric and topological representation of architectural wall networks.
Supports WallSegment, JunctionNode, and bipartite graph mapping to IFC / DXF.
"""

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Optional
import torch


@dataclass
class WallSegment:
    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float = 100.0   # 200.0 for exterior, 100.0 for interior (mm)
    is_exterior: bool = False

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def is_horizontal(self) -> bool:
        return abs(self.y2 - self.y1) < abs(self.x2 - self.x1)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.x1, self.y1, self.x2, self.y2,
            self.thickness / 200.0,
            1.0 if self.is_exterior else 0.0
        ], dtype=torch.float32)


@dataclass
class JunctionNode:
    id: int
    x: float
    y: float
    junction_type: str = "L"  # 'end', 'L', 'T', 'X'
    degree: int = 2

    TYPE_TO_ID = {"end": 0, "L": 1, "T": 2, "X": 3}
    ID_TO_TYPE = {0: "end", 1: "L", 2: "T", 3: "X"}

    def to_tensor(self) -> torch.Tensor:
        t_id = self.TYPE_TO_ID.get(self.junction_type, 1)
        return torch.tensor([self.x, self.y, float(t_id)], dtype=torch.float32)


class WallGraph:
    """
    Topological wall-junction bipartite graph.
    Walls connect junctions; junctions connect wall segments.
    """
    def __init__(self):
        self.walls: List[WallSegment] = []
        self.junctions: List[JunctionNode] = []
        # Incidence: list of (wall_idx, junction_idx)
        self.incidence: List[Tuple[int, int]] = []

    def add_wall(self, x1: float, y1: float, x2: float, y2: float, thickness: float = 100.0, is_exterior: bool = False) -> int:
        w_id = len(self.walls)
        wall = WallSegment(
            id=w_id,
            x1=float(x1), y1=float(y1),
            x2=float(x2), y2=float(y2),
            thickness=float(thickness),
            is_exterior=bool(is_exterior)
        )
        self.walls.append(wall)
        return w_id

    def add_junction(self, x: float, y: float, junction_type: str = "L", degree: int = 2) -> int:
        j_id = len(self.junctions)
        junction = JunctionNode(
            id=j_id,
            x=float(x), y=float(y),
            junction_type=junction_type,
            degree=degree
        )
        self.junctions.append(junction)
        return j_id

    def add_incidence(self, wall_id: int, junction_id: int):
        self.incidence.append((wall_id, junction_id))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "walls": [asdict(w) for w in self.walls],
            "junctions": [asdict(j) for j in self.junctions],
            "incidence": self.incidence
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WallGraph":
        wg = cls()
        for w in d.get("walls", []):
            wg.add_wall(
                x1=w["x1"], y1=w["y1"],
                x2=w["x2"], y2=w["y2"],
                thickness=w.get("thickness", 100.0),
                is_exterior=w.get("is_exterior", False)
            )
        for j in d.get("junctions", []):
            wg.add_junction(
                x=j["x"], y=j["y"],
                junction_type=j.get("junction_type", "L"),
                degree=j.get("degree", 2)
            )
        for inc in d.get("incidence", []):
            wg.add_incidence(inc[0], inc[1])
        return wg

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          - wall_tensor: [N, 6]
          - junction_tensor: [J, 3]
          - incidence_tensor: [2, E]
        """
        if len(self.walls) == 0:
            wall_tensor = torch.zeros((0, 6), dtype=torch.float32)
        else:
            wall_tensor = torch.stack([w.to_tensor() for w in self.walls])

        if len(self.junctions) == 0:
            junction_tensor = torch.zeros((0, 3), dtype=torch.float32)
        else:
            junction_tensor = torch.stack([j.to_tensor() for j in self.junctions])

        if len(self.incidence) == 0:
            incidence_tensor = torch.zeros((2, 0), dtype=torch.long)
        else:
            incidence_tensor = torch.tensor(self.incidence, dtype=torch.long).t()

        return wall_tensor, junction_tensor, incidence_tensor
