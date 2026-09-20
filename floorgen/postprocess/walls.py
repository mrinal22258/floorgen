"""
Explicit Wall Topology and Structural Junction Engine for FloorGen.
Inspired by GSDiff (Hu et al., AAAI 2025) and architectural CAD standards.
Extracts explicit wall centerlines, parameterizes exterior (200mm) and interior (100mm) wall thicknesses,
and classifies corner junctions (L-junction, T-junction, X-junction).
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class WallSegment:
    def __init__(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        is_exterior: bool,
        thickness: float,
        rooms_adjacent: List[int]
    ):
        # Normalize direction: start always has smaller coordinate
        if (start[0] > end[0]) or (math.isclose(start[0], end[0], abs_tol=1e-3) and start[1] > end[1]):
            self.start = end
            self.end = start
        else:
            self.start = start
            self.end = end
        self.is_exterior = is_exterior
        self.thickness = thickness
        self.rooms_adjacent = rooms_adjacent
        self.length = math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])
        self.is_horizontal = math.isclose(self.start[1], self.end[1], abs_tol=1.0)
        self.is_vertical = math.isclose(self.start[0], self.end[0], abs_tol=1.0)

    @property
    def key(self) -> Tuple[float, float, float, float]:
        return (round(self.start[0], 1), round(self.start[1], 1), round(self.end[0], 1), round(self.end[1], 1))

    def to_polygon(self) -> List[Tuple[float, float]]:
        """Constructs an oriented thick wall polygon from the centerline."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        length = math.hypot(dx, dy)
        if length < 1e-4:
            return []
        
        nx = -dy / length * (self.thickness / 2.0)
        ny = dx / length * (self.thickness / 2.0)

        p1 = (round(self.start[0] + nx, 2), round(self.start[1] + ny, 2))
        p2 = (round(self.end[0] + nx, 2), round(self.end[1] + ny, 2))
        p3 = (round(self.end[0] - nx, 2), round(self.end[1] - ny, 2))
        p4 = (round(self.start[0] - nx, 2), round(self.start[1] - ny, 2))
        return [p1, p2, p3, p4]


class WallTopologyGraph:
    """
    Extracts, merges, and classifies architectural wall networks from floorplan layouts.
    """
    def __init__(
        self,
        exterior_thickness: float = 6.0, # in pixel units (~200mm at 16 px/m)
        interior_thickness: float = 3.2  # in pixel units (~100mm at 16 px/m)
    ):
        self.ext_thick = exterior_thickness
        self.int_thick = interior_thickness

    def extract_wall_network(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts all wall segments, computes exterior perimeter, and classifies junctions.
        """
        rooms = plan.get("rooms", [])
        boundary = plan.get("boundary", [])
        raw_segments: Dict[Tuple, WallSegment] = {}

        # 1. Collect all boundary edges from room boxes
        for idx, r in enumerate(rooms):
            b = r.get("bbox") or r.get("box")
            if not b or len(b) < 4:
                continue
            r_id = r.get("id", idx)
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            edges = [
                ((x1, y1), (x2, y1)), # Top
                ((x2, y1), (x2, y2)), # Right
                ((x1, y2), (x2, y2)), # Bottom
                ((x1, y1), (x1, y2))  # Left
            ]
            for start, end in edges:
                seg = WallSegment(start, end, is_exterior=False, thickness=self.int_thick, rooms_adjacent=[r_id])
                if seg.key in raw_segments:
                    existing = raw_segments[seg.key]
                    existing.rooms_adjacent.append(r_id)
                    existing.is_exterior = False
                    existing.thickness = self.int_thick
                else:
                    raw_segments[seg.key] = seg

        # 2. Determine exterior facade walls (segments with only 1 adjacent room)
        for seg in raw_segments.values():
            if len(seg.rooms_adjacent) == 1:
                seg.is_exterior = True
                seg.thickness = self.ext_thick

        # 3. Classify junctions
        node_degrees: Dict[Tuple[float, float], int] = {}
        for seg in raw_segments.values():
            s_pt = (round(seg.start[0], 1), round(seg.start[1], 1))
            e_pt = (round(seg.end[0], 1), round(seg.end[1], 1))
            node_degrees[s_pt] = node_degrees.get(s_pt, 0) + 1
            node_degrees[e_pt] = node_degrees.get(e_pt, 0) + 1

        junctions = []
        for pt, deg in node_degrees.items():
            if deg == 2:
                jtype = "L_junction"
            elif deg == 3:
                jtype = "T_junction"
            elif deg >= 4:
                jtype = "X_junction"
            else:
                jtype = "terminal"
            junctions.append({
                "pos": [pt[0], pt[1]],
                "degree": deg,
                "type": jtype
            })

        # 4. Generate structured wall geometry output
        wall_dicts = []
        for seg in raw_segments.values():
            if seg.length < 5.0: # Filter microscopic edge artifacts
                continue
            wall_dicts.append({
                "start": [seg.start[0], seg.start[1]],
                "end": [seg.end[0], seg.end[1]],
                "length": round(seg.length, 1),
                "is_exterior": seg.is_exterior,
                "thickness": round(seg.thickness, 2),
                "adjacent_rooms": seg.rooms_adjacent,
                "polygon": seg.to_polygon()
            })

        return {
            "total_walls": len(wall_dicts),
            "exterior_walls": sum(1 for w in wall_dicts if w["is_exterior"]),
            "interior_walls": sum(1 for w in wall_dicts if not w["is_exterior"]),
            "junctions": junctions,
            "walls": wall_dicts
        }


def extract_wall_topology(
    rooms_or_plan: Any,
    exterior_thickness: float = 6.0,
    interior_thickness: float = 3.2
) -> Dict[str, Any]:
    """
    Convenience function to extract wall topology directly from either a room list or a plan dict.
    """
    if isinstance(rooms_or_plan, dict):
        plan = rooms_or_plan
    else:
        plan = {"rooms": rooms_or_plan}
    
    graph = WallTopologyGraph(exterior_thickness=exterior_thickness, interior_thickness=interior_thickness)
    return graph.extract_wall_network(plan)

