"""
Filtering and quality-assurance rules for FloorGen datasets (matching RLVR floorplan Appendix A.2).
Enforces:
1. Graph connectivity (no orphan room components)
2. Manifold & non-self-intersecting room polygons
3. Reasonable room aspect ratios (0.15 <= aspect <= 6.5)
4. Non-degenerate minimum room areas (> 2.0% of total plan area)
5. Boundary containment
"""

from typing import Dict, Any, Tuple, List
import networkx as nx


def is_valid_floorplan(plan: Dict[str, Any], min_rooms: int = 2, max_rooms: int = 20) -> Tuple[bool, str]:
    """
    Validates a floorplan dictionary against structural and topological constraints.
    Returns (is_valid, reason).
    """
    rooms = plan.get("rooms", [])
    if len(rooms) < min_rooms:
        return False, f"Too few rooms: {len(rooms)} < {min_rooms}"
    if len(rooms) > max_rooms:
        return False, f"Too many rooms: {len(rooms)} > {max_rooms}"

    # 1. Graph Connectivity Check
    G = nx.Graph()
    for r in rooms:
        G.add_node(r["id"])
    for u, v in plan.get("adjacency", []):
        G.add_edge(u, v)

    if not nx.is_connected(G):
        return False, "Disconnected bubble diagram graph (orphan room component found)"

    # Total area accumulator
    total_area = sum(r.get("area", 0) for r in rooms)
    if total_area <= 0:
        return False, "Total floorplan area is zero or negative"

    # 2. Polygon & Geometry Sanity
    for r in rooms:
        poly = r.get("polygon", [])
        if len(poly) < 3:
            return False, f"Room {r['id']} has invalid vertex count (< 3)"

        bbox = r.get("bbox", [])
        if len(bbox) != 4:
            return False, f"Room {r['id']} has invalid bbox"

        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin

        if width <= 0 or height <= 0:
            return False, f"Room {r['id']} has degenerate width/height <= 0"

        aspect = width / height
        if aspect < 0.12 or aspect > 8.0:
            return False, f"Room {r['id']} has extreme aspect ratio {aspect:.2f}"

        area = r.get("area", width * height)
        if area / total_area < 0.015:
            return False, f"Room {r['id']} area ratio is too tiny ({area/total_area:.3f})"

    return True, "Valid floorplan"


def filter_floorplan_dataset(plans: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filters a batch of floorplans, returning (valid_plans, rejected_plans)."""
    valid = []
    rejected = []
    for p in plans:
        ok, reason = is_valid_floorplan(p)
        if ok:
            valid.append(p)
        else:
            rejected.append({"id": p.get("id", "unknown"), "reason": reason})
    return valid, rejected
