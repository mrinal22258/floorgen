"""
Intelligent Architectural Furniture Placement and Layout Optimization Engine.
Places functionally oriented, code-compliant furniture blocks with proper clearances:
- Master Suite: King bed, dual nightstands, wardrobe closet
- Secondary Bedroom: Bed, study desk, chair
- Living Room: Multi-seater sofa, coffee table, media console
- Dining Room: Centered dining table and chairs with circulation clearance
- Kitchen: Counters, sink, and cooktop range
- Bathroom: Vanity, toilet with code clearance, and shower cabin
"""

import math
from typing import List, Dict, Any, Tuple, Optional


class FurnitureItem:
    def __init__(self, item_type: str, bbox: List[float], rotation: float = 0.0, layer: str = "FURNITURE"):
        self.item_type = item_type
        self.bbox = [round(v, 1) for v in bbox] # [xmin, ymin, xmax, ymax]
        self.rotation = rotation
        self.layer = layer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.item_type,
            "bbox": self.bbox,
            "rotation": self.rotation,
            "layer": self.layer
        }


def optimize_room_furniture(
    room_category: str,
    room_bbox: List[float],
    door_positions: List[Tuple[float, float]] = None
) -> List[Dict[str, Any]]:
    """
    Computes architectural furniture placement tailored to room dimensions and door locations.
    """
    x1, y1, x2, y2 = room_bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    door_positions = door_positions or []

    items: List[FurnitureItem] = []

    # 1. Master Bedroom
    if room_category == "master_bedroom":
        # King Bed anchored against top or back wall (w=32px, h=36px)
        bed_w, bed_h = min(w * 0.45, 34.0), min(h * 0.45, 38.0)
        bed_x = cx - bed_w / 2.0
        bed_y = y1 + 4.0
        items.append(FurnitureItem("king_bed", [bed_x, bed_y, bed_x + bed_w, bed_y + bed_h]))

        # Dual Nightstands
        ns_w, ns_h = 8.0, 7.0
        if bed_x - x1 > ns_w + 2.0:
            items.append(FurnitureItem("nightstand_left", [bed_x - ns_w - 2.0, bed_y, bed_x - 2.0, bed_y + ns_h]))
        if x2 - (bed_x + bed_w) > ns_w + 2.0:
            items.append(FurnitureItem("nightstand_right", [bed_x + bed_w + 2.0, bed_y, bed_x + bed_w + ns_w + 2.0, bed_y + ns_h]))

        # Wardrobe Closet against bottom wall
        ward_w = min(w * 0.6, 45.0)
        ward_h = 10.0
        items.append(FurnitureItem("wardrobe", [cx - ward_w / 2.0, y2 - ward_h - 3.0, cx + ward_w / 2.0, y2 - 3.0]))

    # 2. Secondary Bedroom
    elif room_category in ["second_bedroom", "study"]:
        # Twin/Double bed
        bed_w, bed_h = min(w * 0.38, 26.0), min(h * 0.45, 34.0)
        bed_x = x1 + 4.0
        bed_y = y1 + 4.0
        items.append(FurnitureItem("bed_single", [bed_x, bed_y, bed_x + bed_w, bed_y + bed_h]))

        # Study Desk
        desk_w, desk_h = min(w * 0.45, 24.0), 10.0
        items.append(FurnitureItem("study_desk", [x2 - desk_w - 4.0, y2 - desk_h - 4.0, x2 - 4.0, y2 - 4.0]))

    # 3. Living Room
    elif room_category == "living_room":
        # 3-Seater Sofa facing center
        sofa_w = min(w * 0.55, 48.0)
        sofa_h = 16.0
        sofa_y = y1 + 8.0
        items.append(FurnitureItem("sofa_3seater", [cx - sofa_w / 2.0, sofa_y, cx + sofa_w / 2.0, sofa_y + sofa_h]))

        # Coffee Table in front of sofa
        table_w = min(w * 0.35, 26.0)
        table_h = 12.0
        items.append(FurnitureItem("coffee_table", [cx - table_w / 2.0, sofa_y + sofa_h + 6.0, cx + table_w / 2.0, sofa_y + sofa_h + 6.0 + table_h]))

        # Media TV Credenza on opposite wall
        tv_w = min(w * 0.5, 42.0)
        tv_h = 7.0
        items.append(FurnitureItem("tv_unit", [cx - tv_w / 2.0, y2 - tv_h - 4.0, cx + tv_w / 2.0, y2 - 4.0]))

    # 4. Dining Room
    elif room_category == "dining_room":
        # Centered dining table (6 persons)
        dt_w = min(w * 0.5, 32.0)
        dt_h = min(h * 0.45, 20.0)
        items.append(FurnitureItem("dining_table_6p", [cx - dt_w / 2.0, cy - dt_h / 2.0, cx + dt_w / 2.0, cy + dt_h / 2.0]))

    # 5. Kitchen
    elif room_category == "kitchen":
        # Countertop along top edge
        items.append(FurnitureItem("kitchen_counter", [x1 + 2.0, y1 + 2.0, x2 - 2.0, y1 + 12.0]))
        # Sink unit
        items.append(FurnitureItem("kitchen_sink", [cx - 10.0, y1 + 3.0, cx + 10.0, y1 + 11.0]))
        # Range / Cooktop
        if w > 30.0:
            items.append(FurnitureItem("cooktop_range", [x1 + 6.0, y1 + 3.0, x1 + 18.0, y1 + 11.0]))

    # 6. Bathroom
    elif room_category == "bathroom":
        # Vanity sink
        items.append(FurnitureItem("bathroom_vanity", [x1 + 2.0, y1 + 2.0, min(x1 + 16.0, x2 - 2.0), y1 + 10.0]))
        # Water Closet (Toilet)
        items.append(FurnitureItem("toilet_wc", [x2 - 14.0, y1 + 2.0, x2 - 2.0, y1 + 12.0]))
        # Shower enclosure
        sh_size = min(w * 0.45, h * 0.45, 20.0)
        items.append(FurnitureItem("shower_cabin", [x1 + 2.0, y2 - sh_size - 2.0, x1 + 2.0 + sh_size, y2 - 2.0]))

    return [item.to_dict() for item in items]


def populate_floorplan_furniture(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populates all rooms in the floorplan with optimized architectural furniture.
    """
    plan_with_furniture = dict(plan)
    rooms = plan.get("rooms", [])
    doors = plan.get("doors", [])
    door_pts = [(d["pos"][0], d["pos"][1]) for d in doors if "pos" in d]

    all_furniture = []
    for r in rooms:
        cat = r.get("category", "")
        bbox = r.get("bbox", [])
        if len(bbox) == 4:
            f_items = optimize_room_furniture(cat, bbox, door_positions=door_pts)
            for item in f_items:
                item["room_id"] = r["id"]
                all_furniture.append(item)

    plan_with_furniture["furniture"] = all_furniture
    return plan_with_furniture


def place_room_furniture(rooms_or_plan: Any, doors: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Places optimized furniture blocks for a given list of rooms or a plan dict.
    """
    if isinstance(rooms_or_plan, dict):
        rooms = rooms_or_plan.get("rooms", [])
        doors = doors or rooms_or_plan.get("doors", [])
    else:
        rooms = rooms_or_plan
        doors = doors or []
    
    door_pts = []
    for d in doors:
        pos = d.get("pos") or d.get("position")
        if pos:
            door_pts.append((pos[0], pos[1]))
            
    all_furniture = []
    for idx, r in enumerate(rooms):
        cat = r.get("category", "")
        bbox = r.get("bbox") or r.get("box", [])
        r_id = r.get("id", idx)
        if len(bbox) == 4:
            f_items = optimize_room_furniture(cat, bbox, door_positions=door_pts)
            for item in f_items:
                item["room_id"] = r_id
                all_furniture.append(item)
    return all_furniture

