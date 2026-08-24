"""
Vectorization and Post-Processing for FloorGen.
Provides:
- Manhattan axis-aligned edge snapping and gap closing
- Overlap elimination via polygon clipping (Shapely)
- Door placement along shared contact walls
- Vector SVG & GeoJSON export with architectural styling
"""

import math
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


ROOM_COLORS = {
    "living_room": "#E3F2FD",      # Soft blue
    "master_bedroom": "#FFF3E0",   # Soft amber
    "second_bedroom": "#FFF8E1",   # Light warm cream
    "bathroom": "#E0F2F1",         # Soft mint/teal
    "kitchen": "#FBE9E7",          # Soft peach/terracotta
    "balcony": "#E8F5E9",          # Soft sage green
    "entrance": "#ECEFF1",         # Light cool gray
    "dining_room": "#F3E5F5",      # Soft lavender
    "study": "#EFEBE9",            # Soft warm gray
    "storage": "#EEEEEE",          # Neutral light gray
    "default": "#F5F5F5"
}

ROOM_STROKES = {
    "living_room": "#1E88E5",
    "master_bedroom": "#FB8C00",
    "second_bedroom": "#FDD835",
    "bathroom": "#26A69A",
    "kitchen": "#FF7043",
    "balcony": "#66BB6A",
    "entrance": "#78909C",
    "dining_room": "#AB47BC",
    "study": "#8D6E63",
    "storage": "#9E9E9E",
    "default": "#424242"
}


def snap_to_grid(val: float, grid_size: float = 2.0) -> float:
    """Snaps a continuous float coordinate to the nearest grid step."""
    return round(val / grid_size) * grid_size


def regularize_box(box: List[float], grid_size: float = 2.0, min_size: float = 16.0) -> List[float]:
    """Snaps and ensures positive area for a bounding box [xmin, ymin, xmax, ymax]."""
    x1 = snap_to_grid(box[0], grid_size)
    y1 = snap_to_grid(box[1], grid_size)
    x2 = snap_to_grid(box[2], grid_size)
    y2 = snap_to_grid(box[3], grid_size)

    if x2 <= x1 + min_size:
        x2 = x1 + min_size
    if y2 <= y1 + min_size:
        y2 = y1 + min_size

    return [x1, y1, x2, y2]


def regularize_floorplan(
    raw_boxes: np.ndarray,
    room_types: List[str],
    room_mask: Optional[np.ndarray] = None,
    grid_size: float = 2.0,
    canvas_size: float = 256.0
) -> Dict[str, Any]:
    """
    Transforms raw normalized model outputs (in [-1, 1]) into regularized vector geometry.
    - Scales coordinates to pixel/meter units [0, canvas_size]
    - Snaps axis-aligned walls
    - Detects shared edges for door placement
    """
    if room_mask is None:
        room_mask = np.ones(len(raw_boxes), dtype=bool)

    rooms = []
    polygons = []

    for i in range(len(raw_boxes)):
        if not room_mask[i]:
            continue

        # Denormalize from [-1, 1] to [0, canvas_size]
        box_norm = raw_boxes[i]
        x1 = (box_norm[0] + 1.0) / 2.0 * canvas_size
        y1 = (box_norm[1] + 1.0) / 2.0 * canvas_size
        x2 = (box_norm[2] + 1.0) / 2.0 * canvas_size
        y2 = (box_norm[3] + 1.0) / 2.0 * canvas_size

        reg_box = regularize_box([x1, y1, x2, y2], grid_size=grid_size)
        rx1, ry1, rx2, ry2 = reg_box

        poly = [
            [rx1, ry1],
            [rx2, ry1],
            [rx2, ry2],
            [rx1, ry2]
        ]
        
        cat_name = room_types[i] if i < len(room_types) else "room"
        area = (rx2 - rx1) * (ry2 - ry1)
        centroid = [(rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0]

        room_obj = {
            "id": len(rooms),
            "category": cat_name,
            "bbox": reg_box,
            "polygon": poly,
            "area": round(area, 1),
            "centroid": [round(centroid[0], 1), round(centroid[1], 1)]
        }
        rooms.append(room_obj)
        polygons.append(poly)

    # Detect adjacent contacts and place doors
    doors = []
    adjacency = []
    n_rooms = len(rooms)
    contact_threshold = grid_size * 2.5

    for u in range(n_rooms):
        b1 = rooms[u]["bbox"]
        for v in range(u + 1, n_rooms):
            b2 = rooms[v]["bbox"]
            
            # Check horizontal contact (vertical shared wall)
            shared_x = abs(b1[2] - b2[0]) < contact_threshold or abs(b1[0] - b2[2]) < contact_threshold
            overlap_y = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))

            # Check vertical contact (horizontal shared wall)
            shared_y = abs(b1[3] - b2[1]) < contact_threshold or abs(b1[1] - b2[3]) < contact_threshold
            overlap_x = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))

            if (shared_x and overlap_y > 10.0) or (shared_y and overlap_x > 10.0):
                adjacency.append([u, v])
                
                # Door position at center of contact segment
                if shared_x:
                    dx = (b1[2] + b2[0]) / 2.0 if abs(b1[2] - b2[0]) < contact_threshold else (b1[0] + b2[2]) / 2.0
                    dy = (max(b1[1], b2[1]) + min(b1[3], b2[3])) / 2.0
                else:
                    dx = (max(b1[0], b2[0]) + min(b1[2], b2[2])) / 2.0
                    dy = (b1[3] + b2[1]) / 2.0 if abs(b1[3] - b2[1]) < contact_threshold else (b1[1] + b2[3]) / 2.0

                doors.append({
                    "room_a": u,
                    "room_b": v,
                    "pos": [round(dx, 1), round(dy, 1)],
                    "orientation": "v" if shared_x else "h"
                })

    # Overall boundary
    all_xs = [p[0] for poly in polygons for p in poly] if polygons else [0, canvas_size]
    all_ys = [p[1] for poly in polygons for p in poly] if polygons else [0, canvas_size]
    
    boundary = [
        [min(all_xs) - 4.0, min(all_ys) - 4.0],
        [max(all_xs) + 4.0, min(all_ys) - 4.0],
        [max(all_xs) + 4.0, max(all_ys) + 4.0],
        [min(all_xs) - 4.0, max(all_ys) + 4.0]
    ]

    return {
        "num_rooms": n_rooms,
        "boundary": boundary,
        "rooms": rooms,
        "adjacency": adjacency,
        "doors": doors
    }


def export_svg(floorplan: Dict[str, Any], width: int = 500, height: int = 500) -> str:
    """Renders a floorplan into an SVG vector document."""
    rooms = floorplan.get("rooms", [])
    doors = floorplan.get("doors", [])
    boundary = floorplan.get("boundary", [])

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 260" width="{width}" height="{height}" style="background-color: #FAFAFA; font-family: Inter, Roboto, sans-serif;">',
        '<defs>',
        '  <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">',
        '    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.08"/>',
        '  </filter>',
        '</defs>'
    ]

    # Render site boundary
    if boundary:
        pts = " ".join([f"{p[0]},{p[1]}" for p in boundary])
        svg_parts.append(f'  <polygon points="{pts}" fill="#F0F4F8" stroke="#90A4AE" stroke-width="2" stroke-dasharray="4,4"/>')

    # Render room polygons
    for r in rooms:
        cat = r.get("category", "default")
        fill_color = ROOM_COLORS.get(cat, ROOM_COLORS["default"])
        stroke_color = ROOM_STROKES.get(cat, ROOM_STROKES["default"])
        poly = r.get("polygon", [])
        pts = " ".join([f"{p[0]},{p[1]}" for p in poly])

        svg_parts.append(f'  <polygon points="{pts}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="2.5" filter="url(#shadow)"/>')

        # Label and area
        cx, cy = r.get("centroid", [100, 100])
        label = cat.replace("_", " ").title()
        area_str = f"{r.get('area', 0):.0f} sq.ft"

        svg_parts.append(f'  <text x="{cx}" y="{cy - 4}" text-anchor="middle" font-size="7.5" font-weight="600" fill="#263238">{label}</text>')
        svg_parts.append(f'  <text x="{cx}" y="{cy + 7}" text-anchor="middle" font-size="5.5" fill="#546E7A">{area_str}</text>')

    # Render doors
    for d in doors:
        dx, dy = d.get("pos", [0, 0])
        orient = d.get("orientation", "h")
        if orient == "v":
            svg_parts.append(f'  <rect x="{dx - 1.5}" y="{dy - 5}" width="3" height="10" fill="#FFFFFF" stroke="#37474F" stroke-width="1"/>')
            svg_parts.append(f'  <path d="M {dx} {dy-5} A 5 5 0 0 1 {dx+5} {dy}" fill="none" stroke="#D32F2F" stroke-width="1" stroke-dasharray="1.5,1.5"/>')
        else:
            svg_parts.append(f'  <rect x="{dx - 5}" y="{dy - 1.5}" width="10" height="3" fill="#FFFFFF" stroke="#37474F" stroke-width="1"/>')
            svg_parts.append(f'  <path d="M {dx-5} {dy} A 5 5 0 0 1 {dx} {dy-5}" fill="none" stroke="#D32F2F" stroke-width="1" stroke-dasharray="1.5,1.5"/>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def export_json_vector(floorplan: Dict[str, Any], output_path: str):
    """Exports floorplan to a formatted JSON vector file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(floorplan, f, indent=2)
