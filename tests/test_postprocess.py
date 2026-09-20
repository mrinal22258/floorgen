"""
Unit tests for FloorGen Post-Processing and Vectorization (Stage 4).
"""

import numpy as np
import pytest
from floorgen.postprocess.vectorize import regularize_floorplan, export_svg


def test_regularize_and_svg_export():
    raw_boxes = np.array([
        [-0.8, -0.8, 0.0, 0.8],   # living room
        [0.0, -0.8, 0.8, 0.0],    # bedroom
        [0.0, 0.0, 0.8, 0.8]      # kitchen
    ])
    room_types = ["living_room", "master_bedroom", "kitchen"]
    mask = np.array([True, True, True])

    res = regularize_floorplan(raw_boxes, room_types, mask, grid_size=2.0)
    assert res["num_rooms"] == 3
    assert len(res["rooms"]) == 3
    assert len(res["boundary"]) == 4

    svg_str = export_svg(res)
    assert "<svg" in svg_str
    assert "</svg>" in svg_str
    assert "Living Room" in svg_str


def test_wall_topology_and_furniture():
    from floorgen.postprocess.walls import extract_wall_topology
    from floorgen.postprocess.furniture import place_room_furniture
    from floorgen.postprocess.export_ifc import export_ifc

    rooms = [
        {"name": "living_room", "category": "living_room", "box": [0, 0, 50, 40], "width": 50, "height": 40},
        {"name": "master_bedroom", "category": "master_bedroom", "box": [50, 0, 90, 40], "width": 40, "height": 40}
    ]
    doors = [{"from_room": "living_room", "to_room": "master_bedroom", "position": [50, 20]}]

    # Wall topology
    wall_data = extract_wall_topology(rooms, exterior_thickness=0.2, interior_thickness=0.1)
    assert "walls" in wall_data
    assert len(wall_data["walls"]) > 0
    assert any(w["is_exterior"] for w in wall_data["walls"])

    # Furniture placement
    furniture = place_room_furniture(rooms)
    assert len(furniture) > 0
    categories = [f["type"] for f in furniture]
    assert "sofa" in categories or "king_bed" in categories

    # IFC BIM Export
    plan_dict = {
        "rooms": rooms,
        "boundary": [0, 0, 90, 40],
        "doors": doors,
        "windows": [],
        "walls": wall_data,
        "furniture": furniture
    }
    ifc_str = export_ifc(plan_dict, project_name="UnitTestHouse")
    assert "ISO-10303-21;" in ifc_str
    assert "IFCWALLSTANDARDCASE" in ifc_str
    assert "IFCSPACE" in ifc_str

