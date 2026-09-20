"""
Unit tests for CP-SAT Furniture Layout Optimizer.
"""

import pytest
from floorgen.constraints.furniture_optimizer import optimize_room_furniture_cpsat


def test_furniture_optimizer_master_bedroom():
    # 50x40 pixel bedroom
    room_box = [0, 0, 80, 70]
    doors = [(80, 35)] # Door on right wall

    placed = optimize_room_furniture_cpsat("master_bedroom", room_box, door_positions=doors)
    assert len(placed) > 0
    types = [p["type"] for p in placed]
    assert "king_bed" in types

    # Check non-overlap between placed items
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            b1 = placed[i]["bbox"]
            b2 = placed[j]["bbox"]
            # Disjoint in at least one axis
            overlap_x = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            overlap_y = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            assert overlap_x == 0 or overlap_y == 0, f"Overlap detected between {placed[i]['type']} and {placed[j]['type']}"


def test_furniture_optimizer_living_room():
    room_box = [0, 0, 120, 90]
    placed = optimize_room_furniture_cpsat("living_room", room_box)
    assert len(placed) > 0
    types = [p["type"] for p in placed]
    assert "sofa_3seater" in types
