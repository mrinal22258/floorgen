"""
Unit tests for FloorGen Building Code Compliance (IRC/IBC/ADA) and Brief Parser.
"""

import pytest
from floorgen.constraints.compliance import validate_building_code_compliance
from floorgen.rag.brief_parser import parse_architectural_brief


def test_building_code_compliance_valid():
    rooms = [
        {"name": "living_room", "category": "living_room", "box": [0, 0, 60, 50], "width": 60, "height": 50},
        {"name": "master_bedroom", "category": "master_bedroom", "box": [60, 0, 120, 50], "width": 60, "height": 50},
        {"name": "bathroom", "category": "bathroom", "box": [120, 0, 150, 30], "width": 30, "height": 30},
        {"name": "kitchen", "category": "kitchen", "box": [0, 50, 50, 90], "width": 50, "height": 40}
    ]
    doors = [
        {"from_room": "living_room", "to_room": "master_bedroom", "position": [60, 25]},
        {"from_room": "master_bedroom", "to_room": "bathroom", "position": [120, 15]},
        {"from_room": "living_room", "to_room": "kitchen", "position": [25, 50]}
    ]


    plan = {"rooms": rooms, "doors": doors, "windows": []}
    report = validate_building_code_compliance(plan)
    assert report["passed"] is True
    assert report["compliance_score"] >= 0.8
    assert report["total_rooms_inspected"] == 4
    assert len(report["checks"]) > 0


def test_building_code_compliance_violations():
    # Intentionally sub-code room (tiny bedroom < 70 sq ft)
    rooms = [
        {"name": "master_bedroom", "category": "master_bedroom", "box": [0, 0, 10, 10], "width": 10, "height": 10}
    ]
    plan = {"rooms": rooms, "doors": [], "windows": []}
    report = validate_building_code_compliance(plan)
    assert report["passed"] is False
    assert len(report["violations"]) > 0
    assert any("Habitable Room Area" in v["rule"] for v in report["violations"])



def test_parse_architectural_brief():
    brief = "Spacious 3 bedroom corner apartment with open kitchen, south facing balcony, and ensuite bathroom"
    parsed = parse_architectural_brief(brief)

    assert "master_bedroom" in parsed["required_rooms"]
    assert "kitchen" in parsed["required_rooms"]
    assert "balcony" in parsed["required_rooms"]
    assert parsed["style_preference"] == "open_plan"
    assert parsed["aspect_ratio_preference"] == "spacious"
    assert len(parsed["adjacency_constraints"]) > 0
