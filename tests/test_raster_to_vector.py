"""
Unit tests for OpenCV raster-to-vector refinement and Manhattan snapping.
"""

import numpy as np
import cv2
import pytest
from floorgen.postprocess.raster_to_vector import raster_to_vector, snap_polygon_to_manhattan


def test_snap_polygon_to_manhattan():
    # Nearly rectangular polygon with slight tilt
    poly = [[10.0, 10.0], [100.0, 10.5], [99.5, 80.0], [10.2, 79.8]]
    snapped = snap_polygon_to_manhattan(poly, angle_tol_deg=15.0)
    assert len(snapped) == 4
    for pt in snapped:
        assert isinstance(pt[0], float)
        assert isinstance(pt[1], float)


def test_raster_to_vector_detection():
    # Create synthetic image with 2 room compartments surrounded by dark walls
    img = np.full((256, 256, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (110, 110), (0, 0, 0), 3)
    cv2.rectangle(img, (110, 20), (200, 110), (0, 0, 0), 3)

    res = raster_to_vector(img, min_room_area=100.0)
    assert "rooms" in res
    assert "adjacency" in res
    assert len(res["rooms"]) >= 1
    for r in res["rooms"]:
        assert "bbox" in r
        assert "polygon" in r
        assert r["area"] > 0


def test_raster_to_vector_with_candidate_plan():
    img = np.full((256, 256, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (120, 120), (0, 0, 0), 3)

    candidate = {
        "id": "cand_01",
        "rooms": [
            {"category": "living_room", "zone": "public", "bbox": [20, 20, 120, 120]}
        ]
    }
    res = raster_to_vector(img, candidate_plan=candidate, min_room_area=100.0)
    assert res["id"] == "cand_01"
    assert len(res["rooms"]) >= 1
    # Check that candidate's semantic category was associated
    cats = [r["category"] for r in res["rooms"]]
    assert "living_room" in cats
