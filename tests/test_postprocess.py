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
