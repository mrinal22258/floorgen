import os
import numpy as np
import pytest
from floorgen.constraints.solver import FloorplanConstraintSolver
from floorgen.postprocess.vectorize import export_dxf, export_svg


def test_cp_sat_solver():
    solver = FloorplanConstraintSolver()
    raw_boxes = np.array([
        [-0.5, -0.5, 0.2, 0.2],
        [-0.4, -0.4, 0.3, 0.3],
        [0.0, 0.0, 0.5, 0.5]
    ])
    room_types = ["living_room", "master_bedroom", "bathroom"]
    res = solver.solve(raw_boxes, room_types)
    assert len(res) == 3
    # Check that solved areas are positive
    for r in res:
        assert r["area"] > 0
        assert len(r["polygon"]) == 4


def test_dxf_and_svg_export(tmp_path):
    plan = {
        "rooms": [
            {"id": 0, "category": "living_room", "polygon": [[10, 10], [100, 10], [100, 80], [10, 80]], "bbox": [10, 10, 100, 80], "area": 6300, "centroid": [55, 45]},
            {"id": 1, "category": "master_bedroom", "polygon": [[100, 10], [180, 10], [180, 80], [100, 80]], "bbox": [100, 10, 180, 80], "area": 5600, "centroid": [140, 45]}
        ],
        "doors": [
            {"pos": [100, 45], "orientation": "v"}
        ],
        "boundary": [[0, 0], [200, 0], [200, 100], [0, 100]]
    }
    dxf_path = str(tmp_path / "test.dxf")
    export_dxf(plan, dxf_path)
    assert os.path.exists(dxf_path)

    svg = export_svg(plan)
    assert "<svg" in svg
    assert "Living Room" in svg
