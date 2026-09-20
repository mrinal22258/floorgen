"""
Unit and integration tests for FloorGen v2.0 SOTA Pipeline and Wall Graph Diffusion.
"""

import os
import pytest
import torch
from floorgen.pipeline_sota import SOTAPipeline, generate_sota
from floorgen.models.diffusion_graph.wall_graph import WallGraph
from floorgen.models.diffusion_graph.wall_graph_diffusion import WallGraphDiffusion
from floorgen.models.diffusion_graph.junctions import classify_junction


def test_classify_junction_geometry():
    # L junction: two perpendicular rays
    walls_L = [(10, 10, 10, 50), (10, 10, 50, 10)]
    j_type = classify_junction(walls_L, (10, 10))
    assert j_type in ["L", "end"]

    # End junction: single ray
    walls_end = [(10, 10, 50, 10)]
    assert classify_junction(walls_end, (10, 10)) == "end"


def test_wall_graph_data_structure():
    wg = WallGraph()
    w0 = wg.add_wall(0, 0, 100, 0, thickness=200.0, is_exterior=True)
    w1 = wg.add_wall(100, 0, 100, 100, thickness=200.0, is_exterior=True)
    j0 = wg.add_junction(100, 0, "L", 2)
    wg.add_incidence(w0, j0)
    wg.add_incidence(w1, j0)

    d = wg.to_dict()
    assert len(d["walls"]) == 2
    assert len(d["junctions"]) == 1
    assert len(d["incidence"]) == 2

    w_t, j_t, inc_t = wg.get_tensors()
    assert w_t.shape == (2, 6)
    assert j_t.shape == (1, 3)
    assert inc_t.shape == (2, 2)


def test_sota_pipeline_generation(tmp_path):
    output_dir = str(tmp_path / "sota_out")
    res = generate_sota("Modern 2BHK with balcony", output_dir=output_dir)

    assert "parsed_brief" in res
    assert "vector_plan" in res
    assert "wall_graph" in res
    assert "raster_512" in res
    assert "compliance" in res
    assert "files" in res

    # Verify generated files on disk
    files = res["files"]
    assert "json" in files and os.path.exists(files["json"])
    assert "svg" in files and os.path.exists(files["svg"])
    assert "raster_512" in files and os.path.exists(files["raster_512"])


def test_sota_pipeline_generate_raster():
    """Verify SOTAPipeline.generate_raster executes end-to-end with correct output shapes."""
    import numpy as np
    pipeline = SOTAPipeline(device="cpu")
    synthetic_plan = {
        "id": "synthetic_test_001",
        "rooms": [
            {"id": 0, "category": "living_room", "bbox": [30, 30, 120, 120]},
            {"id": 1, "category": "master_bedroom", "bbox": [120, 30, 210, 120]},
            {"id": 2, "category": "bathroom", "bbox": [30, 120, 100, 190]},
            {"id": 3, "category": "kitchen", "bbox": [100, 120, 210, 190]}
        ],
        "adjacency": [[0, 1], [0, 2], [0, 3]]
    }
    blended_bgr, combined_edges = pipeline.generate_raster(synthetic_plan)
    assert blended_bgr.shape == (512, 512, 3)
    assert combined_edges.shape == (512, 512)
    assert blended_bgr.dtype == np.uint8
    assert combined_edges.dtype == np.uint8
