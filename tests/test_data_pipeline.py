"""
Unit tests for FloorGen Data Pipeline (Stage 0).
"""

import pytest
import os
import shutil
import tempfile
from floorgen.data.scripts.generate_sample_data import generate_dataset, generate_single_plan
from floorgen.data.scripts.filter_invalid import is_valid_floorplan, filter_floorplan_dataset
from floorgen.data.scripts.parse_rplan import init_sqlite_db, save_plan_to_sqlite, load_plans_from_dir
from floorgen.data.dataset import FloorplanDataset, create_dataloader


def test_generate_single_plan():
    plan = generate_single_plan(plan_id=1, archetype_idx=0)
    assert plan["id"] == "plan_00001"
    assert len(plan["rooms"]) >= 3
    assert len(plan["boundary"]) >= 4
    ok, msg = is_valid_floorplan(plan)
    assert ok, f"Generated plan failed validation: {msg}"


def test_filter_invalid():
    valid_plan = generate_single_plan(plan_id=2, archetype_idx=1)
    ok, _ = is_valid_floorplan(valid_plan)
    assert ok

    # Create invalid plan (disconnected rooms)
    invalid_plan = dict(valid_plan)
    invalid_plan["adjacency"] = [] # no connections
    ok, msg = is_valid_floorplan(invalid_plan)
    assert not ok
    assert "Disconnected" in msg


def test_sqlite_and_dir_loading():
    temp_dir = tempfile.mkdtemp()
    try:
        plans = generate_dataset(num_samples=5, output_dir=temp_dir)
        assert len(plans) == 5

        db_path = os.path.join(temp_dir, "test_floorplans.db")
        conn = init_sqlite_db(db_path)
        for p in plans:
            save_plan_to_sqlite(p, conn)
        conn.close()

        loaded = load_plans_from_dir(temp_dir)
        assert len(loaded) == 5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_pytorch_dataset():
    plans = [generate_single_plan(plan_id=i, archetype_idx=i) for i in range(4)]
    dataset = FloorplanDataset(plans=plans, max_rooms=10)
    assert len(dataset) == 4

    item = dataset[0]
    assert item["room_types"].shape == (10,)
    assert item["room_boxes"].shape == (10, 4)
    assert item["room_mask"].shape == (10,)
    assert item["adj_matrix"].shape == (10, 10)

    loader = create_dataloader(plans, batch_size=2, max_rooms=10)
    batch = next(iter(loader))
    assert batch["room_types"].shape == (2, 10)

