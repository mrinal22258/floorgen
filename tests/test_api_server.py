"""
Unit and Integration tests for FloorGen FastAPI Production Server.
"""

import pytest
from fastapi.testclient import TestClient
from floorgen.api.server import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data


def test_ready_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["indexed_plans"] > 0
    assert data["checkpoint_epoch"] >= 80


def test_rooms_endpoint():
    response = client.get("/api/v1/rooms")
    assert response.status_code == 200
    data = response.json()
    assert "living_room" in data["rooms"]
    assert "master_bedroom" in data["rooms"]


def test_generate_endpoint_validation():
    # Invalid room category
    response = client.post("/api/v1/generate", json={
        "rooms": ["space_station_docking_bay"]
    })
    assert response.status_code == 422


def test_generate_endpoint_success():
    response = client.post("/api/v1/generate", json={
        "rooms": ["living_room", "master_bedroom", "bathroom", "kitchen"],
        "steps": 10,
        "batch": 1,
        "solver": True
    })
    assert response.status_code == 200
    data = response.json()
    assert data["realism_score"] > 80.0
    assert data["circulation"] == 1.0
    assert "<svg" in data["svg"]
    assert "rooms" in data["json_spec"]
    assert len(data["json_spec"]["rooms"]) == 4


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "floorgen_http_requests_total" in response.text
    assert "floorgen_generation_requests_total" in response.text


def test_generate_with_ifc_and_compliance():
    response = client.post("/api/v1/generate", json={
        "rooms": ["living_room", "master_bedroom", "bathroom", "kitchen"],
        "steps": 10,
        "batch": 1,
        "solver": True,
        "export_ifc": True
    })
    assert response.status_code == 200
    data = response.json()
    assert data["ifc_base64"] is not None
    assert data["compliance"] is not None
    assert "compliance_score" in data["compliance"]
    assert "walls" in data["json_spec"]
    assert "furniture" in data["json_spec"]


def test_job_store_persistence(tmp_path):
    from floorgen.api.job_store import PersistentJobStore

    db_path = str(tmp_path / "test_jobs.db")
    store = PersistentJobStore(db_path=db_path)
    store.save_job("test_job_1", {"status": "running", "total_requests": 2})
    
    # Read back from same store
    job = store.get_job("test_job_1")
    assert job is not None
    assert job["status"] == "running"
    assert job["total_requests"] == 2

    # Create a separate store instance from same DB file to verify disk persistence
    store2 = PersistentJobStore(db_path=db_path)
    job2 = store2.get_job("test_job_1")
    assert job2 is not None
    assert job2["status"] == "running"
    assert job2["total_requests"] == 2


