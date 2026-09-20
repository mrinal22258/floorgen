"""
Pydantic Schemas and Request/Response Models for FloorGen API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


VALID_ROOMS = {
    "living_room",
    "master_bedroom",
    "second_bedroom",
    "bathroom",
    "kitchen",
    "balcony",
    "entrance",
    "dining_room",
    "study",
    "storage"
}


class GenerateRequest(BaseModel):
    rooms: List[str] = Field(
        default=["living_room", "master_bedroom", "second_bedroom", "bathroom", "kitchen", "balcony"],
        description="List of desired room categories",
        min_length=1,
        max_length=20
    )
    brief: Optional[str] = Field(
        default=None,
        description="Optional natural-language architectural brief",
        max_length=2000
    )
    top_k: int = Field(default=5, ge=1, le=10, description="Exemplars retrieved from RPLAN corpus")
    batch: int = Field(default=1, ge=1, le=10, description="Number of candidate layouts to sample")
    sampling: str = Field(default="ddim", pattern="^(ddim|ddpm)$", description="Sampling method")
    steps: int = Field(default=20, ge=5, le=50, description="Diffusion steps")
    solver: bool = Field(default=True, description="Apply Google OR-Tools CP-SAT solver")
    export_dxf: bool = Field(default=True, description="Export AutoCAD DXF 9-layer deliverable")
    export_ifc: bool = Field(default=True, description="Export ISO-16739 standard IFC BIM deliverable")

    @field_validator("rooms")
    @classmethod
    def validate_room_types(cls, v: List[str]) -> List[str]:
        cleaned = []
        for r in v:
            rc = r.strip().lower()
            if rc not in VALID_ROOMS:
                raise ValueError(f"Invalid room type: '{rc}'. Must be one of: {sorted(list(VALID_ROOMS))}")
            cleaned.append(rc)
        return cleaned


class ExemplarItem(BaseModel):
    plan_id: str
    score: float
    archetype: str


class GenerateResponse(BaseModel):
    realism_score: float
    circulation: float
    aspect_ratio: float
    generation_time_ms: float
    device: str
    checkpoint_epoch: int
    exemplars: List[ExemplarItem]
    json_spec: Dict[str, Any]
    svg: str
    dxf_base64: Optional[str] = None
    ifc_base64: Optional[str] = None
    compliance: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    device: str
    cuda_available: bool


class ReadyResponse(BaseModel):
    ready: bool
    indexed_plans: int
    checkpoint_epoch: int


class BatchJobResponse(BaseModel):
    job_id: str
    status: str
    total_requests: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    results: Optional[List[GenerateResponse]] = None
    error: Optional[str] = None
