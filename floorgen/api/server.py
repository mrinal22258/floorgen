"""
FastAPI Production Server for FloorGen.
Provides high-performance, asynchronous endpoints for architectural floorplan generation,
batch synthesis, health checks, Prometheus telemetry, and CAD deliverables.
"""

import os
import uuid
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, PlainTextResponse
import torch

from floorgen import __version__
from floorgen.config import default_config
from floorgen.logging_config import get_logger, setup_logging
from floorgen.pipeline import FloorGenPipeline
from floorgen.api.job_store import PersistentJobStore
from floorgen.api.middleware import SecurityAndObservabilityMiddleware, metrics
from floorgen.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ReadyResponse,
    BatchJobResponse,
    JobStatusResponse,
    ExemplarItem,
    VALID_ROOMS
)

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warms the generative pipeline and index during startup."""
    setup_logging(level=default_config.log_level, json_format=(default_config.log_format == "json"))
    log.info(f"FloorGen API Server v{__version__} starting up...")
    pipeline = FloorGenPipeline.get_instance(default_config)
    job_store = PersistentJobStore.get_instance()
    log.info(f"FloorGen Pipeline pre-warmed on device: {pipeline.device}")
    yield
    log.info("FloorGen API Server shutting down...")


app = FastAPI(
    title="FloorGen API",
    description="Production REST API for Retrieval-Augmented Generative Floorplan Synthesis",
    version=__version__,
    lifespan=lifespan
)

# Enable CORS for cross-origin web apps / frontends
cors_origins_env = os.getenv("FLOORGEN_CORS_ORIGINS", "*").split(",")
cors_origins = [o.strip() for o in cors_origins_env if o.strip()]
# When wildcard origin '*' is used, allow_credentials must be False per browser security specs
allow_credentials = False if "*" in cors_origins else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security, rate-limiting, and telemetry middleware
app.add_middleware(SecurityAndObservabilityMiddleware)



def _process_batch_job(job_id: str, requests: List[GenerateRequest]):
    """Background task worker for batch synthesis jobs."""
    job_store = PersistentJobStore.get_instance()
    try:
        pipeline = FloorGenPipeline.get_instance()
        results = []
        for req in requests:
            res = pipeline.generate(
                rooms=req.rooms,
                brief=req.brief,
                top_k=req.top_k,
                batch=req.batch,
                sampling=req.sampling,
                steps=req.steps,
                solver=req.solver,
                export_dxf_flag=req.export_dxf,
                export_ifc_flag=req.export_ifc
            )
            results.append(GenerateResponse(
                realism_score=res.realism_score,
                circulation=res.circulation,
                aspect_ratio=res.aspect_ratio,
                compliance=res.compliance,
                generation_time_ms=res.generation_time_ms,
                device=res.device,
                checkpoint_epoch=res.checkpoint_epoch,
                exemplars=[ExemplarItem(
                    plan_id=e.get("plan_id", "plan"),
                    score=round(float(e.get("score", 0.0)), 4),
                    archetype=e.get("plan", {}).get("archetype", "unknown")
                ) for e in res.exemplars],
                json_spec=res.json_spec,
                svg=res.svg,
                dxf_base64=res.dxf_base64,
                ifc_base64=res.ifc_base64
            ).model_dump())

        job_store.update_job_status(job_id, "completed", results=results)
    except Exception as e:
        log.error(f"Batch job {job_id} failed: {e}", exc_info=True)
        job_store.update_job_status(job_id, "failed", error=str(e))


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health():
    """Health check for load balancers and orchestrators."""
    pipeline = FloorGenPipeline.get_instance()
    return HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=(pipeline.model is not None),
        device=str(pipeline.device),
        cuda_available=torch.cuda.is_available()
    )


@app.get("/ready", response_model=ReadyResponse, tags=["Observability"])
async def ready():
    """Readiness probe checking corpus indexing and checkpoint loading."""
    pipeline = FloorGenPipeline.get_instance()
    if not pipeline.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline not initialized")
    indexed_count = len(pipeline.retriever.corpus_plans) if pipeline.retriever else 0
    return ReadyResponse(
        ready=True,
        indexed_plans=indexed_count,
        checkpoint_epoch=pipeline.checkpoint_epoch
    )


@app.get("/metrics", tags=["Observability"])
async def get_metrics():
    """Prometheus-compatible scrape endpoint."""
    return PlainTextResponse(metrics.to_prometheus_format(), media_type="text/plain; version=0.0.4")


@app.get("/api/v1/rooms", tags=["Floorplan Generation"])
async def list_supported_rooms():
    """Returns the set of supported architectural room categories."""
    return {"rooms": sorted(list(VALID_ROOMS))}


@app.post("/api/v1/generate", response_model=GenerateResponse, tags=["Floorplan Generation"])
async def generate_floorplan(request: GenerateRequest):
    """
    Synchronous floorplan synthesis.
    Retrieves nearest exemplars, generates layout via vector diffusion, applies CP-SAT solver,
    and returns SVG, JSON, DXF, and evaluation metrics.
    """
    try:
        pipeline = FloorGenPipeline.get_instance()
        res = pipeline.generate(
            rooms=request.rooms,
            brief=request.brief,
            top_k=request.top_k,
            batch=request.batch,
            sampling=request.sampling,
            steps=request.steps,
            solver=request.solver,
            export_dxf_flag=request.export_dxf,
            export_ifc_flag=request.export_ifc
        )

        return GenerateResponse(
            realism_score=res.realism_score,
            circulation=res.circulation,
            aspect_ratio=res.aspect_ratio,
            compliance=res.compliance,
            generation_time_ms=res.generation_time_ms,
            device=res.device,
            checkpoint_epoch=res.checkpoint_epoch,
            exemplars=[ExemplarItem(
                plan_id=e.get("plan_id", "plan"),
                score=round(float(e.get("score", 0.0)), 4),
                archetype=e.get("plan", {}).get("archetype", "unknown")
            ) for e in res.exemplars],
            json_spec=res.json_spec,
            svg=res.svg,
            dxf_base64=res.dxf_base64,
            ifc_base64=res.ifc_base64
        )
    except Exception as e:
        log.error(f"Generation request failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during floorplan synthesis. Please inspect server logs."
        )



@app.post("/api/v1/generate/batch", response_model=BatchJobResponse, tags=["Batch Synthesis"])
async def generate_batch(requests: List[GenerateRequest], background_tasks: BackgroundTasks):
    """Queues an asynchronous batch generation job."""
    if len(requests) > 50:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Batch size limited to 50 requests")
    
    job_id = str(uuid.uuid4())
    job_store = PersistentJobStore.get_instance()
    job_store.create_job(job_id, len(requests))
    background_tasks.add_task(_process_batch_job, job_id, requests)
    return BatchJobResponse(job_id=job_id, status="queued", total_requests=len(requests))


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["Batch Synthesis"])
async def get_job_status(job_id: str):
    """Polls the status and results of a background batch generation job."""
    job_store = PersistentJobStore.get_instance()
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    
    raw_results = job.get("results")
    results = [GenerateResponse(**r) for r in raw_results] if raw_results else None

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        results=results,
        error=job.get("error")
    )


@app.get("/api/v1/export/svg/{job_id}", tags=["Deliverables"])
async def export_svg_file(job_id: str):
    """Directly downloads the generated SVG vector image."""
    job_store = PersistentJobStore.get_instance()
    job = job_store.get_job(job_id)
    if not job or not job.get("results"):
        raise HTTPException(status_code=404, detail="Job or results not found")
    svg_data = job["results"][0]["svg"]
    return Response(content=svg_data, media_type="image/svg+xml")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "floorgen.api.server:app",
        host=default_config.host,
        port=default_config.port,
        workers=default_config.workers,
        reload=False
    )
