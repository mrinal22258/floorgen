"""
Configuration Management for FloorGen.
Supports environment variable overrides, pydantic validation, and production defaults.
"""

import os
import warnings
from typing import Optional
import torch
from pydantic import BaseModel, Field, ConfigDict, model_validator



class FloorGenConfig(BaseModel):
    # Paths
    data_dir: str = Field(default_factory=lambda: os.getenv("FLOORGEN_DATA_DIR", "data/processed"))
    checkpoint_path: str = Field(default_factory=lambda: os.getenv("FLOORGEN_CHECKPOINT", "checkpoints/rag_diffusion.pt"))
    raw_dir: str = Field(default_factory=lambda: os.getenv("FLOORGEN_RAW_DIR", "data/raw"))
    
    # Model
    hidden_dim: int = Field(default_factory=lambda: int(os.getenv("FLOORGEN_HIDDEN_DIM", "256")))
    num_layers: int = Field(default_factory=lambda: int(os.getenv("FLOORGEN_NUM_LAYERS", "4")))
    device: str = Field(default_factory=lambda: os.getenv("FLOORGEN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Sampling & Diffusion
    default_sampling: str = Field(default_factory=lambda: os.getenv("FLOORGEN_SAMPLING", "ddim"))
    default_steps: int = Field(default_factory=lambda: int(os.getenv("FLOORGEN_STEPS", "20")))
    default_batch: int = Field(default_factory=lambda: int(os.getenv("FLOORGEN_BATCH", "1")))
    
    # RAG Retrieval
    top_k: int = Field(default_factory=lambda: int(os.getenv("FLOORGEN_TOP_K", "5")))
    lambda_mult: float = Field(default_factory=lambda: float(os.getenv("FLOORGEN_LAMBDA_MULT", "0.65")))
    enable_retrieval_cache: bool = Field(default_factory=lambda: os.getenv("FLOORGEN_ENABLE_CACHE", "true").lower() == "true")
    
    # Solver
    enable_solver: bool = Field(default_factory=lambda: os.getenv("FLOORGEN_ENABLE_SOLVER", "true").lower() == "true")
    solver_time_limit: float = Field(default_factory=lambda: float(os.getenv("FLOORGEN_SOLVER_TIME_LIMIT", "2.0")))
    pixels_per_meter: float = Field(default_factory=lambda: float(os.getenv("FLOORGEN_PIXELS_PER_METER", "16.0")))
    
    # Server & API
    host: str = Field(default_factory=lambda: os.getenv("FLOORGEN_HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("FLOORGEN_PORT", "8000")))
    workers: int = Field(default_factory=lambda: int(os.getenv("FLOORGEN_WORKERS", "1")))
    api_prefix: str = Field(default_factory=lambda: os.getenv("FLOORGEN_API_PREFIX", "/api/v1"))
    
    # Logging
    log_level: str = Field(default_factory=lambda: os.getenv("FLOORGEN_LOG_LEVEL", "INFO"))
    log_format: str = Field(default_factory=lambda: os.getenv("FLOORGEN_LOG_FORMAT", "standard"))

    # Security & Rate Limiting
    auth_enabled: bool = Field(default_factory=lambda: os.getenv("FLOORGEN_AUTH_ENABLED", "false").lower() == "true")
    api_key: str = Field(default_factory=lambda: os.getenv("FLOORGEN_API_KEY", "floorgen-secret-key-prod"))
    rate_limit_rpm: int = Field(default_factory=lambda: int(os.getenv("FLOORGEN_RATE_LIMIT_RPM", "120")))

    # Metrics
    metrics_enabled: bool = Field(default_factory=lambda: os.getenv("FLOORGEN_METRICS_ENABLED", "true").lower() == "true")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_security(self) -> "FloorGenConfig":
        if self.auth_enabled and self.api_key == "floorgen-secret-key-prod":
            warnings.warn(
                "CRITICAL SECURITY WARNING: FLOORGEN_AUTH_ENABLED is True, but FLOORGEN_API_KEY is set to the "
                "insecure default literal 'floorgen-secret-key-prod'. Please set a cryptographically secure "
                "FLOORGEN_API_KEY environment variable for production deployments.",
                category=RuntimeWarning,
                stacklevel=2,
            )
        return self



# Global default configuration instance
default_config = FloorGenConfig()


def get_settings() -> FloorGenConfig:
    """Returns global default configuration settings."""
    return default_config


def get_config() -> FloorGenConfig:
    return default_config
