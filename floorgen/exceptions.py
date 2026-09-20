"""
Custom Exception Hierarchy for FloorGen.
Provides typed exceptions for model inference, constraint solving, data ingestion, and authentication.
"""

class FloorGenError(Exception):
    """Base exception for all FloorGen operations."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConstraintViolationError(FloorGenError):
    """Raised when CP-SAT solver fails to find a physically admissible layout."""
    pass


class ModelInferenceError(FloorGenError):
    """Raised when neural vector diffusion fails during sampling."""
    pass


class DataIngestionError(FloorGenError):
    """Raised when parsing or ingesting floorplan datasets fails."""
    pass


class AuthenticationError(FloorGenError):
    """Raised when API client fails authentication or rate limiting."""
    pass


class InvalidRoomSpecificationError(FloorGenError):
    """Raised when requested room categories are malformed or invalid."""
    pass
