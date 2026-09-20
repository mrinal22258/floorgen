"""
Production Security, Rate-Limiting, and Telemetry Middleware for FloorGen.
"""

import time
import hmac
from typing import Dict, List
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from floorgen.config import default_config
from floorgen.logging_config import get_logger

log = get_logger("middleware")

# Sliding-window rate limiter state: IP -> list of timestamps
_RATE_LIMIT_STORE = defaultdict(list)

# Simple Prometheus metrics registry in memory
class MetricsRegistry:
    def __init__(self):
        self.request_count = defaultdict(int)
        self.request_latency_sum = 0.0
        self.generation_count = 0
        self.generation_time_sum = 0.0

    def record_request(self, method: str, path: str, status_code: int, duration: float):
        key = f"{method}_{path}_{status_code}"
        self.request_count[key] += 1
        self.request_latency_sum += duration
        if "generate" in path:
            self.generation_count += 1
            self.generation_time_sum += duration

    def to_prometheus_format(self) -> str:
        lines = [
            "# HELP floorgen_http_requests_total Total number of HTTP requests processed",
            "# TYPE floorgen_http_requests_total counter"
        ]
        for key, count in self.request_count.items():
            parts = key.split("_")
            method = parts[0]
            status = parts[-1]
            path = "_".join(parts[1:-1])
            lines.append(f'floorgen_http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}')
            
        lines.extend([
            "# HELP floorgen_generation_requests_total Total floorplan generation requests",
            "# TYPE floorgen_generation_requests_total counter",
            f"floorgen_generation_requests_total {self.generation_count}",
            "# HELP floorgen_generation_latency_seconds_total Total duration of generation requests in seconds",
            "# TYPE floorgen_generation_latency_seconds_total counter",
            f"floorgen_generation_latency_seconds_total {self.generation_time_sum:.4f}"
        ])
        return "\n".join(lines) + "\n"

metrics = MetricsRegistry()


class SecurityAndObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # 1. Bypass public endpoints (health, docs, metrics)
        is_public = path in ["/health", "/ready", "/docs", "/redoc", "/openapi.json", "/metrics"]

        # 2. Rate Limiting Check (if enabled)
        if not is_public and default_config.rate_limit_rpm > 0:
            now = time.time()
            window_start = now - 60.0
            # Prune old timestamps
            _RATE_LIMIT_STORE[client_ip] = [t for t in _RATE_LIMIT_STORE[client_ip] if t > window_start]
            if len(_RATE_LIMIT_STORE[client_ip]) >= default_config.rate_limit_rpm:
                log.warning(f"Rate limit exceeded for IP: {client_ip}")
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again in 1 minute."}
                )
            _RATE_LIMIT_STORE[client_ip].append(now)

        # 3. Authentication Check (if enabled)
        if default_config.auth_enabled and not is_public:
            api_key = request.headers.get("X-API-Key")
            auth_header = request.headers.get("Authorization")
            
            token = api_key
            if not token and auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split("Bearer ")[1].strip()

            if not token or not hmac.compare_digest(token, default_config.api_key):
                log.warning(f"Unauthorized access attempt from {client_ip} on {path}")
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key. Provide 'X-API-Key' header."}
                )

        # 4. Process Request
        response: Response = await call_next(request)

        # 5. Telemetry & Timing Header
        duration = time.time() - start_time
        response.headers["X-Process-Time-Ms"] = f"{duration * 1000:.2f}"
        metrics.record_request(request.method, path, response.status_code, duration)

        return response
