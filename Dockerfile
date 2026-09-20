# Production Multi-Stage Dockerfile for FloorGen
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

# Install Python & system graphics libraries for OpenCV
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Install dependencies first for maximum layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .
RUN pip install --no-cache-dir -e .

# Non-root service user
RUN useradd -m -u 1000 floorgen && chown -R floorgen:floorgen /app
USER floorgen

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "floorgen.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
