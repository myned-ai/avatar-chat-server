# ==============================================================================
# Avatar Chat Server - CPU-Only Dockerfile (Benchmark)
# ==============================================================================
# Multi-stage build optimized for CPU inference with PyTorch optimizations
# This Dockerfile is designed for benchmarking CPU performance to help determine
# if quantization or other optimization techniques are needed.
#
# Build:
#   docker build -f Dockerfile.cpu -t avatar-chat-server:cpu .
#
# Run:
#   docker run -p 8080:8080 --env-file .env -e USE_GPU=false avatar-chat-server:cpu
#
# Benchmark notes:
# - Uses PyTorch CPU-only builds (smaller, optimized)
# - Enables Intel MKL for better CPU performance
# - Sets optimal threading for inference
# - Remove GPU dependencies entirely
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Base image without CUDA
# ------------------------------------------------------------------------------
FROM ubuntu:22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# ------------------------------------------------------------------------------
# Stage 2: Dependencies installation with uv (CPU-only PyTorch)
# ------------------------------------------------------------------------------
FROM base AS dependencies

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for better caching
COPY pyproject.toml ./

# Create a CPU-only pyproject.toml override
RUN cat > pyproject.cpu.toml << 'EOF'
[project]
name = "avatar-chat-server"
version = "1.0.0"
description = "Real-time voice-to-avatar interaction server combining OpenAI Realtime API with Audio to Expression model"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }

dependencies = [
    # FastAPI Framework
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic-settings>=2.1.0",

    # WebSocket support (for OpenAI Realtime client)
    "websockets>=12.0",

    # Performance - Fast JSON serialization
    "orjson>=3.9.0",

    # Audio processing
    "numpy>=1.24.0",
    "librosa>=0.10.0",
    "scipy>=1.10.0",

    # Environment variables
    "python-dotenv>=1.0.0",

    # Audio2Expression model dependencies (CPU-only versions)
    "torch>=2.1.0",
    "torchaudio>=2.1.0",
    "transformers==4.36.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.25.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
# Use PyTorch CPU-only wheels for smaller image and CPU optimizations
torch = { index = "pytorch-cpu" }
torchaudio = { index = "pytorch-cpu" }

[tool.hatch.build.targets.wheel]
packages = ["src"]
EOF

# Install Python dependencies with uv using CPU-only PyTorch
RUN mv pyproject.cpu.toml pyproject.toml && \
    uv sync --frozen --no-dev || uv sync --no-dev

# ------------------------------------------------------------------------------
# Stage 3: Production image with CPU optimizations
# ------------------------------------------------------------------------------
FROM dependencies AS production

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy application code
COPY --chown=appuser:appuser . .

# CPU Optimization Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8080 \
    USE_GPU=false \
    # PyTorch CPU optimizations
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TORCH_NUM_THREADS=4 \
    # Use Intel MKL for optimized math operations
    MKL_THREADING_LAYER=GNU \
    # Disable TensorFloat32 (not applicable for CPU but good practice)
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Start the server (using uv run to use the managed environment)
CMD ["uv", "run", "python", "main.py"]

# ------------------------------------------------------------------------------
# Alternative: Development image (with hot reload)
# ------------------------------------------------------------------------------
FROM dependencies AS development

WORKDIR /app

# Mount point for source code
VOLUME ["/app"]

# CPU Optimization for development
ENV PYTHONUNBUFFERED=1 \
    DEBUG=true \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8080 \
    USE_GPU=false \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TORCH_NUM_THREADS=4

EXPOSE 8080

# Start with hot reload (using uv run)
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
