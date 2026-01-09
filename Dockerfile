# ==============================================================================
# Avatar Chat Server - Dockerfile
# ==============================================================================
# Multi-stage build for optimized production image with GPU support
# Uses uv for fast, reliable Python package management
#
# Build:
#   docker build -t avatar-chat-server .
#
# Run:
#   docker run -p 8080:8080 --gpus all --env-file .env avatar-chat-server
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Base image with CUDA support
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

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
# Stage 2: Dependencies installation with uv
# ------------------------------------------------------------------------------
FROM base AS dependencies

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* ./

# Install Python dependencies with uv (much faster than pip)
RUN uv sync --frozen --no-dev || uv sync --no-dev

# ------------------------------------------------------------------------------
# Stage 3: Production image
# ------------------------------------------------------------------------------
FROM dependencies AS production

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8080

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

ENV PYTHONUNBUFFERED=1 \
    DEBUG=true \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8080

EXPOSE 8080

# Start with hot reload (using uv run)
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
