# ==============================================================================
# Avatar Chat Server - CPU-Only Dockerfile
# ==============================================================================
# Multi-stage build optimized for ONNX CPU inference.
#
# Build:
#   docker build -t avatar-chat-server .
#
# Run:
#   docker run -p 8080:8080 --env-file .env avatar-chat-server
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Base image with non-root user
# ------------------------------------------------------------------------------
FROM python:3.10-slim AS base

# Prevent interactive prompts and set Python to not buffer output
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

# Install system dependencies and clean up in same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user early so all subsequent files are owned by them
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory and give ownership to appuser
WORKDIR /app
RUN chown appuser:appuser /app

RUN pip install -U "huggingface_hub[cli]" \
    && mkdir -p pretrained_models \
    && huggingface-cli download myned-ai/wav2arkit_cpu --local-dir pretrained_models

# ------------------------------------------------------------------------------
# Stage 2: Dependencies installation with uv (as appuser)
# ------------------------------------------------------------------------------
FROM base AS dependencies

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Switch to non-root user BEFORE installing dependencies
# This ensures .venv is owned by appuser from the start
USER appuser

# Copy dependency files first for better caching
COPY --chown=appuser:appuser pyproject.toml README.md ./

# Install Python dependencies and clean uv cache in same layer
RUN uv sync --frozen --no-dev || uv sync --no-dev \
    && rm -rf ~/.cache/uv

# ------------------------------------------------------------------------------
# Stage 3: Production image
# ------------------------------------------------------------------------------
FROM dependencies AS production

# Copy application code (already running as appuser)
COPY --chown=appuser:appuser ./src ./src
RUN mv ./pretrained_models/ ./src/

# Environment variables
ENV SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8080 \
    PYTHONPATH=/app/src

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Start the server
CMD ["uv", "run", "--no-dev", "python", "src/main.py"]

# ------------------------------------------------------------------------------
# Stage 4: Development image (with hot reload)
# ------------------------------------------------------------------------------
FROM dependencies AS development

# Copy application code
COPY --chown=appuser:appuser ./src ./src
RUN mv ./pretrained_models/ ./src/

# Mount point for source code (overrides the COPY above when mounted)
VOLUME ["/app"]

# Environment variables for development
ENV DEBUG=true \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8080 \
    PYTHONPATH=/app/src

# Expose port
EXPOSE 8080

# Start with hot reload
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
