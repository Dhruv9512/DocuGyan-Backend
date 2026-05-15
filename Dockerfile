# ============================================================
# Unified Dockerfile — Daphne web server OR Celery worker
# Uses uv for fast, deterministic dependency installation
# ============================================================

# Stage 1: uv binary
FROM ghcr.io/astral-sh/uv:latest AS uv-source

# Stage 2: App image
FROM python:3.11-slim

# --- Environment ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell uv to install into the system Python (no venv needed in Docker)
    UV_SYSTEM_PYTHON=1 \
    # Disable uv's progress bars in CI/build logs
    UV_NO_PROGRESS=1

WORKDIR /app

# --- Copy uv binary from stage 1 ---
COPY --from=uv-source /uv /usr/local/bin/uv

# --- System dependencies ---
# git     → needed for the github-hosted docugyan-shared-models package
# libpq   → psycopg2-binary runtime link
# build-essential → compile any packages without pre-built wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies via uv ---
# Copy requirements first so Docker layer cache is preserved on code changes
COPY requirements.txt .

# uv pip install is ~10-100x faster than pip for large dependency trees
RUN uv pip install --no-cache -r requirements.txt

# --- Application code ---
COPY . .

# --- Expose Daphne port ---
EXPOSE 8000

# No CMD here — the start command is injected by Render / Koyeb
# Web (Render):   daphne -b 0.0.0.0 -p 8000 DocuGyan.asgi:application
# Worker (Koyeb): celery -A DocuGyan worker -l info --concurrency=2