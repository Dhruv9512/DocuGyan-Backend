# ============================================================
# Unified Dockerfile — Daphne web server OR Celery worker
# Uses uv for fast, deterministic dependency installation
# ============================================================

# Stage 1: Pull uv binary
FROM ghcr.io/astral-sh/uv:latest AS uv-source

# Stage 2: Main app image
FROM python:3.11-slim

# --- Environment ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Install into system Python — no virtualenv needed inside Docker
    UV_SYSTEM_PYTHON=1 \
    UV_NO_PROGRESS=1

WORKDIR /app

# --- Copy uv binary from stage 1 ---
COPY --from=uv-source /uv /usr/local/bin/uv

# --- System dependencies ---
# git           → needed for docugyan-shared-models (git+https install)
# libpq-dev     → psycopg2-binary runtime
# build-essential → compile wheels that have no pre-built binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies ---
# Copy requirements first so Docker layer cache survives code-only changes
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# --- Application code ---
COPY . .

# --- Entrypoint ---
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# --- Port ---
EXPOSE 8000

# Default: runs migrate + collectstatic + daphne (web server)
# To run Celery instead (Koyeb Docker Command field):
#   celery -A DocuGyan worker -l info --concurrency=2
CMD ["./entrypoint.sh"]