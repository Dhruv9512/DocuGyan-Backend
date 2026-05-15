# ============================================================
# Unified Dockerfile — Daphne web server OR Celery worker
# ============================================================

# Stage 1: Pull uv binary
FROM ghcr.io/astral-sh/uv:latest AS uv-source

# Stage 2: Main app image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_PROGRESS=1 \
    GIT_TERMINAL_PROMPT=0

WORKDIR /app

# --- Copy uv binary ---
COPY --from=uv-source /uv /usr/local/bin/uv

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies ---
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# --- Application code ---
COPY . .

# --- Entrypoint ---
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

# Default: web server
# Koyeb Celery override: celery -A DocuGyan worker -l info --concurrency=2
CMD ["./entrypoint.sh"]