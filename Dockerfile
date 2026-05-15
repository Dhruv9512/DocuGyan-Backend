# ============================================================
# Unified Dockerfile — Daphne web server OR Celery worker
# collectstatic runs at BUILD time so static files are baked
# into the image (Django admin CSS works out of the box)
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

# --- Collect static files at build time ---
# A dummy SECRET_KEY is used only for this build step.
# The real SECRET_KEY from Render env vars is used at runtime.
# DATABASE_URL is NOT needed for collectstatic — it only copies files.
RUN SECRET_KEY=build-only-dummy-key-not-used-at-runtime \
    DJANGO_SETTINGS_MODULE=DocuGyan.settings \
    python manage.py collectstatic --noinput

# --- Entrypoint ---
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

# Default: web server
# Koyeb Celery override: celery -A DocuGyan worker -l info --concurrency=2
CMD ["./entrypoint.sh"]