#!/bin/sh
# ==============================================================
# entrypoint.sh — web service startup
# Static files are already collected during Docker build.
# ==============================================================
set -e

export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::PendingDeprecationWarning"

echo "-----> Checking required environment variables..."
: "${SECRET_KEY:?ERROR: SECRET_KEY is not set}"
: "${DATABASE_URL:?ERROR: DATABASE_URL is not set}"

echo "-----> Running database migrations..."
python manage.py migrate --noinput

echo "-----> Starting Daphne..."
exec daphne -b 0.0.0.0 -p 8000 DocuGyan.asgi:application