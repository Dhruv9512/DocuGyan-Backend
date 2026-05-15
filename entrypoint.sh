#!/bin/sh
# ==============================================================
# entrypoint.sh — web service startup
# ==============================================================
set -e

# Suppress LangChain/LangGraph deprecation warnings during startup
# so they don't clutter migration output (they are not errors)
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::PendingDeprecationWarning"

echo "-----> Checking required environment variables..."
: "${SECRET_KEY:?ERROR: SECRET_KEY is not set}"
: "${DATABASE_URL:?ERROR: DATABASE_URL is not set}"

echo "-----> Running database migrations..."
python manage.py migrate --noinput

echo "-----> Collecting static files..."
python manage.py collectstatic --noinput

# Restore normal warnings for the running app
unset PYTHONWARNINGS

echo "-----> Starting Daphne (ASGI server)..."
exec daphne -b 0.0.0.0 -p 8000 DocuGyan.asgi:application