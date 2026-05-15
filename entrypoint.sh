#!/bin/sh
# ==============================================================
# entrypoint.sh — web service startup
# ==============================================================
set -e

export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::PendingDeprecationWarning"

echo "-----> Checking required environment variables..."
: "${SECRET_KEY:?ERROR: SECRET_KEY is not set}"
: "${DATABASE_URL:?ERROR: DATABASE_URL is not set}"

echo "-----> Running database migrations..."
python manage.py migrate --noinput

echo "-----> Collecting static files..."
# collectstatic automatically creates the staticfiles/ folder if it doesn't exist.
# All Django admin CSS/JS is copied into it from installed packages.
python manage.py collectstatic --noinput --clear

echo "-----> Starting Daphne..."
exec daphne -b 0.0.0.0 -p 8000 DocuGyan.asgi:application