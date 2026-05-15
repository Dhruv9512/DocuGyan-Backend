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
python manage.py collectstatic --noinput --clear

echo "-----> Starting Celery worker..."
celery -A DocuGyan worker -l info -Q DocuGyan,DocuAgent_tasks,DocuChat_tasks,DocuMail_tasks --concurrency=2 &

echo "-----> Starting Daphne on port ${PORT:-8000}..."
exec daphne -b 0.0.0.0 -p ${PORT:-8000} DocuGyan.asgi:application