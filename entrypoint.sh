#!/bin/sh
# ==============================================================
# entrypoint.sh — runs on every container start (web service only)
# Celery workers on Koyeb skip this entirely via Docker Command override
# ==============================================================
set -e

echo "-----> Running database migrations..."
python manage.py migrate --noinput

echo "-----> Collecting static files..."
python manage.py collectstatic --noinput

echo "-----> Starting Daphne (ASGI server)..."
# exec replaces the shell process with Daphne so that Docker SIGTERM
# signals reach Daphne directly (clean shutdown, no zombie processes)
exec daphne -b 0.0.0.0 -p 8000 DocuGyan.asgi:application