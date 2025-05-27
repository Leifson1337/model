#!/bin/sh
# entrypoint.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Optional: Add any setup tasks here ---
# Example: Run database migrations (if applicable)
# echo "Running database migrations..."
# python manage.py migrate --noinput

# Example: Wait for a database to be ready (if applicable)
# if [ -n "$DATABASE_HOST" ] && [ -n "$DATABASE_PORT" ]; then
#   echo "Waiting for database at $DATABASE_HOST:$DATABASE_PORT..."
#   # Use a tool like wait-for-it.sh or netcat
#   # ./wait-for-it.sh "$DATABASE_HOST:$DATABASE_PORT" --timeout=30 --strict -- echo "Database is up."
# fi

echo "Entrypoint script executed. Container is ready."

# Execute the command passed as CMD in Dockerfile or docker-compose.yml
# "$@" is an array-like construct of all positional parameters, $0, $1, ...
# This allows the container to run any command specified in `docker run` or `docker-compose command:`.
exec "$@"
