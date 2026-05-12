#!/bin/sh
set -e

DATA=/app/data
mkdir -p "$DATA"

if [ -n "$DATA_BOOTSTRAP_URL" ] && [ ! -f "$DATA/pool.json" ]; then
  echo "[bootstrap] fetching $DATA_BOOTSTRAP_URL"
  curl -L --fail -o "$DATA/_bootstrap.tgz" "$DATA_BOOTSTRAP_URL"
  tar -xzf "$DATA/_bootstrap.tgz" -C "$DATA"
  rm -f "$DATA/_bootstrap.tgz"
  echo "[bootstrap] done"
fi

exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 4 --timeout 300 server:app
