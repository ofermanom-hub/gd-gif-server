#!/bin/sh
set -e

DATA=/app/data
mkdir -p "$DATA" "$DATA/gifs" "$DATA/frames"

RELEASE_BASE="${GIF_RELEASE_BASE:-https://github.com/ofermanom-hub/gd-gif-server/releases/download/v0.1-data}"

# Bootstrap only the small metadata files. Gifs and frames are fetched lazily
# by server.py on first request (see _fetch_gif_from_release / _fetch_frame_from_release).
fetch_if_missing() {
  name="$1"
  if [ ! -f "$DATA/$name" ]; then
    echo "[bootstrap] $name"
    curl -L --fail -sS -o "$DATA/$name" "$RELEASE_BASE/$name" || \
      echo "[bootstrap] WARN: failed to fetch $name (continuing)"
  fi
}

fetch_if_missing pool.json
fetch_if_missing candidates.json
fetch_if_missing meta-bundle.json

# Legacy: if DATA_BOOTSTRAP_URL is set and pool.json still missing, fall back to tarball bootstrap.
if [ -n "$DATA_BOOTSTRAP_URL" ] && [ ! -f "$DATA/pool.json" ]; then
  echo "[bootstrap] fallback tarball $DATA_BOOTSTRAP_URL"
  curl -L --fail -o "$DATA/_bootstrap.tgz" "$DATA_BOOTSTRAP_URL"
  tar -xzf "$DATA/_bootstrap.tgz" -C "$DATA"
  rm -f "$DATA/_bootstrap.tgz"
fi

exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 4 --timeout 300 server:app
