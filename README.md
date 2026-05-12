# GIF Curator Server

Flask service that fetches GIFs from Giphy, extracts subject silhouettes (rembg), computes convex-hull polygons, and serves them to the Geometry Dash Godot game.

## Run locally

```sh
cp .env.example .env   # fill in keys
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py       # http://localhost:8080
```

## Deploy to Render

`render.yaml` is a Render Blueprint — Render auto-builds the Docker image and redeploys on every push to `main`.

One-time setup:

1. Open https://dashboard.render.com/blueprints → **New Blueprint Instance** → pick `ofermanom-hub/gd-gif-server` → apply.
2. In the new service's **Environment** tab, fill in the secrets (already declared in `render.yaml` as `sync: false`):
   - `GIPHY_API_KEY`
   - `GOOGLE_CLIENT_ID`
   - `GOOGLE_CLIENT_SECRET`
3. Done — Render builds and exposes the service at `https://gd-gif-server.onrender.com`.

### Caveats on the free plan

- 512 MB RAM — rembg + u2net is right at the edge; may OOM on heavy GIFs. Upgrade to Starter if so.
- No persistent disk — `/app/data` resets on every deploy/restart. Curated state is ephemeral until you upgrade.
- Service spins down after 15 min idle (~30s cold boot on next request).
