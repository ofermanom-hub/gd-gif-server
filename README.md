# GIF Curator Server

Flask service that fetches GIFs from Giphy, extracts subject silhouettes (rembg), computes convex-hull polygons, and serves them to the Geometry Dash Godot game.

## Run locally

```sh
cp .env.example .env   # fill in keys
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py       # http://localhost:8080
```

## Deploy to Fly.io via GitHub Actions

One-time setup:

1. `gh auth login` — re-auth GitHub CLI
2. Install [flyctl](https://fly.io/docs/flyctl/install/) and `fly auth signup`
3. Create the app + volume:
   ```sh
   fly apps create gd-gif-server
   fly volumes create gif_data --region fra --size 1
   fly secrets set GIPHY_API_KEY=xxx
   fly secrets set GOOGLE_CLIENT_ID=xxx GOOGLE_CLIENT_SECRET=xxx
   ```
4. Generate a Fly deploy token: `fly tokens create deploy -x 999999h`
5. Add it as `FLY_API_TOKEN` secret in the GitHub repo (Settings → Secrets → Actions)

Every push to `main` then builds a Docker image to `ghcr.io/<you>/<repo>` and deploys it to Fly.

## Persistent data

The Fly volume `gif_data` is mounted at `/app/data` (pool, frames, rembg model cache).
