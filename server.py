#!/usr/bin/env python3
"""GIF Curation Server for Geometry Dash Godot.

Usage:
  cp .env.example .env      # add your GIPHY_API_KEY
  .venv312/bin/python server.py
  open http://localhost:8080
"""

import os, json, hashlib, time, io, threading, base64, glob, random, subprocess, shutil, tempfile
import requests
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from flask import Flask, request, jsonify, render_template, send_file, Response
from dotenv import dotenv_values
from scipy.spatial import ConvexHull

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit
app.config["TEMPLATES_AUTO_RELOAD"] = True

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA       = os.path.join(BASE, "data")
GIFS_DIR   = os.path.join(DATA, "gifs")
FRAMES_DIR = os.path.join(DATA, "frames")
POOL_FILE  = os.path.join(DATA, "pool.json")
CAND_FILE  = os.path.join(DATA, "candidates.json")

for d in [DATA, GIFS_DIR, FRAMES_DIR]:
    os.makedirs(d, exist_ok=True)

cfg = dotenv_values(os.path.join(BASE, ".env"))
GIPHY_KEY = cfg.get("GIPHY_API_KEY", "")

OBS_SIZE   = (64, 64)
BG_SIZE    = (640, 180)
MAX_FRAMES = 20
POLY_PTS   = 20   # target polygon vertices per frame after simplification

# Global rembg session (lazy-loaded, shared across requests)
_rembg_session = None
_rembg_lock    = threading.Lock()

# Subject-processing progress tracker
_subject_progress = {"running": False, "done": 0, "total": 0, "current": ""}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_pool():
    pool = load_json(POOL_FILE, {"obstacles": [], "backgrounds": []})
    for key in ("obstacles", "backgrounds"):
        for entry in pool[key]:
            meta_path = os.path.join(FRAMES_DIR, entry["id"], "meta.json")
            meta = load_json(meta_path, {})
            entry["polygons"]    = meta.get("polygons", [])
            entry["has_subject"] = bool(entry["polygons"])
            # Fill missing preview_url for uploaded/approved GIFs
            if not entry.get("preview_url"):
                gif_path = os.path.join(GIFS_DIR, f"{entry['id']}.gif")
                if os.path.exists(gif_path):
                    entry["preview_url"] = f"/api/gif/{entry['id']}"
    return pool

def load_candidates():
    return load_json(CAND_FILE, [])


# ── GIF frame extraction ──────────────────────────────────────────────────────

def extract_frames(gif_path, target_size):
    gif = Image.open(gif_path)
    try:
        duration_ms = gif.info.get("duration", 100) or 100
        fps = round(1000.0 / max(duration_ms, 10), 2)
    except Exception:
        fps = 10.0

    raw = []
    i = 0
    while True:
        try:
            gif.seek(i)
            raw.append(gif.convert("RGBA").resize(target_size, Image.LANCZOS))
            i += 1
        except EOFError:
            break

    if not raw:
        raise ValueError("No frames found in GIF")

    if len(raw) > MAX_FRAMES:
        step = len(raw) / MAX_FRAMES
        raw = [raw[int(j * step)] for j in range(MAX_FRAMES)]

    return raw, fps


def build_spritesheet(frames, gif_id, filename="spritesheet.png"):
    w, h = frames[0].size
    sheet = Image.new("RGBA", (w * len(frames), h))
    for idx, frame in enumerate(frames):
        sheet.paste(frame, (idx * w, 0))
    path = os.path.join(FRAMES_DIR, gif_id, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path, "PNG")
    return path


def process_gif(gif_path, gif_id, gif_type):
    meta_path = os.path.join(FRAMES_DIR, gif_id, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)

    size = OBS_SIZE if gif_type == "obstacle" else BG_SIZE
    frames, fps = extract_frames(gif_path, size)
    build_spritesheet(frames, gif_id)

    meta = {
        "id":           gif_id,
        "frame_count":  len(frames),
        "fps":          fps,
        "frame_width":  size[0],
        "frame_height": size[1],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return meta


def download_gif(url, gif_id):
    path = os.path.join(GIFS_DIR, f"{gif_id}.gif")
    if not os.path.exists(path):
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


# ── Subject extraction (rembg + convex hull) ──────────────────────────────────

def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        with _rembg_lock:
            if _rembg_session is None:
                from rembg import new_session
                _rembg_session = new_session("u2netp")  # fast lightweight model
    return _rembg_session


def _douglas_peucker(points, epsilon):
    """Simplify a polygon using Douglas-Peucker. points: Nx2 numpy array."""
    if len(points) <= 2:
        return points
    # Find the point with the maximum distance from the line start→end
    start, end = points[0], points[-1]
    line = end - start
    line_len = np.linalg.norm(line)
    if line_len == 0:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        dists = np.abs(np.cross(line, start - points)) / line_len
    idx = np.argmax(dists)
    if dists[idx] > epsilon:
        left  = _douglas_peucker(points[:idx + 1], epsilon)
        right = _douglas_peucker(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.array([start, end])


def _alpha_to_polygon(rgba_img, target_pts=POLY_PTS):
    """
    Given an RGBA PIL image (after rembg), extract the subject silhouette
    as a normalized polygon. Returns list of [x,y] pairs in [0,1] range,
    or [] if subject is too sparse.
    """
    alpha = np.array(rgba_img)[:, :, 3]  # H x W
    mask  = alpha > 15                    # threshold

    coords = np.argwhere(mask)  # (row, col) = (y, x)
    if len(coords) < 20:
        return []  # not enough subject pixels

    # Build convex hull from foreground pixel coords
    try:
        hull = ConvexHull(coords)
        hull_pts = coords[hull.vertices]  # Mx2 in (row,col) order
    except Exception:
        return []

    # Convert (row,col) → (x,y) and normalize
    h, w = alpha.shape
    xy = hull_pts[:, ::-1].astype(float)  # flip to (col,row) = (x,y)
    xy[:, 0] /= w
    xy[:, 1] /= h

    # Order hull counter-clockwise and simplify
    cx, cy = xy.mean(axis=0)
    angles  = np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)
    order   = np.argsort(angles)
    xy      = xy[order]

    # Douglas-Peucker to reduce to ~target_pts points
    epsilon = 0.02
    for _ in range(15):
        simplified = _douglas_peucker(np.vstack([xy, xy[0]]), epsilon)
        if len(simplified) <= target_pts + 2:
            break
        epsilon *= 1.3

    pts = simplified[:-1].tolist()  # remove the closing duplicate
    return [[round(p[0], 4), round(p[1], 4)] for p in pts]


def extract_subjects(gif_path, gif_id, gif_type):
    """
    Run rembg on every frame of the GIF, extract per-frame polygons,
    save a subject_spritesheet.png (RGBA, transparent bg), and return
    the list of per-frame polygon lists.
    Skips if already cached.
    """
    meta_path    = os.path.join(FRAMES_DIR, gif_id, "meta.json")
    subject_path = os.path.join(FRAMES_DIR, gif_id, "subject_spritesheet.png")
    meta         = load_json(meta_path, {})

    if meta.get("polygons") and os.path.exists(subject_path):
        return meta["polygons"]

    gif_file = os.path.join(GIFS_DIR, f"{gif_id}.gif")
    if not os.path.exists(gif_file):
        return []

    size = OBS_SIZE if gif_type == "obstacle" else BG_SIZE

    try:
        frames, _ = extract_frames(gif_file, size)
    except Exception as e:
        print(f"[subject] frame extraction failed for {gif_id}: {e}")
        return []

    try:
        from rembg import remove
        session = _get_rembg_session()
    except Exception as e:
        print(f"[subject] rembg unavailable: {e}")
        return []

    subject_frames = []
    polygons       = []

    for i, frame in enumerate(frames):
        try:
            removed = remove(frame, session=session, alpha_matting=False)
            subject_frames.append(removed)
            poly = _alpha_to_polygon(removed)
            polygons.append(poly)
        except Exception as e:
            print(f"[subject] frame {i} failed for {gif_id}: {e}")
            subject_frames.append(frame)
            polygons.append([])

    # Save subject spritesheet
    build_spritesheet(subject_frames, gif_id, "subject_spritesheet.png")

    # Persist polygons into meta.json
    meta["polygons"] = polygons
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return polygons


# ── Preview composite image ───────────────────────────────────────────────────

def _neon_composite(subject_frame, poly_norm, size=192):
    """Generate a dark-bg neon-style composite PNG bytes for the preview."""
    bg = Image.new("RGBA", (size, size), (10, 10, 20, 255))

    # Scale and centre subject
    subj = subject_frame.resize((size, size), Image.LANCZOS)
    bg.paste(subj, (0, 0), subj)

    if poly_norm:
        draw = ImageDraw.Draw(bg)
        pts  = [(p[0] * size, p[1] * size) for p in poly_norm]
        pts_closed = pts + [pts[0]]
        # Glow: draw several times with decreasing opacity/increasing width
        for width, alpha in [(12, 60), (7, 120), (4, 200), (2, 255)]:
            color = (180, 80, 255, alpha)
            draw.line(pts_closed, fill=color, width=width)
        # Bright core line
        draw.line(pts_closed, fill=(230, 160, 255, 255), width=1)

    buf = io.BytesIO()
    bg.convert("RGB").save(buf, "PNG")
    buf.seek(0)
    return buf


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    from flask import make_response
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/search")
def search():
    q        = request.args.get("q", "italian brainrot")
    gif_type = request.args.get("type", "obstacle")
    limit    = min(int(request.args.get("limit", 24)), 50)

    if not GIPHY_KEY:
        return jsonify({"error": "GIPHY_API_KEY not set in gif_server/.env"}), 500

    try:
        r = requests.get(
            "https://api.giphy.com/v1/gifs/search",
            params={"api_key": GIPHY_KEY, "q": q, "limit": limit, "rating": "pg-13"},
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    pool       = load_pool()
    approved   = {g["id"] for g in pool["obstacles"] + pool["backgrounds"]}
    candidates = load_candidates()
    cand_index = {c["id"]: c for c in candidates}

    results = []
    for gif in r.json().get("data", []):
        gid    = gif["id"]
        imgs   = gif.get("images", {})
        dl_url = (imgs.get("fixed_height") or {}).get("url", "")
        prev   = (imgs.get("preview_gif") or imgs.get("fixed_height_small") or {}).get("url", "")
        title  = gif.get("title", "")

        if not dl_url:
            continue

        if gid not in cand_index and gid not in approved:
            entry = {"id": gid, "type": gif_type, "preview_url": prev, "gif_url": dl_url, "title": title}
            candidates.append(entry)
            cand_index[gid] = entry

        results.append({"id": gid, "preview_url": prev, "gif_url": dl_url,
                         "title": title, "approved": gid in approved})

    save_json(CAND_FILE, candidates)
    return jsonify({"results": results})


@app.route("/api/approve/<gif_id>", methods=["POST"])
def approve(gif_id):
    gif_type   = request.args.get("type", "obstacle")
    candidates = load_candidates()
    cand       = next((c for c in candidates if c["id"] == gif_id), None)

    gif_url = (request.get_json(silent=True) or {}).get("gif_url", "")
    if cand:
        gif_url = cand.get("gif_url", gif_url)

    if not gif_url:
        return jsonify({"error": "gif_url not found"}), 400

    try:
        gif_path = download_gif(gif_url, gif_id)
        meta     = process_gif(gif_path, gif_id, gif_type)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    pool = load_pool()
    key  = "obstacles" if gif_type == "obstacle" else "backgrounds"
    pool[key] = [g for g in pool[key] if g["id"] != gif_id]
    # Strip enriched fields before saving
    entry = {**meta, "preview_url": cand.get("preview_url", "") if cand else ""}
    # Preserve existing polygons key from meta if present
    raw_meta = load_json(os.path.join(FRAMES_DIR, gif_id, "meta.json"), {})
    if raw_meta.get("polygons"):
        entry["polygons"] = raw_meta["polygons"]
    pool[key].append(entry)
    save_json(POOL_FILE, pool)

    return jsonify({"ok": True, "meta": meta})


@app.route("/api/reject/<gif_id>", methods=["POST"])
def reject(gif_id):
    pool = load_pool()
    pool["obstacles"]   = [g for g in pool["obstacles"]   if g["id"] != gif_id]
    pool["backgrounds"] = [g for g in pool["backgrounds"] if g["id"] != gif_id]
    save_json(POOL_FILE, pool)
    candidates = [c for c in load_candidates() if c["id"] != gif_id]
    save_json(CAND_FILE, candidates)
    return jsonify({"ok": True})


@app.route("/api/pool")
def get_pool():
    return jsonify(load_pool())


@app.route("/api/gif/<gif_id>")
def get_raw_gif(gif_id):
    path = os.path.join(GIFS_DIR, f"{gif_id}.gif")
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/gif")


@app.route("/api/spritesheet/<gif_id>")
def get_spritesheet(gif_id):
    path = os.path.join(FRAMES_DIR, gif_id, "spritesheet.png")
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/subject/<gif_id>")
def get_subject(gif_id):
    path = os.path.join(FRAMES_DIR, gif_id, "subject_spritesheet.png")
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/subject_frame/<gif_id>")
def get_subject_frame(gif_id):
    """Return the first frame of the subject spritesheet as a standalone PNG."""
    meta_path = os.path.join(FRAMES_DIR, gif_id, "meta.json")
    subj_path = os.path.join(FRAMES_DIR, gif_id, "subject_spritesheet.png")
    if not os.path.exists(subj_path):
        return jsonify({"error": "not found"}), 404
    meta = load_json(meta_path, {})
    fw = int(meta.get("frame_width", 64))
    fh = int(meta.get("frame_height", 64))
    sheet = Image.open(subj_path).convert("RGBA")
    frame = sheet.crop((0, 0, fw, fh))
    buf = io.BytesIO()
    frame.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/subject_gif/<gif_id>")
def get_subject_gif(gif_id):
    """Return subject spritesheet as an animated GIF (RGBA frames on dark bg)."""
    meta_path = os.path.join(FRAMES_DIR, gif_id, "meta.json")
    subj_path = os.path.join(FRAMES_DIR, gif_id, "subject_spritesheet.png")
    if not os.path.exists(subj_path):
        return jsonify({"error": "not found"}), 404
    meta  = load_json(meta_path, {})
    fw    = int(meta.get("frame_width",  64))
    fh    = int(meta.get("frame_height", 64))
    fps   = float(meta.get("fps", 10) or 10)
    sheet = Image.open(subj_path).convert("RGBA")
    n     = sheet.width // fw
    frames = []
    for i in range(n):
        f = sheet.crop((i * fw, 0, (i + 1) * fw, fh))
        bg = Image.new("RGBA", (fw, fh), (10, 10, 20, 255))
        bg.paste(f, (0, 0), f)
        frames.append(bg.convert("P", palette=Image.ADAPTIVE, colors=256))
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   loop=0, duration=int(1000 / fps))
    buf.seek(0)
    resp = send_file(buf, mimetype="image/gif")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/preview/<gif_id>")
def get_preview(gif_id):
    """Return a composite preview PNG: original | subject cut | neon sim."""
    meta_path = os.path.join(FRAMES_DIR, gif_id, "meta.json")
    meta      = load_json(meta_path, {})

    fw = int(meta.get("frame_width",  64))
    fh = int(meta.get("frame_height", 64))

    PANEL = 192  # each panel rendered at this square size
    GAP   = 8
    PANELS = 3
    W = PANEL * PANELS + GAP * (PANELS - 1)
    H = PANEL + 60  # extra space for labels

    out = Image.new("RGB", (W, H), (12, 12, 20))
    draw = ImageDraw.Draw(out)

    def paste_panel(img, idx, label):
        x = idx * (PANEL + GAP)
        thumb = img.resize((PANEL, PANEL), Image.LANCZOS)
        if thumb.mode == "RGBA":
            bg = Image.new("RGB", (PANEL, PANEL), (20, 20, 35))
            bg.paste(thumb, (0, 0), thumb)
            thumb = bg
        out.paste(thumb, (x, 0))
        draw.text((x + PANEL // 2 - len(label) * 3, PANEL + 6), label, fill=(150, 150, 200))

    # Panel 1 — original first frame
    orig_path = os.path.join(FRAMES_DIR, gif_id, "spritesheet.png")
    if os.path.exists(orig_path):
        sheet = Image.open(orig_path)
        orig  = sheet.crop((0, 0, fw, fh))
        paste_panel(orig, 0, "Original")
    else:
        draw.text((4, 4), "No spritesheet", fill=(200, 80, 80))

    # Panel 2 — subject cut (first frame)
    subj_path = os.path.join(FRAMES_DIR, gif_id, "subject_spritesheet.png")
    if os.path.exists(subj_path):
        ssheet = Image.open(subj_path)
        subj   = ssheet.crop((0, 0, fw, fh))
        paste_panel(subj, 1, "Subject cut")
    else:
        draw.text((PANEL + GAP + 4, 4), "Not processed", fill=(200, 200, 80))

    # Panel 3 — neon simulation
    if os.path.exists(subj_path):
        polys = meta.get("polygons", [])
        poly0 = polys[0] if polys else []
        neon_buf = _neon_composite(subj, poly0, PANEL)
        neon_img = Image.open(neon_buf)
        out.paste(neon_img, (2 * (PANEL + GAP), 0))
        draw.text((2 * (PANEL + GAP) + PANEL // 2 - 18, PANEL + 6), "Neon preview", fill=(150, 150, 200))

    buf = io.BytesIO()
    out.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/process_all_subjects", methods=["POST"])
def process_all_subjects():
    """Queue subject extraction for all approved GIFs lacking polygon data."""
    if _subject_progress["running"]:
        return jsonify({"ok": False, "error": "Already running"}), 409

    pool = load_pool()
    all_entries = []
    for entry in pool["obstacles"]:
        all_entries.append((entry["id"], "obstacle"))
    for entry in pool["backgrounds"]:
        all_entries.append((entry["id"], "background"))

    # Filter to only those that need processing
    to_process = []
    for gif_id, gif_type in all_entries:
        meta = load_json(os.path.join(FRAMES_DIR, gif_id, "meta.json"), {})
        subj = os.path.join(FRAMES_DIR, gif_id, "subject_spritesheet.png")
        if not (meta.get("polygons") and os.path.exists(subj)):
            to_process.append((gif_id, gif_type))

    if not to_process:
        return jsonify({"ok": True, "message": "All GIFs already processed"})

    def _run():
        _subject_progress["running"] = True
        _subject_progress["total"]   = len(to_process)
        _subject_progress["done"]    = 0
        for gif_id, gif_type in to_process:
            _subject_progress["current"] = gif_id
            print(f"[subject] processing {gif_id} ({gif_type})…")
            extract_subjects(os.path.join(GIFS_DIR, f"{gif_id}.gif"), gif_id, gif_type)
            _subject_progress["done"] += 1
        _subject_progress["running"] = False
        _subject_progress["current"] = ""
        print("[subject] all done")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "queued": len(to_process)})


@app.route("/api/subject_progress")
def subject_progress():
    return jsonify(_subject_progress)


@app.route("/api/upload", methods=["POST"])
def upload():
    gif_type = request.form.get("type", "obstacle")
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file provided"}), 400

    raw = f.read()
    gif_id = "upload_" + hashlib.md5(raw + str(time.time()).encode()).hexdigest()[:12]
    gif_path = os.path.join(GIFS_DIR, f"{gif_id}.gif")
    with open(gif_path, "wb") as fh:
        fh.write(raw)

    try:
        meta = process_gif(gif_path, gif_id, gif_type)
        if not meta:
            return jsonify({"error": "Could not extract frames — make sure it's a GIF file"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    pool = load_pool()
    key  = "obstacles" if gif_type == "obstacle" else "backgrounds"
    pool[key] = [g for g in pool[key] if g["id"] != gif_id]
    pool[key].append({**meta, "preview_url": f"/api/gif/{gif_id}"})
    save_json(POOL_FILE, pool)

    return jsonify({"ok": True, "meta": meta})


@app.route("/edit/<gif_id>")
def edit_gif(gif_id):
    meta = load_json(os.path.join(FRAMES_DIR, gif_id, "meta.json"), {})
    if not meta:
        return "GIF not found in frames directory", 404
    return render_template("editor.html", gif_id=gif_id, meta=meta)


@app.route("/api/frame/<gif_id>/<int:idx>")
def get_frame_single(gif_id, idx):
    meta = load_json(os.path.join(FRAMES_DIR, gif_id, "meta.json"), {})
    fw   = int(meta.get("frame_width",  64))
    fh   = int(meta.get("frame_height", 64))
    path = os.path.join(FRAMES_DIR, gif_id, "spritesheet.png")
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    img   = Image.open(path).convert("RGBA")
    frame = img.crop((idx * fw, 0, (idx + 1) * fw, fh))
    buf   = io.BytesIO()
    frame.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/subject_frame/<gif_id>/<int:idx>")
def get_subject_frame_single(gif_id, idx):
    meta = load_json(os.path.join(FRAMES_DIR, gif_id, "meta.json"), {})
    fw   = int(meta.get("frame_width",  64))
    fh   = int(meta.get("frame_height", 64))
    path = os.path.join(FRAMES_DIR, gif_id, "subject_spritesheet.png")
    if not os.path.exists(path):
        return ("", 204)
    img   = Image.open(path).convert("RGBA")
    frame = img.crop((idx * fw, 0, (idx + 1) * fw, fh))
    buf   = io.BytesIO()
    frame.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/save_mask/<gif_id>", methods=["POST"])
def save_mask(gif_id):
    """
    Body: { "masks": ["<base64-png>", ...] }  — one per frame.
    Each mask PNG: green pixels (G>100, R<100) = keep, red (R>100, G<100) = remove,
    alpha=0 = leave the rembg result unchanged.
    Applies masks to existing subject spritesheet, rebuilds it, recomputes polygons.
    """
    data       = request.get_json(force=True)
    masks_b64  = data.get("masks", [])

    meta_path  = os.path.join(FRAMES_DIR, gif_id, "meta.json")
    meta       = load_json(meta_path, {})
    fw         = int(meta.get("frame_width",  64))
    fh         = int(meta.get("frame_height", 64))
    frame_count= int(meta.get("frame_count",  1))

    orig_path  = os.path.join(FRAMES_DIR, gif_id, "spritesheet.png")
    subj_path  = os.path.join(FRAMES_DIR, gif_id, "subject_spritesheet.png")

    if not os.path.exists(orig_path):
        return jsonify({"error": "spritesheet not found"}), 404

    orig_sheet = Image.open(orig_path).convert("RGBA")
    subj_sheet = Image.open(subj_path).convert("RGBA") if os.path.exists(subj_path) else None

    new_frames = []
    new_polys  = []

    for i in range(frame_count):
        orig_f = orig_sheet.crop((i * fw, 0, (i + 1) * fw, fh))
        subj_f = subj_sheet.crop((i * fw, 0, (i + 1) * fw, fh)).convert("RGBA") \
                 if subj_sheet else orig_f.copy()

        if i < len(masks_b64) and masks_b64[i]:
            try:
                raw      = base64.b64decode(masks_b64[i])
                mask_img = Image.open(io.BytesIO(raw)).convert("RGBA").resize((fw, fh), Image.NEAREST)
                s_arr    = np.array(subj_f)
                m_arr    = np.array(mask_img)
                o_arr    = np.array(orig_f)

                painted   = m_arr[:, :, 3] > 32
                keep_px   = painted & (m_arr[:, :, 1] > 100) & (m_arr[:, :, 0] < 100)
                remove_px = painted & (m_arr[:, :, 0] > 100) & (m_arr[:, :, 1] < 100)

                s_arr[keep_px]    = o_arr[keep_px]
                s_arr[keep_px, 3] = 255
                s_arr[remove_px, 3] = 0

                subj_f = Image.fromarray(s_arr, "RGBA")
            except Exception as e:
                print(f"[mask] frame {i} error: {e}")

        new_frames.append(subj_f)
        new_polys.append(_alpha_to_polygon(subj_f))

    build_spritesheet(new_frames, gif_id, "subject_spritesheet.png")
    meta["polygons"] = new_polys
    save_json(meta_path, meta)

    return jsonify({"ok": True, "polygons": new_polys})


# ── Google Photos integration ─────────────────────────────────────────────────

try:
    import imageio_ffmpeg as _iio_ffmpeg
    _FFMPEG = _iio_ffmpeg.get_ffmpeg_exe()
except Exception:
    # Use the known bundled binary from the system imageio_ffmpeg install
    _BUNDLED = "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-x86_64-v7.1"
    _FFMPEG = _BUNDLED if os.path.exists(_BUNDLED) else "ffmpeg"

print(f"[server] ffmpeg: {_FFMPEG}")

_gphotos_progress = {
    "running": False, "stage": "idle",
    "done": 0, "total": 0, "current_year": "",
    "errors": 0, "total_videos": 0,
}

GPHOTOS_CLIPS_DIR        = os.path.join(DATA, "gphotos_clips")
GPHOTOS_CAND_FILE        = os.path.join(DATA, "gphotos_candidates.json")
GPHOTOS_TOKEN_FILE       = os.path.join(DATA, "gphotos_token.json")
GPHOTOS_PICKER_FILE      = os.path.join(DATA, "gphotos_picker_session.json")
GP_SCOPES                = ["https://www.googleapis.com/auth/photospicker.mediaitems.readonly"]
GP_REDIRECT_URI          = "http://localhost:8080/api/gphotos/callback"

os.makedirs(GPHOTOS_CLIPS_DIR, exist_ok=True)


def _gp_client_config():
    return {"web": {
        "client_id":     cfg.get("GOOGLE_CLIENT_ID", ""),
        "client_secret": cfg.get("GOOGLE_CLIENT_SECRET", ""),
        "redirect_uris": [GP_REDIRECT_URI],
        "auth_uri":  "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }}


def _load_gp_creds():
    if not os.path.exists(GPHOTOS_TOKEN_FILE):
        return None
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        creds = Credentials.from_authorized_user_file(GPHOTOS_TOKEN_FILE, GP_SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _save_gp_creds(creds)
        return creds if creds.valid else None
    except Exception as e:
        print(f"[gphotos] creds error: {e}")
        return None


def _save_gp_creds(creds):
    with open(GPHOTOS_TOKEN_FILE, "w") as f:
        f.write(creds.to_json())


def _gp_session(creds):
    from google.auth.transport.requests import AuthorizedSession
    return AuthorizedSession(creds)


def _list_all_videos(creds):
    session = _gp_session(creds)
    videos, page_token = [], None
    while True:
        body = {
            "filters": {"mediaTypeFilter": {"mediaTypes": ["VIDEO"]}},
            "pageSize": 100,
        }
        if page_token:
            body["pageToken"] = page_token
        r = session.post(
            "https://photoslibrary.googleapis.com/v1/mediaItems:search",
            json=body, timeout=30)
        if r.status_code != 200:
            print(f"[gphotos] list error {r.status_code}: {r.text[:300]}")
            break
        data = r.json()
        videos.extend(data.get("mediaItems", []))
        _gphotos_progress["total_videos"] = len(videos)
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return videos


def _stratified_sample(videos, target):
    from collections import defaultdict
    by_year = defaultdict(list)
    for v in videos:
        yr = (v.get("mediaMetadata", {}).get("creationTime") or "0000")[:4]
        by_year[yr].append(v)
    years = sorted(by_year)
    if not years:
        return []
    base, rem = divmod(target, len(years))
    sampled = []
    for i, yr in enumerate(years):
        slots = base + (1 if i < rem else 0)
        sampled.extend(random.sample(by_year[yr], min(slots, len(by_year[yr]))))
    random.shuffle(sampled)
    return sampled[:target]


def _get_video_duration(path):
    try:
        import av
        with av.open(path) as c:
            return float(c.duration) / 1e6 if c.duration else None
    except Exception:
        return None


def _extract_clip_frames(video_path, start_sec, duration=3.0, scale_w=320):
    tmpdir = tempfile.mkdtemp()
    try:
        out_pat = os.path.join(tmpdir, "%04d.png")
        cmd = [
            _FFMPEG,
            "-ss", str(max(0.0, start_sec)),
            "-i", video_path,
            "-t", str(duration),
            "-vf", f"fps=10,scale={scale_w}:-2:flags=lanczos",
            out_pat, "-y",
        ]
        result = subprocess.run(cmd, timeout=90, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[gphotos] ffmpeg exit {result.returncode}: {result.stderr[-300:]}")
        elif not glob.glob(os.path.join(tmpdir, "*.png")):
            print(f"[gphotos] ffmpeg ok but 0 frames.\nSTDERR:\n{result.stderr[:2000]}")
        files = sorted(glob.glob(os.path.join(tmpdir, "*.png")))
        return [Image.open(f).convert("RGBA").copy() for f in files]
    except Exception as e:
        print(f"[gphotos] ffmpeg error: {e}")
        return []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _save_preview_gif(frames, clip_id):
    path = os.path.join(GPHOTOS_CLIPS_DIR, f"{clip_id}.gif")
    if not frames:
        return None
    rgb = [f.convert("RGB") for f in frames]
    rgb[0].save(path, save_all=True, append_images=rgb[1:], loop=0, duration=100)
    return path


def _save_clip_frames(frames, clip_id):
    d = os.path.join(GPHOTOS_CLIPS_DIR, f"{clip_id}_frames")
    os.makedirs(d, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(d, f"{i:04d}.png"))
    return d


def _run_picker_processing(items):
    _gphotos_progress.update({
        "running": True, "stage": "processing",
        "done": 0, "total": len(items), "current_year": "",
        "errors": 0, "total_videos": len(items),
    })
    try:
        candidates = load_json(GPHOTOS_CAND_FILE, [])
        existing   = {c["gp_id"] for c in candidates}

        for item in items:
            gp_id      = item["id"]
            media_file = item.get("mediaFile", {})
            meta       = media_file.get("mediaFileMetadata", {})
            year       = (meta.get("creationTime") or item.get("createTime") or "")[:4]
            filename   = media_file.get("filename", "")
            base_url   = media_file.get("baseUrl", "")

            _gphotos_progress["current_year"] = year

            if gp_id in existing:
                _gphotos_progress["done"] += 1
                continue

            if not base_url:
                _gphotos_progress["errors"] += 1
                _gphotos_progress["done"]   += 1
                continue

            try:
                creds   = _load_gp_creds()
                session = _gp_session(creds)

                # For video items, =dv gives the actual video; base_url alone
                # returns only a thumbnail preview image (~25-50KB).
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                downloaded = False
                for video_url in [base_url + "=dv", base_url]:
                    try:
                        head = session.head(video_url, timeout=10)
                        cl   = int(head.headers.get("Content-Length", 0))
                        if cl > 300 * 1024 * 1024:
                            print(f"[gphotos] {filename} too large ({cl//1024//1024}MB), skipping")
                            break
                    except Exception:
                        pass

                    resp = session.get(video_url, stream=True, timeout=180)
                    print(f"[gphotos] {filename} url={video_url[:60]}… status={resp.status_code}")
                    if resp.status_code == 200:
                        for chunk in resp.iter_content(65536):
                            tmp.write(chunk)
                        downloaded = True
                        break
                tmp.close()

                file_size = os.path.getsize(tmp.name)
                if not downloaded or file_size < 500 * 1024:
                    print(f"[gphotos] {filename} download failed or too small ({file_size//1024}KB — likely thumbnail)")
                    _gphotos_progress["errors"] += 1
                    _gphotos_progress["done"]   += 1
                    os.unlink(tmp.name)
                    continue
                print(f"[gphotos] {filename} downloaded {file_size//1024}KB")

                dur   = _get_video_duration(tmp.name)
                start = 0.0
                if dur and dur > 4.0:
                    start = random.uniform(0, dur - 3.5)

                frames = _extract_clip_frames(tmp.name, start, duration=3.0, scale_w=320)
                os.unlink(tmp.name)

                if not frames:
                    print(f"[gphotos] {filename} no frames extracted (dur={dur})")
                    _gphotos_progress["errors"] += 1
                    _gphotos_progress["done"]   += 1
                    continue

                clip_id = "gp_" + hashlib.md5(gp_id.encode()).hexdigest()[:12]
                _save_preview_gif(frames, clip_id)
                _save_clip_frames(frames, clip_id)

                candidates.append({
                    "clip_id":     clip_id,
                    "gp_id":       gp_id,
                    "year":        year,
                    "filename":    filename,
                    "duration":    round(dur or 0, 1),
                    "start_sec":   round(start, 2),
                    "fps":         10.0,
                    "frame_count": len(frames),
                    "preview_url": f"/api/gphotos/preview/{clip_id}",
                    "status":      "pending",
                })
                existing.add(gp_id)
                save_json(GPHOTOS_CAND_FILE, candidates)

            except Exception as e:
                import traceback
                print(f"[gphotos] {filename or gp_id} error: {e}")
                traceback.print_exc()
                _gphotos_progress["errors"] += 1

            _gphotos_progress["done"] += 1

    finally:
        _gphotos_progress.update({"running": False, "stage": "idle", "current_year": ""})
        print("[gphotos] picker processing complete")


# ── Google Photos routes ───────────────────────────────────────────────────────

@app.route("/api/gphotos/status")
def gphotos_status():
    has_creds = bool(cfg.get("GOOGLE_CLIENT_ID") and cfg.get("GOOGLE_CLIENT_SECRET"))
    authed    = _load_gp_creds() is not None
    return jsonify({"configured": has_creds, "authed": authed})


@app.route("/api/gphotos/auth")
def gphotos_auth():
    from google_auth_oauthlib.flow import Flow
    from flask import redirect
    if not cfg.get("GOOGLE_CLIENT_ID") or not cfg.get("GOOGLE_CLIENT_SECRET"):
        return "Set GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET in gif_server/.env then restart.", 400
    flow = Flow.from_client_config(
        _gp_client_config(), scopes=GP_SCOPES, redirect_uri=GP_REDIRECT_URI)
    auth_url, state = flow.authorization_url(access_type="offline", prompt="consent")
    with open(os.path.join(DATA, "gphotos_state.txt"), "w") as f:
        f.write(state)
    # Persist PKCE verifier so the callback can restore it on the new Flow instance
    verifier = getattr(flow, "code_verifier", None)
    with open(os.path.join(DATA, "gphotos_verifier.txt"), "w") as f:
        f.write(verifier or "")
    return redirect(auth_url)


@app.route("/api/gphotos/callback")
def gphotos_callback():
    from google_auth_oauthlib.flow import Flow
    from flask import redirect
    state_path    = os.path.join(DATA, "gphotos_state.txt")
    verifier_path = os.path.join(DATA, "gphotos_verifier.txt")
    state    = open(state_path).read().strip()    if os.path.exists(state_path)    else None
    verifier = open(verifier_path).read().strip() if os.path.exists(verifier_path) else None
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # allow http on localhost
    flow = Flow.from_client_config(
        _gp_client_config(), scopes=GP_SCOPES, state=state, redirect_uri=GP_REDIRECT_URI)
    if verifier:
        flow.code_verifier = verifier
    flow.fetch_token(authorization_response=request.url)
    _save_gp_creds(flow.credentials)
    return redirect("/#gphotos")


@app.route("/api/gphotos/disconnect", methods=["POST"])
def gphotos_disconnect():
    if os.path.exists(GPHOTOS_TOKEN_FILE):
        os.unlink(GPHOTOS_TOKEN_FILE)
    return jsonify({"ok": True})


@app.route("/api/gphotos/create_picker_session", methods=["POST"])
def gphotos_create_picker_session():
    creds = _load_gp_creds()
    if not creds:
        return jsonify({"ok": False, "error": "Not authenticated"}), 401
    session = _gp_session(creds)
    r = session.post("https://photospicker.googleapis.com/v1/sessions", json={}, timeout=30)
    if r.status_code != 200:
        return jsonify({"ok": False, "error": f"Picker API {r.status_code}: {r.text[:200]}"}), 500
    data = r.json()
    session_id  = data.get("id")
    picker_uri  = data.get("pickerUri")
    save_json(GPHOTOS_PICKER_FILE, {"session_id": session_id, "picker_uri": picker_uri})
    return jsonify({"ok": True, "session_id": session_id, "picker_uri": picker_uri})


@app.route("/api/gphotos/picker_session_status")
def gphotos_picker_session_status():
    ps = load_json(GPHOTOS_PICKER_FILE, {})
    session_id = ps.get("session_id")
    if not session_id:
        return jsonify({"ok": False, "no_session": True})
    creds = _load_gp_creds()
    if not creds:
        return jsonify({"ok": False, "error": "Not authenticated"})
    sess = _gp_session(creds)
    r = sess.get(f"https://photospicker.googleapis.com/v1/sessions/{session_id}", timeout=15)
    if r.status_code != 200:
        return jsonify({"ok": False, "error": f"Session API {r.status_code}"})
    data = r.json()
    return jsonify({
        "ok": True,
        "session_id": session_id,
        "media_items_set": data.get("mediaItemsSet", False),
        "picker_uri": data.get("pickerUri") or ps.get("picker_uri", ""),
    })


@app.route("/api/gphotos/process_picker", methods=["POST"])
def gphotos_process_picker():
    if _gphotos_progress.get("running"):
        return jsonify({"ok": False, "error": "Already running"}), 409
    ps = load_json(GPHOTOS_PICKER_FILE, {})
    session_id = ps.get("session_id")
    if not session_id:
        return jsonify({"ok": False, "error": "No active picker session"}), 400
    creds = _load_gp_creds()
    if not creds:
        return jsonify({"ok": False, "error": "Not authenticated"}), 401
    sess = _gp_session(creds)
    all_items, page_token = [], None
    while True:
        params = {"sessionId": session_id, "pageSize": 100}
        if page_token:
            params["pageToken"] = page_token
        r = sess.get("https://photospicker.googleapis.com/v1/mediaItems", params=params, timeout=30)
        if r.status_code != 200:
            return jsonify({"ok": False, "error": f"Items API {r.status_code}: {r.text[:200]}"}), 500
        data = r.json()
        all_items.extend(i for i in data.get("mediaItems", []) if i.get("type") == "VIDEO")
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    if not all_items:
        return jsonify({"ok": False, "error": "No video items selected in picker"}), 400

    count = int(request.args.get("count", 0)) or len(all_items)
    if count < len(all_items):
        # Stratified sample by year from the picked videos
        from collections import defaultdict
        by_year = defaultdict(list)
        for v in all_items:
            ct = (v.get("mediaFile", {}).get("mediaFileMetadata", {}).get("creationTime")
                  or v.get("createTime") or "0000")
            by_year[ct[:4]].append(v)
        years = sorted(by_year)
        base, rem = divmod(count, len(years))
        sampled = []
        for i, yr in enumerate(years):
            slots = base + (1 if i < rem else 0)
            sampled.extend(random.sample(by_year[yr], min(slots, len(by_year[yr]))))
        random.shuffle(sampled)
        all_items = sampled[:count]

    threading.Thread(target=_run_picker_processing, args=(all_items,), daemon=True).start()
    return jsonify({"ok": True, "queued": len(all_items), "total_picked": len(all_items)})


@app.route("/api/gphotos/sample_progress")
def gphotos_sample_progress():
    return jsonify(_gphotos_progress)


@app.route("/api/gphotos/candidates")
def gphotos_candidates_list():
    cands = load_json(GPHOTOS_CAND_FILE, [])
    cands = [c for c in cands
             if c.get("status") != "pending"
             or os.path.exists(os.path.join(GPHOTOS_CLIPS_DIR, f"{c['clip_id']}.gif"))]
    return jsonify(cands)


@app.route("/api/gphotos/preview/<clip_id>")
def gphotos_preview(clip_id):
    path = os.path.join(GPHOTOS_CLIPS_DIR, f"{clip_id}.gif")
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    resp = send_file(path, mimetype="image/gif")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/gphotos/approve", methods=["POST"])
def gphotos_approve():
    data      = request.get_json(force=True)
    clip_id   = data.get("clip_id")
    clip_type = data.get("type", "obstacle")   # obstacle | background | both

    cands = load_json(GPHOTOS_CAND_FILE, [])
    cand  = next((c for c in cands if c["clip_id"] == clip_id), None)
    if not cand:
        return jsonify({"error": "clip not found"}), 404

    frames_dir  = os.path.join(GPHOTOS_CLIPS_DIR, f"{clip_id}_frames")
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_files:
        return jsonify({"error": "source frames missing"}), 404
    frames_raw = [Image.open(f).convert("RGBA") for f in frame_files]

    result = {}

    def _add(t):
        size   = OBS_SIZE if t == "obstacle" else BG_SIZE
        frames = [f.resize(size, Image.LANCZOS) for f in frames_raw]
        gif_id = "gp_" + hashlib.md5((clip_id + t).encode()).hexdigest()[:14]

        build_spritesheet(frames, gif_id)
        meta = {
            "id": gif_id, "frame_count": len(frames),
            "fps": cand.get("fps", 10.0),
            "frame_width": size[0], "frame_height": size[1],
        }
        save_json(os.path.join(FRAMES_DIR, gif_id, "meta.json"), meta)

        pool_raw = load_json(POOL_FILE, {"obstacles": [], "backgrounds": []})
        key = "obstacles" if t == "obstacle" else "backgrounds"
        pool_raw[key] = [g for g in pool_raw[key] if g["id"] != gif_id]
        pool_raw[key].append({**meta, "preview_url": f"/api/gif/{gif_id}"})
        save_json(POOL_FILE, pool_raw)

        # Store a GIF file so rembg can process it
        src = os.path.join(GPHOTOS_CLIPS_DIR, f"{clip_id}.gif")
        dst = os.path.join(GIFS_DIR, f"{gif_id}.gif")
        if os.path.exists(src):
            shutil.copy(src, dst)

        if t == "obstacle":
            threading.Thread(
                target=extract_subjects, args=(dst, gif_id, "obstacle"),
                daemon=True).start()
        return gif_id

    if clip_type in ("obstacle", "both"):
        result["gif_id"]    = _add("obstacle")
    if clip_type in ("background", "both"):
        result["gif_id_bg"] = _add("background")

    for c in cands:
        if c["clip_id"] == clip_id:
            c["status"] = clip_type
    save_json(GPHOTOS_CAND_FILE, cands)

    return jsonify({"ok": True, **result})


@app.route("/api/gphotos/skip", methods=["POST"])
def gphotos_skip():
    data    = request.get_json(force=True)
    clip_id = data.get("clip_id")
    cands   = load_json(GPHOTOS_CAND_FILE, [])
    for c in cands:
        if c["clip_id"] == clip_id:
            c["status"] = "skip"
    save_json(GPHOTOS_CAND_FILE, cands)
    for p in [
        os.path.join(GPHOTOS_CLIPS_DIR, f"{clip_id}.gif"),
        os.path.join(GPHOTOS_CLIPS_DIR, f"{clip_id}_frames"),
    ]:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.exists(p):
            os.unlink(p)
    return jsonify({"ok": True})


# ── NTFY sound picker ─────────────────────────────────────────────────────────

CLAUDE_SETTINGS = os.path.expanduser("~/.claude/settings.json")

def _read_current_ntfy_sound():
    import re
    try:
        text = open(CLAUDE_SETTINGS).read()
        # handles both unescaped "Sound: x" and JSON-escaped \"Sound: x\"
        m = re.search(r'-H\s+\\"Sound:\s+([^\\]+)\\"', text) or \
            re.search(r'-H\s+"Sound:\s+([^"]+)"', text)
        return m.group(1) if m else "default"
    except Exception:
        return "default"

@app.route("/ntfy-sounds")
def ntfy_sounds_page():
    return render_template("ntfy_sounds.html", current_sound=_read_current_ntfy_sound())

@app.route("/api/set_ntfy_sound", methods=["POST"])
def set_ntfy_sound():
    import re
    data = request.get_json(force=True)
    sound = data.get("sound", "").strip()
    valid = {"ding", "juntos", "pristine", "dadum", "pop", "pop-swoosh", "beep"}
    if sound not in valid:
        return jsonify({"ok": False, "error": f"Unknown sound: {sound}"}), 400
    try:
        text = open(CLAUDE_SETTINGS).read()
        # replace in JSON-escaped form (\\\"Sound: old\\\") and plain form
        new_text = re.sub(r'(-H\s+\\"Sound:\s+)([^\\"]+)(\\")', lambda m: m.group(1) + sound + m.group(3), text)
        if new_text == text:
            new_text = re.sub(r'(-H\s+"Sound:\s+)([^"]+)(")', lambda m: m.group(1) + sound + m.group(3), text)
        if new_text == text:
            return jsonify({"ok": False, "error": "No Sound header found in settings.json"}), 400
        open(CLAUDE_SETTINGS, "w").write(new_text)
        return jsonify({"ok": True, "sound": sound})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print(" GIF Curator — http://localhost:8080")
    if not GIPHY_KEY:
        print(" ⚠  GIPHY_API_KEY not set — copy .env.example to .env")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8080, debug=False)
