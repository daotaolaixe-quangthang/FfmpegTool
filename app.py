"""
app.py
======
Flask web UI for FfmpegTool.
Wraps the existing pipeline (main.py) in a beautiful web interface.

Usage:
    python app.py
    Then open: http://localhost:5000

Dependencies:
    pip install flask
"""

import os
import re
import sys
import uuid
import json
import html
import queue
import threading
import subprocess
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, send_from_directory

# ── Ensure FfmpegTool dir is in path ──
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOL_DIR)

app = Flask(__name__, template_folder=os.path.join(TOOL_DIR, "templates"))

# ── In-memory job store ──
# job_id -> {"queue": Queue, "status": str, "stats": dict}
JOBS: dict[str, dict] = {}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def build_command(data: dict) -> list[str]:
    """Build the main.py CLI command from UI form data."""
    cmd = [sys.executable, os.path.join(TOOL_DIR, "main.py")]

    # Input source
    if data.get("url", "").strip():
        cmd += ["--url", data["url"].strip()]
    elif data.get("batch", False):
        cmd += ["--batch", data["input"].strip()]
    else:
        cmd += ["--input", data["input"].strip()]

    cmd += ["--output", data["output"].strip()]

    # Extraction
    cmd += ["--mode", data.get("mode", "fps")]
    if data.get("mode", "fps") == "fps":
        cmd += ["--fps", str(data.get("fps", 5))]

    # Filters
    cmd += ["--blur", str(data.get("blur", 80))]
    cmd += ["--sim",  str(data.get("sim",  0.70))]
    cmd += ["--method", data.get("method", "phash")]

    # Scorer
    if data.get("top_n", 0) > 0:
        cmd += ["--top", str(data["top_n"])]

    # Options
    if data.get("keep_raw"):
        cmd += ["--keep-raw"]
    if data.get("no_html"):
        cmd += ["--no-html"]

    return cmd


def load_result_stats(output_dir: str, video_name: str) -> dict | None:
    """Try to load report.json from the output folder."""
    report_path = os.path.join(output_dir, video_name, "report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def resolve_single_video_dir(stats: dict) -> Path | None:
    """Resolve the per-video output folder from stats written by the CLI pipeline."""
    output_folder = stats.get("output_folder")
    if not output_folder:
        return None

    output_path = Path(os.path.normpath(output_folder))
    if output_path.name == "unique_frames":
        return output_path.parent
    return output_path


def resolve_preview_path(stats: dict) -> str:
    """Resolve the preview HTML path for either batch or single-video jobs."""
    if stats.get("is_batch"):
        preview_path = stats.get("batch_preview_path", "")
        return os.path.normpath(preview_path) if preview_path else ""

    video_dir = resolve_single_video_dir(stats)
    if not video_dir:
        return ""
    return os.path.normpath(str(video_dir / "preview.html"))


def resolve_open_folder(stats: dict) -> str:
    """Resolve the folder that should open in Explorer for the current job."""
    if stats.get("is_batch"):
        folder = stats.get("output_folder", "")
        return os.path.normpath(folder) if folder else ""

    video_dir = resolve_single_video_dir(stats)
    return os.path.normpath(str(video_dir)) if video_dir else ""


def resolve_preview_root(stats: dict) -> Path | None:
    """Resolve the directory that contains the preview HTML and its relative assets."""
    if stats.get("is_batch"):
        folder = stats.get("output_folder", "")
        return Path(os.path.normpath(folder)) if folder else None

    video_dir = resolve_single_video_dir(stats)
    return video_dir if video_dir else None


def load_batch_stats(output_dir: str) -> dict | None:
    """
    Read _batch_summary.json produced by main.py's process_batch().
    Falls back to scanning report.json files if summary doesn't exist.
    """
    output_path = Path(output_dir)
    if not output_path.is_dir():
        return None

    # ── Primary: use _batch_summary.json written by main.py ──
    summary_file = output_path / "_batch_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                s = json.load(f)

            video_results = s.get("video_results", [])
            total_raw    = sum(v.get("total_raw_frames",    0) for v in video_results)
            total_blur   = sum(v.get("removed_blurry",      0) for v in video_results)
            total_dup    = sum(v.get("removed_duplicate",   0) for v in video_results)
            total_unique = sum(v.get("final_unique_frames", 0) for v in video_results)

            derived_success = sum(1 for v in video_results if v.get("status") == "success")
            derived_skipped = sum(1 for v in video_results if v.get("status") == "skipped")
            derived_error   = sum(1 for v in video_results if v.get("status") == "error")

            success_n = s.get("success", derived_success)
            skipped_n = s.get("skipped", derived_skipped)
            error_n   = s.get("error", derived_error)
            failed_n  = s.get("failed", skipped_n + error_n)

            reduction = round(
                100.0 * (total_raw - total_unique) / total_raw, 1
            ) if total_raw > 0 else 0.0

            batch_html  = output_path / "_batch_preview.html"
            return {
                "is_batch":             True,
                "total_videos":         s.get("total_videos", success_n + skipped_n + error_n),
                "videos_processed":     success_n,
                "videos_skipped":       skipped_n,
                "videos_failed":        error_n,
                "videos_non_success":   failed_n,
                "total_raw_frames":     total_raw,
                "removed_blurry":       total_blur,
                "removed_duplicate":    total_dup,
                "final_unique_frames":  total_unique,
                "reduction_rate":       f"{reduction}%",
                "output_folder":        output_dir,
                "batch_preview_path":   str(batch_html),
                "batch_preview_exists": batch_html.exists(),
                "video_results":        video_results,  # includes status + error field
                "generated_at":         s.get("generated_at"),
            }
        except Exception:
            pass  # fall through to legacy scan

    # ── Fallback: scan individual report.json files (legacy behavior) ──
    total_raw    = 0
    total_blur   = 0
    total_dup    = 0
    total_unique = 0
    videos_found = 0
    video_results = []

    for sub in sorted(output_path.iterdir()):
        if not sub.is_dir():
            continue
        report_file = sub / "report.json"
        if not report_file.exists():
            continue
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                r = json.load(f)
            total_raw    += r.get("total_raw_frames",    0)
            total_blur   += r.get("removed_blurry",      0)
            total_dup    += r.get("removed_duplicate",   0)
            total_unique += r.get("final_unique_frames", 0)
            videos_found += 1
            video_results.append({
                "index":               videos_found,
                "video_name":          r.get("video_name", sub.name),
                "status":              "success",
                "total_raw_frames":    r.get("total_raw_frames",    0),
                "removed_blurry":      r.get("removed_blurry",      0),
                "removed_duplicate":   r.get("removed_duplicate",   0),
                "final_unique_frames": r.get("final_unique_frames", 0),
                "top_n_selected":      r.get("top_n_selected"),
                "output_folder":       r.get("output_folder", str(sub / "unique_frames")),
                "error":               None,
            })
        except Exception:
            continue

    if videos_found == 0:
        return None

    reduction = round(
        100.0 * (total_raw - total_unique) / total_raw, 1
    ) if total_raw > 0 else 0.0

    batch_html = output_path / "_batch_preview.html"
    return {
        "is_batch":             True,
        "total_videos":         videos_found,
        "videos_processed":     videos_found,
        "videos_skipped":       0,
        "videos_failed":        0,
        "videos_non_success":   0,
        "total_raw_frames":     total_raw,
        "removed_blurry":       total_blur,
        "removed_duplicate":    total_dup,
        "final_unique_frames":  total_unique,
        "reduction_rate":       f"{reduction}%",
        "output_folder":        output_dir,
        "batch_preview_path":   str(batch_html),
        "batch_preview_exists": batch_html.exists(),
        "video_results":        video_results,
        "generated_at":         None,
    }


def _is_tqdm_line(line: str) -> bool:
    """
    Detect tqdm progress bar lines to skip them in the web terminal.
    tqdm lines look like: '  Blur check:  33%|████      | 105/322 [00:01<00:02]'
    They contain a digit% followed immediately by a pipe character.
    NOTE: do NOT filter blank lines — they are used as visual separators in the terminal.
    """
    stripped = line.strip()
    if not stripped:
        return False  # keep blank lines as visual separators in terminal
    return bool(re.search(r'\d+%\|', stripped))


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    """Start a pipeline job. Returns job_id immediately."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    # Validate required fields
    has_input = data.get("url") or data.get("input")
    if not has_input:
        return jsonify({"error": "Provide an input file path or URL"}), 400
    if not data.get("output"):
        return jsonify({"error": "Output folder is required"}), 400

    job_id = str(uuid.uuid4())[:8]
    q: queue.Queue = queue.Queue()
    JOBS[job_id] = {"queue": q, "status": "running", "stats": None}

    cmd = build_command(data)

    def run_pipeline():
        JOBS[job_id]["cmd"] = " ".join(cmd)
        try:
            env = {
                **os.environ,
                "PYTHONIOENCODING": "utf-8",   # subprocess writes utf-8
                "PYTHONUNBUFFERED": "1",        # flush output immediately
            }
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # IMPORTANT: explicit utf-8 + replace so block chars don't crash
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=TOOL_DIR,
                env=env
            )
            for line in proc.stdout:
                line = line.rstrip()
                # Skip tqdm progress bar lines — they contain % and | and \r
                # and look like: "  Blur check:  66%|██████    | 212/322"
                if _is_tqdm_line(line):
                    continue
                q.put(("log", line))
            proc.wait()
            exit_code = proc.returncode
        except Exception as e:
            q.put(("log", f"[ERROR] {e}"))
            exit_code = 1

        # Try to load stats
        output_dir = data.get("output", "")

        # Determine if this was a batch run:
        #   - explicit batch flag, or
        #   - --input pointed at a directory (auto-batch)
        is_batch = (
            data.get("batch", False)
            or (
                not data.get("url", "").strip()
                and data.get("input", "").strip()
                and os.path.isdir(data.get("input", "").strip())
            )
        )

        if is_batch:
            stats = load_batch_stats(output_dir)
        elif data.get("url", "").strip():
            # URL mode saves file as 'downloaded_video.mp4'
            stats = load_result_stats(output_dir, "downloaded_video")
        else:
            video_name = Path(data.get("input", "video")).stem
            stats = load_result_stats(output_dir, video_name)

        JOBS[job_id]["stats"] = stats
        JOBS[job_id]["status"] = "done" if exit_code == 0 else "error"

        q.put(("done", str(exit_code)))

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    # Keep JOBS dict from growing unboundedly (keep last 20 jobs)
    if len(JOBS) > 20:
        oldest_key = next(iter(JOBS))
        del JOBS[oldest_key]

    return jsonify({"job_id": job_id, "command": " ".join(cmd)})


@app.route("/api/stream/<job_id>")
def api_stream(job_id: str):
    """Server-Sent Events stream for real-time log output."""
    job = JOBS.get(job_id)
    if not job:
        return "Job not found", 404

    q = job["queue"]

    def generate():
        while True:
            try:
                msg_type, msg = q.get(timeout=30)
                if msg_type == "log":
                    # Escape for SSE
                    safe = msg.replace("\n", " ").replace("\r", "")
                    yield f"data: {json.dumps({'type': 'log', 'text': safe})}\n\n"
                elif msg_type == "done":
                    stats = JOBS[job_id].get("stats")
                    yield f"data: {json.dumps({'type': 'done', 'exit_code': msg, 'stats': stats})}\n\n"
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.route("/api/job/<job_id>")
def api_job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"status": job["status"], "stats": job.get("stats")})


@app.route("/api/open-folder", methods=["POST"])
def api_open_folder():
    """Open a folder in Windows Explorer."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    folder = data.get("folder", "")
    if not folder:
        return jsonify({"error": "No folder specified"}), 400
    if os.path.isdir(folder):
        subprocess.Popen(["explorer", os.path.normpath(folder)])
        return jsonify({"ok": True})
    return jsonify({"error": f"Folder not found: {folder}"}), 404


@app.route("/api/preview/<job_id>")
def api_preview(job_id: str):
    """Return HTML preview path for the job."""
    job = JOBS.get(job_id)
    if not job or not job.get("stats"):
        return jsonify({"error": "No stats"}), 404
    stats = job["stats"]

    preview = resolve_preview_path(stats)

    return jsonify({"path": preview, "exists": os.path.exists(preview)})


@app.route("/api/serve-preview/<job_id>")
def api_serve_preview(job_id: str):
    """
    Serve the HTML preview file content directly from Flask.
    This avoids browser security restrictions on file:// URLs.
    The frontend opens /api/serve-preview/<job_id> in a new tab.
    """
    job = JOBS.get(job_id)
    if not job or not job.get("stats"):
        return "Job not found or no stats available.", 404

    stats = job["stats"]
    preview_path = resolve_preview_path(stats)
    preview_root = resolve_preview_root(stats)

    if not preview_path or not os.path.exists(preview_path):
        safe_preview_path = html.escape(preview_path)
        return (
            "<html><body style='font-family:sans-serif;padding:40px;background:#0d1117;color:#e6edf3'>"
            "<h2>⚠ Preview file not found</h2>"
            f"<p>Expected: <code>{safe_preview_path}</code></p>"
            "<p>Make sure <b>Skip HTML Preview Generation</b> is NOT checked.</p>"
            "</body></html>",
            404,
            {"Content-Type": "text/html; charset=utf-8"}
        )

    with open(preview_path, "r", encoding="utf-8") as f:
        preview_html = f.read()

    if preview_root:
        base_href = f"/api/serve-preview-assets/{job_id}/"
        if "<head>" in preview_html:
            preview_html = preview_html.replace(
                "<head>",
                f'<head>\n  <base href="{html.escape(base_href, quote=True)}">',
                1,
            )

    return Response(preview_html, mimetype="text/html")


@app.route("/api/serve-preview-assets/<job_id>/<path:asset_path>")
def api_serve_preview_asset(job_id: str, asset_path: str):
    """Serve assets referenced by preview HTML when opened through Flask."""
    job = JOBS.get(job_id)
    if not job or not job.get("stats"):
        return "Job not found or no stats available.", 404

    preview_root = resolve_preview_root(job["stats"])
    if not preview_root or not preview_root.exists():
        return "Preview root not found.", 404

    asset_full_path = (preview_root / asset_path).resolve()
    try:
        asset_full_path.relative_to(preview_root.resolve())
    except ValueError:
        return "Asset path is outside preview root.", 403

    if not asset_full_path.exists() or not asset_full_path.is_file():
        return "Preview asset not found.", 404

    return send_from_directory(str(preview_root), asset_path)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser
    url = "http://localhost:5000"
    print(f"\n  FfmpegTool Web UI")
    print(f"  Open: {url}")
    print(f"  Press Ctrl+C to stop\n")
    # Open browser after a short delay
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
