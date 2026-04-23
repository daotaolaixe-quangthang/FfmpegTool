"""
reporter.py
===========
Generate two types of output after filtering:

  1. JSON report   — machine-readable stats & file lists
  2. HTML preview  — visual gallery to review results in browser
"""

import os
import json
import base64
from html import escape
from datetime import datetime
from pathlib import Path


def save_json_report(stats: dict, output_dir: str, video_name: str) -> str:
    """Save processing stats to a JSON report file."""
    report = {
        "video"             : video_name,
        "generated_at"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_raw_frames"  : stats["total_raw"],
        "removed_blurry"    : stats["removed_blur"],
        "removed_duplicate" : stats["removed_duplicate"],
        "final_unique_frames": stats["final_count"],
        "reduction_rate"    : f"{(1 - stats['final_count'] / max(stats['total_raw'], 1)) * 100:.1f}%",
        "scorer_enabled"    : stats.get("scorer_enabled", False),
        "top_n_requested"   : stats.get("top_n_requested"),
        "top_n_selected"    : stats.get("top_n_selected"),   # None if scorer disabled
        "top_frames_dir"    : stats.get("top_frames_dir"),
        "score_report_path" : stats.get("score_report_path"),
        "output_folder"     : stats["output_dir"],
        "blur_threshold_used"       : stats.get("blur_threshold"),
        "similarity_threshold_used" : stats.get("similarity_threshold"),
        "blurry_files_removed"      : stats["blur_removed_list"],
        "duplicate_files_removed"   : stats["dup_removed_list"],
    }

    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report_path


def save_html_preview(final_paths: list[str], output_dir: str,
                      video_name: str, stats: dict, columns: int = 5) -> str:
    """
    Generate an HTML visual gallery of final unique frames.
    Opens directly in any browser — no server needed.
    """

    def img_to_data_uri(path: str) -> str:
        # REPORTER-2 FIX: catch OSError if a frame file was deleted between
        # dedup and preview generation (race condition / external tool).
        # Returns empty string → browser shows broken image icon instead of crash.
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f"data:image/jpeg;base64,{data}"
        except OSError:
            return ""

    safe_video_name = escape(video_name)

    # Build image grid HTML — limit to 150 frames to keep file size manageable
    # (base64 encoding: ~100KB/frame × 200 frames = 20MB HTML = slow browser)
    MAX_PREVIEW = 150
    preview_paths = final_paths[:MAX_PREVIEW]
    truncated = len(final_paths) > MAX_PREVIEW
    truncated_msg = ""
    if truncated:
        truncated_msg = (
            f'<div style="text-align:center;padding:16px;color:#f59e0b;font-size:0.85rem">'
            f'Showing first {MAX_PREVIEW} of {len(final_paths)} frames. '
            f'View all in the unique_frames/ folder.</div>'
        )

    cards_html = ""
    for i, path in enumerate(preview_paths):
        fname = os.path.basename(path)
        safe_fname = escape(fname)
        data_uri = img_to_data_uri(path)
        cards_html += f"""
        <div class="card">
          <img src="{data_uri}" alt="{safe_fname}" loading="lazy">
          <div class="label">{i+1:03d} — {safe_fname}</div>
        </div>"""

    reduction = (1 - stats['final_count'] / max(stats['total_raw'], 1)) * 100

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Frame Preview — {safe_video_name}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0f0f13;
      color: #e0e0e0;
      padding: 20px;
    }}
    header {{
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      border: 1px solid #2a2a4a;
      border-radius: 12px;
      padding: 24px 32px;
      margin-bottom: 24px;
    }}
    header h1 {{
      font-size: 1.5rem;
      color: #7eb8f7;
      margin-bottom: 12px;
    }}
    .stats {{
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .stat {{
      background: #0f0f1f;
      border: 1px solid #2a2a4a;
      border-radius: 8px;
      padding: 10px 18px;
      text-align: center;
    }}
    .stat .num {{
      font-size: 1.8rem;
      font-weight: 700;
      color: #7eb8f7;
      display: block;
    }}
    .stat .lbl {{
      font-size: 0.75rem;
      color: #888;
      margin-top: 2px;
    }}
    .stat.red .num {{ color: #f77e7e; }}
    .stat.green .num {{ color: #7ef7a0; }}

    .grid {{
      display: grid;
      grid-template-columns: repeat({columns}, 1fr);
      gap: 10px;
    }}
    .card {{
      background: #1a1a2e;
      border: 1px solid #2a2a4a;
      border-radius: 8px;
      overflow: hidden;
      transition: transform 0.2s, border-color 0.2s;
    }}
    .card:hover {{
      transform: scale(1.03);
      border-color: #7eb8f7;
    }}
    .card img {{
      width: 100%;
      aspect-ratio: 16/9;
      object-fit: cover;
      display: block;
    }}
    .label {{
      font-size: 0.65rem;
      color: #666;
      padding: 4px 6px;
      text-align: center;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    footer {{
      text-align: center;
      color: #444;
      font-size: 0.75rem;
      margin-top: 24px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Frame Preview — {safe_video_name}</h1>
    <div class="stats">
      <div class="stat">
        <span class="num">{stats['total_raw']}</span>
        <span class="lbl">Raw Frames</span>
      </div>
      <div class="stat red">
        <span class="num">{stats['removed_blur']}</span>
        <span class="lbl">Blurry Removed</span>
      </div>
      <div class="stat red">
        <span class="num">{stats['removed_duplicate']}</span>
        <span class="lbl">Duplicates Removed</span>
      </div>
      <div class="stat green">
        <span class="num">{stats['final_count']}</span>
        <span class="lbl">Unique Frames Kept</span>
      </div>
      <div class="stat">
        <span class="num">{reduction:.0f}%</span>
        <span class="lbl">Reduction</span>
      </div>
    </div>
  </header>

  {truncated_msg}
  <div class="grid">
    {cards_html}
  </div>

  <footer>
    Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} — FfmpegTool
  </footer>
</body>
</html>"""

    html_path = os.path.join(output_dir, "preview.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # REPORTER-1: warn user if preview HTML is large (base64 frames can balloon fast)
    size_mb = os.path.getsize(html_path) / (1024 * 1024)
    if size_mb > 10:
        print(f"[WARN] HTML preview is {size_mb:.0f}MB — may load slowly in browser. "
              "Consider raising jpeg_quality (e.g. 5-8) or reducing fps.")

    return html_path


def save_batch_html_preview(batch_output_dir: str) -> str:
    """
    Generate a consolidated master HTML gallery for all videos processed in a batch.
    Scans batch_output_dir for subfolders containing unique_frames/ and builds
    one overview page with all frames grouped by video.

    Uses relative <img src> paths (no base64) to keep the file small and fast.
    The HTML file must remain in batch_output_dir to resolve relative paths correctly.
    """
    batch_dir = Path(batch_output_dir)

    # Collect all video sections: {video_name: [frame_paths]}
    sections = []
    total_frames = 0

    for video_dir in sorted(batch_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        unique_dir = video_dir / "unique_frames"
        if not unique_dir.exists():
            continue
        frame_files = sorted(unique_dir.glob("*.jpg")) + sorted(unique_dir.glob("*.png"))
        if not frame_files:
            continue
        # Convert to relative paths from batch_output_dir
        rel_paths = [f.relative_to(batch_dir).as_posix() for f in frame_files]
        sections.append({"name": video_dir.name, "frames": rel_paths})
        total_frames += len(rel_paths)

    if not sections:
        return None

    safe_batch_name = escape(batch_dir.name)

    # Build per-video section HTML
    sections_html = ""
    for sec in sections:
        safe_section_name = escape(sec["name"])
        cards = ""
        for i, rel_path in enumerate(sec["frames"]):
            fname = Path(rel_path).name
            safe_fname = escape(fname)
            safe_rel_attr = escape(rel_path, quote=True)
            cards += f"""
        <div class="card" data-src="{safe_rel_attr}" onclick="openModal(this.dataset.src)">
          <img src="{safe_rel_attr}" alt="{safe_fname}" loading="lazy">
          <div class="label">{i+1:03d} — {safe_fname}</div>
        </div>"""

        sections_html += f"""
  <section class="video-section">
    <div class="section-header" onclick="toggleSection(this)">
      <span class="toggle-icon">▾</span>
      <h2>{safe_section_name}</h2>
      <span class="badge">{len(sec['frames'])} frames</span>
    </div>
    <div class="frames-grid">
      {cards}
    </div>
  </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Batch Preview — {safe_batch_name}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0a0a10;
      color: #e0e0e0;
      padding: 24px;
    }}

    /* ── Top bar ── */
    .topbar {{
      background: linear-gradient(135deg, #13131f 0%, #1a1a2e 100%);
      border: 1px solid #2a2a4a;
      border-radius: 14px;
      padding: 28px 36px;
      margin-bottom: 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 16px;
    }}
    .topbar h1 {{
      font-size: 1.6rem;
      color: #7eb8f7;
      letter-spacing: -0.02em;
    }}
    .topbar .subtitle {{
      font-size: 0.82rem;
      color: #666;
      margin-top: 4px;
    }}
    .pills {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .pill {{
      background: #0f0f1f;
      border: 1px solid #2a2a4a;
      border-radius: 999px;
      padding: 8px 20px;
      text-align: center;
    }}
    .pill .num {{
      font-size: 1.5rem;
      font-weight: 700;
      color: #7eb8f7;
      display: block;
    }}
    .pill .lbl {{
      font-size: 0.7rem;
      color: #888;
    }}

    /* ── Section ── */
    .video-section {{
      background: #12121c;
      border: 1px solid #222238;
      border-radius: 12px;
      margin-bottom: 20px;
      overflow: hidden;
    }}
    .section-header {{
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 16px 22px;
      cursor: pointer;
      user-select: none;
      background: #16162a;
      transition: background 0.2s;
    }}
    .section-header:hover {{ background: #1e1e3a; }}
    .section-header h2 {{
      font-size: 0.95rem;
      font-weight: 600;
      color: #a0bef7;
      flex: 1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .toggle-icon {{
      color: #555;
      font-size: 1rem;
      transition: transform 0.25s;
      flex-shrink: 0;
    }}
    .badge {{
      background: #1f2a45;
      color: #7eb8f7;
      border-radius: 999px;
      font-size: 0.72rem;
      padding: 3px 12px;
      white-space: nowrap;
      flex-shrink: 0;
    }}
    .frames-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 8px;
      padding: 12px;
    }}

    /* ── Card ── */
    .card {{
      background: #1a1a2e;
      border: 1px solid #2a2a4a;
      border-radius: 8px;
      overflow: hidden;
      cursor: pointer;
      transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    }}
    .card:hover {{
      transform: scale(1.03);
      border-color: #7eb8f7;
      box-shadow: 0 4px 18px rgba(126,184,247,0.15);
    }}
    .card img {{
      width: 100%;
      aspect-ratio: 9/16;
      object-fit: cover;
      display: block;
    }}
    .label {{
      font-size: 0.6rem;
      color: #555;
      padding: 4px 6px;
      text-align: center;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    /* ── Lightbox ── */
    #modal {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.92);
      z-index: 1000;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 12px;
    }}
    #modal.open {{ display: flex; }}
    #modal img {{
      max-width: 92vw;
      max-height: 88vh;
      border-radius: 8px;
      box-shadow: 0 8px 48px rgba(0,0,0,0.8);
    }}
    #modal-caption {{
      color: #888;
      font-size: 0.78rem;
    }}
    #modal-close {{
      position: fixed;
      top: 18px;
      right: 24px;
      background: none;
      border: none;
      color: #aaa;
      font-size: 2rem;
      cursor: pointer;
      line-height: 1;
    }}
    #modal-close:hover {{ color: #fff; }}

    footer {{
      text-align: center;
      color: #333;
      font-size: 0.72rem;
      margin-top: 28px;
    }}
  </style>
</head>
<body>

  <div class="topbar">
    <div>
      <h1>📁 Batch Preview — {safe_batch_name}</h1>
      <div class="subtitle">All extracted frames across all videos · click any frame to enlarge</div>
    </div>
    <div class="pills">
      <div class="pill">
        <span class="num">{len(sections)}</span>
        <span class="lbl">Videos</span>
      </div>
      <div class="pill">
        <span class="num">{total_frames}</span>
        <span class="lbl">Total Frames</span>
      </div>
    </div>
  </div>

  {sections_html}

  <!-- Lightbox modal -->
  <div id="modal" onclick="closeModal()">
    <button id="modal-close" onclick="closeModal()">✕</button>
    <!-- REPORTER-3 FIX: stopPropagation so clicking the image itself does not
         bubble up to the modal backdrop and close the lightbox. -->
    <img id="modal-img" src="" alt="" onclick="event.stopPropagation()">
    <div id="modal-caption"></div>
  </div>

  <footer>Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} — FfmpegTool Batch Preview</footer>

  <script>
    function toggleSection(header) {{
      const grid = header.nextElementSibling;
      const icon = header.querySelector('.toggle-icon');
      const collapsed = grid.style.display === 'none';
      grid.style.display = collapsed ? 'grid' : 'none';
      icon.style.transform = collapsed ? 'rotate(0deg)' : 'rotate(-90deg)';
    }}

    function openModal(src) {{
      document.getElementById('modal-img').src = src;
      document.getElementById('modal-caption').textContent = src.split('/').pop();
      document.getElementById('modal').classList.add('open');
    }}

    function closeModal() {{
      document.getElementById('modal').classList.remove('open');
      document.getElementById('modal-img').src = '';
    }}

    document.addEventListener('keydown', e => {{
      if (e.key === 'Escape') closeModal();
    }});
  </script>
</body>
</html>"""

    html_path = batch_dir / "_batch_preview.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[REPORT] Batch master HTML: {html_path}")
    return str(html_path)


def print_summary(stats: dict, video_name: str):
    """Print a clean summary table to console."""
    reduction = (1 - stats['final_count'] / max(stats['total_raw'], 1)) * 100
    sep = "=" * 55
    div = "-" * 47
    print(f"\n{sep}")
    print(f"  RESULTS -- {video_name}")
    print(sep)
    print(f"  Raw frames extracted    : {stats['total_raw']:>6}")
    print(f"  Removed (blurry)        : {stats['removed_blur']:>6}")
    print(f"  Removed (duplicate)     : {stats['removed_duplicate']:>6}")
    print(f"  {div}")
    print(f"  UNIQUE FRAMES KEPT      : {stats['final_count']:>6}")
    print(f"  Reduction rate          : {reduction:>5.1f}%")
    print(f"  Output folder           : {stats['output_dir']}")
    print(sep)

