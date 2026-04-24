"""
main.py
=======
FfmpegTool -- Video Frame Extractor & Smart Filter
==================================================

Full pipeline:
  1. [Optional] Download video from TikTok/YouTube via --url
  2. Extract frames using FFmpeg (fps mode or scene-detection mode)
  3. Filter blurry frames (Laplacian variance)
  4. Filter duplicate/similar frames (pHash or SSIM)
  5. [Optional] Score frames aesthetically and select top-N
  6. Generate JSON report + HTML visual preview gallery

Phase 1 additions:
  --preset NAME     Load a named preset from presets/ folder
  --list-presets    Show all available presets and exit
  --no-probe        Skip pre-flight ffprobe scan (batch mode)
  Hardware acceleration auto-detected from config hardware.encoder
  Output naming pattern: config output.naming_pattern

Phase 4 additions:
  --workers N       Parallel workers for batch mode (default 1 = sequential)
  --dag FILE        Run one source -> multiple preset branches (DAG spec JSON)
  --dag-workers N   Parallel workers for DAG branches (default 1 = sequential)

Usage:
  # Single video (local file)
  python main.py --input "G:/Videos/clip.mp4" --output "G:/Frames"

  # With preset
  python main.py --input clip.mp4 --output G:/Frames --preset tiktok_pack

  # List available presets
  python main.py --list-presets

  # Parallel batch (2 workers)
  python main.py --batch "G:/Videos/" --output "G:/Frames" --workers 2

  # DAG: 1 source -> multiple presets
  python main.py --dag dag_spec.json --output G:/Frames

  # From TikTok / YouTube URL
  python main.py --url "https://www.tiktok.com/..." --output "G:/Frames"
"""

import os
import re
import sys
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

from extractor import extract_frames
from filters import run_filter_pipeline
from reporter import save_json_report, save_html_preview, save_batch_html_preview, print_summary
from downloader import download_video
from scorer import score_all_frames, select_top_n, save_score_report, print_top_scores
from preset_loader import apply_preset, print_presets_table, list_presets
from hw_detect import resolve_encoder, get_ffmpeg_hwaccel_args
from probe_first import scan_batch
from queue_manager import QueueManager, QueueItem, print_queue_table
from normalizer import normalize_video
from cache_manager import CacheManager


# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.json") -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, config_path)
    if os.path.exists(full_path):
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            # MAIN-3 FIX: catch malformed config.json (e.g. user edit error) and
            # fall back to defaults rather than crashing with an ugly traceback.
            print(f"[WARN] config.json has invalid JSON ({e}). Using built-in defaults.")
    return {}


def apply_defaults(cfg: dict) -> dict:
    """Ensure all config keys have sensible defaults."""
    cfg.setdefault("extraction", {})
    cfg.setdefault("filter", {})
    cfg.setdefault("scorer", {})
    cfg.setdefault("output", {})
    cfg.setdefault("hardware", {})
    cfg.setdefault("batch", {})

    e = cfg["extraction"]
    e.setdefault("mode", "fps")
    e.setdefault("fps", 5)
    e.setdefault("jpeg_quality", 2)
    e.setdefault("scene_threshold", 27.0)

    f = cfg["filter"]
    f.setdefault("blur_threshold", 80.0)
    f.setdefault("similarity_threshold", 0.70)
    f.setdefault("dedup_method", "phash")
    f.setdefault("phash_size", 16)

    s = cfg["scorer"]
    s.setdefault("enabled", False)
    s.setdefault("top_n", 30)
    s.setdefault("save_score_report", True)

    o = cfg["output"]
    o.setdefault("keep_raw", False)
    o.setdefault("generate_html_preview", True)
    o.setdefault("preview_columns", 5)
    o.setdefault("report_json", True)
    o.setdefault("naming_pattern", "{video_name}")
    o.setdefault("campaign", "")
    o.setdefault("lang", "")
    o.setdefault("ratio", "")

    h = cfg["hardware"]
    h.setdefault("encoder", "auto")
    h.setdefault("enable_hwaccel", True)

    b = cfg["batch"]
    b.setdefault("probe_before_run", True)

    n = cfg.setdefault("normalize", {})
    n.setdefault("enabled", True)

    return cfg


# ─────────────────────────────────────────────
# Output naming resolution  (Phase 1)
# ─────────────────────────────────────────────

def resolve_video_output_name(
    video_path: str,
    cfg: dict,
    batch_index: int = 0,
) -> str:
    """
    Resolve the output folder name for a single video.

    Supports these tokens in cfg["output"]["naming_pattern"]:
        {video_name}  — original filename stem   (e.g. "clip")
        {index}       — zero-padded batch index  (e.g. "003")
        {campaign}    — cfg["output"]["campaign"] (e.g. "summer_sale")
        {lang}        — cfg["output"]["lang"]     (e.g. "vi")
        {ratio}       — cfg["output"]["ratio"]    (e.g. "9x16")
        {date}        — current date              (e.g. "20260423")

    Default pattern is '{video_name}', which preserves the original behavior.
    """
    pattern    = cfg["output"].get("naming_pattern", "{video_name}")
    video_name = Path(video_path).stem

    tokens = {
        "{video_name}": video_name,
        "{index}":      f"{batch_index:03d}",
        "{campaign}":   cfg["output"].get("campaign", ""),
        "{lang}":       cfg["output"].get("lang",     ""),
        "{ratio}":      cfg["output"].get("ratio",    ""),
        "{date}":       datetime.now().strftime("%Y%m%d"),
    }

    result = pattern
    for token, value in tokens.items():
        result = result.replace(token, value)

    # Collapse multiple underscores/hyphens, strip leading/trailing separators
    result = re.sub(r"[_\-]{2,}", "_", result).strip("_- ")

    # Fallback to original stem if pattern resolves to empty string
    return result or video_name


def materialize_cached_frames(cached_frames: list[str], unique_dir: str) -> list[str]:
    """Copy cached frames into the current run's unique_frames folder."""
    os.makedirs(unique_dir, exist_ok=True)
    materialized = []
    for src in sorted(cached_frames):
        dst = os.path.join(unique_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        materialized.append(dst)
    return materialized


def build_batch_result_entry(index: int | None, video_name: str, status: str,
                             stats: dict | None = None, error: str | None = None) -> dict:
    """Normalize per-video batch results for summary JSON and app loading."""
    stats = stats or {}
    return {
        "index":               index,
        "video_name":          video_name,
        "status":              status,
        "total_raw_frames":    stats.get("total_raw", 0),
        "removed_blurry":      stats.get("removed_blur", 0),
        "removed_duplicate":   stats.get("removed_duplicate", 0),
        "final_unique_frames": stats.get("final_count", 0),
        "top_n_selected":      stats.get("top_n_selected"),
        "output_folder":       stats.get("output_dir", ""),
        "error":               error,
    }


def write_batch_summary(output_dir: str, input_folder: str, total_videos: int,
                        preflight_skipped: list[dict], video_results: list[dict]) -> str:
    """Write the canonical batch summary JSON consumed by the app."""
    all_results = preflight_skipped + video_results
    success_count = sum(1 for r in video_results if r["status"] == "success")
    skipped_count = sum(1 for r in all_results if r["status"] == "skipped")
    error_count = sum(1 for r in all_results if r["status"] == "error")

    print(f"\n[BATCH] Done. Success: {success_count} | Skipped: {skipped_count} | Errors: {error_count}")
    non_success_names = [v["video_name"] for v in all_results if v["status"] != "success"]
    if non_success_names:
        print(f"[BATCH] Non-success files: {', '.join(non_success_names)}")

    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "total_videos":      total_videos,
        "success":           success_count,
        "skipped":           skipped_count,
        "error":             error_count,
        "failed":            skipped_count + error_count,
        "input_folder":      input_folder,
        "output_folder":     output_dir,
        "generated_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_results":     all_results,
    }
    summary_path = os.path.join(output_dir, "_batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[BATCH_SUMMARY] {summary_path}")

    if success_count > 0:
        html_path = save_batch_html_preview(output_dir)
        if html_path:
            print(f"[BATCH] Master preview: {html_path}")

    return summary_path


def write_url_result_marker(output_dir: str, stats: dict | None):
    """Persist the most recent URL-run report location for the web app."""
    marker_path = os.path.join(output_dir, "_last_url_result.json")
    if not stats or not stats.get("output_dir"):
        if os.path.exists(marker_path):
            os.remove(marker_path)
        return

    video_dir = os.path.dirname(stats["output_dir"])
    report_path = os.path.join(video_dir, "report.json")
    payload = {
        "report_path": report_path,
        "output_folder": stats.get("output_dir", ""),
        "written_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# Single video pipeline
# ─────────────────────────────────────────────

def process_video(video_path: str, output_dir: str, cfg: dict, batch_index: int = 0) -> dict:
    """
    Full pipeline for a single video:
      0. [Optional] Auto-normalize input codec/pix_fmt
      0.5. [Optional] Cache hit check -- skip extraction if frames already cached
      1. Extract frames (raw)
      2. Filter blurry frames
      3. Filter duplicate frames (pHash or SSIM)
      4. [Optional] Score + select top-N aesthetically best frames
      5. Generate report + HTML preview
    """
    video_name       = resolve_video_output_name(video_path, cfg, batch_index)
    video_output_dir = os.path.join(output_dir, video_name)
    raw_dir          = os.path.join(video_output_dir, "_raw")
    unique_dir       = os.path.join(video_output_dir, "unique_frames")
    top_frames_dir   = os.path.join(video_output_dir, "top_frames")

    for transient_dir in (raw_dir, unique_dir, top_frames_dir):
        shutil.rmtree(transient_dir, ignore_errors=True)

    # Write an in-progress marker so --clean-empty skips active output dirs.
    os.makedirs(video_output_dir, exist_ok=True)
    _processing_marker = os.path.join(video_output_dir, "_processing")
    try:
        open(_processing_marker, "w").close()
    except OSError:
        _processing_marker = None  # non-fatal; --clean-empty will be conservative

    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  Processing: {video_name}")
    print(sep)

    # BUG-C2 FIX: cache key must always use the ORIGINAL video path (before
    # normalization), because the normalized temp file is deleted after the
    # pipeline by norm_result.cleanup(). If we keyed on the temp path, every
    # subsequent run would get an OSError on getmtime → mtime="0" → perpetual
    # cache miss.
    original_video_path = video_path

    # ── Step 0 (Phase 2): Auto-normalize input codec/pix_fmt ──
    norm_result = None
    if cfg.get("normalize", {}).get("enabled", True):
        from probe_first import probe_video
        probe = probe_video(video_path)
        # Only normalize if probe succeeded; broken files fail gracefully at step 1
        if probe.ok:
            norm_result = normalize_video(video_path, video_output_dir, cfg, probe=probe)
            if norm_result.was_transcoded:
                video_path = norm_result.path  # use normalized file for extraction

    # BUG-H4 FIX: create a single CacheManager instance reused for both lookup
    # and store (was two separate CacheManager() calls before).
    _cache_mgr = CacheManager() if not cfg.get("no_cache", False) else None

    # ── Step 0.5 (Phase 3): Cache hit check ──
    if _cache_mgr is not None:
        cached_frames = _cache_mgr.get_cached_frames(original_video_path, cfg)
        if cached_frames:
            print(f"[CACHE] Cache hit -- skipping extraction for {video_name}")
            final_paths = materialize_cached_frames(cached_frames, unique_dir)
            stats = {
                "total_raw":            len(final_paths),
                "after_blur_filter":    len(final_paths),
                "after_dedup_filter":   len(final_paths),
                "removed_blur":         0,
                "removed_duplicate":    0,
                "final_count":          len(final_paths),
                "final_paths":          final_paths,
                "output_dir":           unique_dir,
                "blur_threshold":       cfg["filter"].get("blur_threshold"),
                "similarity_threshold": cfg["filter"].get("similarity_threshold"),
                "dedup_method":         cfg["filter"].get("dedup_method", "phash"),
                "scorer_enabled":       False,
                "top_n_requested":      None,
                "top_n_selected":       None,
                "top_frames_dir":       None,
                "score_report_path":    None,
                "cache_hit":            True,
                "blur_removed_list":    [],
                "dup_removed_list":     [],
            }
            if cfg["output"].get("report_json", True):
                rpath = save_json_report(stats, video_output_dir, video_name)
                print(f"[REPORT] JSON: {rpath}")
            if cfg["output"].get("generate_html_preview", True) and stats["final_paths"]:
                cols = cfg["output"].get("preview_columns", 5)
                hpath = save_html_preview(
                    stats["final_paths"], video_output_dir, video_name, stats, columns=cols
                )
                print(f"[REPORT] HTML: {hpath}")
            print_summary(stats, video_name)
            if norm_result and norm_result.was_transcoded:
                norm_result.cleanup()
            return stats

    # ── Step 1: Extract ──
    raw_frames = extract_frames(video_path, raw_dir, cfg["extraction"])

    if not raw_frames:
        print("[WARN] No frames extracted - skipping this video (no output folder created).")
        return {}

    # Create output dir only AFTER confirming we have frames to process
    os.makedirs(video_output_dir, exist_ok=True)

    try:
        # ── Steps 2 & 3: Filter (blur + dedup) ──
        stats = run_filter_pipeline(raw_frames, video_output_dir, cfg["filter"])
        stats["blur_threshold"]       = cfg["filter"].get("blur_threshold")
        stats["similarity_threshold"] = cfg["filter"].get("similarity_threshold")
        stats["dedup_method"]         = cfg["filter"].get("dedup_method", "phash")

        # ── Step 4 (Optional): Aesthetic scoring ──
        scorer_cfg = cfg["scorer"]
        stats["scorer_enabled"] = bool(scorer_cfg.get("enabled"))
        stats["top_n_requested"] = scorer_cfg.get("top_n") if stats["scorer_enabled"] else None
        stats["top_n_selected"] = 0 if stats["scorer_enabled"] else None
        stats["top_frames_dir"] = None
        stats["score_report_path"] = None

        if stats["scorer_enabled"] and stats["final_paths"]:
            top_n = scorer_cfg.get("top_n", 30)
            top_n = min(top_n, len(stats["final_paths"]))  # can't select more than available
            stats["top_n_requested"] = scorer_cfg.get("top_n", 30)

            scored = score_all_frames(stats["final_paths"])
            print_top_scores(scored, n=min(10, top_n))

            top_paths = select_top_n(scored, video_output_dir, top_n)
            stats["top_n_selected"] = len(top_paths)
            stats["top_frames_dir"] = os.path.join(video_output_dir, "top_frames")

            if scorer_cfg.get("save_score_report"):
                rp = save_score_report(scored, video_output_dir)
                stats["score_report_path"] = rp
                print(f"[SCORE] Score report saved: {rp}")

    finally:
        # MAIN-1 FIX: always clean up _raw/ even if an exception is raised
        # mid-pipeline (e.g. during filter or scoring). Without this, partial
        # raw frames accumulate on disk silently on error.
        if not cfg.get("output", {}).get("keep_raw", False):
            shutil.rmtree(raw_dir, ignore_errors=True)
        # Phase 2: clean up normalized temp file after pipeline completes
        if norm_result and norm_result.was_transcoded:
            norm_result.cleanup()
        # Remove in-progress marker now that the pipeline has finished.
        if _processing_marker:
            try:
                os.remove(_processing_marker)
            except OSError:
                pass

    # ── Step 0.5b (Phase 3): Store frames in cache after successful extraction ──
    if _cache_mgr is not None and stats.get("final_paths"):
        try:
            _cache_mgr.store_frames(original_video_path, cfg, stats["final_paths"])
        except Exception as _cache_exc:
            print(f"[CACHE] Warning: failed to cache frames: {_cache_exc}")

    # ── Step 6: Reports ──
    if cfg["output"].get("report_json", True):
        rpath = save_json_report(stats, video_output_dir, video_name)
        print(f"[REPORT] JSON: {rpath}")

    if cfg["output"].get("generate_html_preview", True) and stats["final_paths"]:
        cols = cfg["output"].get("preview_columns", 5)
        hpath = save_html_preview(
            stats["final_paths"], video_output_dir, video_name, stats, columns=cols
        )
        print(f"[REPORT] HTML: {hpath}")

    print_summary(stats, video_name)
    return stats


# ─────────────────────────────────────────────
# Batch mode
# ─────────────────────────────────────────────

def process_batch(input_folder: str, output_dir: str, cfg: dict):
    """Find all videos in a folder and process each one."""
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    video_files = sorted([
        str(f) for f in Path(input_folder).iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ])

    if not video_files:
        print(f"[WARN] No video files found in: {input_folder}")
        return

    total = len(video_files)
    print(f"[BATCH] Found {total} videos in: {input_folder}")

    # ── Phase 0: Pre-flight probe scan ──
    preflight_skipped: list[dict] = []
    if cfg.get("batch", {}).get("probe_before_run", True):
        video_files, bad_results = scan_batch(video_files)
        for bad in bad_results:
            preflight_skipped.append({
                "index":               None,
                "video_name":          Path(bad.path).name,
                "status":              "skipped",
                "total_raw_frames":    0,
                "removed_blurry":      0,
                "removed_duplicate":   0,
                "final_unique_frames": 0,
                "top_n_selected":      None,
                "output_folder":       "",
                "error":               " | ".join(bad.issues),
            })
        if not video_files:
            print("[BATCH] No valid video files to process after pre-flight scan. Aborting.")
            return

    # ── Per-video result tracking ──
    video_results: list[dict] = []

    for i, video_path in enumerate(video_files, 1):
        fname = os.path.basename(video_path)
        print(f"\n[BATCH] {i}/{len(video_files)}: {fname}")
        try:
            stats = process_video(video_path, output_dir, cfg, batch_index=i)
            if stats:  # process_video returns {} on 0 frames
                video_results.append(build_batch_result_entry(i, fname, "success", stats=stats))
            else:
                video_results.append(build_batch_result_entry(
                    i,
                    fname,
                    "skipped",
                    error="No frames extracted (video may be too short or unreadable)",
                ))
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Failed on {fname}: {error_msg}")
            video_results.append(build_batch_result_entry(i, fname, "error", error=error_msg))

    write_batch_summary(output_dir, input_folder, total, preflight_skipped, video_results)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="FfmpegTool",
        description="Smart video frame extractor: extract -> blur filter -> dedup -> score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Local file:
    python main.py --input "G:/Videos/clip.mp4" --output "G:/Frames"

  TikTok / YouTube URL:
    python main.py --url "https://www.tiktok.com/..." --output "G:/Frames"

  Scene-aware (smarter than fps):
    python main.py --input clip.mp4 --output G:/Frames --mode scene

  Custom filters:
    python main.py --input clip.mp4 --output G:/Frames --fps 3 --blur 60 --sim 0.65

  SSIM dedup (more accurate, slower):
    python main.py --input clip.mp4 --output G:/Frames --method ssim

  Score + select top 20 frames:
    python main.py --input clip.mp4 --output G:/Frames --top 20

  Batch folder:
    python main.py --batch "G:/Videos/" --output "G:/Frames"
        """
    )

    # ── Preset (Phase 1) ──
    parser.add_argument("--preset", "-p",
                        metavar="NAME",
                        help="Load a named preset from presets/ folder (e.g. tiktok_pack)")
    parser.add_argument("--list-presets", action="store_true",
                        help="Show all available presets and exit")

    # ── Input source ──
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--input", "-i",
                     help="Path to a local video file")
    src.add_argument("--url", "-u",
                     help="TikTok / YouTube / Instagram URL to download first")
    src.add_argument("--batch", "-b",
                     help="Folder of videos to process in batch")

    # ── Output ──
    parser.add_argument("--output", "-o",
                        help="Output root folder")

    # ── Extraction ──
    parser.add_argument("--mode",
                        choices=["fps", "scene"], default=None,
                        help="fps = extract N frames/sec | scene = 1 frame per scene change")
    parser.add_argument("--fps", type=float, default=None,
                        help="Frames per second (fps mode). Default: 5")

    # ── Filters ──
    parser.add_argument("--blur", type=float, default=None,
                        help="Blur threshold (Laplacian variance). Default: 80. Lower = remove more")
    parser.add_argument("--sim", type=float, default=None,
                        help="Similarity threshold 0-1. Frames >= this removed. Default: 0.70")
    parser.add_argument("--method",
                        choices=["phash", "ssim"], default=None,
                        help="Dedup method: phash (fast, default) or ssim (accurate, slow)")

    # ── Scorer ──
    parser.add_argument("--top", type=int, default=None,
                        help="Enable scorer: select top-N aesthetically best frames into top_frames/")

    # ── Output options ──
    parser.add_argument("--keep-raw", action="store_true",
                        help="Keep raw extracted frames after filtering")
    parser.add_argument("--no-html", action="store_true",
                        help="Skip HTML preview generation")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip JSON report generation")
    parser.add_argument("--no-probe", action="store_true",
                        help="Skip pre-flight ffprobe scan in batch mode")
    parser.add_argument("--gen-batch-html", action="store_true",
                        help="(Re)generate master batch HTML from an existing --output folder. "
                             "No video processing — scans unique_frames/ in existing subfolders only.")
    parser.add_argument("--clean-empty", action="store_true",
                        help="Remove empty output subfolders (no unique_frames/) from --output dir.")
    parser.add_argument("--hw-report", action="store_true",
                        help="Print hardware encoder availability report and exit")

    # ── Phase 2: Draft mode ──
    parser.add_argument("--draft", action="store_true",
                        help="Draft/preview mode: scale to 360p for fast extraction")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip auto-normalization of input codec/pix_fmt (step 0)")

    # ── Phase 2: Queue management ──
    parser.add_argument("--queue-add", action="store_true",
                        help="Add current --input/--output as a queue item and exit")
    parser.add_argument("--queue-run", action="store_true",
                        help="Process all pending queue items sequentially and exit")
    parser.add_argument("--queue-list", action="store_true",
                        help="Print the current queue as a table and exit")
    parser.add_argument("--queue-retry", metavar="ID",
                        help="Retry a failed/skipped queue item by ID and exit")
    parser.add_argument("--queue-remove", metavar="ID",
                        help="Remove a pending queue item by ID and exit")

    # ── Phase 3: Template runner ──
    parser.add_argument("--template", metavar="FILE",
                        help="CSV or JSON template file with batch job definitions")

    # ── Phase 3: Cache ──
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass frame cache (always re-extract)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Purge all cached frames and exit")

    # ── Phase 4: Parallel batch ──
    parser.add_argument("--workers", type=int, default=1, metavar="N",
                        help="Parallel workers for batch mode (default 1 = sequential)")

    # ── Phase 4: DAG runner ──
    parser.add_argument("--dag", metavar="FILE",
                        help="DAG spec JSON file: 1 source -> multiple preset branches")
    parser.add_argument("--dag-workers", type=int, default=1, metavar="N",
                        help="Parallel workers for DAG branches (default 1 = sequential)")

    return parser


def apply_cli_overrides(cfg: dict, args) -> dict:
    """Override config with any explicitly passed CLI arguments."""
    if args.mode:
        cfg["extraction"]["mode"] = args.mode
    if args.fps is not None:
        if args.fps <= 0:
            print(f"[ERROR] --fps must be > 0, got {args.fps}. Using default.")
        else:
            cfg["extraction"]["fps"] = args.fps
    if args.blur is not None:
        if args.blur < 0:
            print(f"[ERROR] --blur must be >= 0, got {args.blur}. Using default.")
        else:
            cfg["filter"]["blur_threshold"] = args.blur
    if args.sim is not None:
        # MAIN-5 FIX: validate range. --sim > 1.0 means sim >= threshold is
        # never true, so NO frames would ever be removed as duplicates.
        # --sim <= 0.0 means all frames would be immediately removed.
        if not (0.0 < args.sim < 1.0):
            print(f"[ERROR] --sim must be between 0.0 and 1.0 (exclusive), got {args.sim}. "
                  "Using default of 0.70.")
        else:
            cfg["filter"]["similarity_threshold"] = args.sim
    if args.method:
        cfg["filter"]["dedup_method"] = args.method
    if args.top is not None:
        if args.top <= 0:
            print(f"[ERROR] --top must be > 0, got {args.top}. Scorer disabled.")
        else:
            cfg["scorer"]["enabled"] = True
            cfg["scorer"]["top_n"] = args.top
    if args.keep_raw:
        cfg["output"]["keep_raw"] = True
    if args.no_html:
        cfg["output"]["generate_html_preview"] = False
    if args.no_report:
        cfg["output"]["report_json"] = False
    if args.no_probe:
        cfg["batch"]["probe_before_run"] = False
    if hasattr(args, "no_normalize") and args.no_normalize:
        cfg["normalize"]["enabled"] = False
    if hasattr(args, "draft") and args.draft:
        cfg["extraction"]["draft"] = True
    if hasattr(args, "no_cache") and args.no_cache:
        cfg["no_cache"] = True
    return cfg


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Standalone: list presets ──
    if args.list_presets:
        print_presets_table()
        sys.exit(0)

    # ── Standalone: hardware report ──
    if args.hw_report:
        from hw_detect import print_hw_report
        print_hw_report()
        sys.exit(0)

    # ── Phase 3: Clear cache ──
    if args.clear_cache:
        cm = CacheManager()
        purged = cm.purge_all()
        print(f"[CACHE] Purged {purged} cache entries.")
        sys.exit(0)

    # ── Phase 2: Queue standalone utilities (before cfg load, no ffmpeg needed) ──
    _qm = QueueManager()

    if args.queue_list:
        items = _qm.list_items()
        print_queue_table(items)
        summary = _qm.summary()
        print(f"  Total: {summary['total']} | Pending: {summary['pending']} | "
              f"Running: {summary['running']} | Done: {summary['done']} | "
              f"Failed: {summary['failed']} | Skipped: {summary['skipped']}")
        sys.exit(0)

    if args.queue_retry:
        try:
            _qm.retry(args.queue_retry)
            print(f"[QUEUE] Item {args.queue_retry} reset to pending.")
        except (KeyError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
        sys.exit(0)

    if args.queue_remove:
        try:
            removed = _qm.remove(args.queue_remove)
            if removed:
                print(f"[QUEUE] Item {args.queue_remove} removed.")
            else:
                print(f"[QUEUE] Item {args.queue_remove} not found.")
                sys.exit(1)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
        sys.exit(0)

    cfg = apply_defaults(load_config("config.json"))

    # ── Apply preset BEFORE CLI overrides (CLI wins over preset) ──
    if args.preset:
        cfg = apply_preset(cfg, args.preset)

    cfg = apply_cli_overrides(cfg, args)

    # ── Resolve hardware encoder and inject hwaccel args ──
    if cfg["hardware"].get("enable_hwaccel", True):
        hw_key = resolve_encoder(cfg["hardware"])
        cfg["hardware"]["_resolved_key"] = hw_key
        cfg["extraction"]["_hwaccel_args"] = get_ffmpeg_hwaccel_args(hw_key)
    else:
        cfg["hardware"]["_resolved_key"]   = "cpu"
        cfg["extraction"]["_hwaccel_args"] = []

    input_sources = [args.input, args.url, args.batch]
    input_source_count = sum(1 for value in input_sources if value)
    is_standalone_utility = args.gen_batch_html or args.clean_empty

    if is_standalone_utility and input_source_count > 0:
        parser.error("--gen-batch-html and --clean-empty cannot be combined with --input, --url, or --batch")

    # ── Standalone utility: regenerate master HTML from existing output folder ──
    if args.gen_batch_html:
        if not args.output or not os.path.isdir(args.output):
            print(f"[ERROR] --gen-batch-html requires a valid --output folder.")
            sys.exit(1)
        html_path = save_batch_html_preview(args.output)
        if html_path:
            print(f"[OK] Master batch HTML generated: {html_path}")
        else:
            print("[WARN] No subfolders with unique_frames found in output dir.")
        sys.exit(0)

    # ── Standalone utility: remove empty output subfolders ──
    if args.clean_empty:
        if not args.output or not os.path.isdir(args.output):
            print(f"[ERROR] --clean-empty requires a valid --output folder.")
            sys.exit(1)
        removed_count = 0
        skipped_count = 0
        for sub in sorted(Path(args.output).iterdir()):
            if not sub.is_dir():
                continue
            # Skip folders with an active in-progress marker (pipeline is running).
            if (sub / "_processing").exists():
                print(f"[CLEAN] Skipped (in-progress): {sub.name}")
                skipped_count += 1
                continue
            unique_dir = sub / "unique_frames"
            # Recheck existence inside try/except to reduce TOCTOU window.
            try:
                empty = not unique_dir.exists() or not any(unique_dir.iterdir())
                if empty:
                    shutil.rmtree(sub)
                    print(f"[CLEAN] Removed empty folder: {sub.name}")
                    removed_count += 1
            except (OSError, StopIteration):
                # Folder modified concurrently -- leave it alone.
                print(f"[CLEAN] Skipped (concurrent write): {sub.name}")
                skipped_count += 1
        print(f"[CLEAN] Done -- removed {removed_count} empty folder(s), skipped {skipped_count}.")
        sys.exit(0)

    # ── Phase 2: Queue add ──
    if args.queue_add:
        if not (args.input or args.url or args.batch):
            parser.error("--queue-add requires --input, --url, or --batch")
        if not args.output:
            parser.error("--queue-add requires --output")
        src = args.input or args.url or args.batch
        item_id = _qm.add(src, args.output)
        print(f"[QUEUE] Added item {item_id}: {src} -> {args.output}")
        sys.exit(0)

    # ── Phase 2: Queue run (process all pending) ──
    if args.queue_run:
        pending = _qm.pending_count()
        if pending == 0:
            print("[QUEUE] No pending items in queue.")
            sys.exit(0)
        print(f"[QUEUE] Running {pending} pending item(s)...")
        batch_index = 0
        while _qm.pending_count() > 0:
            batch_index += 1
            _qm.run_next(cfg, process_video, batch_index=batch_index)
        print("[QUEUE] All pending items processed.")
        sys.exit(0)

    # ── Phase 3: Template runner ──
    if args.template:
        from template_runner import load_template, run_template
        from preset_loader import list_presets as _list_presets
        known = [p["file"] for p in _list_presets()]
        rows = load_template(args.template)
        base_out = args.output or None
        result = run_template(rows, _qm, base_output=base_out, known_presets=known)
        print(f"[TEMPLATE] Queued: {result['queued']} | Skipped: {result['skipped']}")
        for err in result["errors"]:
            print(f"  [ROW {err['row_index']}] {err['video_src']}: {', '.join(err['errors'])}")
        if result["queued"] > 0:
            print(f"[TEMPLATE] Run queue with: python main.py --queue-run")
        sys.exit(0)

    if input_source_count != 1:
        parser.error("one of the arguments --input/-i --url/-u --batch/-b is required")
    if not args.output:
        parser.error("the following arguments are required: --output/-o")

    # Print active config summary
    sep = "=" * 58
    print(f"\n{sep}")
    print("  FfmpegTool  --  Video Frame Extractor & Smart Filter")
    print(sep)
    print(f"  Extract mode : {cfg['extraction']['mode'].upper()}", end="")
    if cfg["extraction"]["mode"] == "fps":
        print(f" ({cfg['extraction']['fps']} fps)")
    else:
        print()
    print(f"  Blur thresh  : {cfg['filter']['blur_threshold']}")
    print(f"  Sim thresh   : {cfg['filter']['similarity_threshold']:.0%}")
    print(f"  Dedup method : {cfg['filter']['dedup_method'].upper()}")
    if cfg["scorer"]["enabled"]:
        print(f"  Scorer       : ON  --  top {cfg['scorer']['top_n']} frames")
    if cfg["extraction"].get("draft"):
        print(f"  Draft mode   : ON  (360p fast preview)")
    if not cfg.get("normalize", {}).get("enabled", True):
        print(f"  Normalize    : OFF (--no-normalize)")
    print(sep)

    # ── Determine video path ──
    # ── Phase 4: DAG mode ──
    if args.dag:
        from dag_runner import load_dag_spec, run_dag
        try:
            spec = load_dag_spec(args.dag)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] DAG spec error: {exc}")
            sys.exit(1)
        # CLI --output overrides spec-level output (spec branch output still wins per-branch)
        if args.output and not spec.get("output"):
            spec["output"] = args.output
        dag_workers = getattr(args, "dag_workers", 1)
        result = run_dag(spec, cfg, default_output=args.output or "",
                         workers=dag_workers)
        success = result["success"]
        failed  = result["failed"]
        skipped = result["skipped"]
        total   = success + failed + skipped
        print(f"\n[DAG] Finished {total} branch(es): {success} OK, "
              f"{skipped} skipped, {failed} failed")
        sys.exit(0 if failed == 0 else 1)

    if args.url:
        os.makedirs(args.output, exist_ok=True)
        try:
            video_path = download_video(args.url, args.output, filename="downloaded_video")
        except RuntimeError as e:
            print(f"[ERROR] Download failed: {e}")
            sys.exit(1)
        stats = process_video(video_path, args.output, cfg)
        write_url_result_marker(args.output, stats)

    elif args.batch:
        if not os.path.isdir(args.batch):
            print(f"[ERROR] Batch folder not found: {args.batch}")
            sys.exit(1)
        # ── Phase 4: Parallel batch ──
        n_workers = getattr(args, "workers", 1)
        if n_workers > 1:
            from parallel_runner import run_parallel_batch, resolve_max_workers
            from pathlib import Path as _Path
            _video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
            video_files = sorted([
                str(f) for f in _Path(args.batch).iterdir()
                if f.is_file() and f.suffix.lower() in _video_exts
            ])
            preflight_skipped = []
            if cfg.get("batch", {}).get("probe_before_run", True):
                original_files = list(video_files)
                video_files, bad_results = scan_batch(video_files)
                for bad in bad_results:
                    preflight_skipped.append(build_batch_result_entry(
                        None,
                        Path(bad.path).name,
                        "skipped",
                        error=" | ".join(bad.issues),
                    ))
            eff_workers = resolve_max_workers(n_workers)
            print(f"[BATCH] Parallel mode: {eff_workers} workers")
            results = run_parallel_batch(video_files, args.output, cfg, workers=n_workers)
            video_results = []
            for i, result in enumerate(results, 1):
                video_results.append(build_batch_result_entry(
                    i,
                    result["video_name"],
                    result["status"],
                    stats=result.get("stats"),
                    error=result.get("error"),
                ))
            write_batch_summary(args.output, args.batch, len(original_files) if cfg.get("batch", {}).get("probe_before_run", True) else len(video_files), preflight_skipped, video_results)
        else:
            process_batch(args.batch, args.output, cfg)

    else:
        input_path = args.input

        if not os.path.exists(input_path):
            print(f"[ERROR] Path not found: {input_path}")
            sys.exit(1)

        # ── Auto-detect: if a folder is passed to --input, treat as batch ──
        if os.path.isdir(input_path):
            print(f"[INFO] --input is a directory. Switching to batch mode automatically.")
            print(f"[INFO] Scanning folder: {input_path}")
            n_workers = getattr(args, "workers", 1)
            if n_workers > 1:
                from parallel_runner import run_parallel_batch, resolve_max_workers
                from pathlib import Path as _Path
                _video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
                video_files = sorted([
                    str(f) for f in _Path(input_path).iterdir()
                    if f.is_file() and f.suffix.lower() in _video_exts
                ])
                preflight_skipped = []
                if cfg.get("batch", {}).get("probe_before_run", True):
                    original_files = list(video_files)
                    video_files, bad_results = scan_batch(video_files)
                    for bad in bad_results:
                        preflight_skipped.append(build_batch_result_entry(
                            None,
                            Path(bad.path).name,
                            "skipped",
                            error=" | ".join(bad.issues),
                        ))
                eff_workers = resolve_max_workers(n_workers)
                print(f"[BATCH] Parallel mode: {eff_workers} workers")
                results = run_parallel_batch(video_files, args.output, cfg, workers=n_workers)
                video_results = []
                for i, result in enumerate(results, 1):
                    video_results.append(build_batch_result_entry(
                        i,
                        result["video_name"],
                        result["status"],
                        stats=result.get("stats"),
                        error=result.get("error"),
                    ))
                write_batch_summary(args.output, input_path, len(original_files) if cfg.get("batch", {}).get("probe_before_run", True) else len(video_files), preflight_skipped, video_results)
            else:
                process_batch(input_path, args.output, cfg)

        elif os.path.isfile(input_path):
            process_video(input_path, args.output, cfg)

        else:
            print(f"[ERROR] --input is neither a file nor a folder: {input_path}")
            sys.exit(1)


if __name__ == "__main__":
    main()

