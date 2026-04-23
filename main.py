"""
main.py
=======
FfmpegTool — Video Frame Extractor & Smart Filter
==================================================

Full pipeline:
  1. [Optional] Download video from TikTok/YouTube via --url
  2. Extract frames using FFmpeg (fps mode or scene-detection mode)
  3. Filter blurry frames (Laplacian variance)
  4. Filter duplicate/similar frames (pHash or SSIM)
  5. [Optional] Score frames aesthetically and select top-N
  6. Generate JSON report + HTML visual preview gallery

Usage:
  # Single video (local file)
  python main.py --input "G:/Videos/clip.mp4" --output "G:/Frames"

  # From TikTok / YouTube URL
  python main.py --url "https://www.tiktok.com/..." --output "G:/Frames"

  # Scene-aware extraction (smarter than fps)
  python main.py --input "G:/Videos/clip.mp4" --output "G:/Frames" --mode scene

  # Custom thresholds
  python main.py --input clip.mp4 --output G:/Frames --fps 3 --blur 60 --sim 0.65

  # Use SSIM for duplicate detection (more accurate, slower)
  python main.py --input clip.mp4 --output G:/Frames --method ssim

  # Keep only top 20 aesthetically best frames
  python main.py --input clip.mp4 --output G:/Frames --top 20

  # Batch: process all videos in a folder
  python main.py --batch "G:/Videos/" --output "G:/Frames"
"""

import os
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


# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.json") -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, config_path)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def apply_defaults(cfg: dict) -> dict:
    """Ensure all config keys have sensible defaults."""
    cfg.setdefault("extraction", {})
    cfg.setdefault("filter", {})
    cfg.setdefault("scorer", {})
    cfg.setdefault("output", {})

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

    return cfg


# ─────────────────────────────────────────────
# Single video pipeline
# ─────────────────────────────────────────────

def process_video(video_path: str, output_dir: str, cfg: dict) -> dict:
    """
    Full pipeline for a single video:
      1. Extract frames (raw)
      2. Filter blurry frames
      3. Filter duplicate frames (pHash or SSIM)
      4. [Optional] Score + select top-N aesthetically best frames
      5. Generate report + HTML preview
    """
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    raw_dir = os.path.join(video_output_dir, "_raw")
    unique_dir = os.path.join(video_output_dir, "unique_frames")
    top_frames_dir = os.path.join(video_output_dir, "top_frames")

    for transient_dir in (raw_dir, unique_dir, top_frames_dir):
        shutil.rmtree(transient_dir, ignore_errors=True)

    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  Processing: {video_name}")
    print(sep)

    # ── Step 1: Extract ──
    raw_frames = extract_frames(video_path, raw_dir, cfg["extraction"])

    if not raw_frames:
        print("[WARN] No frames extracted — skipping this video (no output folder created).")
        return {}

    # Create output dir only AFTER confirming we have frames to process
    os.makedirs(video_output_dir, exist_ok=True)

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

    # ── Step 5: Cleanup raw ──
    if not cfg["output"].get("keep_raw", False):
        shutil.rmtree(raw_dir, ignore_errors=True)

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

    # Per-video result tracking
    video_results: list[dict] = []
    success_count, skipped_count, error_count = 0, 0, 0

    for i, video_path in enumerate(video_files, 1):
        fname = os.path.basename(video_path)
        print(f"\n[BATCH] {i}/{total}: {fname}")
        try:
            stats = process_video(video_path, output_dir, cfg)
            if stats:  # process_video returns {} on 0 frames
                success_count += 1
                video_results.append({
                    "index":               i,
                    "video_name":          fname,
                    "status":              "success",
                    "total_raw_frames":    stats.get("total_raw",        0),
                    "removed_blurry":      stats.get("removed_blur",     0),
                    "removed_duplicate":   stats.get("removed_duplicate",0),
                    "final_unique_frames": stats.get("final_count",      0),
                    "top_n_selected":      stats.get("top_n_selected"),
                    "output_folder":       stats.get("output_dir",       ""),
                    "error":               None,
                })
            else:
                skipped_count += 1
                video_results.append({
                    "index":               i,
                    "video_name":          fname,
                    "status":              "skipped",
                    "total_raw_frames":    0,
                    "removed_blurry":      0,
                    "removed_duplicate":   0,
                    "final_unique_frames": 0,
                    "top_n_selected":      None,
                    "output_folder":       "",
                    "error":               "No frames extracted (video may be too short or unreadable)",
                })
        except Exception as e:
            error_count += 1
            error_msg = str(e)
            print(f"[ERROR] Failed on {fname}: {error_msg}")
            video_results.append({
                "index":               i,
                "video_name":          fname,
                "status":              "error",
                "total_raw_frames":    0,
                "removed_blurry":      0,
                "removed_duplicate":   0,
                "final_unique_frames": 0,
                "top_n_selected":      None,
                "output_folder":       "",
                "error":               error_msg,
            })

    print(f"\n[BATCH] Done. Success: {success_count} | Skipped: {skipped_count} | Errors: {error_count}")
    non_success_names = [v["video_name"] for v in video_results if v["status"] != "success"]
    if non_success_names:
        print(f"[BATCH] Non-success files: {', '.join(non_success_names)}")

    # ── Save _batch_summary.json for app.py to read ──
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "total_videos":      total,
        "success":           success_count,
        "skipped":           skipped_count,
        "error":             error_count,
        "failed":            skipped_count + error_count,
        "input_folder":      input_folder,
        "output_folder":     output_dir,
        "generated_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_results":     video_results,
    }
    summary_path = os.path.join(output_dir, "_batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    # Print structured marker so app.py can locate the file
    print(f"[BATCH_SUMMARY] {summary_path}")

    # ── Generate consolidated master HTML for all processed videos ──
    if success_count > 0:
        html_path = save_batch_html_preview(output_dir)
        if html_path:
            print(f"[BATCH] Master preview: {html_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="FfmpegTool",
        description="Smart video frame extractor: extract → blur filter → dedup → score",
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
    parser.add_argument("--gen-batch-html", action="store_true",
                        help="(Re)generate master batch HTML from an existing --output folder. "
                             "No video processing — scans unique_frames/ in existing subfolders only.")
    parser.add_argument("--clean-empty", action="store_true",
                        help="Remove empty output subfolders (no unique_frames/) from --output dir.")

    return parser


def apply_cli_overrides(cfg: dict, args) -> dict:
    """Override config with any explicitly passed CLI arguments."""
    if args.mode:
        cfg["extraction"]["mode"] = args.mode
    if args.fps is not None:
        cfg["extraction"]["fps"] = args.fps
    if args.blur is not None:
        cfg["filter"]["blur_threshold"] = args.blur
    if args.sim is not None:
        cfg["filter"]["similarity_threshold"] = args.sim
    if args.method:
        cfg["filter"]["dedup_method"] = args.method
    if args.top is not None:
        cfg["scorer"]["enabled"] = True
        cfg["scorer"]["top_n"] = args.top
    if args.keep_raw:
        cfg["output"]["keep_raw"] = True
    if args.no_html:
        cfg["output"]["generate_html_preview"] = False
    if args.no_report:
        cfg["output"]["report_json"] = False
    return cfg


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = apply_defaults(load_config("config.json"))
    cfg = apply_cli_overrides(cfg, args)

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
        for sub in sorted(Path(args.output).iterdir()):
            if not sub.is_dir():
                continue
            unique_dir = sub / "unique_frames"
            if not unique_dir.exists() or not any(unique_dir.iterdir()):
                shutil.rmtree(sub)
                print(f"[CLEAN] Removed empty folder: {sub.name}")
                removed_count += 1
        print(f"[CLEAN] Done — removed {removed_count} empty folder(s).")
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
    print(sep)

    # ── Determine video path ──
    if args.url:
        os.makedirs(args.output, exist_ok=True)
        try:
            video_path = download_video(args.url, args.output, filename="downloaded_video")
        except RuntimeError as e:
            print(f"[ERROR] Download failed: {e}")
            sys.exit(1)
        process_video(video_path, args.output, cfg)

    elif args.batch:
        if not os.path.isdir(args.batch):
            print(f"[ERROR] Batch folder not found: {args.batch}")
            sys.exit(1)
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
            process_batch(input_path, args.output, cfg)

        elif os.path.isfile(input_path):
            process_video(input_path, args.output, cfg)

        else:
            print(f"[ERROR] --input is neither a file nor a folder: {input_path}")
            sys.exit(1)


if __name__ == "__main__":
    main()
