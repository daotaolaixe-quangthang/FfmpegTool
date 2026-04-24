"""
parallel_runner.py
==================
FfmpegTool Phase 4 -- Smart Parallelism.

Runs multiple process_video() calls concurrently using
concurrent.futures.ProcessPoolExecutor.

Design principles:
  - max_workers = min(cpu_count // 2, 4)  (conservative default)
  - Memory guard: skip fork if free RAM < threshold (requires psutil)
  - Crash isolation: 1 worker exception does NOT abort remaining workers
  - Thread-safe result collection via futures
  - QueueManager is thread-safe (uses Lock) -- safe from caller thread

Usage:
    from parallel_runner import run_parallel_batch, check_memory_ok

    results = run_parallel_batch(
        video_files=["a.mp4", "b.mp4", "c.mp4"],
        output_dir="/output",
        cfg=cfg,
        workers=2,
    )
    # results: list of {"video": path, "stats": dict|None, "error": str|None}
"""

import os
import sys
import copy
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Safe defaults ──
_DEFAULT_MAX_WORKERS = 2
_MIN_FREE_RAM_MB     = 512       # Skip new fork if free RAM below this

# ── Supported video extensions (mirrors process_batch in main.py) ──
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# ── psutil is optional (memory guard) ──
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# ─────────────────────────────────────────────
# Memory guard
# ─────────────────────────────────────────────

def check_memory_ok(threshold_mb: int = _MIN_FREE_RAM_MB) -> bool:
    """
    Return True if free RAM >= threshold_mb.

    If psutil is not installed, always returns True (skip guard silently).
    """
    if not _PSUTIL_AVAILABLE:
        return True
    try:
        free_mb = psutil.virtual_memory().available // (1024 * 1024)
        return free_mb >= threshold_mb
    except Exception:
        return True


def get_free_ram_mb() -> Optional[int]:
    """Return free RAM in MB, or None if psutil is unavailable."""
    if not _PSUTIL_AVAILABLE:
        return None
    try:
        return psutil.virtual_memory().available // (1024 * 1024)
    except Exception:
        return None


# ─────────────────────────────────────────────
# Worker function (top-level for pickling)
# ─────────────────────────────────────────────

def _worker(video_path: str, output_dir: str, cfg: dict, batch_index: int) -> dict:
    """
    Top-level worker function executed in a subprocess.

    Must be defined at module level so ProcessPoolExecutor can pickle it.

    Returns:
        {"video": video_path, "stats": stats_dict, "error": None}
        or
        {"video": video_path, "stats": None, "error": "error message"}
    """
    # Ensure the repo root is on sys.path inside the subprocess.
    # (ProcessPoolExecutor forks/spawns a fresh Python process on Windows.)
    tool_dir = os.path.dirname(os.path.abspath(__file__))
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)

    try:
        from main import process_video  # noqa: PLC0415
        stats = process_video(video_path, output_dir, cfg, batch_index)
        return {"video": video_path, "stats": stats, "error": None}
    except Exception as exc:
        return {"video": video_path, "stats": None, "error": str(exc)}


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def resolve_max_workers(requested: int) -> int:
    """
    Clamp requested workers to a safe value.

    Rule: min(requested, cpu_count // 2, 4)
    Always returns at least 1.
    """
    cpu = os.cpu_count() or 2
    safe_max = min(cpu // 2, 4)
    safe_max = max(safe_max, 1)
    return max(1, min(requested, safe_max))


def run_parallel_batch(
    video_files: list,
    output_dir: str,
    cfg: dict,
    workers: int = 1,
    memory_threshold_mb: int = _MIN_FREE_RAM_MB,
    progress_callback=None,
) -> list:
    """
    Process a list of video files in parallel.

    Args:
        video_files:         List of video file paths.
        output_dir:          Root output directory.
        cfg:                 Full config dict (deep-copied per worker).
        workers:             Requested parallel workers (clamped to safe max).
        memory_threshold_mb: Minimum free RAM in MB before forking new worker.
        progress_callback:   Optional callable(result_dict) called after each file
                             finishes (from the main thread).

    Returns:
        List of result dicts, one per video:
            {
                "video":       str,    # input video path
                "video_name":  str,    # filename
                "stats":       dict|None,
                "status":      "success"|"skipped"|"error",
                "error":       str|None,
            }
    """
    if not video_files:
        return []

    effective_workers = resolve_max_workers(workers)

    # Fall back to sequential if only 1 worker requested
    if effective_workers == 1:
        return _run_sequential(
            video_files, output_dir, cfg, progress_callback=progress_callback
        )

    return _run_parallel(
        video_files,
        output_dir,
        cfg,
        effective_workers,
        memory_threshold_mb,
        progress_callback,
    )


def _run_sequential(
    video_files: list,
    output_dir: str,
    cfg: dict,
    progress_callback=None,
) -> list:
    """
    Sequential fallback -- calls process_video directly in the current process.
    Used when workers=1.
    """
    from main import process_video  # noqa: PLC0415

    results = []
    for i, video_path in enumerate(video_files, 1):
        fname = os.path.basename(video_path)
        print(f"\n[PARALLEL] {i}/{len(video_files)}: {fname}")
        try:
            stats = process_video(video_path, output_dir, copy.deepcopy(cfg), i)
            status = "success" if stats else "skipped"
            r = {
                "video":      video_path,
                "video_name": fname,
                "stats":      stats,
                "status":     status,
                "error":      None if stats else "No frames extracted",
            }
        except Exception as exc:
            r = {
                "video":      video_path,
                "video_name": fname,
                "stats":      None,
                "status":     "error",
                "error":      str(exc),
            }
        results.append(r)
        if progress_callback:
            try:
                progress_callback(r)
            except Exception:
                pass
    return results


def _run_parallel(
    video_files: list,
    output_dir: str,
    cfg: dict,
    workers: int,
    memory_threshold_mb: int,
    progress_callback=None,
) -> list:
    """
    True parallel execution via ProcessPoolExecutor.

    Each video gets its own subprocess. Crash isolation: futures that raise
    are caught individually and converted to error results without aborting
    the remaining workers.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415

    print(f"[PARALLEL] Starting {len(video_files)} video(s) with {workers} worker(s)")
    if _PSUTIL_AVAILABLE:
        free = get_free_ram_mb()
        print(f"[PARALLEL] Free RAM: {free} MB (threshold: {memory_threshold_mb} MB)")

    total = len(video_files)
    results_map: dict = {}      # video_path -> result dict
    futures_map: dict = {}      # future -> video_path

    # Submit all videos; respect memory guard before each submission
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, video_path in enumerate(video_files, 1):
            fname = os.path.basename(video_path)

            # Memory guard: skip fork if RAM is critically low
            if not check_memory_ok(memory_threshold_mb):
                free_mb = get_free_ram_mb()
                msg = (
                    f"Skipped (low RAM: {free_mb} MB < {memory_threshold_mb} MB)"
                )
                print(f"[PARALLEL] WARN {fname}: {msg}")
                results_map[video_path] = {
                    "video":      video_path,
                    "video_name": fname,
                    "stats":      None,
                    "status":     "skipped",
                    "error":      msg,
                }
                continue

            # Deep-copy cfg so each worker gets its own mutable copy
            worker_cfg = copy.deepcopy(cfg)
            future = executor.submit(_worker, video_path, output_dir, worker_cfg, i)
            futures_map[future] = video_path
            print(f"[PARALLEL] Submitted {i}/{total}: {fname}")

        # Collect results as they complete
        done_count = 0
        for future in as_completed(futures_map):
            video_path = futures_map[future]
            fname = os.path.basename(video_path)
            done_count += 1
            try:
                result = future.result()
                stats  = result.get("stats")
                error  = result.get("error")
                if error:
                    status = "error"
                    print(f"[PARALLEL] ERROR {fname}: {error}")
                elif not stats:
                    status = "skipped"
                    error  = "No frames extracted"
                    print(f"[PARALLEL] SKIPPED {fname}")
                else:
                    status = "success"
                    print(f"[PARALLEL] Done {done_count}/{total}: {fname}")

                r = {
                    "video":      video_path,
                    "video_name": fname,
                    "stats":      stats,
                    "status":     status,
                    "error":      error,
                }
            except Exception as exc:
                # Worker process crashed (e.g. OOM, killed) — isolate the failure
                r = {
                    "video":      video_path,
                    "video_name": fname,
                    "stats":      None,
                    "status":     "error",
                    "error":      f"Worker crash: {exc}",
                }
                print(f"[PARALLEL] CRASH {fname}: {exc}")

            results_map[video_path] = r
            if progress_callback:
                try:
                    progress_callback(r)
                except Exception:
                    pass

    # Return results in original input order
    ordered = []
    for video_path in video_files:
        if video_path in results_map:
            ordered.append(results_map[video_path])
        else:
            # Should not happen, but be defensive
            ordered.append({
                "video":      video_path,
                "video_name": os.path.basename(video_path),
                "stats":      None,
                "status":     "error",
                "error":      "Unknown: no result recorded",
            })
    return ordered


# ─────────────────────────────────────────────
# Convenience: scan folder + run parallel
# ─────────────────────────────────────────────

def run_parallel_folder(
    input_folder: str,
    output_dir: str,
    cfg: dict,
    workers: int = 1,
    memory_threshold_mb: int = _MIN_FREE_RAM_MB,
) -> list:
    """
    Collect all supported video files from input_folder and run parallel_batch.

    Returns same list format as run_parallel_batch().
    """
    video_files = sorted([
        str(f)
        for f in Path(input_folder).iterdir()
        if f.is_file() and f.suffix.lower() in _VIDEO_EXTS
    ])
    if not video_files:
        print(f"[PARALLEL] No video files found in: {input_folder}")
        return []
    print(f"[PARALLEL] Found {len(video_files)} video(s) in: {input_folder}")
    return run_parallel_batch(video_files, output_dir, cfg, workers=workers,
                              memory_threshold_mb=memory_threshold_mb)
