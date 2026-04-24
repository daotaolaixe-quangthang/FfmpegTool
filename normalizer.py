"""
normalizer.py
=============
Auto-normalize input videos for FfmpegTool Phase 2 (#7).

Runs ffprobe (reuses probe_first.ProbeResult) to check codec and pixel format.
If the video uses an unsupported/problematic codec, transcodes it to
H.264 / yuv420p (universally compatible) as a pre-processing step 0 before
the main extraction pipeline.

This module is a WRAPPER — it does NOT touch extractor.py, filters.py, etc.
The normalized temp file is cleaned up after the pipeline finishes
(caller is responsible for cleanup, indicated by NormalizeResult.tmp_path).

Usage (in main.py process_video, step 0):
    from normalizer import normalize_video, needs_normalization
    from probe_first import probe_video

    probe = probe_video(video_path)
    needed, reason = needs_normalization(probe)
    norm_result = normalize_video(video_path, raw_dir, cfg, probe=probe)
    video_path = norm_result.path   # use normalized path for rest of pipeline
    ...
    # After pipeline:
    if norm_result.tmp_path:
        norm_result.cleanup()
"""

import os
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from probe_first import ProbeResult


# ─────────────────────────────────────────────
# Codec compatibility tables
# ─────────────────────────────────────────────

# Codecs that FFmpeg handles well without transcoding
SUPPORTED_CODECS = {
    "h264", "h265", "hevc",
    "vp8", "vp9", "av1",
    "mpeg4", "mpeg2video",
}

# Pixel formats that our filter pipeline (OpenCV/Pillow) handles reliably
# Formats NOT in this set will trigger normalization
SUPPORTED_PIX_FMTS = {
    "yuv420p",
    "yuvj420p",   # JPEG variant of yuv420p — safe
}

# Formats known to cause subtle issues (logged but not always fatal)
PROBLEMATIC_PIX_FMTS = {
    "yuv422p", "yuv444p",
    "yuv420p10le", "yuv422p10le", "yuv444p10le",
    "rgb24", "bgr24",
    "gbrp",
}


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class NormalizeResult:
    """Result of a normalize_video() call."""
    path:          str            # path to use for extraction (original or normalized)
    was_transcoded: bool          # True if a new file was created
    reason:        str            # why normalization was/wasn't done
    original_path: str            # always the original input video path
    tmp_path:      Optional[str]  = None   # path to temp file (same as path if transcoded)

    def cleanup(self):
        """Remove the temporary normalized file if it exists."""
        if self.tmp_path and os.path.exists(self.tmp_path):
            try:
                os.remove(self.tmp_path)
            except OSError:
                pass


# ─────────────────────────────────────────────
# Needs-normalization check
# ─────────────────────────────────────────────

def needs_normalization(probe) -> tuple[bool, str]:
    """
    Decide whether the video needs to be transcoded before processing.

    Args:
        probe: ProbeResult from probe_first.probe_video()

    Returns:
        (True, reason_string) if normalization is needed,
        (False, "")           if the video is already compatible.
    """
    if not probe.ok:
        # Let the main pipeline handle broken files (they'll fail at extraction)
        return False, ""

    codec = (probe.video_codec or "").lower()
    pix   = (probe.pixel_format or "").lower()

    if codec and codec not in SUPPORTED_CODECS:
        return True, f"Unsupported codec '{codec}' — transcoding to H.264"

    if pix and pix in PROBLEMATIC_PIX_FMTS:
        return True, f"Pixel format '{pix}' may cause filter issues — converting to yuv420p"

    return False, ""


# ─────────────────────────────────────────────
# Normalizer
# ─────────────────────────────────────────────

def normalize_video(
    video_path:  str,
    work_dir:    str,
    cfg:         dict,
    probe=None,
) -> NormalizeResult:
    """
    Transcode a video to H.264/yuv420p if needed.

    Args:
        video_path: Input video path.
        work_dir:   Directory where the normalized temp file is written.
                    Should be the video-level output dir (e.g. output/clip/).
        cfg:        Full config dict. Reads cfg["normalize"]["enabled"].
        probe:      Optional pre-computed ProbeResult. If None, probe is run here.

    Returns:
        NormalizeResult. If was_transcoded=False, .path == video_path (no copy made).
    """
    # Allow caller to disable via cfg
    norm_cfg = cfg.get("normalize", {})
    if not norm_cfg.get("enabled", True):
        return NormalizeResult(
            path           = video_path,
            was_transcoded = False,
            reason         = "Normalization disabled via config",
            original_path  = video_path,
        )

    # Probe if not provided
    if probe is None:
        from probe_first import probe_video
        probe = probe_video(video_path)

    needed, reason = needs_normalization(probe)

    if not needed:
        return NormalizeResult(
            path           = video_path,
            was_transcoded = False,
            reason         = "No normalization needed",
            original_path  = video_path,
        )

    # Build output path for normalized file
    stem       = Path(video_path).stem
    norm_name  = f"{stem}_normalized.mp4"
    os.makedirs(work_dir, exist_ok=True)
    norm_path  = os.path.join(work_dir, norm_name)

    print(f"[NORMALIZE] {reason}")
    size_mb = _file_size_mb(video_path)
    if size_mb:
        print(f"[NORMALIZE] Input: {size_mb:.1f} MB — transcoding to H.264/yuv420p...")

    cmd = [
        "ffmpeg", "-y",
        "-i",        video_path,
        "-c:v",      "libx264",
        "-pix_fmt",  "yuv420p",
        "-preset",   "fast",
        "-crf",      "18",         # high quality, lossless enough for frame extraction
        "-c:a",      "copy",       # keep audio unchanged (not needed for extraction)
        "-loglevel", "warning",
        norm_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        from error_parser import parse_ffmpeg_error, format_error
        parsed = parse_ffmpeg_error(result.stderr)
        raise RuntimeError(
            f"Normalization failed:\n{format_error(parsed)}"
        )

    out_mb = _file_size_mb(norm_path)
    print(f"[NORMALIZE] Done -> {norm_name} ({out_mb:.1f} MB)" if out_mb else
          f"[NORMALIZE] Done -> {norm_name}")

    return NormalizeResult(
        path           = norm_path,
        was_transcoded = True,
        reason         = reason,
        original_path  = video_path,
        tmp_path       = norm_path,
    )


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _file_size_mb(path: str) -> Optional[float]:
    """Return file size in MB, or None if file doesn't exist."""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return None
