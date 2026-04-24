"""
probe_first.py
==============
Pre-flight validation: run ffprobe on all input videos BEFORE starting
the main processing pipeline.

Key benefit: discover bad/unsupported files EARLY, not midway through a
2-hour batch (avoids the "failed at file 73 after 90 minutes" situation).

Usage in main.py process_batch():
    from probe_first import scan_batch

    ok_files, bad_results = scan_batch(video_files)
    # ok_files  → proceed with normal processing
    # bad_results → already reported to stdout; stored in summary JSON

Each ProbeResult has:
    .path          (str)            absolute path to video
    .ok            (bool)           True if no issues found
    .width / .height  (int|None)    resolution
    .fps           (float|None)     frame rate
    .duration      (float|None)     duration in seconds
    .video_codec   (str|None)       e.g. "h264", "hevc"
    .audio_codec   (str|None)       e.g. "aac", "mp3"
    .pixel_format  (str|None)       e.g. "yuv420p"
    .issues        (list[str])      human-readable problem descriptions
"""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class ProbeResult:
    """Structured output of a single ffprobe call."""
    path:         str
    ok:           bool               = False
    width:        Optional[int]      = None
    height:       Optional[int]      = None
    fps:          Optional[float]    = None
    duration:     Optional[float]    = None
    video_codec:  Optional[str]      = None
    audio_codec:  Optional[str]      = None
    pixel_format: Optional[str]      = None
    file_size_mb: Optional[float]    = None
    issues:       list[str]          = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to plain dict (for JSON reporting)."""
        return {
            "path":         self.path,
            "ok":           self.ok,
            "width":        self.width,
            "height":       self.height,
            "fps":          self.fps,
            "duration":     self.duration,
            "video_codec":  self.video_codec,
            "audio_codec":  self.audio_codec,
            "pixel_format": self.pixel_format,
            "file_size_mb": self.file_size_mb,
            "issues":       self.issues,
        }


# ─────────────────────────────────────────────
# Single-file probe
# ─────────────────────────────────────────────

def probe_video(video_path: str, timeout: int = 30) -> ProbeResult:
    """
    Run ffprobe on a single video and return a structured ProbeResult.

    Args:
        video_path: Absolute or relative path to the video file.
        timeout:    Max seconds to wait for ffprobe (default 30).

    Returns:
        ProbeResult with .ok=True if no issues, .ok=False with .issues list.
    """
    result = ProbeResult(path=video_path)

    # Check file existence first (fast path)
    if not Path(video_path).exists():
        result.issues.append("File not found")
        return result

    # Record file size
    try:
        size_bytes = Path(video_path).stat().st_size
        result.file_size_mb = round(size_bytes / (1024 * 1024), 2)
    except Exception:
        pass

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if proc.returncode != 0:
            # Include only first 200 chars of stderr to keep output clean
            err_snippet = (proc.stderr or "").strip()[:200]
            result.issues.append(f"ffprobe error: {err_snippet or 'unknown error'}")
            return result

        data = json.loads(proc.stdout)

    except subprocess.TimeoutExpired:
        result.issues.append(f"ffprobe timed out (>{timeout}s) — file may be corrupt or very large")
        return result
    except json.JSONDecodeError:
        result.issues.append("ffprobe output was not valid JSON — file may be corrupt")
        return result
    except FileNotFoundError:
        result.issues.append("ffprobe not found in PATH — is FFmpeg installed?")
        return result
    except Exception as e:
        result.issues.append(f"Unexpected probe error: {e}")
        return result

    # ── Parse streams ──
    streams     = data.get("streams", [])
    fmt         = data.get("format", {})
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if not video_stream:
        result.issues.append("No video stream found — may be audio-only or corrupt")
        return result

    # Resolution
    result.width  = video_stream.get("width")
    result.height = video_stream.get("height")

    # FPS (r_frame_rate is more reliable than avg_frame_rate for VFR)
    fps_str = video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate", "0/1")
    try:
        num, den = fps_str.split("/")
        result.fps = round(float(num) / float(den), 3) if float(den) != 0 else 0.0
    except Exception:
        result.fps = 0.0

    # Codec and pixel format
    result.video_codec   = video_stream.get("codec_name")
    result.pixel_format  = video_stream.get("pix_fmt")
    if audio_stream:
        result.audio_codec = audio_stream.get("codec_name")

    # Duration (prefer format-level, falls back to stream-level)
    duration_str = fmt.get("duration") or video_stream.get("duration", "0")
    try:
        result.duration = float(duration_str)
    except (ValueError, TypeError):
        result.duration = 0.0

    # ── Validation checks ──
    if not result.width or result.width <= 0 or not result.height or result.height <= 0:
        result.issues.append(f"Invalid resolution: {result.width}x{result.height}")

    if result.fps == 0.0:
        result.issues.append("FPS could not be determined (VFR or corrupt header)")

    if result.duration is not None and result.duration < 0.5:
        result.issues.append(
            f"Duration very short ({result.duration:.2f}s) — "
            "pipeline may extract 0 frames and skip this video"
        )

    if result.file_size_mb is not None and result.file_size_mb < 0.001:
        result.issues.append("File size is 0 bytes — empty file")

    result.ok = len(result.issues) == 0
    return result


# ─────────────────────────────────────────────
# Batch scan
# ─────────────────────────────────────────────

def scan_batch(
    video_files: list[str],
    verbose: bool = True,
) -> tuple[list[str], list[ProbeResult]]:
    """
    Pre-flight scan all candidate video files using ffprobe.

    Prints a formatted report to stdout. Files with issues are excluded
    from the returned ok_files list so the batch can proceed safely.

    Args:
        video_files: List of video file paths to scan.
        verbose:     If True, print progress + summary to stdout.

    Returns:
        (ok_files, bad_results)
        ok_files    : paths with no detected issues
        bad_results : ProbeResult objects for problematic files
    """
    total       = len(video_files)
    ok_files    = []
    bad_results = []

    if verbose:
        sep = "-" * 62
        print(f"\n{sep}")
        print(f"  [PROBE] Pre-flight scan - {total} file(s)")
        print(sep)

    for i, path in enumerate(video_files, 1):
        name   = Path(path).name
        result = probe_video(path)

        if result.ok:
            ok_files.append(path)
            if verbose:
                res_str = f"{result.width}x{result.height}"  if result.width else "?x?"
                fps_str = f"{result.fps}fps"                 if result.fps   else "?fps"
                dur_str = f"{result.duration:.1f}s"          if result.duration is not None else "?s"
                mb_str  = f"{result.file_size_mb}MB"         if result.file_size_mb is not None else ""
                info    = f"{res_str} {fps_str} {dur_str} {mb_str}".strip()
                print(f"  [{i:3d}/{total}] OK   {name}  ({info})")
        else:
            bad_results.append(result)
            if verbose:
                print(f"  [{i:3d}/{total}] BAD  {name}")
                for issue in result.issues:
                    print(f"                  -> {issue}")

    if verbose:
        ok_n  = len(ok_files)
        bad_n = len(bad_results)
        print(sep)
        print(f"  [PROBE] OK: {ok_n}  |  Problematic (will be SKIPPED): {bad_n}")
        if bad_n > 0:
            bad_names = ", ".join(Path(r.path).name for r in bad_results)
            print(f"  [PROBE] Skipped: {bad_names}")
        print(f"{sep}\n")

    return ok_files, bad_results

