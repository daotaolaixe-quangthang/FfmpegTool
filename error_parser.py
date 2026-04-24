"""
error_parser.py
===============
Smart FFmpeg stderr parser for FfmpegTool Phase 2.

Converts raw ffmpeg error output into human-readable, actionable messages
grouped by category. Used by extractor.py to surface friendly error messages
instead of raw stderr dumps.

Usage:
    from error_parser import parse_ffmpeg_error, format_error

    parsed = parse_ffmpeg_error(ffmpeg_stderr_string)
    print(format_error(parsed))
    # e.g. "[CODEC ERROR] Decoder 'hevc_cuvid' not found. Try --no-hw or use CPU mode."
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# Categories
# ─────────────────────────────────────────────

class ErrorCategory(Enum):
    PERMISSION          = "permission"
    CODEC_UNSUPPORTED   = "codec_unsupported"
    FILE_CORRUPT        = "file_corrupt"
    CONTAINER_MISMATCH  = "container_mismatch"
    GPU_OOM             = "gpu_oom"
    DISK_FULL           = "disk_full"
    TIMEOUT             = "timeout"
    FFMPEG_MISSING      = "ffmpeg_missing"
    FILTER_ERROR        = "filter_error"
    UNKNOWN             = "unknown"


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class ParsedError:
    """Result of parsing an ffmpeg/ffprobe stderr string."""
    category:    ErrorCategory
    message:     str            # human-readable explanation
    raw:         str            # original stderr (truncated to 500 chars)
    hint:        Optional[str]  = None  # optional fix suggestion


# ─────────────────────────────────────────────
# Pattern registry
# ─────────────────────────────────────────────

# Each entry: (regex pattern, ErrorCategory, human message, optional hint)
_PATTERNS: list[tuple[str, ErrorCategory, str, Optional[str]]] = [
    # FFmpeg/FFprobe binary missing
    (
        r"ffmpeg.*not found|ffprobe.*not found|No such file.*ffmpeg",
        ErrorCategory.FFMPEG_MISSING,
        "FFmpeg or FFprobe was not found in PATH.",
        "Install FFmpeg: https://ffmpeg.org/download.html and make sure it is on PATH.",
    ),

    # Permission denied
    (
        r"Permission denied|Access is denied|cannot open",
        ErrorCategory.PERMISSION,
        "Permission denied — cannot read the input or write to the output.",
        "Check file/folder permissions and ensure the output directory is writable.",
    ),

    # GPU out of memory
    (
        r"CUDA_ERROR_OUT_OF_MEMORY|out of memory|Cannot allocate|cudaMalloc",
        ErrorCategory.GPU_OOM,
        "GPU ran out of memory during hardware-accelerated processing.",
        "Try disabling hardware acceleration (set encoder=cpu in config.json).",
    ),

    # Codec/decoder not found
    (
        r"Decoder .* not found|codec not currently supported|"
        r"Unknown encoder|no such encoder|Encoder .* not found|"
        r"codec.*not supported|hwcontext",
        ErrorCategory.CODEC_UNSUPPORTED,
        "A required codec or hardware encoder was not found.",
        "Try using --no-hw or set hardware.encoder=cpu in config.json.",
    ),

    # Corrupt/invalid file
    (
        r"Invalid data found|moov atom not found|"
        r"error while decoding|Invalid stream|"
        r"Truncated file|End of file|corrupt(?:ed)?\s+(?:file|data|stream|packet|header)",
        ErrorCategory.FILE_CORRUPT,
        "The input file appears to be corrupt or incomplete.",
        "Re-download the file or verify it plays correctly in a media player.",
    ),

    # Container/format mismatch
    (
        r"Could not find tag for codec|"
        r"Muxer .* does not support|"
        r"Unable to find a suitable output format|"
        r"matches no streams",
        ErrorCategory.CONTAINER_MISMATCH,
        "The video format or container is incompatible with the output settings.",
        "Try a different output format or use --no-normalize to skip transcoding.",
    ),

    # Video filter error
    (
        r"Error while filtering|filter graph|"
        r"Impossible to convert between|"
        r"scale.*invalid|Invalid option|"
        r"No such filter",
        ErrorCategory.FILTER_ERROR,
        "FFmpeg encountered an error in the video filter chain.",
        "Check that --draft mode scale/fps values are valid.",
    ),

    # Disk full
    (
        r"No space left|Disk quota exceeded|"
        r"not enough space|ENOSPC",
        ErrorCategory.DISK_FULL,
        "Disk is full — cannot write output frames or temporary files.",
        "Free up disk space in the output directory and retry.",
    ),

    # Timeout (internal, raised by our code not ffmpeg directly)
    (
        r"timed out|TimeoutExpired|timeout expired",
        ErrorCategory.TIMEOUT,
        "FFmpeg process timed out.",
        "The video may be very large or the system is under heavy load. Try again.",
    ),
]


# ─────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────

def parse_ffmpeg_error(stderr: str) -> ParsedError:
    """
    Parse raw ffmpeg stderr into a structured ParsedError.

    Tries each pattern in order; returns the first match.
    Falls back to UNKNOWN if no pattern matches.

    Args:
        stderr: Raw stderr string from ffmpeg/ffprobe subprocess.

    Returns:
        ParsedError with category, message, raw, and optional hint.
    """
    raw_truncated = (stderr or "").strip()[:500]

    if not raw_truncated:
        return ParsedError(
            category = ErrorCategory.UNKNOWN,
            message  = "FFmpeg exited with an error but produced no stderr output.",
            raw      = "",
            hint     = "Enable verbose logging with ffmpeg -loglevel verbose for details.",
        )

    for pattern, category, message, hint in _PATTERNS:
        if re.search(pattern, raw_truncated, re.IGNORECASE):
            return ParsedError(
                category = category,
                message  = message,
                raw      = raw_truncated,
                hint     = hint,
            )

    # No pattern matched
    # Extract the last meaningful error line for the message
    lines = [l.strip() for l in raw_truncated.splitlines() if l.strip()]
    last_line = lines[-1] if lines else raw_truncated[:120]
    return ParsedError(
        category = ErrorCategory.UNKNOWN,
        message  = f"FFmpeg error: {last_line}",
        raw      = raw_truncated,
        hint     = "Check FFmpeg output above for details.",
    )


def format_error(parsed: ParsedError) -> str:
    """
    Format a ParsedError into a single human-readable string for print()/raise.

    Example output:
        [CODEC ERROR] A required codec or hardware encoder was not found.
        Hint: Try using --no-hw or set hardware.encoder=cpu in config.json.
    """
    tag = parsed.category.name.replace("_", " ")
    lines = [f"[{tag}] {parsed.message}"]
    if parsed.hint:
        lines.append(f"  Hint: {parsed.hint}")
    return "\n".join(lines)


def format_error_short(parsed: ParsedError) -> str:
    """
    Single-line version — for use in JSON reports and batch summaries.

    Example: "[CODEC_UNSUPPORTED] A required codec or hardware encoder was not found."
    """
    return f"[{parsed.category.value}] {parsed.message}"
