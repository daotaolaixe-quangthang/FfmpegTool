"""
hw_detect.py
============
Auto-detect the best available FFmpeg hardware encoder/decoder on this machine.

Priority order: NVIDIA NVENC → Intel QSV → AMD AMF → CPU x264 (fallback)

Usage in main.py:
    from hw_detect import resolve_encoder, get_ffmpeg_hwaccel_args, print_hw_report

    hw_key = resolve_encoder(cfg["hardware"])          # e.g. "nvenc"
    hwaccel_args = get_ffmpeg_hwaccel_args(hw_key)     # e.g. ["-hwaccel", "cuda"]
    # Inject into cfg so extractor.py can use them
    cfg["extraction"]["_hwaccel_args"] = hwaccel_args

Probe strategy:
    For each encoder, attempt to encode a 1-frame RGB null source.
    If ffmpeg returns exit code 0, the encoder is considered available.
    Timeout: 10 seconds per probe to avoid hanging on driver issues.

Note on frame extraction (decode side):
    FFmpeg frame extraction uses hardware-accelerated DECODE when hwaccel is set.
    The frames are still saved as JPEG (CPU), but the decode step is GPU-offloaded.
    Net effect: faster extraction on large/4K videos, no quality difference.
"""

import subprocess
from typing import Optional


# ─────────────────────────────────────────────
# Encoder profiles
# ─────────────────────────────────────────────

# Full info for each supported hardware encoder
HW_PROFILES: dict[str, dict] = {
    "nvenc": {
        "label":    "NVIDIA NVENC (CUDA)",
        "hwaccel":  "cuda",
        "encoder":  "h264_nvenc",
        "priority": 1,
    },
    "qsv": {
        "label":    "Intel Quick Sync Video",
        "hwaccel":  "qsv",
        "encoder":  "h264_qsv",
        "priority": 2,
    },
    "amf": {
        "label":    "AMD AMF (D3D11VA)",
        "hwaccel":  "d3d11va",
        "encoder":  "h264_amf",
        "priority": 3,
    },
    "cpu": {
        "label":    "CPU x264 (fallback)",
        "hwaccel":  None,
        "encoder":  "libx264",
        "priority": 99,
    },
}

# Probe command: encode a 1-frame null input with each encoder
# Using -f null as output format avoids any disk write
_NULL_INPUT = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", "color=black:s=64x64:r=1:d=0.1",
               "-vframes", "1"]

_PROBE_CMDS: dict[str, list[str]] = {
    "nvenc": _NULL_INPUT + ["-c:v", "h264_nvenc", "-f", "null", "-"],
    "qsv":   _NULL_INPUT + ["-c:v", "h264_qsv",   "-f", "null", "-"],
    "amf":   _NULL_INPUT + ["-c:v", "h264_amf",   "-f", "null", "-"],
}


# ─────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────

def probe_encoder(encoder_key: str) -> bool:
    """
    Test if a hardware encoder is available on this machine.

    Returns True if ffmpeg exits cleanly with that encoder.
    Returns False on any error, timeout, or unavailability.
    """
    cmd = _PROBE_CMDS.get(encoder_key)
    if not cmd:
        return False  # cpu doesn't need probing

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except FileNotFoundError:
        # ffmpeg not in PATH
        return False
    except Exception:
        return False


def detect_best_encoder() -> str:
    """
    Probe all hardware encoders and return the best available key.

    Priority: nvenc → qsv → amf → cpu (always available as fallback).
    """
    for key in ("nvenc", "qsv", "amf"):
        if probe_encoder(key):
            return key
    return "cpu"


def get_hw_profile(encoder_key: str) -> dict:
    """Return the full hardware profile dict for a given encoder key."""
    return HW_PROFILES.get(encoder_key, HW_PROFILES["cpu"])


def get_ffmpeg_hwaccel_args(encoder_key: str) -> list[str]:
    """
    Return FFmpeg input-side args for hardware-accelerated decode.

    These args are prepended BEFORE -i in the ffmpeg command:
      nvenc → ["-hwaccel", "cuda"]
      qsv   → ["-hwaccel", "qsv"]
      amf   → ["-hwaccel", "d3d11va"]
      cpu   → []   (no hardware acceleration)

    Note: Hardware decode gives speed benefit during frame extraction.
    The output (JPEG frames) is always written by CPU regardless.
    """
    profile = get_hw_profile(encoder_key)
    hwaccel = profile.get("hwaccel")
    if hwaccel:
        return ["-hwaccel", hwaccel]
    return []


def resolve_encoder(hw_cfg: dict) -> str:
    """
    Resolve the encoder key to use, based on the hardware config section.

    hw_cfg["encoder"] can be:
      "auto"   → probe all encoders and pick the best (default)
      "nvenc"  → force NVIDIA NVENC
      "qsv"    → force Intel Quick Sync
      "amf"    → force AMD AMF
      "cpu"    → force CPU x264

    Args:
        hw_cfg: cfg["hardware"] dict

    Returns:
        One of: "nvenc", "qsv", "amf", "cpu"
    """
    requested = hw_cfg.get("encoder", "auto").lower().strip()

    if requested == "auto":
        detected = detect_best_encoder()
        profile  = get_hw_profile(detected)
        print(f"[HW] Auto-detected: {profile['label']}")
        return detected

    if requested not in HW_PROFILES:
        print(f"[HW] ⚠ Unknown encoder '{requested}'. Falling back to CPU.")
        return "cpu"

    # User explicitly requested an encoder — verify it's available (except cpu)
    if requested != "cpu" and not probe_encoder(requested):
        profile = get_hw_profile(requested)
        print(f"[HW] ⚠ '{profile['label']}' not available on this machine. Falling back to CPU.")
        return "cpu"

    profile = get_hw_profile(requested)
    print(f"[HW] Using: {profile['label']}")
    return requested


def print_hw_report() -> None:
    """Print a hardware availability report to stdout (for diagnostics)."""
    print("\n[HW] Hardware Encoder Availability Report:")
    print(f"  {'ENCODER':<30} STATUS")
    print(f"  {'─'*30} ──────")
    for key in ("nvenc", "qsv", "amf"):
        profile   = get_hw_profile(key)
        available = probe_encoder(key)
        status    = "✓ Available" if available else "✗ Not found"
        print(f"  {profile['label']:<30} {status}")
    print(f"  {'CPU x264 (fallback)':<30} ✓ Always available")
    print()
