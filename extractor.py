"""
extractor.py
============
Extract frames from a video using FFmpeg.
Supports two modes:
  - fps   : extract N frames per second (fast, simple)
  - scene : extract at scene change boundaries (smart, uses PySceneDetect)

Hardware acceleration:
  Pass cfg["_hwaccel_args"] (set by hw_detect.resolve_encoder) to enable
  hardware-accelerated decode. Example: ["-hwaccel", "cuda"] for NVENC.
  Falls back to CPU silently if not set.

Phase 2 additions:
  Draft mode: cfg["draft"] = True injects scale=640:360 before fps filter.
  Error parser: ffmpeg failures raise RuntimeError with human-readable message.
"""

import os
import subprocess
from pathlib import Path

from error_parser import parse_ffmpeg_error, format_error


def extract_by_fps(
    video_path: str,
    raw_dir: str,
    fps: float,
    jpeg_quality: int,
    hwaccel_args: list[str] | None = None,
    draft: bool = False,
) -> list[str]:
    """Extract frames at a fixed FPS rate using FFmpeg.

    Args:
        hwaccel_args: Optional list of hardware decode args, e.g. ["-hwaccel", "cuda"].
                      Injected before -i. Defaults to [] (CPU decode).
        draft:        If True, scale to 640x360 before extraction (fast preview mode).
                      Preserves aspect ratio via scale=640:-2 capped to 360p height.
    """
    os.makedirs(raw_dir, exist_ok=True)
    output_pattern = os.path.join(raw_dir, "frame_%05d.jpg")
    hw_args = hwaccel_args or []

    # Draft mode: scale down to 360p for fast previews
    # scale=-2:360 keeps aspect ratio; fps filter applied after scale
    if draft:
        vf_filter = f"scale=-2:360,fps={fps}"
    else:
        vf_filter = f"fps={fps}"

    # QSV decode yields GPU frames; download them before CPU filters.
    if "qsv" in hw_args:
        vf_filter = f"hwdownload,format=nv12,{vf_filter}"

    cmd = [
        "ffmpeg", "-y",           # -y (overwrite) MUST be a global option before -i
        *hw_args,                  # hardware decode acceleration (empty = CPU)
        "-i", video_path,
        "-vf", vf_filter,
        "-q:v", str(jpeg_quality),
        "-loglevel", "warning",
        output_pattern,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                                timeout=300)  # BUG-M1 FIX: 5-min ceiling prevents hung ffmpeg blocking forever
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out after 300s extracting frames from: {video_path}")
    if result.returncode != 0:
        parsed = parse_ffmpeg_error(result.stderr)
        raise RuntimeError(format_error(parsed))

    frames = sorted(Path(raw_dir).glob("frame_*.jpg"))
    return [str(f) for f in frames]


def extract_by_scene(
    video_path: str,
    raw_dir: str,
    threshold: float,
    jpeg_quality: int,
    hwaccel_args: list[str] | None = None,
) -> list[str]:
    """
    Extract the middle frame of each detected scene using PySceneDetect.
    Much smarter than fps mode — captures true scene changes only.

    Args:
        hwaccel_args: Optional hardware decode args (same as extract_by_fps).
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import AdaptiveDetector
    except ImportError:
        raise ImportError("scenedetect not installed. Run: pip install scenedetect[opencv]")

    os.makedirs(raw_dir, exist_ok=True)

    print("  [SCENE] Detecting scene boundaries...")
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=threshold / 10))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    print(f"  [SCENE] Found {len(scenes)} scenes")

    if not scenes:
        print("  [SCENE] No scenes detected -- falling back to 1 fps extraction")
        # BUG-H2 FIX: forward hwaccel_args and draft to the fallback call so
        # that hardware acceleration and draft mode are not silently dropped
        # when scene detection finds zero scenes.
        return extract_by_fps(
            video_path, raw_dir, fps=1.0,
            jpeg_quality=jpeg_quality,
            hwaccel_args=hwaccel_args,
            draft=False,  # draft is meaningless for 1-fps fallback (no speed gain)
        )

    hw_args = hwaccel_args or []
    frame_paths = []
    for i, (start, end) in enumerate(scenes):
        # Use the middle of each scene (most stable frame)
        mid_sec = (start.get_seconds() + end.get_seconds()) / 2
        output_path = os.path.join(raw_dir, f"frame_{i:05d}.jpg")

        cmd = [
            "ffmpeg", "-y",       # -y (overwrite) MUST be a global option before -ss/-i
            *hw_args,              # hardware decode acceleration
            "-ss", f"{mid_sec:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", str(jpeg_quality),
            "-loglevel", "quiet",
            output_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                                    timeout=60)  # BUG-M1 FIX: per-frame 60s ceiling for scene extraction
        except subprocess.TimeoutExpired:
            print(f"  [SCENE] Frame {i} timed out after 60s - skipping")
            continue
        if result.returncode == 0 and os.path.exists(output_path):
            frame_paths.append(output_path)
        elif result.returncode != 0:
            # Log but don't abort the whole scene loop for single-frame failures
            parsed = parse_ffmpeg_error(result.stderr)
            print(f"  [SCENE] Frame {i} failed: {parsed.message}")

    return frame_paths


def extract_frames(video_path: str, raw_dir: str, cfg: dict) -> list[str]:
    """
    Main extraction entry point. Reads mode from config.

    Args:
        video_path : path to input video
        raw_dir    : folder to save raw extracted frames
        cfg        : extraction config dict
                     cfg["_hwaccel_args"] (optional): hardware decode args
                     injected by hw_detect and resolve_encoder in main.py.
                     cfg["draft"] (optional bool): enable 360p draft mode.

    Returns:
        list of extracted frame file paths
    """
    mode            = cfg.get("mode", "fps")
    fps             = cfg.get("fps", 5)
    quality         = cfg.get("jpeg_quality", 2)
    scene_threshold = cfg.get("scene_threshold", 27.0)
    hwaccel_args    = cfg.get("_hwaccel_args", [])  # set by hw_detect in main.py
    draft           = cfg.get("draft", False)        # Phase 2: draft/preview mode

    hw_label    = f" [HW:{' '.join(hwaccel_args)}]" if hwaccel_args else ""
    draft_label = " [DRAFT 360p]" if draft else ""
    print(f"\n[EXTRACT] Mode: {mode.upper()}{hw_label}{draft_label} | Video: {os.path.basename(video_path)}")

    if mode == "scene":
        frames = extract_by_scene(video_path, raw_dir, scene_threshold, quality, hwaccel_args)
    else:
        frames = extract_by_fps(video_path, raw_dir, fps, quality, hwaccel_args, draft=draft)

    print(f"[EXTRACT] Done -- {len(frames)} raw frames extracted")
    return frames
