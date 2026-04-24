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
"""

import os
import subprocess
from pathlib import Path


def extract_by_fps(
    video_path: str,
    raw_dir: str,
    fps: float,
    jpeg_quality: int,
    hwaccel_args: list[str] | None = None,
) -> list[str]:
    """Extract frames at a fixed FPS rate using FFmpeg.

    Args:
        hwaccel_args: Optional list of hardware decode args, e.g. ["-hwaccel", "cuda"].
                      Injected before -i. Defaults to [] (CPU decode).
    """
    os.makedirs(raw_dir, exist_ok=True)
    output_pattern = os.path.join(raw_dir, "frame_%05d.jpg")
    hw_args = hwaccel_args or []

    cmd = [
        "ffmpeg", "-y",           # -y (overwrite) MUST be a global option before -i
        *hw_args,                  # hardware decode acceleration (empty = CPU)
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", str(jpeg_quality),
        "-loglevel", "warning",
        output_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")

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
        return extract_by_fps(video_path, raw_dir, fps=1.0, jpeg_quality=jpeg_quality)

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
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
        if result.returncode == 0 and os.path.exists(output_path):
            frame_paths.append(output_path)

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

    Returns:
        list of extracted frame file paths
    """
    mode            = cfg.get("mode", "fps")
    fps             = cfg.get("fps", 5)
    quality         = cfg.get("jpeg_quality", 2)
    scene_threshold = cfg.get("scene_threshold", 27.0)
    hwaccel_args    = cfg.get("_hwaccel_args", [])  # set by hw_detect in main.py

    hw_label = f" [HW:{' '.join(hwaccel_args)}]" if hwaccel_args else ""
    print(f"\n[EXTRACT] Mode: {mode.upper()}{hw_label} | Video: {os.path.basename(video_path)}")

    if mode == "scene":
        frames = extract_by_scene(video_path, raw_dir, scene_threshold, quality, hwaccel_args)
    else:
        frames = extract_by_fps(video_path, raw_dir, fps, quality, hwaccel_args)

    print(f"[EXTRACT] Done -- {len(frames)} raw frames extracted")
    return frames
