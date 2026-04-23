"""
extractor.py
============
Extract frames from a video using FFmpeg.
Supports two modes:
  - fps   : extract N frames per second (fast, simple)
  - scene : extract at scene change boundaries (smart, uses PySceneDetect)
"""

import os
import subprocess
from pathlib import Path


def extract_by_fps(video_path: str, raw_dir: str, fps: float, jpeg_quality: int) -> list[str]:
    """Extract frames at a fixed FPS rate using FFmpeg."""
    os.makedirs(raw_dir, exist_ok=True)
    output_pattern = os.path.join(raw_dir, "frame_%05d.jpg")

    cmd = [
        "ffmpeg", "-y",           # -y (overwrite) MUST be a global option before -i
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


def extract_by_scene(video_path: str, raw_dir: str, threshold: float, jpeg_quality: int) -> list[str]:
    """
    Extract the middle frame of each detected scene using PySceneDetect.
    Much smarter than fps mode — captures true scene changes only.
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

    frame_paths = []
    for i, (start, end) in enumerate(scenes):
        # Use the middle of each scene (most stable frame)
        mid_sec = (start.get_seconds() + end.get_seconds()) / 2
        output_path = os.path.join(raw_dir, f"frame_{i:05d}.jpg")

        cmd = [
            "ffmpeg", "-y",       # -y (overwrite) MUST be a global option before -ss/-i
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

    Returns:
        list of extracted frame file paths
    """
    mode = cfg.get("mode", "fps")
    fps = cfg.get("fps", 5)
    quality = cfg.get("jpeg_quality", 2)
    scene_threshold = cfg.get("scene_threshold", 27.0)

    print(f"\n[EXTRACT] Mode: {mode.upper()} | Video: {os.path.basename(video_path)}")

    if mode == "scene":
        frames = extract_by_scene(video_path, raw_dir, scene_threshold, quality)
    else:
        frames = extract_by_fps(video_path, raw_dir, fps, quality)

    print(f"[EXTRACT] Done -- {len(frames)} raw frames extracted")
    return frames
