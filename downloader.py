"""
downloader.py
=============
Download video from TikTok, YouTube, Instagram using yt-dlp.
Logic migrated and improved from: video_frame_extractor.py (legacy)
"""

import os
import sys
import subprocess
from pathlib import Path


SUPPORTED_SITES = ["tiktok.com", "youtube.com", "youtu.be", "instagram.com", "reels"]


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def download_video(url: str, output_dir: str, filename: str = "downloaded_video") -> str:
    """
    Download a video from TikTok / YouTube / Instagram using yt-dlp.

    Args:
        url        : video URL
        output_dir : directory to save the video
        filename   : output filename without extension

    Returns:
        Full path to downloaded .mp4 file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.mp4")

    print(f"[DOWNLOAD] Source : {url}")
    print(f"[DOWNLOAD] Saving : {output_path}")

    cmd = [
        "yt-dlp",
        # Best quality mp4, fallback to best available
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        # Output path
        "-o", output_path,
        # Skip if already downloaded
        "--no-overwrites",
        # Suppress progress bar clutter
        "--newline",
        url
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"[ERROR] yt-dlp failed. Make sure yt-dlp is installed:")
        print(f"        pip install yt-dlp")
        sys.exit(1)

    if not os.path.exists(output_path):
        # yt-dlp may have saved with a different name — find it
        mp4_files = list(Path(output_dir).glob(f"{filename}*.mp4"))
        if mp4_files:
            output_path = str(mp4_files[0])
        else:
            print("[ERROR] Download completed but output file not found.")
            sys.exit(1)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Downloaded: {os.path.basename(output_path)} ({size_mb:.1f} MB)")
    return output_path
