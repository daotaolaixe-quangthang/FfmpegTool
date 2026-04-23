"""
downloader.py
=============
Download video from TikTok, YouTube, Instagram using yt-dlp.
Logic migrated and improved from: video_frame_extractor.py (legacy)
"""

import os
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

    # BUG-6 FIX: yt-dlp may return non-zero exit code even on success
    # (e.g. --no-overwrites returns 1 if the file already exists).
    # Check file existence FIRST before deciding to bail out.
    if not os.path.exists(output_path):
        # yt-dlp may have saved with a different name — find it
        mp4_files = sorted(Path(output_dir).glob(f"{filename}*.mp4"))
        if mp4_files:
            output_path = str(mp4_files[0])
        elif result.returncode != 0:
            # Only treat as failure when file is truly missing AND yt-dlp reported error
            # SEC-3 FIX: raise instead of sys.exit so callers can catch the error
            raise RuntimeError(
                f"yt-dlp failed (exit code {result.returncode}). "
                "Make sure yt-dlp is installed: pip install yt-dlp"
            )
        else:
            raise RuntimeError("yt-dlp reported success but output file not found.")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Downloaded: {os.path.basename(output_path)} ({size_mb:.1f} MB)")
    return output_path
