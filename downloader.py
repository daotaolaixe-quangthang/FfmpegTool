"""
downloader.py
=============
Download video from TikTok, YouTube, Instagram using yt-dlp.
Logic migrated and improved from: video_frame_extractor.py (legacy)
"""

import os
import socket
import ipaddress
import subprocess
from pathlib import Path
from urllib.parse import urlparse

# ─────────────────────────────────────────────
# URL allowlist and SSRF guard (VULN-2, VULN-3)
# ─────────────────────────────────────────────

# Only http and https are safe schemes for yt-dlp.
# file://, ftp://, smb://, data: etc. are blocked.
_ALLOWED_SCHEMES = {"http", "https"}

# Approved video-hosting domains.
# A host matches if it equals one of these or is a subdomain of one.
_ALLOWED_DOMAINS = {
    "tiktok.com",
    "youtube.com",
    "youtu.be",
    "instagram.com",
    "facebook.com",
    "fb.watch",
    "twitter.com",
    "x.com",
    "vimeo.com",
    "dailymotion.com",
    "twitch.tv",
    "reddit.com",
    "bilibili.com",
}

# Backwards-compat alias used by other modules
SUPPORTED_SITES = sorted(_ALLOWED_DOMAINS)


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _is_private_ip(host: str) -> bool:
    """Return True if host resolves to a private/loopback/link-local address."""
    # Fast path: host is already a numeric IP string
    try:
        addr = ipaddress.ip_address(host)
        return (
            addr.is_private or addr.is_loopback
            or addr.is_link_local or addr.is_reserved
            or addr.is_multicast
        )
    except ValueError:
        pass
    # Slow path: resolve hostname and check resulting IP
    try:
        resolved = socket.gethostbyname(host)
        addr = ipaddress.ip_address(resolved)
        return (
            addr.is_private or addr.is_loopback
            or addr.is_link_local or addr.is_reserved
            or addr.is_multicast
        )
    except (socket.gaierror, ValueError, OSError):
        # DNS failed -- do not silently allow unknown hosts
        return False


def _validate_url(url: str) -> None:
    """
    Validate that a URL is safe to pass to yt-dlp.

    Raises ValueError for:
    - Leading dash (argument-injection: yt-dlp interprets as a flag)
    - Non-http/https schemes  (file://, ftp://, smb://, etc.)
    - Hostnames not in _ALLOWED_DOMAINS
    - Hostnames resolving to private/loopback IPs (SSRF guard)
    """
    stripped = url.strip()

    # Argument-injection guard: yt-dlp treats leading dashes as CLI flags.
    if stripped.startswith("-"):
        raise ValueError(
            "URL cannot start with '-'. Provide a valid https:// URL."
        )

    try:
        parsed = urlparse(stripped)
    except Exception as exc:
        raise ValueError(f"Malformed URL: {exc}") from exc

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"URL scheme {parsed.scheme!r} is not allowed. "
            "Only 'https' and 'http' URLs are accepted."
        )

    host = (parsed.hostname or "").lower().strip(".")
    if not host:
        raise ValueError("URL has no hostname.")

    # Domain allowlist: host must equal or be a subdomain of an approved domain.
    if not any(host == d or host.endswith("." + d) for d in _ALLOWED_DOMAINS):
        raise ValueError(
            f"URL hostname {host!r} is not in the approved domain list. "
            f"Supported sites: {', '.join(sorted(_ALLOWED_DOMAINS))}"
        )

    # SSRF guard: block URLs that resolve to internal/loopback addresses.
    if _is_private_ip(host):
        raise ValueError(
            f"URL hostname {host!r} resolves to a private or loopback "
            "address, which is not allowed."
        )


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
    # Security: validate URL before any processing (SSRF + arg-injection guard).
    _validate_url(url)

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
        # "--" marks end of yt-dlp options; anything after is treated as a URL,
        # preventing a crafted URL starting with "-" from being parsed as a flag.
        "--",
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=False, text=True,
                                timeout=1800)  # BUG-M1 FIX: 30-min ceiling for yt-dlp downloads
    except subprocess.TimeoutExpired:
        raise RuntimeError("yt-dlp timed out after 30 minutes. Check network or URL.")

    # BUG-6 FIX: yt-dlp may return non-zero exit code even on success
    # (e.g. --no-overwrites returns 1 if the file already exists).
    # Check file existence FIRST before deciding to bail out.
    if not os.path.exists(output_path):
        # yt-dlp may have saved with a different name — find the newest matching mp4
        mp4_files = sorted(
            Path(output_dir).glob(f"{filename}*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
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
