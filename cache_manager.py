"""
cache_manager.py
================
FfmpegTool Phase 3 -- Cache Intermediate Renders

Caches extracted frames to avoid re-processing the same video+config combination.

Cache key:
    sha256(abs_path + "|" + str(mtime) + "|" + config_sig)[:16]
    where config_sig = sha256(json.dumps(relevant_cfg, sort_keys=True))[:8]

Cache layout (relative to cache_dir, default: {TOOL_DIR}/cache/):
    {key}_meta.json   -- metadata (paths, timestamps, config_sig)
    {key}_raw/        -- copies of raw extracted frames
    {key}_unique/     -- copies of unique (filtered) frames

Usage:
    from cache_manager import CacheManager

    cm = CacheManager()
    frames = cm.get_cached_frames("/v/clip.mp4", cfg)
    if frames is None:
        frames = extract_and_filter(...)
        cm.store_frames("/v/clip.mp4", cfg, frames)
"""

import os
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Paths ──
TOOL_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(TOOL_DIR, "cache")

# Config keys that affect frame output — anything else is irrelevant to cache
_CACHE_CFG_KEYS = {
    "extraction": ("mode", "fps", "scene_threshold", "draft"),
    "filter":     ("blur_threshold", "similarity_threshold", "dedup_method", "phash_size"),
}


# ─────────────────────────────────────────────
# CacheManager
# ─────────────────────────────────────────────

class CacheManager:
    """
    Manages a disk-based cache of extracted video frames.

    Thread-safety note: reads are safe to call concurrently; writes (store_frames,
    invalidate, purge_all) should be serialised by the caller if multiple threads
    process the same video simultaneously (unlikely in practice).
    """

    def __init__(self, cache_dir: str = CACHE_DIR):
        self._cache_dir = cache_dir

    # ── Public API ──

    def get_cached_frames(self, video_path: str, cfg: dict) -> Optional[list[str]]:
        """
        Return cached unique-frame paths if a valid cache entry exists, else None.

        A cache entry is valid only if:
          - The meta file exists and is parseable
          - The video file mtime still matches the cached mtime
          - The config signature still matches
          - All frame files in the cached unique/ folder still exist on disk

        Args:
            video_path: Absolute path to the source video.
            cfg:        Config dict (from main.py apply_defaults).

        Returns:
            List of absolute frame paths (sorted), or None on cache miss.
        """
        key = self._cache_key(video_path, cfg)
        meta_path = os.path.join(self._cache_dir, f"{key}_meta.json")

        if not os.path.isfile(meta_path):
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        # Validate mtime
        try:
            current_mtime = os.path.getmtime(video_path)
        except OSError:
            return None

        if abs(meta.get("mtime", -1) - current_mtime) > 0.001:
            return None  # file was modified — cache stale

        # Validate config signature
        if meta.get("cfg_sig") != _config_sig(cfg):
            return None

        # Validate frames still exist
        frames = meta.get("unique_frames", [])
        if not frames:
            return None
        if not all(os.path.isfile(p) for p in frames):
            return None

        return sorted(frames)

    def store_frames(
        self,
        video_path: str,
        cfg: dict,
        unique_frame_paths: list[str],
        raw_frame_paths: Optional[list[str]] = None,
    ) -> str:
        """
        Store frames in the cache.

        Copies all frame files into cache/{key}_unique/ (and optionally _raw/)
        and writes a meta.json.

        Args:
            video_path:        Source video file.
            cfg:               Config dict.
            unique_frame_paths: Paths to unique (filtered) frames to cache.
            raw_frame_paths:    Optional raw frames to also cache.

        Returns:
            Cache key string.
        """
        key = self._cache_key(video_path, cfg)
        unique_dir = os.path.join(self._cache_dir, f"{key}_unique")
        os.makedirs(unique_dir, exist_ok=True)

        # Copy unique frames
        cached_unique: list[str] = []
        for src in sorted(unique_frame_paths):
            dst = os.path.join(unique_dir, os.path.basename(src))
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)
            cached_unique.append(dst)

        # Copy raw frames (optional)
        cached_raw: list[str] = []
        if raw_frame_paths:
            raw_dir = os.path.join(self._cache_dir, f"{key}_raw")
            os.makedirs(raw_dir, exist_ok=True)
            for src in sorted(raw_frame_paths):
                dst = os.path.join(raw_dir, os.path.basename(src))
                if not os.path.isfile(dst):
                    shutil.copy2(src, dst)
                cached_raw.append(dst)

        # Write meta
        try:
            mtime = os.path.getmtime(video_path)
        except OSError:
            mtime = 0.0

        meta = {
            "cache_key":    key,
            "video_path":   video_path,
            "mtime":        mtime,
            "cfg_sig":      _config_sig(cfg),
            "created_at":   _now(),
            "unique_frames": cached_unique,
            "raw_frames":   cached_raw,
        }
        meta_path = os.path.join(self._cache_dir, f"{key}_meta.json")
        _atomic_json_write(meta_path, meta)

        return key

    def invalidate(self, video_path: str) -> int:
        """
        Remove all cache entries for a specific video file.

        Returns the number of entries removed.
        """
        if not os.path.isdir(self._cache_dir):
            return 0

        removed = 0
        abs_path = os.path.abspath(video_path)

        for meta_file in Path(self._cache_dir).glob("*_meta.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if os.path.abspath(meta.get("video_path", "")) == abs_path:
                    key = meta.get("cache_key", meta_file.stem.replace("_meta", ""))
                    _remove_cache_entry(self._cache_dir, key)
                    removed += 1
            except (json.JSONDecodeError, OSError):
                continue

        return removed

    def purge_all(self) -> int:
        """
        Delete the entire cache directory.

        Returns the number of meta entries that were present before purge.
        """
        if not os.path.isdir(self._cache_dir):
            return 0

        count = len(list(Path(self._cache_dir).glob("*_meta.json")))
        shutil.rmtree(self._cache_dir, ignore_errors=True)
        return count

    def stats(self) -> dict:
        """
        Return cache statistics.

        Returns:
            {
                "entries":     int,   # number of cached videos
                "total_files": int,   # total frame files across all entries
                "total_bytes": int,   # total disk usage in bytes
                "cache_dir":   str,
            }
        """
        if not os.path.isdir(self._cache_dir):
            return {"entries": 0, "total_files": 0, "total_bytes": 0,
                    "cache_dir": self._cache_dir}

        entries = 0
        total_files = 0
        total_bytes = 0

        for item in Path(self._cache_dir).iterdir():
            if item.name.endswith("_meta.json"):
                entries += 1
                total_bytes += item.stat().st_size
            elif item.is_dir():
                for f in item.rglob("*"):
                    if f.is_file():
                        total_files += 1
                        total_bytes += f.stat().st_size

        return {
            "entries":     entries,
            "total_files": total_files,
            "total_bytes": total_bytes,
            "cache_dir":   self._cache_dir,
        }

    def _cache_key(self, video_path: str, cfg: dict) -> str:
        """Compute a deterministic 16-hex-char cache key."""
        abs_path = os.path.abspath(video_path)
        try:
            mtime = str(os.path.getmtime(abs_path))
        except OSError:
            mtime = "0"
        sig = _config_sig(cfg)
        raw = f"{abs_path}|{mtime}|{sig}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ─────────────────────────────────────────────
# Module-level singleton helpers
# ─────────────────────────────────────────────

_default_manager: Optional[CacheManager] = None


def get_default_manager() -> CacheManager:
    """Return (and lazily create) the module-level default CacheManager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = CacheManager()
    return _default_manager


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _config_sig(cfg: dict) -> str:
    """Compute an 8-hex-char signature for the cache-relevant config keys."""
    relevant: dict = {}
    for section, keys in _CACHE_CFG_KEYS.items():
        section_cfg = cfg.get(section, {})
        relevant[section] = {k: section_cfg.get(k) for k in keys}
    serialised = json.dumps(relevant, sort_keys=True)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:8]


def _remove_cache_entry(cache_dir: str, key: str):
    """Delete meta file and associated _raw/ and _unique/ directories for a key."""
    meta = os.path.join(cache_dir, f"{key}_meta.json")
    if os.path.isfile(meta):
        os.remove(meta)
    for suffix in ("_raw", "_unique"):
        d = os.path.join(cache_dir, f"{key}{suffix}")
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


def _atomic_json_write(path: str, data: dict):
    """Write JSON to path atomically via a temp file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
