"""
filters.py
==========
Two-stage frame filtering pipeline:

  Stage 1 — Blur Filter
    Removes frames that are blurry or have motion blur.
    Uses Laplacian variance: low variance = blurry image.

  Stage 2 — Duplicate Filter
    Removes frames too similar to any previously kept frame.
    Two methods available:
      phash (default) — Perceptual Hash. Fast, ~100ms/frame. Good for most use cases.
      ssim            — Structural Similarity Index. Slower but more accurate.
                        Migrated from: video_frame_extractor.py (legacy)
    A frame is kept if similarity < similarity_threshold with ALL kept frames.
    Default threshold 0.70 → keep frames that are 30%+ different.
"""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm


# ─────────────────────────────────────────────
# STAGE 1: BLUR DETECTION
# ─────────────────────────────────────────────

def compute_blur_score(image_path: str) -> float:
    """
    Compute sharpness score using Laplacian variance.
    Higher = sharper. Lower = blurrier.
    Typical values:
      < 50  : very blurry
      50-100: somewhat blurry
      > 100 : acceptably sharp
      > 300 : very sharp
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def filter_blurry(frame_paths: list[str], blur_threshold: float) -> tuple[list[str], list[dict]]:
    """
    Remove blurry frames.

    Returns:
        sharp_frames : list of paths that passed blur check
        removed      : list of {path, blur_score} for removed frames
    """
    sharp = []
    removed = []

    print(f"\n[BLUR] Filtering blurry frames (threshold={blur_threshold})...")

    for path in tqdm(frame_paths, desc="  Blur check", ncols=70):
        score = compute_blur_score(path)
        if score >= blur_threshold:
            sharp.append(path)
        else:
            removed.append({"path": path, "blur_score": round(score, 1)})

    print(f"[BLUR] Removed {len(removed)} blurry frames | Kept {len(sharp)}")
    return sharp, removed


# ─────────────────────────────────────────────
# STAGE 2: DUPLICATE DETECTION
# ─────────────────────────────────────────────

def compute_phash(image_path: str, hash_size: int = 16) -> imagehash.ImageHash:
    """Compute perceptual hash of an image."""
    with Image.open(image_path) as img:
        return imagehash.phash(img, hash_size=hash_size)


def hamming_to_similarity(distance: int, max_distance: int = None,
                          hash_size: int = 16) -> float:
    """Convert Hamming distance to similarity ratio (0.0 = different, 1.0 = identical).

    Args:
        max_distance: Total bits in the hash. If None, derived from hash_size.
                      BUG-M6 FIX: was hardcoded to 256 (only correct for
                      hash_size=16). Now defaults to hash_size**2 so direct
                      callers with non-default hash sizes get correct values.
    """
    if max_distance is None:
        max_distance = hash_size * hash_size  # e.g. 16*16=256 for default size
    return 1.0 - (distance / max_distance)


def filter_duplicates(
    frame_paths: list[str],
    similarity_threshold: float,
    phash_size: int
) -> tuple[list[str], list[dict]]:
    """
    Remove frames too similar to any already-kept frame.

    A frame is kept if its similarity to ALL kept frames is < similarity_threshold.
    Example: threshold=0.70 → keep frame only if it is 30%+ different from all kept frames.

    Returns:
        unique_frames : paths of frames that passed duplicate check
        removed       : list of {path, similarity} for removed frames
    """
    unique = []
    kept_hashes = []
    removed = []

    max_dist = phash_size * phash_size  # bits in hash

    print(f"\n[DEDUP] Filtering duplicates (similarity_threshold={similarity_threshold:.0%})...")
    print(f"        Keep frame only if <{similarity_threshold:.0%} similar to ALL kept frames")

    for path in tqdm(frame_paths, desc="  Dedup check", ncols=70):
        try:
            current_hash = compute_phash(path, hash_size=phash_size)
        except Exception:
            removed.append({"path": path, "reason": "hash_error"})
            continue

        is_duplicate = False
        max_similarity_seen = 0.0

        for kept_hash in kept_hashes:
            distance = current_hash - kept_hash
            sim = hamming_to_similarity(distance, max_dist)
            if sim > max_similarity_seen:
                max_similarity_seen = sim
            if sim >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(path)
            kept_hashes.append(current_hash)
        else:
            removed.append({
                "path": path,
                "reason": "duplicate",
                "similarity": round(max_similarity_seen, 3)
            })

    print(f"[DEDUP] Removed {len(removed)} duplicates | Kept {len(unique)}")
    return unique, removed


def filter_duplicates_ssim(
    frame_paths: list[str],
    similarity_threshold: float,
    resize_to: tuple = (256, 256)
) -> tuple[list[str], list[dict]]:
    """
    SSIM-based duplicate filter. More perceptually accurate than pHash,
    but slower (~5-10x). Use for small frame sets (<500 frames) or when
    pHash gives too many false positives.

    Migrated from: video_frame_extractor.py (legacy), method='ssim' branch.
    Requires: pip install scikit-image

    Args:
        frame_paths          : list of frame paths to filter
        similarity_threshold : SSIM score >= this = duplicate (0.0-1.0)
        resize_to            : resize all frames before comparison (for speed)

    Returns:
        unique_frames : paths of frames that passed
        removed       : list of {path, ssim_score} for removed frames
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        raise ImportError("scikit-image required for SSIM: pip install scikit-image")

    unique = []
    kept_imgs = []
    removed = []

    print(f"\n[DEDUP-SSIM] Filtering duplicates (threshold={similarity_threshold:.0%})...")
    print(f"             Note: SSIM is slower than pHash - suitable for <500 frames")

    for path in tqdm(frame_paths, desc="  SSIM check", ncols=70):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            removed.append({"path": path, "reason": "unreadable"})
            continue
        img = cv2.resize(img, resize_to)

        is_duplicate = False
        max_ssim = 0.0

        for kept_img in kept_imgs:
            score = ssim_fn(img, kept_img, data_range=255)
            if score > max_ssim:
                max_ssim = score
            if score >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(path)
            kept_imgs.append(img)
        else:
            removed.append({
                "path": path,
                "reason": "duplicate",
                "similarity": round(max_ssim, 3)   # same key as phash for consistent report
            })

    print(f"[DEDUP-SSIM] Removed {len(removed)} duplicates | Kept {len(unique)}")
    return unique, removed


# ─────────────────────────────────────────────
# MAIN FILTER PIPELINE
# ─────────────────────────────────────────────

def run_filter_pipeline(
    raw_frames: list[str],
    output_dir: str,
    cfg: dict
) -> dict:
    """
    Run Stage 1 (blur) then Stage 2 (duplicate) on raw frames.
    Copy surviving frames to output_dir/unique_frames/.

    Config keys:
      blur_threshold        : Laplacian variance threshold (default 80)
      similarity_threshold  : max similarity to keep frame (default 0.70)
      dedup_method          : 'phash' (default, fast) or 'ssim' (slower, accurate)
      phash_size            : hash grid size for phash (default 16)

    Returns a stats dict for reporting.
    """
    blur_threshold    = cfg.get("blur_threshold", 80.0)
    sim_threshold     = cfg.get("similarity_threshold", 0.70)
    dedup_method      = cfg.get("dedup_method", "phash")
    phash_size        = cfg.get("phash_size", 16)

    total_raw = len(raw_frames)

    # Stage 1: Blur
    sharp_frames, blur_removed = filter_blurry(raw_frames, blur_threshold)

    # Stage 2: Duplicates (method selectable)
    if dedup_method == "ssim":
        unique_frames, dup_removed = filter_duplicates_ssim(sharp_frames, sim_threshold)
    else:
        unique_frames, dup_removed = filter_duplicates(sharp_frames, sim_threshold, phash_size)

    # Copy unique frames to output folder
    unique_dir = os.path.join(output_dir, "unique_frames")
    os.makedirs(unique_dir, exist_ok=True)

    final_paths = []
    for i, src in enumerate(unique_frames):
        ext = Path(src).suffix
        dest = os.path.join(unique_dir, f"unique_{i:04d}{ext}")
        shutil.copy2(src, dest)
        final_paths.append(dest)

    stats = {
        "total_raw"         : total_raw,
        "after_blur_filter" : len(sharp_frames),
        "after_dedup_filter": len(unique_frames),
        "final_count"       : len(final_paths),
        "removed_blur"      : len(blur_removed),
        "removed_duplicate" : len(dup_removed),
        "blur_removed_list" : [os.path.basename(r["path"]) for r in blur_removed],
        "dup_removed_list"  : [
            {"file": os.path.basename(r["path"]), "similarity": r.get("similarity")}
            for r in dup_removed
        ],
        "output_dir"        : unique_dir,
        "final_paths"       : final_paths,
    }

    return stats
