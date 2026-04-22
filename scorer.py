"""
scorer.py
=========
Aesthetic scoring for extracted frames.
Scores each frame on multiple visual quality dimensions,
then allows selecting the top-N best frames.

Scoring dimensions (weighted):
  1. Sharpness    (35%) — Laplacian variance
  2. Colorfulness (20%) — rg-yb color spread
  3. Brightness   (20%) — closeness to ideal mid-brightness
  4. Contrast     (15%) — standard deviation of grayscale
  5. Composition  (10%) — edge mass in center region (rule of thirds proxy)

Score range: 0.0 (worst) → 1.0 (best)
"""

import os
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────
# Individual metrics
# ─────────────────────────────────────────────

def score_sharpness(gray: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: 0 = blurry (<50), 1.0 = very sharp (>600)
    return float(min(lap_var / 600.0, 1.0))


def score_brightness(gray: np.ndarray) -> float:
    """
    Penalize too dark or too bright frames.
    Ideal brightness ~ 100-160 (out of 255).
    """
    mean = float(np.mean(gray))
    # Triangle: peaks at 128, falls to 0 at 0 and 255
    return 1.0 - abs(mean - 128.0) / 128.0


def score_contrast(gray: np.ndarray) -> float:
    """Standard deviation of grayscale — higher = more contrast."""
    std = float(np.std(gray))
    return min(std / 80.0, 1.0)


def score_colorfulness(bgr: np.ndarray) -> float:
    """
    Hasler & Susstrunk (2003) colorfulness metric.
    Higher = more vivid/saturated colors.
    """
    b = bgr[:, :, 0].astype(float)
    g = bgr[:, :, 1].astype(float)
    r = bgr[:, :, 2].astype(float)

    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    colorfulness = np.mean(rg) + np.mean(yb)
    return float(min(colorfulness / 80.0, 1.0))


def score_composition(gray: np.ndarray) -> float:
    """
    Rough composition score: fraction of edge pixels in the central
    two-thirds of the frame (rule-of-thirds proxy).
    """
    edges = cv2.Canny(gray, 80, 180)
    total_edges = float(np.sum(edges))
    if total_edges == 0:
        return 0.0

    h, w = edges.shape
    # Central region: middle third vertically and horizontally
    center = edges[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    center_edges = float(np.sum(center))
    return min(center_edges / (total_edges + 1e-6) * 1.5, 1.0)


# ─────────────────────────────────────────────
# Full frame scorer
# ─────────────────────────────────────────────

WEIGHTS = {
    "sharpness"   : 0.35,
    "colorfulness": 0.20,
    "brightness"  : 0.20,
    "contrast"    : 0.15,
    "composition" : 0.10,
}


def score_frame(image_path: str) -> dict:
    """
    Score a single frame on all aesthetic dimensions.

    Returns:
        dict with individual scores and weighted total (all 0.0–1.0)
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        return {"total": 0.0, "error": "unreadable"}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    raw = {
        "sharpness"   : score_sharpness(gray),
        "colorfulness": score_colorfulness(bgr),
        "brightness"  : score_brightness(gray),
        "contrast"    : score_contrast(gray),
        "composition" : score_composition(gray),
    }

    total = sum(raw[k] * WEIGHTS[k] for k in WEIGHTS)

    return {
        "file"        : os.path.basename(image_path),
        "path"        : image_path,
        "sharpness"   : round(raw["sharpness"], 3),
        "colorfulness": round(raw["colorfulness"], 3),
        "brightness"  : round(raw["brightness"], 3),
        "contrast"    : round(raw["contrast"], 3),
        "composition" : round(raw["composition"], 3),
        "total"       : round(total, 3),
    }


# ─────────────────────────────────────────────
# Batch scorer + top-N selector
# ─────────────────────────────────────────────

def score_all_frames(frame_paths: list[str]) -> list[dict]:
    """Score every frame and return sorted list (best first)."""
    if not frame_paths:
        print("[SCORE] No frames to score.")
        return []

    print(f"\n[SCORE] Scoring {len(frame_paths)} frames on aesthetic quality...")
    results = []
    for path in tqdm(frame_paths, desc="  Scoring", ncols=70):
        results.append(score_frame(path))

    results.sort(key=lambda x: x["total"], reverse=True)
    print(f"[SCORE] Done. Best score: {results[0]['total']:.3f} | "
          f"Worst: {results[-1]['total']:.3f}")
    return results


def select_top_n(
    scored_frames: list[dict],
    output_dir: str,
    top_n: int
) -> list[str]:
    """
    Copy top-N highest-scoring frames to output_dir/top_frames/.
    Returns list of destination paths.
    """
    top_dir = os.path.join(output_dir, "top_frames")
    os.makedirs(top_dir, exist_ok=True)

    selected = scored_frames[:top_n]
    dest_paths = []

    for rank, item in enumerate(selected, 1):
        src = item["path"]
        ext = Path(src).suffix
        dest = os.path.join(top_dir, f"top_{rank:03d}_score{item['total']:.2f}{ext}")
        shutil.copy2(src, dest)
        dest_paths.append(dest)

    print(f"[SCORE] Top {top_n} frames saved to: {top_dir}")
    return dest_paths


def save_score_report(scored_frames: list[dict], output_dir: str) -> str:
    """Save full scoring report as JSON."""
    report_path = os.path.join(output_dir, "score_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(scored_frames, f, indent=2)
    return report_path


def print_top_scores(scored_frames: list[dict], n: int = 10):
    """Print top-N scores as a table."""
    sep = "-" * 72
    print(f"\n[SCORE] Top {min(n, len(scored_frames))} frames by aesthetic score:")
    print(sep)
    print(f"  {'Rank':<5} {'File':<25} {'Total':>6} {'Sharp':>6} {'Color':>6} {'Bright':>7} {'Comp':>6}")
    print(sep)
    for i, r in enumerate(scored_frames[:n], 1):
        print(
            f"  {i:<5} {r['file']:<25} {r['total']:>6.3f} "
            f"{r['sharpness']:>6.3f} {r['colorfulness']:>6.3f} "
            f"{r['brightness']:>7.3f} {r['composition']:>6.3f}"
        )
    print(sep)
