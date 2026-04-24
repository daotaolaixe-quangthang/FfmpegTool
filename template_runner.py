"""
template_runner.py
==================
FfmpegTool Phase 3 -- Template CSV/JSON Batch Runner

Reads a CSV or JSON template file containing a list of video jobs, validates
each row, then enqueues valid jobs into QueueManager.

CSV format:
    video_src,output,preset,campaign,lang,ratio,top_n
    G:/vids/clip1.mp4,G:/out,tiktok_pack,summer_sale,vi,9x16,20

JSON format (list of dicts with same keys):
    [{"video_src": "clip1.mp4", "output": "G:/out", "preset": "tiktok_pack"}]

Usage:
    from template_runner import load_template_csv, load_template_json, run_template
    from queue_manager import QueueManager

    qm = QueueManager()
    rows = load_template_csv("campaign.csv")
    result = run_template(rows, qm, base_output="G:/Output")

CLI:
    python main.py --template campaign.csv --output G:/Output
"""

import os
import csv
import json
from pathlib import Path
from typing import Optional

# ── Required and optional fields ──
REQUIRED_FIELDS = {"video_src", "output"}
OPTIONAL_FIELDS = {"preset", "campaign", "lang", "ratio", "top_n"}
ALL_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS


# ─────────────────────────────────────────────
# Row validation
# ─────────────────────────────────────────────

def validate_row(row: dict, known_presets: list[str] | None = None) -> list[str]:
    """
    Validate a single template row dict.

    Returns a list of error strings (empty list = valid).

    Args:
        row:           Dict with at minimum 'video_src' and 'output' keys.
        known_presets: List of valid preset names; None = skip preset check.
    """
    errors = []

    # ── Required: video_src ──
    video_src = (row.get("video_src") or "").strip()
    if not video_src:
        errors.append("video_src is required")
    elif not os.path.isfile(video_src):
        errors.append(f"video_src not found: {video_src}")

    # ── Required: output ──
    output = (row.get("output") or "").strip()
    if not output:
        errors.append("output is required")

    # ── Optional: preset ──
    preset = (row.get("preset") or "").strip()
    if preset and known_presets is not None:
        if preset not in known_presets:
            errors.append(f"unknown preset: {preset!r} (known: {', '.join(known_presets)})")

    # ── Optional: top_n ──
    top_n_raw = (str(row.get("top_n") or "")).strip()
    if top_n_raw:
        try:
            top_n = int(top_n_raw)
            if top_n <= 0:
                errors.append(f"top_n must be > 0, got {top_n}")
        except ValueError:
            errors.append(f"top_n must be an integer, got {top_n_raw!r}")

    return errors


# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────

def load_template_csv(path: str) -> list[dict]:
    """
    Load a CSV template file and return a list of row dicts.

    The CSV must have a header row. Field names are case-insensitive and
    stripped of whitespace.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if the CSV has no recognised columns.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Template CSV not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Template CSV appears to be empty")

        # Normalise header: lowercase + strip
        normalised = {k: k.lower().strip() for k in reader.fieldnames}
        recognized = {v for v in normalised.values() if v in ALL_FIELDS}
        if not recognized:
            raise ValueError(
                f"Template CSV has no recognised columns. "
                f"Expected at least one of: {sorted(ALL_FIELDS)}"
            )

        for raw_row in reader:
            row = {}
            for orig_key, val in raw_row.items():
                norm_key = normalised.get(orig_key, orig_key.lower().strip())
                row[norm_key] = (val or "").strip()
            rows.append(row)

    return rows


def load_template_json(source: str) -> list[dict]:
    """
    Load a JSON template from a file path or a raw JSON string.

    Returns a list of row dicts (same schema as CSV rows).

    Raises:
        FileNotFoundError: if source looks like a file path but doesn't exist.
        ValueError: if the JSON is not a list of objects.
    """
    # Heuristic: if it doesn't look like inline JSON, treat as path
    stripped = source.strip()
    if not stripped.startswith("[") and not stripped.startswith("{"):
        # Treat as file path
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Template JSON not found: {source}")
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(stripped)

    if not isinstance(data, list):
        raise ValueError("JSON template must be a list of job objects")

    rows = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Each JSON item must be a dict, got: {type(item).__name__}")
        # Normalise keys
        row = {k.lower().strip(): v for k, v in item.items()}
        rows.append(row)

    return rows


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

def run_template(
    rows: list[dict],
    queue_mgr,
    base_output: Optional[str] = None,
    known_presets: Optional[list[str]] = None,
    dry_run: bool = False,
) -> dict:
    """
    Validate rows and enqueue valid jobs into QueueManager.

    Args:
        rows:          List of row dicts (from load_template_csv/json).
        queue_mgr:     QueueManager instance.
        base_output:   Override output folder for all rows that lack one.
        known_presets: Valid preset names for validation; None = skip preset check.
        dry_run:       If True, validate only — do NOT enqueue anything.

    Returns:
        {
            "queued":   int,          # number of rows successfully enqueued
            "skipped":  int,          # number of rows with validation errors
            "errors":   list[dict],   # [{row_index, video_src, errors: [str]}]
            "item_ids": list[str],    # queue item IDs for enqueued jobs
        }
    """
    queued = 0
    skipped = 0
    error_list = []
    item_ids = []

    for i, row in enumerate(rows):
        row = dict(row)  # work on a copy

        # Apply base_output as fallback
        if base_output and not (row.get("output") or "").strip():
            row["output"] = base_output

        errs = validate_row(row, known_presets=known_presets)
        if errs:
            skipped += 1
            error_list.append({
                "row_index": i,
                "video_src": (row.get("video_src") or "").strip(),
                "errors":    errs,
            })
            continue

        if not dry_run:
            # Build cfg_overrides from template metadata fields
            cfg_overrides = _build_cfg_overrides(row)
            item_id = queue_mgr.add(
                input_path=row["video_src"].strip(),
                output_path=row["output"].strip(),
                cfg_overrides=cfg_overrides,
            )
            item_ids.append(item_id)

        queued += 1

    return {
        "queued":   queued,
        "skipped":  skipped,
        "errors":   error_list,
        "item_ids": item_ids,
    }


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _build_cfg_overrides(row: dict) -> dict:
    """
    Convert template row fields into a cfg_overrides dict compatible with QueueManager.

    Supported fields -> config section:
        preset    -> stored at top-level key "preset"
        campaign  -> output.campaign
        lang      -> output.lang
        ratio     -> output.ratio
        top_n     -> scorer.top_n + scorer.enabled=True
    """
    overrides: dict = {}

    preset = (row.get("preset") or "").strip()
    if preset:
        overrides["preset"] = preset

    # Output metadata fields
    output_overrides: dict = {}
    for field in ("campaign", "lang", "ratio"):
        val = (row.get(field) or "").strip()
        if val:
            output_overrides[field] = val
    if output_overrides:
        overrides["output"] = output_overrides

    # Scorer
    top_n_raw = (str(row.get("top_n") or "")).strip()
    if top_n_raw:
        try:
            top_n = int(top_n_raw)
            if top_n > 0:
                overrides["scorer"] = {"enabled": True, "top_n": top_n}
        except ValueError:
            pass

    return overrides


def detect_template_format(path: str) -> str:
    """Return 'csv' or 'json' based on file extension. Raises ValueError for unknown."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext in (".json", ".jsonl"):
        return "json"
    raise ValueError(f"Cannot detect template format from extension: {ext!r}. Use .csv or .json")


def load_template(path: str) -> list[dict]:
    """Auto-detect format and load a template file."""
    fmt = detect_template_format(path)
    if fmt == "csv":
        return load_template_csv(path)
    return load_template_json(path)
