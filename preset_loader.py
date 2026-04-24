"""
preset_loader.py
================
Load, list, and validate processing presets.

Presets are JSON files in the presets/ subfolder next to this script.
Each preset is a partial config that overrides the base config.json values.

Usage in main.py:
    from preset_loader import apply_preset, list_presets
    cfg = apply_preset(cfg, "tiktok_pack")

Preset file format (presets/my_preset.json):
    {
      "name": "My Preset",
      "description": "Short description shown at startup",
      "extraction": { ... },   # any keys from config.json extraction section
      "filter":     { ... },
      "scorer":     { ... },
      "output":     { ... },
      "hardware":   { ... }
    }
"""

import os
import json
from pathlib import Path

# Presets directory sits next to this module file
PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

# Sections that are allowed to be deep-merged from a preset.
# BUG-L4 NOTE: "normalize" and "batch" are intentionally excluded — presets
# are not expected to control normalization or batch probe settings. Add those
# sections here only when Phase 4 presets require it.
MERGEABLE_SECTIONS = ("extraction", "filter", "scorer", "output", "hardware")


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def list_presets() -> list[dict]:
    """
    Return a list of available presets with metadata.

    Each entry: {"file": stem, "name": str, "description": str}
    Returns [] if presets/ folder doesn't exist or is empty.
    """
    if not os.path.isdir(PRESETS_DIR):
        return []

    presets = []
    for path in sorted(Path(PRESETS_DIR).glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            presets.append({
                "file":        path.stem,
                "name":        data.get("name", path.stem),
                "description": data.get("description", ""),
            })
        except Exception as exc:
            print(f"[PRESET] Warning: skipping malformed preset file '{path.name}': {exc}")
            continue
    return presets


def load_preset(preset_name: str) -> dict:
    """
    Load a preset by filename stem (without .json extension).

    Args:
        preset_name: e.g. "tiktok_pack"

    Returns:
        Raw preset dict as loaded from JSON.

    Raises:
        FileNotFoundError: If preset file doesn't exist.
        ValueError: If preset file has invalid JSON.
    """
    preset_path = os.path.join(PRESETS_DIR, f"{preset_name}.json")
    if not os.path.exists(preset_path):
        available = [p["file"] for p in list_presets()]
        hint = f"  Available: {', '.join(available)}" if available else "  (No presets found)"
        raise FileNotFoundError(
            f"Preset not found: '{preset_name}'\n"
            f"  Looked in: {PRESETS_DIR}\n"
            f"{hint}"
        )
    try:
        with open(preset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Preset '{preset_name}' has invalid JSON: {e}")


def _merge_preset_into_cfg(base_cfg: dict, preset: dict, preset_name: str) -> dict:
    """Merge an already-loaded preset dict into base_cfg in place."""
    display_name = preset.get("name", preset_name)
    description  = preset.get("description", "")

    print(f"[PRESET] > {display_name}")
    if description:
        print(f"[PRESET]   {description}")

    applied_sections = []
    for section in MERGEABLE_SECTIONS:
        if section in preset:
            if section not in base_cfg:
                base_cfg[section] = {}
            override_keys = {k: v for k, v in preset[section].items() if not k.startswith("_")}
            base_cfg[section].update(override_keys)
            applied_sections.append(section)

    if applied_sections:
        print(f"[PRESET]   Applied sections: {', '.join(applied_sections)}")

    return base_cfg


def apply_preset(base_cfg: dict, preset_name: str) -> dict:
    """
    Deep-merge a preset onto base_cfg.

    Preset values override base_cfg keys within each section.
    Keys present in base_cfg but absent in preset are preserved.
    Metadata keys (name, description) are not applied to cfg.

    Args:
        base_cfg:    Config dict (already passed through apply_defaults)
        preset_name: Preset filename stem, e.g. "tiktok_pack"

    Returns:
        Updated base_cfg (mutated in-place and returned).
    """
    try:
        preset = load_preset(preset_name)
    except (FileNotFoundError, ValueError) as e:
        print(f"[PRESET] ! {e}")
        print("[PRESET] Continuing with base config.json settings.")
        return base_cfg

    return _merge_preset_into_cfg(base_cfg, preset, preset_name)


def apply_preset_strict(base_cfg: dict, preset_name: str) -> dict:
    """Apply a preset, but raise if the preset cannot be loaded."""
    preset = load_preset(preset_name)
    return _merge_preset_into_cfg(base_cfg, preset, preset_name)


def print_presets_table() -> None:
    """Print a formatted table of all available presets to stdout."""
    presets = list_presets()
    if not presets:
        print("[PRESET] No presets found in:", PRESETS_DIR)
        return

    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  {'PRESET NAME':<22} {'FILE':<22} DESCRIPTION")
    print(sep)
    for p in presets:
        desc = (p["description"][:28] + "...") if len(p["description"]) > 29 else p["description"]
        print(f"  {p['name']:<22} {p['file']:<22} {desc}")
    print(f"{sep}\n")
