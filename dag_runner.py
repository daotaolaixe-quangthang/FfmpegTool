"""
dag_runner.py
=============
FfmpegTool Phase 4 -- DAG Job Graph Workflow.

Run one source video through multiple preset branches in one command.

dag_spec.json format:
    {
        "source": "clip.mp4",
        "output": "/output/dir",   (optional, falls back to CLI --output)
        "branches": [
            {"preset": "tiktok_pack"},
            {"preset": "youtube_shorts", "output": "/custom/out"},
            {"preset": "draft_preview"}
        ]
    }

CLI usage:
    python main.py --dag dag_spec.json --output /default/output

API usage:
    POST /api/dag/run
    Body: same as dag_spec.json  OR  {"spec_file": "/path/to/dag_spec.json"}

Design:
  - Simple dict-based DAG -- no external graph library needed.
  - Each branch is a node: (source, preset, output_dir).
  - Executed sequentially by default; parallel execution optional via workers.
  - 1 branch failure does NOT abort sibling branches.
  - Results dict matches structure returned by run_parallel_batch().
"""

import os
import json
import copy
from pathlib import Path


# ─────────────────────────────────────────────
# Spec loader
# ─────────────────────────────────────────────

def load_dag_spec(source: str) -> dict:
    """
    Load a DAG spec from a JSON file path or a raw JSON string.

    Returns a validated spec dict:
        {"source": str, "output": str|None, "branches": list[dict]}

    Raises:
        FileNotFoundError  -- if source is a path that doesn't exist
        ValueError         -- if the spec is structurally invalid
    """
    # Try to parse as a file path first
    if isinstance(source, str) and not source.strip().startswith("{"):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"DAG spec file not found: {source}")
        with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    else:
        # Raw JSON string
        try:
            spec = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid DAG spec JSON: {exc}") from exc

    return validate_dag_spec(spec)


def validate_dag_spec(spec: dict) -> dict:
    """
    Validate a dag spec dict.  Returns the spec with defaults applied.

    Raises ValueError for structural errors.
    """
    if not isinstance(spec, dict):
        raise ValueError("DAG spec must be a JSON object")

    source = (spec.get("source") or "").strip()
    if not source:
        raise ValueError("DAG spec missing required field: 'source'")

    branches = spec.get("branches")
    if not isinstance(branches, list) or not branches:
        raise ValueError("DAG spec 'branches' must be a non-empty list")

    for i, branch in enumerate(branches):
        if not isinstance(branch, dict):
            raise ValueError(f"DAG branch[{i}] must be a JSON object")
        if not branch.get("preset", "").strip():
            raise ValueError(f"DAG branch[{i}] missing required field: 'preset'")

    return {
        "source":   source,
        "output":   (spec.get("output") or "").strip() or None,
        "branches": branches,
    }


# ─────────────────────────────────────────────
# DAG execution
# ─────────────────────────────────────────────

def run_dag(
    spec: dict,
    cfg: dict,
    default_output: str = "",
    workers: int = 1,
) -> dict:
    """
    Execute all branches of a DAG spec.

    Args:
        spec:           Validated dag spec dict (from load_dag_spec / validate_dag_spec).
        cfg:            Base config dict (from main.py apply_defaults).
        default_output: Fallback output dir if not in spec and not in branch.
        workers:        Parallel workers for branch execution (default: 1=sequential).

    Returns:
        {
            "source":   str,
            "branches": [
                {
                    "preset":  str,
                    "output":  str,
                    "status":  "success"|"skipped"|"error",
                    "stats":   dict|None,
                    "error":   str|None,
                }
            ],
            "success": N,
            "skipped": N,
            "failed":  N,
        }
    """
    source     = spec["source"]
    spec_output = spec.get("output") or default_output or ""
    branches   = spec["branches"]

    # Validate source
    if not os.path.isfile(source):
        raise FileNotFoundError(f"DAG source file not found: {source}")

    print(f"\n[DAG] Source : {source}")
    print(f"[DAG] Branches: {len(branches)}")

    # Build branch result list
    branch_results = []
    branch_video_map = []  # (result_idx, video_path, output_dir, preset, branch_cfg)

    for i, branch in enumerate(branches):
        preset     = branch.get("preset", "").strip()
        branch_out = (branch.get("output") or "").strip() or spec_output or ""

        # Each branch gets a sub-directory indexed by position + preset name
        # to avoid output collision when the same preset appears multiple times.
        if branch_out:
            effective_out = os.path.join(branch_out, f"_dag_{i}_{preset}")
        else:
            raise ValueError(
                f"DAG branch[{i}] (preset={preset}): no output directory specified. "
                "Provide 'output' in spec or pass --output on CLI."
            )

        # Deep-copy + apply preset
        branch_cfg = copy.deepcopy(cfg)
        result_idx = len(branch_results)  # capture index BEFORE appending
        try:
            from preset_loader import apply_preset  # noqa: PLC0415
            branch_cfg = apply_preset(branch_cfg, preset)
        except Exception as exc:
            branch_results.append({
                "preset":  preset,
                "output":  effective_out,
                "status":  "error",
                "stats":   None,
                "error":   f"Preset load failed: {exc}",
            })
            continue

        branch_video_map.append((result_idx, source, effective_out, preset, branch_cfg))
        branch_results.append({
            "preset":  preset,
            "output":  effective_out,
            "status":  "pending",   # placeholder
            "stats":   None,
            "error":   None,
        })

    if workers <= 1:
        # Sequential
        _run_dag_sequential(branch_video_map, branch_results)
    else:
        # Parallel (reuse parallel_runner)
        _run_dag_parallel(branch_video_map, branch_results, workers)

    success = sum(1 for r in branch_results if r["status"] == "success")
    skipped = sum(1 for r in branch_results if r["status"] == "skipped")
    failed  = sum(1 for r in branch_results if r["status"] == "error")

    print(f"\n[DAG] Done. Success: {success} | Skipped: {skipped} | Failed: {failed}")

    return {
        "source":   source,
        "branches": branch_results,
        "success":  success,
        "skipped":  skipped,
        "failed":   failed,
    }


def _run_dag_sequential(branch_video_map: list, branch_results: list):
    """Execute DAG branches sequentially, in-place update branch_results."""
    from main import process_video  # noqa: PLC0415

    for branch_num, (result_idx, source, effective_out, preset, branch_cfg) in enumerate(branch_video_map, 1):
        print(f"\n[DAG] Branch '{preset}' -> {effective_out}")
        try:
            stats = process_video(source, effective_out, branch_cfg, batch_index=branch_num)
            if stats:
                branch_results[result_idx]["status"] = "success"
                branch_results[result_idx]["stats"]  = stats
            else:
                branch_results[result_idx]["status"] = "skipped"
                branch_results[result_idx]["error"]  = "No frames extracted"
        except Exception as exc:
            branch_results[result_idx]["status"] = "error"
            branch_results[result_idx]["error"]  = str(exc)
            print(f"[DAG] Branch '{preset}' failed: {exc}")


def _run_dag_parallel(branch_video_map: list, branch_results: list, workers: int):
    """Execute DAG branches in parallel, in-place update branch_results."""
    from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415
    from parallel_runner import _worker, resolve_max_workers  # noqa: PLC0415

    effective_workers = resolve_max_workers(workers)
    futures_idx = {}   # future -> index in branch_results

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        for branch_num, (result_idx, source, effective_out, preset, branch_cfg) in enumerate(branch_video_map, 1):
            future = executor.submit(_worker, source, effective_out, branch_cfg, branch_num)
            futures_idx[future] = (result_idx, preset)

        for future in as_completed(futures_idx):
            idx, preset = futures_idx[future]
            try:
                r       = future.result()
                stats   = r.get("stats")
                error   = r.get("error")
                if error:
                    branch_results[idx]["status"] = "error"
                    branch_results[idx]["error"]  = error
                elif not stats:
                    branch_results[idx]["status"] = "skipped"
                    branch_results[idx]["error"]  = "No frames extracted"
                else:
                    branch_results[idx]["status"] = "success"
                    branch_results[idx]["stats"]  = stats
            except Exception as exc:
                branch_results[idx]["status"] = "error"
                branch_results[idx]["error"]  = f"Worker crash: {exc}"
