"""
queue_manager.py
================
Persistent batch queue for FfmpegTool Phase 2.

States:  pending -> running -> done
                            -> failed  -> (retry) -> pending
                 -> skipped

Queue file: queue/batch_queue.json  (auto-created, relative to this file)

Usage:
    from queue_manager import QueueManager

    qm = QueueManager()
    item_id = qm.add("/videos/clip.mp4", "/output", cfg_overrides={})
    qm.mark_running(item_id)
    qm.mark_done(item_id, stats={})

    # Retry a failed item
    qm.retry(item_id)

    # Run one pending item
    qm.run_next(base_cfg, process_video_fn)
"""

import os
import json
import uuid
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

# ── Paths ──
TOOL_DIR  = os.path.dirname(os.path.abspath(__file__))
QUEUE_DIR = os.path.join(TOOL_DIR, "queue")
QUEUE_FILE = os.path.join(QUEUE_DIR, "batch_queue.json")

# ── Valid states ──
STATES = {"pending", "running", "done", "failed", "skipped"}


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class QueueItem:
    """Single item in the persistent batch queue."""
    id:           str
    input:        str
    output:       str
    status:       str       = "pending"        # pending/running/done/failed/skipped
    cfg_overrides: dict     = field(default_factory=dict)
    created_at:   str       = ""
    started_at:   Optional[str] = None
    finished_at:  Optional[str] = None
    error:        Optional[str] = None
    stats:        Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "QueueItem":
        return QueueItem(
            id           = d["id"],
            input        = d["input"],
            output       = d["output"],
            status       = d.get("status",       "pending"),
            cfg_overrides= d.get("cfg_overrides", {}),
            created_at   = d.get("created_at",   ""),
            started_at   = d.get("started_at"),
            finished_at  = d.get("finished_at"),
            error        = d.get("error"),
            stats        = d.get("stats"),
        )


# ─────────────────────────────────────────────
# Queue Manager
# ─────────────────────────────────────────────

class QueueManager:
    """
    Thread-safe persistent queue manager.

    All mutations are protected by a single threading.Lock.
    The queue is persisted to JSON after every write operation.
    """

    def __init__(self, queue_file: str = QUEUE_FILE):
        self._file   = queue_file
        self._lock   = threading.Lock()
        self._items: dict[str, QueueItem] = {}
        self._load()

    # ── Public API ──

    def add(
        self,
        input_path:    str,
        output_path:   str,
        cfg_overrides: dict | None = None,
    ) -> str:
        """
        Add a new item to the queue.

        Args:
            input_path:    Path to video file (or folder for batch).
            output_path:   Output root directory.
            cfg_overrides: Optional dict of config keys to override for this item.

        Returns:
            The new item's ID string.
        """
        item_id = str(uuid.uuid4())[:8]
        item = QueueItem(
            id            = item_id,
            input         = input_path,
            output        = output_path,
            status        = "pending",
            cfg_overrides = cfg_overrides or {},
            created_at    = _now(),
        )
        with self._lock:
            self._items[item_id] = item
            self._save()
        return item_id

    def list_items(self) -> list[QueueItem]:
        """Return all queue items (ordered by created_at)."""
        with self._lock:
            return sorted(self._items.values(), key=lambda x: x.created_at)

    def get(self, item_id: str) -> Optional[QueueItem]:
        """Return a single item by ID, or None if not found."""
        with self._lock:
            return self._items.get(item_id)

    def remove(self, item_id: str) -> bool:
        """
        Remove a pending item from the queue.

        Returns True if removed, False if not found.
        Raises ValueError if item is not in a removable state (running/done).
        """
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return False
            if item.status == "running":
                raise ValueError(f"Cannot remove item {item_id}: currently running")
            del self._items[item_id]
            self._save()
        return True

    def retry(self, item_id: str) -> QueueItem:
        """
        Reset a failed/skipped item back to pending so it will be re-processed.

        Raises ValueError if item is not in a retryable state.
        """
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                raise KeyError(f"Queue item not found: {item_id}")
            if item.status not in {"failed", "skipped"}:
                raise ValueError(
                    f"Item {item_id} is '{item.status}' — only failed/skipped items can be retried"
                )
            item.status      = "pending"
            item.error       = None
            item.started_at  = None
            item.finished_at = None
            item.stats       = None
            self._save()
        return item

    def mark_running(self, item_id: str):
        """Mark an item as running (called when processing begins)."""
        with self._lock:
            item = self._items.get(item_id)
            if item:
                item.status     = "running"
                item.started_at = _now()
                self._save()

    def mark_done(self, item_id: str, stats: dict | None = None):
        """Mark an item as done with optional stats dict."""
        with self._lock:
            item = self._items.get(item_id)
            if item:
                item.status      = "done"
                item.finished_at = _now()
                item.stats       = stats
                self._save()

    def mark_failed(self, item_id: str, error: str):
        """Mark an item as failed with an error message."""
        with self._lock:
            item = self._items.get(item_id)
            if item:
                item.status      = "failed"
                item.finished_at = _now()
                item.error       = error
                self._save()

    def mark_skipped(self, item_id: str, reason: str):
        """Mark an item as skipped (no frames extracted, or other skip reason)."""
        with self._lock:
            item = self._items.get(item_id)
            if item:
                item.status      = "skipped"
                item.finished_at = _now()
                item.error       = reason
                self._save()

    def run_next(
        self,
        base_cfg: dict,
        process_video_fn: Callable[[str, str, dict, int], dict],
    ) -> Optional[str]:
        """
        Pick the first pending item and process it synchronously.

        Args:
            base_cfg:         Base config dict (from main.py apply_defaults).
            process_video_fn: Callable matching process_video(path, output, cfg, index) -> stats.

        Returns:
            The item ID that was processed, or None if queue is empty.
        """
        # Find first pending item (outside lock for now, check again inside)
        with self._lock:
            pending = [i for i in self._items.values() if i.status == "pending"]
            if not pending:
                return None
            item = pending[0]
            item.status     = "running"
            item.started_at = _now()
            self._save()

        item_id = item.id
        print(f"[QUEUE] Processing item {item_id}: {item.input}")

        # Merge overrides into a copy of base_cfg
        import copy
        cfg = copy.deepcopy(base_cfg)
        for section, values in item.cfg_overrides.items():
            if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                cfg[section].update(values)

        try:
            stats = process_video_fn(item.input, item.output, cfg, 0)
            if stats:
                self.mark_done(item_id, stats)
                print(f"[QUEUE] Done: {item_id}")
            else:
                self.mark_skipped(item_id, "No frames extracted")
                print(f"[QUEUE] Skipped: {item_id} (no frames)")
        except Exception as exc:
            self.mark_failed(item_id, str(exc))
            print(f"[QUEUE] Failed: {item_id} — {exc}")

        return item_id

    def pending_count(self) -> int:
        """Return number of pending items."""
        with self._lock:
            return sum(1 for i in self._items.values() if i.status == "pending")

    def summary(self) -> dict:
        """Return a dict with counts per status."""
        with self._lock:
            counts: dict[str, int] = {s: 0 for s in STATES}
            for item in self._items.values():
                counts[item.status] = counts.get(item.status, 0) + 1
            counts["total"] = len(self._items)
        return counts

    # ── Internal helpers ──

    def _load(self):
        """Load queue from JSON file. Safe if file missing or corrupt."""
        try:
            if os.path.exists(self._file):
                with open(self._file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for d in raw.get("items", []):
                    item = QueueItem.from_dict(d)
                    self._items[item.id] = item
                # Reset any 'running' items back to 'pending' on startup
                # (they were interrupted by a previous crash/restart)
                for item in self._items.values():
                    if item.status == "running":
                        item.status = "pending"
                        item.started_at = None
        except (json.JSONDecodeError, KeyError, TypeError):
            # Corrupt file — start with empty queue
            self._items = {}

    def _save(self):
        """Persist queue to JSON. Must be called inside self._lock."""
        os.makedirs(os.path.dirname(self._file), exist_ok=True)
        payload = {
            "version":    2,
            "updated_at": _now(),
            "items":      [i.to_dict() for i in self._items.values()],
        }
        # Atomic write via temp file
        tmp = self._file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._file)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_queue_table(items: list[QueueItem]):
    """Print queue as an ASCII table (no Unicode box chars)."""
    sep = "-" * 72
    print(f"\n{sep}")
    print(f"  {'ID':<10} {'Status':<10} {'Input':<30} {'Created':<20}")
    print(sep)
    if not items:
        print("  (queue is empty)")
    for item in items:
        input_short = item.input[-28:] if len(item.input) > 28 else item.input
        print(f"  {item.id:<10} {item.status:<10} {input_short:<30} {item.created_at:<20}")
        if item.error:
            print(f"  {'':10} {'':10} -> Error: {item.error[:60]}")
    print(f"{sep}\n")
