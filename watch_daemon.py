"""
watch_daemon.py
===============
FfmpegTool Phase 3 -- Watch Folder Daemon

Monitors a folder for newly dropped video files and automatically enqueues
them into QueueManager for processing.

Features:
  - Uses watchdog library for cross-platform filesystem events
  - 3-second debounce to wait for file write to finish before enqueuing
  - Persists watch state to queue/watch_state.json
  - Can be started as a standalone CLI or embedded in app.py

CLI:
    python watch_daemon.py --watch G:/Inbox --output G:/Output --preset tiktok_pack

API (via app.py):
    POST /api/watch/start  {folder, output, preset}
    POST /api/watch/stop
    GET  /api/watch/status
"""

import os
import sys
import json
import time
import argparse
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Paths ──
TOOL_DIR   = os.path.dirname(os.path.abspath(__file__))
QUEUE_DIR  = os.path.join(TOOL_DIR, "queue")
STATE_FILE = os.path.join(QUEUE_DIR, "watch_state.json")

# ── Supported video extensions ──
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# ── Debounce delay (seconds) ──
DEBOUNCE_SECONDS = 3.0


# ─────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────

def _load_state() -> dict:
    """Load watch state from JSON. Returns empty dict if missing/corrupt."""
    if not os.path.isfile(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict):
    """Persist watch state atomically."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, STATE_FILE)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────────────────────────
# Video event handler (watchdog)
# ─────────────────────────────────────────────

try:
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
    _WATCHDOG_AVAILABLE = True
except ImportError:
    _WATCHDOG_AVAILABLE = False

    # Minimal stubs so the rest of the module is importable without watchdog
    class FileSystemEventHandler:  # type: ignore
        pass

    class FileCreatedEvent:  # type: ignore
        src_path = ""

    class FileMovedEvent:  # type: ignore
        dest_path = ""


class VideoFileHandler(FileSystemEventHandler):
    """
    Watches for new or moved-in video files and enqueues them after a debounce delay.
    """

    def __init__(self, output: str, preset: str, queue_mgr, debounce: float = DEBOUNCE_SECONDS):
        super().__init__()
        self._output   = output
        self._preset   = preset
        self._qm       = queue_mgr
        self._debounce = debounce
        self._pending: dict[str, threading.Timer] = {}  # path -> active timer
        self._enqueued: set[str] = set()  # paths already enqueued this session
        self._lock = threading.Lock()

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_moved(self, event):
        """Handle file moved into watched directory (e.g. atomic save patterns)."""
        if not event.is_directory:
            self._schedule(event.dest_path)

    def _schedule(self, path: str):
        """Schedule enqueue after debounce delay, cancelling any existing timer."""
        if Path(path).suffix.lower() not in VIDEO_EXTENSIONS:
            return
        with self._lock:
            if path in self._pending:
                self._pending[path].cancel()
            timer = threading.Timer(self._debounce, self._enqueue, args=[path])
            self._pending[path] = timer
            timer.daemon = True
            timer.start()

    def _enqueue(self, path: str):
        """Called after debounce: validate file is readable then enqueue."""
        with self._lock:
            self._pending.pop(path, None)
            if path in self._enqueued:
                # Duplicate event burst for the same path -- skip silently.
                return
            self._enqueued.add(path)

        if not os.path.isfile(path):
            with self._lock:
                self._enqueued.discard(path)
            return  # file disappeared before debounce finished

        cfg_overrides = {}
        if self._preset:
            cfg_overrides["preset"] = self._preset

        try:
            item_id = self._qm.add(path, self._output, cfg_overrides=cfg_overrides)
            print(f"[WATCH] Enqueued {os.path.basename(path)} -> {item_id}")
        except Exception as exc:
            with self._lock:
                self._enqueued.discard(path)
            print(f"[WATCH] Failed to enqueue {path}: {exc}")


# ─────────────────────────────────────────────
# WatchDaemon
# ─────────────────────────────────────────────

class WatchDaemon:
    """
    Manages the lifecycle of the watchdog observer thread.

    Only one watch folder is supported at a time. Starting a new watch
    automatically stops the previous one.
    """

    def __init__(self, queue_mgr=None):
        self._qm       = queue_mgr
        self._observer = None
        self._handler  = None
        self._started_at: Optional[str] = None
        self._folder:  Optional[str] = None
        self._output:  Optional[str] = None
        self._preset:  Optional[str] = None
        self._lock = threading.Lock()

    def start(self, folder: str, output: str, preset: str = "") -> dict:
        """
        Start watching a folder.

        Returns the new status dict. Raises RuntimeError if watchdog is not installed,
        or ValueError if the folder doesn't exist.
        """
        if not _WATCHDOG_AVAILABLE:
            raise RuntimeError(
                "watchdog library is not installed. Run: pip install watchdog>=3.0.0"
            )

        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            raise ValueError(f"Watch folder does not exist: {folder}")

        with self._lock:
            self._stop_observer()

            from watchdog.observers import Observer

            self._folder     = folder
            self._output     = output
            self._preset     = preset
            self._started_at = _now()

            self._handler = VideoFileHandler(output, preset, self._qm)
            self._observer = Observer()
            self._observer.schedule(self._handler, folder, recursive=False)
            self._observer.daemon = True
            self._observer.start()

            state = {
                "watching":   True,
                "folder":     folder,
                "output":     output,
                "preset":     preset,
                "started_at": self._started_at,
            }
            _save_state(state)
            print(f"[WATCH] Started watching: {folder}")
            return state

    def stop(self) -> dict:
        """Stop the current watch and update state file."""
        with self._lock:
            self._stop_observer()
            state = {
                "watching":   False,
                "folder":     self._folder or "",
                "output":     self._output or "",
                "preset":     self._preset or "",
                "started_at": None,
                "stopped_at": _now(),
            }
            _save_state(state)
            self._folder     = None
            self._output     = None
            self._preset     = None
            self._started_at = None
            print("[WATCH] Stopped.")
            return state

    def status(self) -> dict:
        """Return current daemon status."""
        with self._lock:
            watching = self._observer is not None and self._observer.is_alive()
            return {
                "watching":   watching,
                "folder":     self._folder  or "",
                "output":     self._output  or "",
                "preset":     self._preset  or "",
                "started_at": self._started_at,
                "watchdog_available": _WATCHDOG_AVAILABLE,
            }

    def resume_from_state(self) -> bool:
        """
        Read watch_state.json and auto-resume if watching=True.
        Called once at app startup.

        Returns True if the daemon was successfully resumed.
        """
        state = _load_state()
        if not state.get("watching"):
            return False
        folder = state.get("folder", "")
        output = state.get("output", "")
        preset = state.get("preset", "")
        if not folder or not os.path.isdir(folder):
            return False
        try:
            self.start(folder, output, preset)
            print(f"[WATCH] Resumed watching: {folder}")
            return True
        except Exception as exc:
            print(f"[WATCH] Could not resume watch: {exc}")
            return False

    # ── Internal ──

    def _stop_observer(self):
        """Stop and join the observer thread. Must be called inside self._lock."""
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception:
                pass
            self._observer = None
            self._handler  = None


# ─────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────

_daemon_instance: Optional[WatchDaemon] = None


def get_daemon(queue_mgr=None) -> WatchDaemon:
    """Return (and lazily create) the module-level WatchDaemon singleton."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = WatchDaemon(queue_mgr=queue_mgr)
    elif queue_mgr is not None:
        # BUG-M4 FIX: always update _qm when a non-None queue_mgr is passed,
        # even if the daemon already has one — the caller may be passing a
        # fresh instance (e.g. after app restart). Previously the update was
        # silently ignored if _qm was already set, causing stale references.
        _daemon_instance._qm = queue_mgr
        if _daemon_instance._handler is not None:
            _daemon_instance._handler._qm = queue_mgr
    return _daemon_instance


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def _build_cli_parser():
    p = argparse.ArgumentParser(
        prog="watch_daemon",
        description="FfmpegTool Watch Folder Daemon -- auto-enqueue new videos",
    )
    p.add_argument("--watch",  "-w", required=True, help="Folder to monitor for new videos")
    p.add_argument("--output", "-o", required=True, help="Output folder for extracted frames")
    p.add_argument("--preset", "-p", default="",   help="Preset name to apply (e.g. tiktok_pack)")
    return p


def _cli_main():
    if not _WATCHDOG_AVAILABLE:
        print("[ERROR] watchdog library not found. Run: pip install watchdog>=3.0.0")
        sys.exit(1)

    sys.path.insert(0, TOOL_DIR)
    from queue_manager import QueueManager

    parser = _build_cli_parser()
    args   = parser.parse_args()

    qm     = QueueManager()
    daemon = WatchDaemon(queue_mgr=qm)
    daemon.start(args.watch, args.output, args.preset)

    print(f"[WATCH] Monitoring: {args.watch}")
    print(f"[WATCH] Output:     {args.output}")
    if args.preset:
        print(f"[WATCH] Preset:     {args.preset}")
    print("[WATCH] Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        daemon.stop()
        print("[WATCH] Exiting.")


if __name__ == "__main__":
    _cli_main()
