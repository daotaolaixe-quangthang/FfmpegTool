"""
tests/test_senior_audit.py
Senior audit test suite covering main.py, filters.py, cache_manager.py,
queue_manager.py, and app.py.

All tests are fully isolated: no real ffmpeg/ffprobe, no real filesystem writes
outside tmp_path, no shared state, no order dependencies.
"""

import importlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# Path setup - insert project root so imports resolve without installation
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# GROUP A -- main.py: Config & Naming
# ===========================================================================

class TestMainApplyDefaults:
    def test_main_apply_defaults_missing_all_sections(self):
        """apply_defaults on an empty dict must populate every required top-level key."""
        from main import apply_defaults

        result = apply_defaults({})

        required_sections = {"extraction", "filter", "scorer", "output", "hardware", "batch", "normalize"}
        assert required_sections.issubset(result.keys()), "Missing top-level config sections"
        assert result["extraction"]["mode"] == "fps"
        assert result["extraction"]["fps"] == 5
        assert result["filter"]["blur_threshold"] == 80.0
        assert result["filter"]["dedup_method"] == "phash"
        assert result["output"]["naming_pattern"] == "{video_name}"
        assert result["hardware"]["encoder"] == "auto"
        assert result["normalize"]["enabled"] is True


class TestMainResolveNaming:
    def _base_cfg(self, pattern, campaign="camp", lang="vi", ratio="9x16"):
        from main import apply_defaults
        cfg = apply_defaults({})
        cfg["output"]["naming_pattern"] = pattern
        cfg["output"]["campaign"] = campaign
        cfg["output"]["lang"] = lang
        cfg["output"]["ratio"] = ratio
        return cfg

    def test_main_resolve_naming_all_tokens(self):
        """Pattern with all tokens filled must produce no double underscores or leading/trailing separators."""
        from main import resolve_video_output_name

        cfg = self._base_cfg("{video_name}_{index}_{campaign}_{lang}_{ratio}_{date}")
        result = resolve_video_output_name("/videos/clip.mp4", cfg, batch_index=3)

        assert "__" not in result, f"Double underscore found in '{result}'"
        assert not result.startswith("_"), f"Starts with underscore: '{result}'"
        assert not result.endswith("_"), f"Ends with underscore: '{result}'"
        assert "clip" in result
        assert "003" in result
        assert "camp" in result
        assert "vi" in result
        assert "9x16" in result

    def test_main_resolve_naming_empty_result_fallback(self):
        """A pattern that resolves to empty string must fall back to the video stem."""
        from main import apply_defaults, resolve_video_output_name

        cfg = apply_defaults({})
        cfg["output"]["naming_pattern"] = "{campaign}"
        cfg["output"]["campaign"] = ""  # all tokens are empty -> result is ""

        result = resolve_video_output_name("/videos/myclip.mp4", cfg, batch_index=0)

        assert result == "myclip", f"Expected fallback 'myclip', got '{result}'"


class TestMainCliFpsInvalid:
    @pytest.mark.parametrize("fps", [0, -1, -0.001])
    def test_main_cli_fps_invalid_uses_default(self, fps, capsys):
        """apply_cli_overrides with an invalid fps must leave cfg unchanged and print a warning."""
        from main import apply_defaults, apply_cli_overrides

        cfg = apply_defaults({})
        original_fps = cfg["extraction"]["fps"]

        args = SimpleNamespace(
            fps=fps, mode=None, blur=None, sim=None, method=None,
            top=None, keep_raw=False, no_html=False, no_report=False,
            no_probe=False, no_normalize=False, draft=False, no_cache=False,
        )
        apply_cli_overrides(cfg, args)

        assert cfg["extraction"]["fps"] == original_fps, (
            f"fps should remain {original_fps} after invalid value {fps}"
        )
        captured = capsys.readouterr()
        assert "ERROR" in captured.out, "Expected [ERROR] warning in stdout"


# ===========================================================================
# GROUP B -- filters.py: Edge Cases
# ===========================================================================

class TestFiltersBlurAllRemoved:
    def test_filters_blur_all_frames_removed(self, tmp_path):
        """All frames below blur threshold -> final_count == 0, pipeline must not crash."""
        import numpy as np
        import cv2

        # Create 3 flat (blurry) images that will score near 0
        frame_paths = []
        for i in range(3):
            p = tmp_path / f"frame_{i:04d}.jpg"
            img = np.zeros((64, 64, 3), dtype=np.uint8)  # all-black -> lap variance ~0
            cv2.imwrite(str(p), img)
            frame_paths.append(str(p))

        from filters import run_filter_pipeline

        out_dir = tmp_path / "output"
        cfg = {"blur_threshold": 9999.0, "similarity_threshold": 0.70,
               "dedup_method": "phash", "phash_size": 16}

        stats = run_filter_pipeline(frame_paths, str(out_dir), cfg)

        assert stats["final_count"] == 0, f"Expected 0 final frames, got {stats['final_count']}"
        assert stats["removed_blur"] == 3


class TestFiltersDedupSingle:
    def test_filters_dedup_single_frame(self, tmp_path):
        """A single-frame input must pass through dedup unchanged (no crash, no comparison)."""
        import numpy as np
        import cv2

        p = tmp_path / "frame_0000.jpg"
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(p), img)

        from filters import run_filter_pipeline

        out_dir = tmp_path / "output"
        cfg = {"blur_threshold": 0.0, "similarity_threshold": 0.70,
               "dedup_method": "phash", "phash_size": 16}

        stats = run_filter_pipeline([str(p)], str(out_dir), cfg)

        assert stats["final_count"] == 1
        assert stats["removed_duplicate"] == 0


class TestFiltersCorruptedJpeg:
    def test_filters_corrupted_jpeg_skipped(self, tmp_path):
        """A corrupt JPEG must be skipped and counted as removed without crashing."""
        import numpy as np
        import cv2

        # Good frame
        good = tmp_path / "frame_0000.jpg"
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(good), img)

        # Corrupt file (not a valid image)
        bad = tmp_path / "frame_0001.jpg"
        bad.write_bytes(b"\xff\xd8\xff\xe0 this is not a real jpeg")

        from filters import run_filter_pipeline

        out_dir = tmp_path / "output"
        cfg = {"blur_threshold": 0.0, "similarity_threshold": 0.70,
               "dedup_method": "phash", "phash_size": 16}

        stats = run_filter_pipeline([str(good), str(bad)], str(out_dir), cfg)

        total_removed = stats["removed_blur"] + stats["removed_duplicate"]
        assert stats["final_count"] == 1, "Only the valid frame should survive"
        assert total_removed == 1, "The corrupt frame must be counted as removed"


class TestFiltersDedupIdentical:
    @pytest.mark.parametrize("method", ["phash", "ssim"])
    def test_filters_dedup_identical_frames_removes_duplicates(self, tmp_path, method):
        """5 identical frames with dedup must keep exactly 1."""
        import numpy as np
        import cv2

        base_img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        frame_paths = []
        for i in range(5):
            p = tmp_path / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(p), base_img)
            frame_paths.append(str(p))

        from filters import run_filter_pipeline

        out_dir = tmp_path / "output"
        cfg = {"blur_threshold": 0.0, "similarity_threshold": 0.70,
               "dedup_method": method, "phash_size": 16}

        stats = run_filter_pipeline(frame_paths, str(out_dir), cfg)

        assert stats["final_count"] == 1, (
            f"[{method}] Expected 1 unique frame from 5 identical, got {stats['final_count']}"
        )
        assert stats["removed_duplicate"] == 4


# ===========================================================================
# GROUP C -- cache_manager.py: Cache Key & Invalidation
# ===========================================================================

def _minimal_cfg():
    """Return the smallest cfg dict that satisfies _config_sig."""
    return {
        "extraction": {"mode": "fps", "fps": 5, "scene_threshold": 27.0, "draft": False},
        "filter": {"blur_threshold": 80.0, "similarity_threshold": 0.70,
                   "dedup_method": "phash", "phash_size": 16},
    }


class TestCacheMissHit:
    def test_cache_miss_then_hit_same_cfg(self, tmp_path):
        """store_frames then get_cached_frames with identical cfg/mtime returns cached frame paths."""
        from cache_manager import CacheManager

        cache_dir = str(tmp_path / "cache")
        cm = CacheManager(cache_dir=cache_dir)

        # Create a fake video file so mtime is readable
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 16)

        # Create fake frame files
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        frame_files = []
        for i in range(3):
            f = frames_dir / f"unique_{i:04d}.jpg"
            f.write_bytes(b"\xff" * 8)
            frame_files.append(str(f))

        cfg = _minimal_cfg()
        cm.store_frames(str(video), cfg, frame_files)

        result = cm.get_cached_frames(str(video), cfg)

        assert result is not None, "Expected cache hit"
        # store_frames copies frames into the cache dir, so returned paths differ
        # from the originals -- verify count and that all returned files exist.
        assert len(result) == len(frame_files), "Returned frame count must match stored count"
        assert all(os.path.isfile(p) for p in result), "All returned cache paths must exist on disk"

    def test_cache_miss_when_fps_changes(self, tmp_path):
        """Changing fps in cfg invalidates the cache (different config sig)."""
        from cache_manager import CacheManager

        cache_dir = str(tmp_path / "cache")
        cm = CacheManager(cache_dir=cache_dir)

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 16)

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        f = frames_dir / "unique_0000.jpg"
        f.write_bytes(b"\xff" * 8)

        cfg_store = _minimal_cfg()
        cfg_store["extraction"]["fps"] = 5
        cm.store_frames(str(video), cfg_store, [str(f)])

        cfg_get = _minimal_cfg()
        cfg_get["extraction"]["fps"] = 3  # different fps

        result = cm.get_cached_frames(str(video), cfg_get)

        assert result is None, "Expected cache miss when fps changes"

    def test_cache_miss_when_mtime_changes(self, tmp_path):
        """Modifying the video file's mtime invalidates the cache."""
        from cache_manager import CacheManager

        cache_dir = str(tmp_path / "cache")
        cm = CacheManager(cache_dir=cache_dir)

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 16)

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        f = frames_dir / "unique_0000.jpg"
        f.write_bytes(b"\xff" * 8)

        cfg = _minimal_cfg()
        cm.store_frames(str(video), cfg, [str(f)])

        # Advance mtime by at least 1 second
        new_mtime = os.path.getmtime(str(video)) + 2.0
        os.utime(str(video), (new_mtime, new_mtime))

        result = cm.get_cached_frames(str(video), cfg)

        assert result is None, "Expected cache miss after mtime change"

    def test_cache_corrupt_meta_returns_none(self, tmp_path):
        """A corrupt _meta.json must return None without raising an exception."""
        from cache_manager import CacheManager

        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cm = CacheManager(cache_dir=cache_dir)

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 16)

        cfg = _minimal_cfg()

        # Pre-compute the key so we can write garbage to the right meta file
        key = cm._cache_key(str(video), cfg)
        meta_path = Path(cache_dir) / f"{key}_meta.json"
        meta_path.write_text("this is not valid json }{{{", encoding="utf-8")

        result = cm.get_cached_frames(str(video), cfg)

        assert result is None, "Expected None for corrupt meta, got something else"

    def test_cache_purge_all_empty_dir(self, tmp_path):
        """purge_all() on an empty cache dir returns 0 without crashing."""
        from cache_manager import CacheManager

        cache_dir = str(tmp_path / "empty_cache")
        os.makedirs(cache_dir)
        cm = CacheManager(cache_dir=cache_dir)

        count = cm.purge_all()

        assert count == 0, f"Expected 0 purged entries, got {count}"


# ===========================================================================
# GROUP D -- queue_manager.py: State Machine
# ===========================================================================

def _make_qm(tmp_path):
    """Create an isolated QueueManager backed by a temp file."""
    from queue_manager import QueueManager
    queue_file = str(tmp_path / "test_queue.json")
    return QueueManager(queue_file=queue_file)


class TestQueueStateMachine:
    def test_queue_run_next_on_empty_queue(self, tmp_path):
        """run_next() on an empty queue returns None immediately."""
        qm = _make_qm(tmp_path)
        result = qm.run_next({}, lambda *a, **kw: {})
        assert result is None

    def test_queue_retry_pending_raises(self, tmp_path):
        """retry() on a 'pending' item must raise ValueError."""
        qm = _make_qm(tmp_path)
        item_id = qm.add("/input/clip.mp4", "/output")
        # item is now pending

        with pytest.raises(ValueError):
            qm.retry(item_id)

    def test_queue_retry_done_raises(self, tmp_path):
        """retry() on a 'done' item must raise ValueError."""
        qm = _make_qm(tmp_path)
        item_id = qm.add("/input/clip.mp4", "/output")
        qm.mark_running(item_id)
        qm.mark_done(item_id, stats={})

        with pytest.raises(ValueError):
            qm.retry(item_id)

    def test_queue_remove_running_raises(self, tmp_path):
        """remove() on a 'running' item must raise ValueError."""
        qm = _make_qm(tmp_path)
        item_id = qm.add("/input/clip.mp4", "/output")
        qm.mark_running(item_id)

        with pytest.raises(ValueError):
            qm.remove(item_id)

    def test_queue_process_video_exception_marks_failed(self, tmp_path):
        """If process_video_fn raises RuntimeError, the item status must be 'failed'."""
        qm = _make_qm(tmp_path)
        item_id = qm.add("/input/clip.mp4", "/output")

        def boom(path, out, cfg, idx=0):
            raise RuntimeError("simulated crash")

        qm.run_next({}, boom)

        item = qm.get(item_id)
        assert item is not None
        assert item.status == "failed"
        assert "simulated crash" in (item.error or "")

    def test_queue_corrupt_file_starts_empty(self, tmp_path):
        """Loading a QueueManager with corrupt JSON queue file starts with an empty queue."""
        from queue_manager import QueueManager

        queue_file = str(tmp_path / "bad_queue.json")
        Path(queue_file).parent.mkdir(parents=True, exist_ok=True)
        Path(queue_file).write_text("{this is not valid json", encoding="utf-8")

        qm = QueueManager(queue_file=queue_file)

        assert qm._items == {}, f"Expected empty _items, got {qm._items}"

    def test_queue_running_items_reset_to_pending_on_load(self, tmp_path):
        """A 'running' item in a saved queue must be reset to 'pending' on reload."""
        from queue_manager import QueueManager

        queue_file = str(tmp_path / "queue.json")
        Path(queue_file).parent.mkdir(parents=True, exist_ok=True)

        # Write a queue JSON with one running item
        payload = {
            "version": 2,
            "updated_at": "2026-01-01 00:00:00",
            "items": [
                {
                    "id": "abc12345",
                    "input": "/input/clip.mp4",
                    "output": "/output",
                    "status": "running",
                    "cfg_overrides": {},
                    "created_at": "2026-01-01 00:00:00",
                    "started_at": "2026-01-01 00:00:01",
                    "finished_at": None,
                    "error": None,
                    "stats": None,
                }
            ],
        }
        Path(queue_file).write_text(json.dumps(payload), encoding="utf-8")

        qm = QueueManager(queue_file=queue_file)
        item = qm.get("abc12345")

        assert item is not None
        assert item.status == "pending", (
            f"Expected 'pending' after reload, got '{item.status}'"
        )
        assert item.started_at is None


# ===========================================================================
# GROUP E -- app.py: API Validation
# ===========================================================================

# We need to import app with heavy dependencies mocked before Flask starts.
# Strategy: patch the expensive module-level side effects before import.

def _get_test_client():
    """Return a Flask test client with module-level side effects mocked."""
    watch_daemon_mock = MagicMock()
    watch_daemon_mock.resume_from_state = MagicMock()
    watch_daemon_mock.status.return_value = {"watching": False}

    with patch("watch_daemon.get_daemon", return_value=watch_daemon_mock), \
         patch("queue_manager.QueueManager._load", return_value=None), \
         patch("queue_manager.QueueManager._save", return_value=None), \
         patch("subprocess.Popen") as _mock_popen, \
         patch("subprocess.run") as _mock_run:
        import app as _app_module
        _app_module = importlib.reload(_app_module)
        _app_module.app.config["TESTING"] = True
        with _app_module.JOBS_LOCK:
            _app_module.JOBS.clear()
        with _app_module.COMPLETED_STATS_LOCK:
            _app_module.COMPLETED_STATS.clear()
        with _app_module.QUEUE_MGR._lock:
            _app_module.QUEUE_MGR._items.clear()
        return _app_module.app.test_client(), _app_module


@pytest.fixture
def flask_client():
    client, app_module = _get_test_client()
    return client, app_module


class TestAppApiRun:
    def test_app_run_missing_input_returns_400(self, flask_client):
        """POST /api/run with no input or url must return HTTP 400."""
        client, _ = flask_client
        resp = client.post(
            "/api/run",
            json={"output": "/some/output"},
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_app_run_missing_output_returns_400(self, flask_client):
        """POST /api/run with input but no output must return HTTP 400."""
        client, _ = flask_client
        resp = client.post(
            "/api/run",
            json={"input": "/some/video.mp4"},
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestAppStream:
    def test_app_stream_invalid_job_id_returns_404(self, flask_client):
        """GET /api/stream/<nonexistent-id> must return HTTP 404."""
        client, _ = flask_client
        resp = client.get("/api/stream/nonexistent-job-id-xyz")
        assert resp.status_code == 404


class TestAppQueueRetryRemove:
    def test_app_queue_retry_pending_item_returns_400(self, flask_client):
        """POST /api/queue/retry/<id> for a 'pending' item must return HTTP 400."""
        client, app_module = flask_client

        # Inject a pending item directly
        from queue_manager import QueueItem
        item_id = "test_pending_1"
        item = QueueItem(id=item_id, input="/in/v.mp4", output="/out",
                         status="pending", created_at="2026-01-01 00:00:00")
        with app_module.QUEUE_MGR._lock:
            app_module.QUEUE_MGR._items[item_id] = item

        try:
            resp = client.post(f"/api/queue/retry/{item_id}")
            assert resp.status_code == 400
        finally:
            with app_module.QUEUE_MGR._lock:
                app_module.QUEUE_MGR._items.pop(item_id, None)

    def test_app_queue_remove_running_item_returns_400(self, flask_client):
        """DELETE /api/queue/<id> for a 'running' item must return HTTP 400."""
        client, app_module = flask_client

        from queue_manager import QueueItem
        item_id = "test_running_1"
        item = QueueItem(id=item_id, input="/in/v.mp4", output="/out",
                         status="running", created_at="2026-01-01 00:00:00")
        with app_module.QUEUE_MGR._lock:
            app_module.QUEUE_MGR._items[item_id] = item

        try:
            resp = client.delete(f"/api/queue/{item_id}")
            assert resp.status_code == 400
        finally:
            with app_module.QUEUE_MGR._lock:
                app_module.QUEUE_MGR._items.pop(item_id, None)


class TestAppBuildCommand:
    def test_app_build_command_url_mode(self):
        """build_command with a url set must contain '--url', not '--input'."""
        # Import helper directly; no Flask context needed
        with patch("watch_daemon.get_daemon", return_value=MagicMock()), \
             patch("queue_manager.QueueManager._load", return_value=None), \
             patch("queue_manager.QueueManager._save", return_value=None):
            from app import build_command

        data = {
            "url": "https://www.tiktok.com/example",
            "output": "/output",
            "mode": "fps",
            "fps": 5,
            "blur": 80,
            "sim": 0.70,
            "method": "phash",
        }
        cmd = build_command(data)
        flat = " ".join(cmd)

        assert "--url" in flat, f"Expected '--url' in command: {flat}"
        assert "--input" not in flat, f"'--input' must not appear in URL-mode command: {flat}"

    def test_app_build_command_batch_mode(self):
        """build_command with batch=True must contain '--batch'."""
        with patch("watch_daemon.get_daemon", return_value=MagicMock()), \
             patch("queue_manager.QueueManager._load", return_value=None), \
             patch("queue_manager.QueueManager._save", return_value=None):
            from app import build_command

        data = {
            "batch": True,
            "input": "/videos/",
            "output": "/output",
            "mode": "fps",
            "fps": 5,
            "blur": 80,
            "sim": 0.70,
            "method": "phash",
        }
        cmd = build_command(data)
        flat = " ".join(cmd)

        assert "--batch" in flat, f"Expected '--batch' in command: {flat}"


# ===========================================================================
# GROUP F -- main.py: Config Loading, CLI Boundaries, Routing
# ===========================================================================

class TestMainLoadConfig:
    def test_main_load_config_missing_returns_empty_dict(self):
        """load_config() with a missing file must return an empty dict."""
        from main import load_config

        result = load_config("definitely_missing_config_for_test.json")

        assert result == {}

    def test_main_load_config_invalid_json_warns_and_returns_empty_dict(self, tmp_path, capsys):
        """Malformed config.json must print a warning and fall back to {}."""
        import main

        cfg_path = tmp_path / "bad_config.json"
        cfg_path.write_text('{"filter": { invalid json', encoding="utf-8")

        with patch("main.os.path.abspath", return_value=str(tmp_path / "main.py")):
            result = main.load_config(cfg_path.name)

        assert result == {}
        captured = capsys.readouterr()
        assert "invalid JSON" in captured.out
        assert "Using built-in defaults" in captured.out

    def test_main_load_config_preserves_unknown_keys(self, tmp_path):
        """Unknown keys from config.json must load without crashing and survive apply_defaults()."""
        import main

        cfg_path = tmp_path / "custom_config.json"
        payload = {
            "unknown_root": 123,
            "filter": {"unknown_nested": "keep-me"},
            "output": {"naming_pattern": "{video_name}"},
        }
        cfg_path.write_text(json.dumps(payload), encoding="utf-8")

        with patch("main.os.path.abspath", return_value=str(tmp_path / "main.py")):
            loaded = main.load_config(cfg_path.name)

        final = main.apply_defaults(loaded)
        assert final["unknown_root"] == 123
        assert final["filter"]["unknown_nested"] == "keep-me"
        assert final["filter"]["blur_threshold"] == 80.0


class TestMainCliOverrideBoundaries:
    def _args(self, **overrides):
        base = {
            "fps": None, "mode": None, "blur": None, "sim": None, "method": None,
            "top": None, "keep_raw": False, "no_html": False, "no_report": False,
            "no_probe": False, "no_normalize": False, "draft": False, "no_cache": False,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_main_cli_blur_negative_uses_default(self, capsys):
        """Invalid negative blur must leave blur_threshold unchanged."""
        from main import apply_defaults, apply_cli_overrides

        cfg = apply_defaults({})
        original = cfg["filter"]["blur_threshold"]
        apply_cli_overrides(cfg, self._args(blur=-1))

        assert cfg["filter"]["blur_threshold"] == original
        assert "ERROR" in capsys.readouterr().out

    def test_main_cli_blur_zero_is_valid(self):
        """Blur threshold 0 is a valid edge and must be applied."""
        from main import apply_defaults, apply_cli_overrides

        cfg = apply_defaults({})
        apply_cli_overrides(cfg, self._args(blur=0))

        assert cfg["filter"]["blur_threshold"] == 0

    @pytest.mark.parametrize("sim", [0.0, 1.0, -0.1, 1.1])
    def test_main_cli_sim_invalid_boundaries_use_default(self, sim, capsys):
        """Similarity outside the exclusive (0,1) range must be rejected."""
        from main import apply_defaults, apply_cli_overrides

        cfg = apply_defaults({})
        original = cfg["filter"]["similarity_threshold"]
        apply_cli_overrides(cfg, self._args(sim=sim))

        assert cfg["filter"]["similarity_threshold"] == original
        assert "ERROR" in capsys.readouterr().out

    @pytest.mark.parametrize("sim", [0.001, 0.999])
    def test_main_cli_sim_near_edge_valid(self, sim):
        """Similarity values just inside the valid range must be applied."""
        from main import apply_defaults, apply_cli_overrides

        cfg = apply_defaults({})
        apply_cli_overrides(cfg, self._args(sim=sim))

        assert cfg["filter"]["similarity_threshold"] == sim

    @pytest.mark.parametrize("top", [0, -5])
    def test_main_cli_top_invalid_disables_scorer(self, top, capsys):
        """Invalid top-N values must not enable the scorer."""
        from main import apply_defaults, apply_cli_overrides

        cfg = apply_defaults({})
        assert cfg["scorer"]["enabled"] is False

        apply_cli_overrides(cfg, self._args(top=top))

        assert cfg["scorer"]["enabled"] is False
        assert "ERROR" in capsys.readouterr().out

    def test_main_cli_top_positive_enables_scorer(self):
        """Positive top-N must enable scorer and set top_n."""
        from main import apply_defaults, apply_cli_overrides

        cfg = apply_defaults({})
        apply_cli_overrides(cfg, self._args(top=999))

        assert cfg["scorer"]["enabled"] is True
        assert cfg["scorer"]["top_n"] == 999


class TestMainBatchRouting:
    def _base_cfg(self):
        return {
            "extraction": {"mode": "fps", "fps": 5},
            "filter": {"blur_threshold": 80.0, "similarity_threshold": 0.70, "dedup_method": "phash"},
            "scorer": {"enabled": False, "top_n": 30, "save_score_report": True},
            "output": {"keep_raw": False, "generate_html_preview": True, "preview_columns": 5,
                       "report_json": True, "naming_pattern": "{video_name}", "campaign": "", "lang": "", "ratio": ""},
            "hardware": {"encoder": "auto", "enable_hwaccel": False},
            "batch": {"probe_before_run": False},
            "normalize": {"enabled": False},
        }

    def test_main_batch_workers_one_uses_process_batch(self):
        """Batch mode with workers=1 must use process_batch(), not parallel runner."""
        import main

        args = SimpleNamespace(
            list_presets=False, hw_report=False, clear_cache=False,
            queue_list=False, queue_retry=None, queue_remove=None,
            preset=None, queue_add=False, queue_run=False, template=None,
            input=None, url=None, batch="/videos", output="/out",
            mode=None, fps=None, blur=None, sim=None, method=None, top=None,
            keep_raw=False, no_html=False, no_report=False, no_probe=False,
            gen_batch_html=False, clean_empty=False, draft=False,
            no_normalize=False, no_cache=False, workers=1,
            dag=None, dag_workers=1,
        )

        with patch.object(main, "build_parser") as mock_build_parser, \
             patch.object(main, "load_config", return_value={}), \
             patch.object(main, "apply_defaults", return_value=self._base_cfg()), \
             patch.object(main, "apply_cli_overrides", side_effect=lambda cfg, parsed: cfg), \
             patch.object(main, "resolve_encoder") as mock_resolve_encoder, \
             patch.object(main, "get_ffmpeg_hwaccel_args") as mock_get_hwargs, \
             patch.object(main, "process_batch") as mock_process_batch, \
             patch("parallel_runner.run_parallel_batch") as mock_parallel, \
             patch.object(main, "QueueManager"), \
             patch("main.os.path.isdir", side_effect=lambda p: p in {"/videos", "/out"}):
            mock_build_parser.return_value.parse_args.return_value = args
            main.main()

        mock_process_batch.assert_called_once()
        mock_parallel.assert_not_called()
        mock_resolve_encoder.assert_not_called()
        mock_get_hwargs.assert_not_called()

    def test_main_input_directory_workers_two_uses_parallel_batch(self):
        """An input path that is a directory with workers>1 must route to parallel batch mode."""
        import main

        args = SimpleNamespace(
            list_presets=False, hw_report=False, clear_cache=False,
            queue_list=False, queue_retry=None, queue_remove=None,
            preset=None, queue_add=False, queue_run=False, template=None,
            input="/videos_symlink_dir", url=None, batch=None, output="/out",
            mode=None, fps=None, blur=None, sim=None, method=None, top=None,
            keep_raw=False, no_html=False, no_report=False, no_probe=False,
            gen_batch_html=False, clean_empty=False, draft=False,
            no_normalize=False, no_cache=False, workers=2,
            dag=None, dag_workers=1,
        )

        fake_cfg = self._base_cfg()
        fake_results = [{"video_name": "a.mp4", "status": "success", "stats": {"total_raw": 1, "removed_blur": 0, "removed_duplicate": 0, "final_count": 1, "output_dir": "/out/a/unique_frames"}, "error": None}]

        with patch.object(main, "build_parser") as mock_build_parser, \
             patch.object(main, "load_config", return_value={}), \
             patch.object(main, "apply_defaults", return_value=fake_cfg), \
             patch.object(main, "apply_cli_overrides", side_effect=lambda cfg, parsed: cfg), \
             patch.object(main, "resolve_encoder") as mock_resolve_encoder, \
             patch.object(main, "get_ffmpeg_hwaccel_args") as mock_get_hwargs, \
             patch.object(main, "write_batch_summary") as mock_write_summary, \
             patch("main.scan_batch", return_value=(["/videos_symlink_dir/a.mp4"], [])), \
             patch("pathlib.Path.iterdir", return_value=[Path("/videos_symlink_dir/a.mp4")]), \
             patch("parallel_runner.resolve_max_workers", return_value=2), \
             patch("parallel_runner.run_parallel_batch", return_value=fake_results) as mock_parallel, \
             patch.object(main, "QueueManager"), \
             patch("main.os.path.exists", return_value=True), \
             patch("main.os.path.isdir", side_effect=lambda p: p in {"/videos_symlink_dir", "/out"}), \
             patch("main.os.path.isfile", side_effect=lambda p: p == "/videos_symlink_dir/a.mp4"):
            mock_build_parser.return_value.parse_args.return_value = args
            main.main()

        mock_parallel.assert_called_once()
        mock_write_summary.assert_called_once()
        mock_resolve_encoder.assert_not_called()
        mock_get_hwargs.assert_not_called()


# ===========================================================================
# GROUP G -- filters.py: Empty Input & Method Parity
# ===========================================================================

class TestFiltersEmptyAndParity:
    def test_filters_empty_input_returns_zero_stats(self, tmp_path):
        """Empty raw frame list must return zeroed stats and no crash."""
        from filters import run_filter_pipeline

        out_dir = tmp_path / "output"
        cfg = {"blur_threshold": 0.0, "similarity_threshold": 0.70,
               "dedup_method": "phash", "phash_size": 16}

        stats = run_filter_pipeline([], str(out_dir), cfg)

        assert stats["total_raw"] == 0
        assert stats["final_count"] == 0
        assert stats["final_paths"] == []
        assert (out_dir / "unique_frames").is_dir()

    @pytest.mark.parametrize("method", ["phash", "ssim"])
    def test_filters_phash_and_ssim_parity_on_simple_duplicate_set(self, tmp_path, method):
        """pHash and SSIM must make the same keep/remove decision on an obvious duplicate set."""
        import numpy as np
        import cv2
        from filters import run_filter_pipeline

        img_a = np.zeros((96, 96, 3), dtype=np.uint8)
        cv2.rectangle(img_a, (8, 8), (40, 88), (255, 255, 255), -1)
        cv2.circle(img_a, (72, 48), 16, (0, 180, 255), -1)

        img_b = img_a.copy()

        img_c = np.zeros((96, 96, 3), dtype=np.uint8)
        cv2.line(img_c, (8, 8), (88, 88), (255, 255, 255), 6)
        cv2.line(img_c, (88, 8), (8, 88), (255, 255, 255), 6)
        cv2.circle(img_c, (48, 48), 10, (255, 0, 0), -1)

        frame_paths = []
        for idx, img in enumerate([img_a, img_b, img_c]):
            p = tmp_path / f"frame_{idx:04d}.jpg"
            cv2.imwrite(str(p), img)
            frame_paths.append(str(p))

        out_dir = tmp_path / f"out_{method}"
        cfg = {"blur_threshold": 0.0, "similarity_threshold": 0.70,
               "dedup_method": method, "phash_size": 16}

        stats = run_filter_pipeline(frame_paths, str(out_dir), cfg)

        assert stats["final_count"] == 2
        assert stats["removed_duplicate"] == 1


# ===========================================================================
# GROUP H -- cache_manager.py: Missing Keys & Concurrency
# ===========================================================================

class TestCacheMetaAndConcurrency:
    def test_cache_missing_meta_keys_returns_none(self, tmp_path):
        """Meta files missing required keys must behave as cache misses."""
        from cache_manager import CacheManager

        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cm = CacheManager(cache_dir=cache_dir)

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"FAKE")
        cfg = _minimal_cfg()

        key = cm._cache_key(str(video), cfg)
        meta_path = Path(cache_dir) / f"{key}_meta.json"
        meta_path.write_text(json.dumps({"cache_key": key, "video_path": str(video)}), encoding="utf-8")

        result = cm.get_cached_frames(str(video), cfg)
        assert result is None

    def test_cache_concurrent_reads_are_consistent(self, tmp_path):
        """Concurrent read-only cache access from multiple threads must stay consistent."""
        from cache_manager import CacheManager

        cache_dir = str(tmp_path / "cache")
        cm = CacheManager(cache_dir=cache_dir)

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"FAKE_VIDEO")

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        frame_files = []
        for i in range(2):
            f = frames_dir / f"unique_{i:04d}.jpg"
            f.write_bytes(f"FRAME{i}".encode("utf-8"))
            frame_files.append(str(f))

        cfg = _minimal_cfg()
        cm.store_frames(str(video), cfg, frame_files)

        def read_once():
            return cm.get_cached_frames(str(video), cfg)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: read_once(), range(8)))

        assert all(r is not None for r in results)
        normalized = [tuple(r) for r in results]
        assert len(set(normalized)) == 1


# ===========================================================================
# GROUP I -- queue_manager.py: Ordering & Concurrency
# ===========================================================================

class TestQueueOrderingAndConcurrency:
    def test_queue_run_next_processes_oldest_pending_first(self, tmp_path):
        """run_next() must process pending items in created order."""
        qm = _make_qm(tmp_path)
        first = qm.add("/input/a.mp4", "/output")
        second = qm.add("/input/b.mp4", "/output")
        processed = []

        def fake_process(path, out, cfg, batch_index=0):
            processed.append(path)
            return {"final_count": 1}

        result1 = qm.run_next({}, fake_process)
        result2 = qm.run_next({}, fake_process)

        assert result1 == first
        assert result2 == second
        assert processed == ["/input/a.mp4", "/input/b.mp4"]

    def test_queue_concurrent_add_preserves_all_items(self, tmp_path):
        """Concurrent add() calls from multiple threads must not lose queue items."""
        qm = _make_qm(tmp_path)

        def add_one(i):
            return qm.add(f"/input/clip_{i}.mp4", "/output")

        with ThreadPoolExecutor(max_workers=2) as executor:
            ids = list(executor.map(add_one, range(10)))

        items = qm.list_items()
        assert len(items) == 10
        assert len(set(ids)) == 10
        assert all(item.status == "pending" for item in items)


# ===========================================================================
# GROUP J -- app.py: Watch & Preview UX
# ===========================================================================

class TestAppWatchAndPreview:
    def test_app_watch_start_nonexistent_folder_returns_400(self, flask_client):
        """POST /api/watch/start must map daemon ValueError to HTTP 400."""
        client, app_module = flask_client
        app_module.WATCH_DAEMON.start = MagicMock(side_effect=ValueError("Watch folder does not exist: /missing"))

        resp = client.post(
            "/api/watch/start",
            json={"folder": "/missing", "output": "/out"},
            content_type="application/json",
        )

        assert resp.status_code == 400
        assert "Watch folder does not exist" in resp.get_json()["error"]

    def test_app_serve_preview_missing_html_returns_404_help_page(self, flask_client, tmp_path):
        """Serving preview for a completed no-html job must return a helpful 404 HTML page."""
        client, app_module = flask_client
        job_id = "job_no_html"
        video_dir = tmp_path / "clip"
        unique_dir = video_dir / "unique_frames"
        unique_dir.mkdir(parents=True)

        stats = {"output_folder": str(unique_dir)}
        with app_module.JOBS_LOCK:
            app_module.JOBS[job_id] = {"queue": None, "status": "done", "stats": stats}

        try:
            resp = client.get(f"/api/serve-preview/{job_id}")
            body = resp.get_data(as_text=True)
            assert resp.status_code == 404
            assert "Preview file not found" in body
            assert "Skip HTML Preview Generation" in body
        finally:
            with app_module.JOBS_LOCK:
                app_module.JOBS.pop(job_id, None)


# ===========================================================================
# GROUP K -- main.py / app.py: P1 Integration
# ===========================================================================

class TestMainProcessVideoP1Integration:
    def _full_cfg(self):
        return {
            "extraction": {
                "mode": "fps", "fps": 5, "scene_threshold": 27.0,
                "jpeg_quality": 2, "draft": False, "_hwaccel_args": [],
            },
            "filter": {
                "blur_threshold": 80.0, "similarity_threshold": 0.70,
                "dedup_method": "phash", "phash_size": 16,
            },
            "scorer": {"enabled": False, "top_n": 30, "save_score_report": True},
            "output": {
                "keep_raw": False, "generate_html_preview": False,
                "preview_columns": 5, "report_json": True,
                "naming_pattern": "{video_name}", "campaign": "", "lang": "", "ratio": "",
            },
            "hardware": {"encoder": "auto", "enable_hwaccel": False},
            "batch": {"probe_before_run": False},
            "normalize": {"enabled": False},
            "no_cache": False,
        }

    def test_process_video_cache_hit_logs_and_skips_extraction(self, tmp_path, capsys):
        """A cache hit must log the cache message and skip extraction/filter stages."""
        import main

        cfg = self._full_cfg()
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"VIDEO")
        out_dir = tmp_path / "out"
        cache_frame = tmp_path / "cached_0000.jpg"
        cache_frame.write_bytes(b"JPEG")
        materialized = out_dir / "clip" / "unique_frames" / "unique_0000.jpg"

        with patch("main.CacheManager") as mock_cache_cls, \
             patch("main.materialize_cached_frames", return_value=[str(materialized)]) as mock_materialize, \
             patch("main.extract_frames") as mock_extract, \
             patch("main.run_filter_pipeline") as mock_filter, \
             patch("main.save_json_report", return_value=str(out_dir / "clip" / "report.json")), \
             patch("main.print_summary"):
            mock_cache = mock_cache_cls.return_value
            mock_cache.get_cached_frames.return_value = [str(cache_frame)]

            stats = main.process_video(str(video_path), str(out_dir), cfg)

        assert stats["cache_hit"] is True
        assert stats["final_paths"] == [str(materialized)]
        assert "[CACHE] Cache hit" in capsys.readouterr().out
        mock_materialize.assert_called_once()
        mock_extract.assert_not_called()
        mock_filter.assert_not_called()

    def test_process_video_cache_store_permission_error_is_nonfatal(self, tmp_path, capsys):
        """Cache store failures after a successful run must not fail the pipeline."""
        import main

        cfg = self._full_cfg()
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"VIDEO")
        out_dir = tmp_path / "out"

        raw_dir = out_dir / "clip" / "_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_frame = raw_dir / "frame_0000.jpg"
        raw_frame.write_bytes(b"RAW")
        final_path = out_dir / "clip" / "unique_frames" / "unique_0000.jpg"

        fake_stats = {
            "total_raw": 1,
            "after_blur_filter": 1,
            "after_dedup_filter": 1,
            "removed_blur": 0,
            "removed_duplicate": 0,
            "final_count": 1,
            "final_paths": [str(final_path)],
            "output_dir": str(out_dir / "clip" / "unique_frames"),
            "blur_removed_list": [],
            "dup_removed_list": [],
        }

        with patch("main.CacheManager") as mock_cache_cls, \
             patch("main.extract_frames", return_value=[str(raw_frame)]), \
             patch("main.run_filter_pipeline", return_value=fake_stats), \
             patch("main.save_json_report", return_value=str(out_dir / "clip" / "report.json")), \
             patch("main.print_summary"):
            mock_cache = mock_cache_cls.return_value
            mock_cache.get_cached_frames.return_value = None
            mock_cache.store_frames.side_effect = PermissionError("read only")

            stats = main.process_video(str(video_path), str(out_dir), cfg)

        assert stats["final_count"] == 1
        assert "failed to cache frames" in capsys.readouterr().out


class TestBatchSummaryAndUrlResultIntegration:
    def test_load_url_result_stats_finds_downloaded_video_variant(self, tmp_path):
        """URL result lookup must find downloaded_video* variants when exact folder is absent."""
        with patch("watch_daemon.get_daemon", return_value=MagicMock()), \
             patch("queue_manager.QueueManager._load", return_value=None), \
             patch("queue_manager.QueueManager._save", return_value=None):
            from app import load_url_result_stats

        variant_dir = tmp_path / "downloaded_video.f123"
        variant_dir.mkdir(parents=True)
        report_path = variant_dir / "report.json"
        report_path.write_text(json.dumps({"video": "variant", "output_folder": str(variant_dir)}), encoding="utf-8")

        stats = load_url_result_stats(str(tmp_path))
        assert stats["video"] == "variant"

    def test_write_batch_summary_failed_equals_skipped_plus_error(self, tmp_path):
        """Batch summary JSON must keep failed == skipped + error."""
        from main import write_batch_summary

        output_dir = tmp_path / "out"
        preflight_skipped = [
            {
                "index": None,
                "video_name": "bad_preflight.mp4",
                "status": "skipped",
                "total_raw_frames": 0,
                "removed_blurry": 0,
                "removed_duplicate": 0,
                "final_unique_frames": 0,
                "top_n_selected": None,
                "output_folder": "",
                "error": "preflight error",
            }
        ]
        video_results = [
            {
                "index": 1,
                "video_name": "ok.mp4",
                "status": "success",
                "total_raw_frames": 5,
                "removed_blurry": 1,
                "removed_duplicate": 1,
                "final_unique_frames": 3,
                "top_n_selected": None,
                "output_folder": str(output_dir / "ok" / "unique_frames"),
                "error": None,
            },
            {
                "index": 2,
                "video_name": "bad_runtime.mp4",
                "status": "error",
                "total_raw_frames": 0,
                "removed_blurry": 0,
                "removed_duplicate": 0,
                "final_unique_frames": 0,
                "top_n_selected": None,
                "output_folder": "",
                "error": "runtime error",
            },
        ]

        summary_path = write_batch_summary(str(output_dir), "/videos", 3, preflight_skipped, video_results)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["total_videos"] == 3
        assert summary["success"] == 1
        assert summary["skipped"] == 1
        assert summary["error"] == 1
        assert summary["failed"] == summary["skipped"] + summary["error"]
        assert len(summary["video_results"]) == 3
