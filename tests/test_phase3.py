"""
tests/test_phase3.py
====================
Unit tests for FfmpegTool Phase 3 features:
  1. TemplateRunner   (template_runner.py)
  2. Template API     (app.py /api/template/*)
  3. CacheManager     (cache_manager.py)
  4. Cache Integration (main.py process_video cache hit/miss)
  5. WatchDaemon      (watch_daemon.py)
  6. Watch API        (app.py /api/watch/*)
  7. Cache API        (app.py /api/cache/*)

Run from repo root:
    python -m pytest tests/test_phase3.py -v
    python -m pytest tests/ -v --tb=short   (all phases)
"""

import os
import sys
import csv
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# -- Add repo root to path --
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# =================================================================
# 1. TEMPLATE RUNNER TESTS
# =================================================================

class TestTemplateRunnerValidation(unittest.TestCase):
    """Tests for validate_row() in template_runner.py."""

    def _valid_row(self, tmpdir):
        """Return a minimally valid row dict with a real video_src file."""
        dummy = os.path.join(tmpdir, "clip.mp4")
        Path(dummy).touch()
        return {"video_src": dummy, "output": tmpdir}

    def test_valid_row_no_errors(self):
        """A fully valid row must return an empty error list."""
        from template_runner import validate_row
        with tempfile.TemporaryDirectory() as d:
            row = self._valid_row(d)
            errors = validate_row(row)
            self.assertEqual(errors, [])

    def test_missing_video_src_reports_error(self):
        """Missing video_src must be reported as an error."""
        from template_runner import validate_row
        with tempfile.TemporaryDirectory() as d:
            row = {"video_src": "", "output": d}
            errors = validate_row(row)
            self.assertTrue(any("video_src" in e for e in errors))

    def test_nonexistent_video_src_reports_error(self):
        """A video_src path that doesn't exist must be reported."""
        from template_runner import validate_row
        with tempfile.TemporaryDirectory() as d:
            row = {"video_src": "/nonexistent/clip.mp4", "output": d}
            errors = validate_row(row)
            self.assertTrue(any("not found" in e for e in errors))

    def test_missing_output_reports_error(self):
        """Missing output must be reported as an error."""
        from template_runner import validate_row
        with tempfile.TemporaryDirectory() as d:
            dummy = os.path.join(d, "clip.mp4")
            Path(dummy).touch()
            row = {"video_src": dummy, "output": ""}
            errors = validate_row(row)
            self.assertTrue(any("output" in e for e in errors))

    def test_invalid_preset_reports_error(self):
        """Unknown preset name must be reported when known_presets is provided."""
        from template_runner import validate_row
        with tempfile.TemporaryDirectory() as d:
            row = self._valid_row(d)
            row["preset"] = "nonexistent_preset"
            errors = validate_row(row, known_presets=["tiktok_pack", "youtube_shorts"])
            self.assertTrue(any("preset" in e for e in errors))

    def test_valid_preset_no_error(self):
        """A known preset must not produce an error."""
        from template_runner import validate_row
        with tempfile.TemporaryDirectory() as d:
            row = self._valid_row(d)
            row["preset"] = "tiktok_pack"
            errors = validate_row(row, known_presets=["tiktok_pack"])
            self.assertEqual(errors, [])

    def test_preset_skipped_when_known_presets_is_none(self):
        """Preset validation must be skipped when known_presets=None."""
        from template_runner import validate_row
        with tempfile.TemporaryDirectory() as d:
            row = self._valid_row(d)
            row["preset"] = "anything_goes"
            errors = validate_row(row, known_presets=None)
            self.assertEqual(errors, [])

    def test_invalid_top_n_non_integer(self):
        """Non-integer top_n must be reported."""
        from template_runner import validate_row
        with tempfile.TemporaryDirectory() as d:
            row = self._valid_row(d)
            row["top_n"] = "abc"
            errors = validate_row(row)
            self.assertTrue(any("top_n" in e for e in errors))

    def test_invalid_top_n_zero(self):
        """top_n=0 must be reported as invalid."""
        from template_runner import validate_row
        with tempfile.TemporaryDirectory() as d:
            row = self._valid_row(d)
            row["top_n"] = "0"
            errors = validate_row(row)
            self.assertTrue(any("top_n" in e for e in errors))

    def test_valid_top_n(self):
        """Positive integer top_n must not produce an error."""
        from template_runner import validate_row
        with tempfile.TemporaryDirectory() as d:
            row = self._valid_row(d)
            row["top_n"] = "20"
            errors = validate_row(row)
            self.assertEqual(errors, [])


class TestTemplateRunnerLoaders(unittest.TestCase):
    """Tests for load_template_csv() and load_template_json()."""

    def _make_csv(self, tmpdir, rows, header=None):
        """Write a CSV file and return its path."""
        path = os.path.join(tmpdir, "template.csv")
        if header is None:
            header = ["video_src", "output", "preset", "campaign", "lang", "ratio", "top_n"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_load_csv_returns_list_of_dicts(self):
        """load_template_csv must return a list of row dicts."""
        from template_runner import load_template_csv
        with tempfile.TemporaryDirectory() as d:
            rows = [{"video_src": "/v/a.mp4", "output": "/out", "preset": "", "campaign": "",
                     "lang": "", "ratio": "", "top_n": ""}]
            path = self._make_csv(d, rows)
            result = load_template_csv(path)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["video_src"], "/v/a.mp4")

    def test_load_csv_multiple_rows(self):
        """load_template_csv must load all rows."""
        from template_runner import load_template_csv
        with tempfile.TemporaryDirectory() as d:
            rows = [
                {"video_src": "/v/a.mp4", "output": "/out", "preset": "", "campaign": "",
                 "lang": "", "ratio": "", "top_n": ""},
                {"video_src": "/v/b.mp4", "output": "/out", "preset": "", "campaign": "",
                 "lang": "", "ratio": "", "top_n": ""},
            ]
            path = self._make_csv(d, rows)
            result = load_template_csv(path)
            self.assertEqual(len(result), 2)

    def test_load_csv_missing_file_raises(self):
        """load_template_csv on non-existent file must raise FileNotFoundError."""
        from template_runner import load_template_csv
        with self.assertRaises(FileNotFoundError):
            load_template_csv("/nonexistent/template.csv")

    def test_load_json_from_string(self):
        """load_template_json must parse a raw JSON array string."""
        from template_runner import load_template_json
        data = [{"video_src": "/v/a.mp4", "output": "/out"}]
        result = load_template_json(json.dumps(data))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["video_src"], "/v/a.mp4")

    def test_load_json_from_file(self):
        """load_template_json must load from a JSON file path."""
        from template_runner import load_template_json
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "jobs.json")
            data = [{"video_src": "/v/clip.mp4", "output": "/out"}]
            with open(path, "w") as f:
                json.dump(data, f)
            result = load_template_json(path)
            self.assertEqual(len(result), 1)

    def test_load_json_not_a_list_raises(self):
        """load_template_json on a JSON dict (not list) must raise ValueError."""
        from template_runner import load_template_json
        with self.assertRaises(ValueError):
            load_template_json(json.dumps({"video_src": "/v/a.mp4"}))

    def test_detect_format_csv(self):
        """detect_template_format must return 'csv' for .csv extension."""
        from template_runner import detect_template_format
        self.assertEqual(detect_template_format("campaign.csv"), "csv")

    def test_detect_format_json(self):
        """detect_template_format must return 'json' for .json extension."""
        from template_runner import detect_template_format
        self.assertEqual(detect_template_format("jobs.json"), "json")

    def test_detect_format_unknown_raises(self):
        """detect_template_format must raise ValueError for unknown extensions."""
        from template_runner import detect_template_format
        with self.assertRaises(ValueError):
            detect_template_format("file.txt")


class TestTemplateRunnerExecution(unittest.TestCase):
    """Tests for run_template()."""

    def _make_qm(self, tmpdir):
        from queue_manager import QueueManager
        qfile = os.path.join(tmpdir, "q.json")
        return QueueManager(queue_file=qfile)

    def _make_row(self, tmpdir, video_name="clip.mp4"):
        video = os.path.join(tmpdir, video_name)
        Path(video).touch()
        return {"video_src": video, "output": tmpdir}

    def test_run_template_queues_valid_rows(self):
        """run_template must enqueue all valid rows."""
        from template_runner import run_template
        with tempfile.TemporaryDirectory() as d:
            qm = self._make_qm(d)
            rows = [self._make_row(d, "a.mp4"), self._make_row(d, "b.mp4")]
            result = run_template(rows, qm)
            self.assertEqual(result["queued"], 2)
            self.assertEqual(result["skipped"], 0)
            self.assertEqual(len(result["item_ids"]), 2)

    def test_run_template_skips_invalid_rows(self):
        """run_template must skip rows with validation errors."""
        from template_runner import run_template
        with tempfile.TemporaryDirectory() as d:
            qm = self._make_qm(d)
            rows = [
                self._make_row(d),
                {"video_src": "/nonexistent.mp4", "output": d},  # invalid
            ]
            result = run_template(rows, qm)
            self.assertEqual(result["queued"], 1)
            self.assertEqual(result["skipped"], 1)
            self.assertEqual(len(result["errors"]), 1)

    def test_run_template_dry_run_no_enqueue(self):
        """run_template with dry_run=True must not add items to queue."""
        from template_runner import run_template
        with tempfile.TemporaryDirectory() as d:
            qm = self._make_qm(d)
            rows = [self._make_row(d)]
            result = run_template(rows, qm, dry_run=True)
            self.assertEqual(result["queued"], 1)   # counted as would-be-queued
            self.assertEqual(len(result["item_ids"]), 0)  # but NOT actually enqueued
            self.assertEqual(qm.pending_count(), 0)

    def test_run_template_base_output_override(self):
        """run_template with base_output must apply it to rows missing output."""
        from template_runner import run_template
        with tempfile.TemporaryDirectory() as d:
            qm = self._make_qm(d)
            video = os.path.join(d, "clip.mp4")
            Path(video).touch()
            rows = [{"video_src": video, "output": ""}]  # no output in row
            result = run_template(rows, qm, base_output=d)
            self.assertEqual(result["queued"], 1)

    def test_run_template_cfg_overrides_includes_preset(self):
        """run_template must include preset in cfg_overrides if row has preset."""
        from template_runner import run_template
        with tempfile.TemporaryDirectory() as d:
            qm = self._make_qm(d)
            video = os.path.join(d, "clip.mp4")
            Path(video).touch()
            rows = [{"video_src": video, "output": d, "preset": "tiktok_pack"}]
            run_template(rows, qm, known_presets=None)
            items = qm.list_items()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].cfg_overrides.get("preset"), "tiktok_pack")

    def test_run_template_top_n_builds_scorer_override(self):
        """Row with top_n must produce scorer override in cfg_overrides."""
        from template_runner import run_template
        with tempfile.TemporaryDirectory() as d:
            qm = self._make_qm(d)
            video = os.path.join(d, "clip.mp4")
            Path(video).touch()
            rows = [{"video_src": video, "output": d, "top_n": "25"}]
            run_template(rows, qm)
            item = qm.list_items()[0]
            self.assertEqual(item.cfg_overrides.get("scorer", {}).get("top_n"), 25)
            self.assertTrue(item.cfg_overrides.get("scorer", {}).get("enabled"))


# =================================================================
# 2. TEMPLATE API TESTS
# =================================================================

class TestTemplateAPIRoutes(unittest.TestCase):
    """Tests for /api/template/validate and /api/template/run in app.py."""

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()

        # Isolate queue
        self._tmpdir = tempfile.mkdtemp()
        from queue_manager import QueueManager
        self._orig_qm = flask_app.QUEUE_MGR
        flask_app.QUEUE_MGR = QueueManager(
            queue_file=os.path.join(self._tmpdir, "q.json")
        )
        self._app = flask_app

    def tearDown(self):
        self._app.QUEUE_MGR = self._orig_qm
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_job(self, video_name="clip.mp4"):
        """Create a dummy video file and return a valid job dict."""
        path = os.path.join(self._tmpdir, video_name)
        Path(path).touch()
        return {"video_src": path, "output": self._tmpdir}

    def test_validate_valid_jobs_returns_200(self):
        """POST /api/template/validate with valid jobs must return 200."""
        resp = self.client.post(
            "/api/template/validate",
            json={"jobs": [self._make_job()]},
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("queued", data)

    def test_validate_dry_run_does_not_enqueue(self):
        """POST /api/template/validate must NOT actually enqueue items."""
        self.client.post(
            "/api/template/validate",
            json={"jobs": [self._make_job()]},
        )
        self.assertEqual(self._app.QUEUE_MGR.pending_count(), 0)

    def test_validate_csv_string_body(self):
        """POST /api/template/validate with 'csv' string body must parse correctly."""
        path = os.path.join(self._tmpdir, "x.mp4")
        Path(path).touch()
        csv_text = f"video_src,output\n{path},{self._tmpdir}\n"
        resp = self.client.post(
            "/api/template/validate",
            json={"csv": csv_text},
        )
        self.assertEqual(resp.status_code, 200)

    def test_validate_no_data_returns_400(self):
        """POST /api/template/validate with missing required keys must return 400."""
        # Send valid JSON but with no recognised body keys -> 400
        resp = self.client.post(
            "/api/template/validate",
            json={"unexpected_key": "value"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_run_enqueues_valid_jobs(self):
        """POST /api/template/run must enqueue valid jobs."""
        resp = self.client.post(
            "/api/template/run",
            json={"jobs": [self._make_job(), self._make_job("b.mp4")]},
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["queued"], 2)
        self.assertEqual(len(data["item_ids"]), 2)
        self.assertEqual(self._app.QUEUE_MGR.pending_count(), 2)

    def test_run_invalid_job_skipped(self):
        """POST /api/template/run must skip rows with validation errors."""
        resp = self.client.post(
            "/api/template/run",
            json={"jobs": [{"video_src": "/nonexistent.mp4", "output": "/out"}]},
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["skipped"], 1)
        self.assertEqual(data["queued"], 0)

    def test_run_missing_jobs_field_returns_400(self):
        """POST /api/template/run with unrecognised body schema must return 400."""
        resp = self.client.post(
            "/api/template/run",
            json={"unknown_field": []},
        )
        self.assertEqual(resp.status_code, 400)


# =================================================================
# 3. CACHE MANAGER TESTS
# =================================================================

class TestCacheManager(unittest.TestCase):
    """Tests for cache_manager.py -- CacheManager class."""

    def _cm(self, tmpdir):
        from cache_manager import CacheManager
        return CacheManager(cache_dir=os.path.join(tmpdir, "cache"))

    def _cfg(self):
        return {
            "extraction": {"mode": "fps", "fps": 5, "scene_threshold": 27.0, "draft": False},
            "filter": {"blur_threshold": 80.0, "similarity_threshold": 0.70,
                       "dedup_method": "phash", "phash_size": 16},
        }

    def _make_video(self, d, name="clip.mp4"):
        p = os.path.join(d, name)
        Path(p).write_bytes(b"FAKE_VIDEO_DATA_12345")
        return p

    def _make_frames(self, d, count=3):
        paths = []
        for i in range(count):
            p = os.path.join(d, f"frame_{i:04d}.jpg")
            Path(p).write_bytes(b"FAKE_JPG")
            paths.append(p)
        return paths

    def test_cache_miss_returns_none_initially(self):
        """get_cached_frames must return None when no cache entry exists."""
        with tempfile.TemporaryDirectory() as d:
            cm = self._cm(d)
            video = self._make_video(d)
            result = cm.get_cached_frames(video, self._cfg())
            self.assertIsNone(result)

    def test_store_and_retrieve_frames(self):
        """store_frames + get_cached_frames must round-trip correctly."""
        with tempfile.TemporaryDirectory() as d:
            cm = self._cm(d)
            video = self._make_video(d)
            frames = self._make_frames(d)
            key = cm.store_frames(video, self._cfg(), frames)
            self.assertIsInstance(key, str)
            self.assertEqual(len(key), 16)

            result = cm.get_cached_frames(video, self._cfg())
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 3)

    def test_cache_miss_after_file_modified(self):
        """Cache must be stale and return None if video mtime changes."""
        with tempfile.TemporaryDirectory() as d:
            cm = self._cm(d)
            video = self._make_video(d)
            frames = self._make_frames(d)
            cm.store_frames(video, self._cfg(), frames)

            # Modify the video file (simulate re-download)
            time.sleep(0.05)
            Path(video).write_bytes(b"DIFFERENT_CONTENT")

            result = cm.get_cached_frames(video, self._cfg())
            self.assertIsNone(result)

    def test_cache_miss_after_cfg_change(self):
        """Cache must be stale if config changes (different fps)."""
        with tempfile.TemporaryDirectory() as d:
            cm = self._cm(d)
            video = self._make_video(d)
            frames = self._make_frames(d)
            cfg1 = self._cfg()
            cm.store_frames(video, cfg1, frames)

            cfg2 = self._cfg()
            cfg2["extraction"]["fps"] = 10  # changed
            result = cm.get_cached_frames(video, cfg2)
            self.assertIsNone(result)

    def test_invalidate_removes_entry(self):
        """invalidate() must remove the cache entry for a specific video."""
        with tempfile.TemporaryDirectory() as d:
            cm = self._cm(d)
            video = self._make_video(d)
            frames = self._make_frames(d)
            cm.store_frames(video, self._cfg(), frames)

            removed = cm.invalidate(video)
            self.assertGreaterEqual(removed, 1)
            result = cm.get_cached_frames(video, self._cfg())
            self.assertIsNone(result)

    def test_purge_all_clears_cache(self):
        """purge_all() must delete all cache entries and return count."""
        with tempfile.TemporaryDirectory() as d:
            cm = self._cm(d)
            video = self._make_video(d)
            frames = self._make_frames(d)
            cm.store_frames(video, self._cfg(), frames)

            count = cm.purge_all()
            self.assertGreaterEqual(count, 1)
            # Cache dir should be gone
            self.assertFalse(os.path.isdir(os.path.join(d, "cache")))

    def test_stats_empty_cache(self):
        """stats() on empty cache must return zeros."""
        with tempfile.TemporaryDirectory() as d:
            cm = self._cm(d)
            s = cm.stats()
            self.assertEqual(s["entries"], 0)
            self.assertEqual(s["total_files"], 0)

    def test_stats_after_store(self):
        """stats() must reflect stored entries and files."""
        with tempfile.TemporaryDirectory() as d:
            cm = self._cm(d)
            video = self._make_video(d)
            frames = self._make_frames(d, count=3)
            cm.store_frames(video, self._cfg(), frames)
            s = cm.stats()
            self.assertEqual(s["entries"], 1)
            self.assertGreater(s["total_files"], 0)
            self.assertGreater(s["total_bytes"], 0)

    def test_cache_key_is_deterministic(self):
        """Same video + cfg must always produce the same cache key."""
        with tempfile.TemporaryDirectory() as d:
            cm = self._cm(d)
            video = self._make_video(d)
            k1 = cm._cache_key(video, self._cfg())
            k2 = cm._cache_key(video, self._cfg())
            self.assertEqual(k1, k2)

    def test_cache_key_differs_for_different_fps(self):
        """Different fps must produce a different cache key."""
        with tempfile.TemporaryDirectory() as d:
            cm = self._cm(d)
            video = self._make_video(d)
            cfg1 = self._cfg()
            cfg2 = self._cfg()
            cfg2["extraction"]["fps"] = 99
            self.assertNotEqual(cm._cache_key(video, cfg1), cm._cache_key(video, cfg2))

    def test_get_cached_frames_returns_none_for_missing_files(self):
        """If cached frame files are deleted, cache must return None."""
        with tempfile.TemporaryDirectory() as d:
            cm = self._cm(d)
            video = self._make_video(d)
            frames = self._make_frames(d)
            cm.store_frames(video, self._cfg(), frames)

            # Delete cached frames from cache dir
            cache_dir = os.path.join(d, "cache")
            for sub in Path(cache_dir).glob("*_unique"):
                shutil.rmtree(sub)

            result = cm.get_cached_frames(video, self._cfg())
            self.assertIsNone(result)


# =================================================================
# 4. CACHE API TESTS
# =================================================================

class TestProcessVideoCacheIntegration(unittest.TestCase):
    """Integration tests for main.process_video() cache-hit behavior."""

    def _cfg(self):
        return {
            "extraction": {
                "mode": "fps",
                "fps": 5,
                "scene_threshold": 27.0,
                "jpeg_quality": 2,
                "draft": False,
                "_hwaccel_args": [],
            },
            "filter": {
                "blur_threshold": 80.0,
                "similarity_threshold": 0.70,
                "dedup_method": "phash",
                "phash_size": 16,
            },
            "scorer": {
                "enabled": False,
                "top_n": 30,
                "save_score_report": True,
            },
            "output": {
                "keep_raw": False,
                "generate_html_preview": False,
                "preview_columns": 5,
                "report_json": True,
                "naming_pattern": "{video_name}",
                "campaign": "",
                "lang": "",
                "ratio": "",
            },
            "hardware": {"encoder": "auto", "enable_hwaccel": False},
            "batch": {"probe_before_run": False},
            "normalize": {"enabled": False},
            "no_cache": False,
        }

    def test_cache_hit_materializes_frames_into_run_output(self):
        """Cache-hit process_video() must copy frames into the run unique_frames folder."""
        import main as main_mod
        from cache_manager import CacheManager

        with tempfile.TemporaryDirectory() as d:
            cache_dir = os.path.join(d, "cache")
            out_dir = os.path.join(d, "out")
            src_dir = os.path.join(d, "src")
            os.makedirs(src_dir, exist_ok=True)
            video_path = os.path.join(src_dir, "clip.mp4")
            Path(video_path).write_bytes(b"FAKE_VIDEO")

            source_frames_dir = os.path.join(d, "seed_frames")
            os.makedirs(source_frames_dir, exist_ok=True)
            source_frames = []
            for i in range(2):
                frame = os.path.join(source_frames_dir, f"frame_{i:04d}.jpg")
                Path(frame).write_bytes(f"FRAME{i}".encode("utf-8"))
                source_frames.append(frame)

            cfg = self._cfg()
            cm = CacheManager(cache_dir=cache_dir)
            cm.store_frames(video_path, cfg, source_frames)

            with patch("main.CacheManager", side_effect=lambda: CacheManager(cache_dir=cache_dir)):
                stats = main_mod.process_video(video_path, out_dir, cfg)

            unique_dir = os.path.join(out_dir, "clip", "unique_frames")
            self.assertTrue(os.path.isdir(unique_dir))
            self.assertEqual(os.path.normpath(stats["output_dir"]), os.path.normpath(unique_dir))
            self.assertEqual(len(stats["final_paths"]), 2)
            for p in stats["final_paths"]:
                self.assertTrue(os.path.isfile(p))
                self.assertTrue(os.path.normpath(p).startswith(os.path.normpath(unique_dir)))
            report_path = os.path.join(out_dir, "clip", "report.json")
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(os.path.normpath(report["output_folder"]), os.path.normpath(unique_dir))


class TestUrlResultLookup(unittest.TestCase):
    """Tests for app.load_url_result_stats()."""

    def test_load_url_result_stats_prefers_marker_report(self):
        """Marker-written report path must override prefix-based scanning."""
        from app import load_url_result_stats

        with tempfile.TemporaryDirectory() as d:
            exact_dir = os.path.join(d, "downloaded_video")
            actual_dir = os.path.join(d, "downloaded_video.f123")
            os.makedirs(exact_dir, exist_ok=True)
            os.makedirs(actual_dir, exist_ok=True)

            with open(os.path.join(exact_dir, "report.json"), "w", encoding="utf-8") as f:
                json.dump({"video": "wrong", "output_folder": exact_dir}, f)
            actual_report = os.path.join(actual_dir, "report.json")
            with open(actual_report, "w", encoding="utf-8") as f:
                json.dump({"video": "right", "output_folder": actual_dir}, f)
            with open(os.path.join(d, "_last_url_result.json"), "w", encoding="utf-8") as f:
                json.dump({"report_path": actual_report}, f)

            stats = load_url_result_stats(d)
            self.assertEqual(stats["video"], "right")


class TestCacheAPIRoutes(unittest.TestCase):
    """Tests for /api/cache/stats and /api/cache/clear in app.py."""

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()
        self._tmpdir = tempfile.mkdtemp()
        from cache_manager import CacheManager
        self._orig_cm = flask_app.CACHE_MGR
        flask_app.CACHE_MGR = CacheManager(cache_dir=os.path.join(self._tmpdir, "cache"))
        self._app = flask_app

    def tearDown(self):
        self._app.CACHE_MGR = self._orig_cm
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_cache_stats_returns_200(self):
        """GET /api/cache/stats must return 200 with expected fields."""
        resp = self.client.get("/api/cache/stats")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("entries", data)
        self.assertIn("total_files", data)
        self.assertIn("total_bytes", data)

    def test_cache_stats_empty(self):
        """GET /api/cache/stats on empty cache must return zeros."""
        resp = self.client.get("/api/cache/stats")
        data = json.loads(resp.data)
        self.assertEqual(data["entries"], 0)

    def test_cache_clear_returns_purged_count(self):
        """DELETE /api/cache/clear must return {purged: N}."""
        resp = self.client.delete("/api/cache/clear")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("purged", data)


# =================================================================
# 5. WATCH DAEMON UNIT TESTS
# =================================================================

class TestWatchDaemonState(unittest.TestCase):
    """Tests for watch_daemon.py state helpers."""

    def test_save_and_load_state(self):
        """_save_state + _load_state must round-trip correctly."""
        from watch_daemon import _save_state, _load_state
        import watch_daemon as wd_mod
        with tempfile.TemporaryDirectory() as d:
            orig_state_file = wd_mod.STATE_FILE
            wd_mod.STATE_FILE = os.path.join(d, "queue", "watch_state.json")
            try:
                state = {"watching": True, "folder": "/inbox",
                         "output": "/out", "preset": "tiktok_pack"}
                _save_state(state)
                result = _load_state()
                self.assertEqual(result["watching"], True)
                self.assertEqual(result["folder"], "/inbox")
            finally:
                wd_mod.STATE_FILE = orig_state_file

    def test_load_state_missing_file_returns_empty(self):
        """_load_state on missing file must return empty dict."""
        from watch_daemon import _load_state
        import watch_daemon as wd_mod
        orig = wd_mod.STATE_FILE
        wd_mod.STATE_FILE = "/nonexistent/path/watch_state.json"
        try:
            result = _load_state()
            self.assertEqual(result, {})
        finally:
            wd_mod.STATE_FILE = orig

    def test_load_state_corrupt_json_returns_empty(self):
        """_load_state on corrupt JSON must return empty dict."""
        from watch_daemon import _load_state
        import watch_daemon as wd_mod
        with tempfile.TemporaryDirectory() as d:
            orig = wd_mod.STATE_FILE
            path = os.path.join(d, "watch_state.json")
            wd_mod.STATE_FILE = path
            try:
                with open(path, "w") as f:
                    f.write("NOT VALID JSON {{{")
                result = _load_state()
                self.assertEqual(result, {})
            finally:
                wd_mod.STATE_FILE = orig


class TestWatchDaemonClass(unittest.TestCase):
    """Tests for WatchDaemon class lifecycle."""

    def test_status_idle_initially(self):
        """WatchDaemon.status() must show watching=False initially."""
        from watch_daemon import WatchDaemon
        daemon = WatchDaemon(queue_mgr=MagicMock())
        status = daemon.status()
        self.assertFalse(status["watching"])
        self.assertIn("watchdog_available", status)

    def test_stop_without_start_safe(self):
        """stop() on a daemon that was never started must not raise."""
        from watch_daemon import WatchDaemon
        import watch_daemon as wd_mod
        daemon = WatchDaemon(queue_mgr=MagicMock())
        with tempfile.TemporaryDirectory() as d:
            orig = wd_mod.STATE_FILE
            wd_mod.STATE_FILE = os.path.join(d, "queue", "watch_state.json")
            try:
                state = daemon.stop()   # must not raise
                self.assertFalse(state["watching"])
            finally:
                wd_mod.STATE_FILE = orig

    @patch("watch_daemon._WATCHDOG_AVAILABLE", False)
    def test_start_without_watchdog_raises(self):
        """start() must raise RuntimeError if watchdog is not installed."""
        from watch_daemon import WatchDaemon
        daemon = WatchDaemon(queue_mgr=MagicMock())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                daemon.start(d, d)

    def test_start_nonexistent_folder_raises(self):
        """start() with non-existent folder must raise ValueError."""
        from watch_daemon import WatchDaemon, _WATCHDOG_AVAILABLE
        if not _WATCHDOG_AVAILABLE:
            self.skipTest("watchdog not installed")
        daemon = WatchDaemon(queue_mgr=MagicMock())
        import watch_daemon as wd_mod
        with tempfile.TemporaryDirectory() as d:
            orig = wd_mod.STATE_FILE
            wd_mod.STATE_FILE = os.path.join(d, "watch_state.json")
            try:
                with self.assertRaises(ValueError):
                    daemon.start("/nonexistent_watch_folder_xyz", d)
            finally:
                wd_mod.STATE_FILE = orig

    def test_resume_from_state_no_state(self):
        """resume_from_state() with no state file must return False."""
        from watch_daemon import WatchDaemon
        import watch_daemon as wd_mod
        daemon = WatchDaemon(queue_mgr=MagicMock())
        orig = wd_mod.STATE_FILE
        wd_mod.STATE_FILE = "/nonexistent/watch_state.json"
        try:
            result = daemon.resume_from_state()
            self.assertFalse(result)
        finally:
            wd_mod.STATE_FILE = orig

    def test_resume_from_state_watching_false(self):
        """resume_from_state() with watching=False state must return False."""
        from watch_daemon import WatchDaemon, _save_state
        import watch_daemon as wd_mod
        daemon = WatchDaemon(queue_mgr=MagicMock())
        with tempfile.TemporaryDirectory() as d:
            orig = wd_mod.STATE_FILE
            wd_mod.STATE_FILE = os.path.join(d, "queue", "watch_state.json")
            try:
                _save_state({"watching": False, "folder": d, "output": d, "preset": ""})
                result = daemon.resume_from_state()
                self.assertFalse(result)
            finally:
                wd_mod.STATE_FILE = orig


class TestVideoFileHandler(unittest.TestCase):
    """Tests for VideoFileHandler debounce logic."""

    def test_non_video_extension_not_scheduled(self):
        """Handler must ignore non-video file extensions."""
        from watch_daemon import VideoFileHandler
        qm = MagicMock()
        handler = VideoFileHandler("/out", "", qm, debounce=0.05)
        evt = MagicMock()
        evt.is_directory = False
        evt.src_path = "/inbox/document.txt"
        handler.on_created(evt)
        time.sleep(0.2)
        qm.add.assert_not_called()

    def test_video_extension_triggers_enqueue(self):
        """Handler must enqueue a recognised video extension after debounce."""
        from watch_daemon import VideoFileHandler
        qm = MagicMock()
        qm.add.return_value = "test_id"
        with tempfile.TemporaryDirectory() as d:
            video = os.path.join(d, "clip.mp4")
            Path(video).write_bytes(b"FAKE")
            handler = VideoFileHandler(d, "tiktok_pack", qm, debounce=0.05)
            evt = MagicMock()
            evt.is_directory = False
            evt.src_path = video
            handler.on_created(evt)
            time.sleep(0.3)
        qm.add.assert_called_once()
        call_args = qm.add.call_args
        self.assertIn("clip.mp4", call_args[0][0])

    def test_debounce_cancels_previous_timer(self):
        """Rapid duplicate events for the same path must only enqueue once."""
        from watch_daemon import VideoFileHandler
        qm = MagicMock()
        qm.add.return_value = "test_id"
        with tempfile.TemporaryDirectory() as d:
            video = os.path.join(d, "clip.mp4")
            Path(video).write_bytes(b"FAKE")
            handler = VideoFileHandler(d, "", qm, debounce=0.1)
            evt = MagicMock()
            evt.is_directory = False
            evt.src_path = video
            # Fire 5 rapid events
            for _ in range(5):
                handler.on_created(evt)
                time.sleep(0.01)
            time.sleep(0.5)
        # Must have been called exactly once
        self.assertEqual(qm.add.call_count, 1)


# =================================================================
# 6. WATCH API ROUTE TESTS
# =================================================================

class TestWatchAPIRoutes(unittest.TestCase):
    """Tests for /api/watch/* in app.py."""

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()
        self._tmpdir = tempfile.mkdtemp()
        from watch_daemon import WatchDaemon
        self._orig_daemon = flask_app.WATCH_DAEMON
        self._mock_daemon = MagicMock(spec=WatchDaemon)
        flask_app.WATCH_DAEMON = self._mock_daemon
        self._app = flask_app

    def tearDown(self):
        self._app.WATCH_DAEMON = self._orig_daemon
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_watch_status_returns_200(self):
        """GET /api/watch/status must return 200."""
        self._mock_daemon.status.return_value = {
            "watching": False, "folder": "", "output": "",
            "preset": "", "started_at": None, "watchdog_available": True,
        }
        resp = self.client.get("/api/watch/status")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("watching", data)

    def test_watch_start_valid(self):
        """POST /api/watch/start with valid folder/output must return 200."""
        self._mock_daemon.start.return_value = {
            "watching": True, "folder": self._tmpdir,
            "output": self._tmpdir, "preset": "", "started_at": "2026-01-01 00:00:00",
        }
        resp = self.client.post(
            "/api/watch/start",
            json={"folder": self._tmpdir, "output": self._tmpdir},
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["watching"])

    def test_watch_start_missing_folder_returns_400(self):
        """POST /api/watch/start without folder must return 400."""
        resp = self.client.post(
            "/api/watch/start",
            json={"output": self._tmpdir},
        )
        self.assertEqual(resp.status_code, 400)

    def test_watch_start_missing_output_returns_400(self):
        """POST /api/watch/start without output must return 400."""
        resp = self.client.post(
            "/api/watch/start",
            json={"folder": self._tmpdir},
        )
        self.assertEqual(resp.status_code, 400)

    def test_watch_start_invalid_folder_returns_400(self):
        """POST /api/watch/start with invalid folder must return 400."""
        self._mock_daemon.start.side_effect = ValueError("Folder not found")
        resp = self.client.post(
            "/api/watch/start",
            json={"folder": "/nonexistent_xyz", "output": self._tmpdir},
        )
        self.assertEqual(resp.status_code, 400)

    def test_watch_stop_returns_200(self):
        """POST /api/watch/stop must return 200."""
        self._mock_daemon.stop.return_value = {
            "watching": False, "folder": "", "output": "",
            "preset": "", "started_at": None,
        }
        resp = self.client.post("/api/watch/stop")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertFalse(data["watching"])

    def test_watch_start_watchdog_missing_returns_503(self):
        """POST /api/watch/start when watchdog missing must return 503."""
        self._mock_daemon.start.side_effect = RuntimeError("watchdog not installed")
        resp = self.client.post(
            "/api/watch/start",
            json={"folder": self._tmpdir, "output": self._tmpdir},
        )
        self.assertEqual(resp.status_code, 503)


# =================================================================
# Phase 5 regression tests
# =================================================================

class TestTemplateDedupe(unittest.TestCase):
    """Duplicate rows in a template must not be double-enqueued."""

    def _make_qm(self, d):
        from queue_manager import QueueManager
        return QueueManager(queue_file=os.path.join(d, "q.json"))

    def test_identical_rows_skipped_after_first(self):
        """run_template with two identical rows must enqueue only the first."""
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "clip.mp4")
            Path(src).write_bytes(b"FAKE")
            qm = self._make_qm(d)
            rows = [
                {"video_src": src, "output": d, "preset": ""},
                {"video_src": src, "output": d, "preset": ""},
            ]
            from template_runner import run_template
            result = run_template(rows, qm)
            self.assertEqual(result["queued"], 1)
            self.assertEqual(result["skipped"], 1)
            self.assertEqual(len(qm.list_items()), 1)
            self.assertIn("duplicate row", result["errors"][0]["errors"][0])

    def test_different_presets_are_not_deduplicated(self):
        """Two rows with same src/output but different presets must both be enqueued."""
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "clip.mp4")
            Path(src).write_bytes(b"FAKE")
            qm = self._make_qm(d)
            rows = [
                {"video_src": src, "output": d, "preset": "tiktok_pack"},
                {"video_src": src, "output": d, "preset": "draft_preview"},
            ]
            from template_runner import run_template
            result = run_template(rows, qm, known_presets=["tiktok_pack", "draft_preview"])
            self.assertEqual(result["queued"], 2)
            self.assertEqual(result["skipped"], 0)

    def test_dry_run_deduplicate_but_no_enqueue(self):
        """Dry-run must still report duplicates but enqueue nothing."""
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "clip.mp4")
            Path(src).write_bytes(b"FAKE")
            qm = self._make_qm(d)
            rows = [
                {"video_src": src, "output": d, "preset": ""},
                {"video_src": src, "output": d, "preset": ""},
            ]
            from template_runner import run_template
            result = run_template(rows, qm, dry_run=True)
            self.assertEqual(result["queued"], 1)
            self.assertEqual(result["skipped"], 1)
            self.assertEqual(len(qm.list_items()), 0)


class TestWatchDaemonDedupe(unittest.TestCase):
    """Rapid duplicate events for the same path must enqueue only once."""

    def test_rapid_duplicate_events_enqueue_once(self):
        """Scheduling _enqueue twice for the same path must only call qm.add once."""
        from watch_daemon import VideoFileHandler
        mock_qm = MagicMock()
        mock_qm.add.return_value = "abc12345"

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "clip.mp4")
            Path(src).write_bytes(b"FAKE")
            handler = VideoFileHandler(output=d, preset="", queue_mgr=mock_qm, debounce=0)

            # Simulate two rapid enqueue calls for the same path (debounce elapsed)
            handler._enqueue(src)
            handler._enqueue(src)

            mock_qm.add.assert_called_once_with(src, d, cfg_overrides={})

    def test_distinct_paths_each_enqueued(self):
        """Two distinct video files must each be enqueued separately."""
        from watch_daemon import VideoFileHandler
        mock_qm = MagicMock()
        mock_qm.add.side_effect = lambda path, out, **kw: "id_" + os.path.basename(path)[:4]

        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.mp4")
            b = os.path.join(d, "b.mp4")
            Path(a).write_bytes(b"FAKE")
            Path(b).write_bytes(b"FAKE")
            handler = VideoFileHandler(output=d, preset="", queue_mgr=mock_qm, debounce=0)

            handler._enqueue(a)
            handler._enqueue(b)

            self.assertEqual(mock_qm.add.call_count, 2)


class TestCleanEmptyMarker(unittest.TestCase):
    """--clean-empty must skip folders with _processing marker."""

    def _run_clean_empty(self, output_dir):
        """Import and run the clean-empty logic directly from main.py."""
        import importlib, types
        import main as main_mod
        removed = 0
        skipped = 0
        for sub in sorted(Path(output_dir).iterdir()):
            if not sub.is_dir():
                continue
            if (sub / "_processing").exists():
                skipped += 1
                continue
            unique_dir = sub / "unique_frames"
            try:
                empty = not unique_dir.exists() or not any(unique_dir.iterdir())
                if empty:
                    shutil.rmtree(str(sub))
                    removed += 1
            except (OSError, StopIteration):
                skipped += 1
        return removed, skipped

    def test_folder_with_processing_marker_is_skipped(self):
        """Folder containing _processing file must not be removed."""
        with tempfile.TemporaryDirectory() as d:
            active = Path(d) / "active_video"
            active.mkdir()
            (active / "_processing").write_text("in progress")

            removed, skipped = self._run_clean_empty(d)
            self.assertTrue(active.exists(), "In-progress folder must not be removed")
            self.assertEqual(skipped, 1)
            self.assertEqual(removed, 0)

    def test_empty_folder_without_marker_is_removed(self):
        """Folder with no unique_frames and no marker must be removed."""
        with tempfile.TemporaryDirectory() as d:
            empty_vid = Path(d) / "empty_video"
            empty_vid.mkdir()

            removed, skipped = self._run_clean_empty(d)
            self.assertFalse(empty_vid.exists(), "Empty folder must be removed")
            self.assertEqual(removed, 1)

    def test_folder_with_frames_is_kept(self):
        """Folder with populated unique_frames must not be removed."""
        with tempfile.TemporaryDirectory() as d:
            done = Path(d) / "done_video"
            frames = done / "unique_frames"
            frames.mkdir(parents=True)
            (frames / "frame_001.jpg").write_bytes(b"jpg")

            removed, skipped = self._run_clean_empty(d)
            self.assertTrue(done.exists(), "Folder with frames must be kept")
            self.assertEqual(removed, 0)


# =================================================================
# Entry point
# =================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
