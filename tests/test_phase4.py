"""
tests/test_phase4.py
====================
Unit tests for FfmpegTool Phase 4 features:

  1. ParallelRunner      (parallel_runner.py)
  2. Parallel Batch API  (app.py  POST /api/batch/run)
  3. DAGRunner           (dag_runner.py)
  4. DAG API             (app.py  POST /api/dag/run)
  5. System Status API   (app.py  GET  /api/system/status)
  6. CLI integration     (main.py --workers, --dag)

Run from repo root:
    python -m pytest tests/test_phase4.py -v
    python -m pytest tests/ -v --tb=short   (all phases)
"""

import os
import sys
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
# 1. PARALLEL RUNNER UNIT TESTS
# =================================================================

class TestResolveMaxWorkers(unittest.TestCase):
    """Tests for parallel_runner.resolve_max_workers()."""

    def test_returns_1_for_1_requested(self):
        """Requesting 1 worker must always return 1."""
        from parallel_runner import resolve_max_workers
        self.assertEqual(resolve_max_workers(1), 1)

    def test_clamps_to_safe_max(self):
        """Requesting > safe_max must be clamped down."""
        from parallel_runner import resolve_max_workers
        # safe_max = min(cpu//2, 4); even on a 1-core machine must be >= 1
        result = resolve_max_workers(999)
        self.assertGreaterEqual(result, 1)
        self.assertLessEqual(result, 4)

    def test_minimum_is_1(self):
        """resolve_max_workers must always return >= 1."""
        from parallel_runner import resolve_max_workers
        self.assertGreaterEqual(resolve_max_workers(0), 1)

    def test_cpu_half_cap(self):
        """Return must not exceed cpu_count // 2 (or 4, whichever is lower)."""
        from parallel_runner import resolve_max_workers
        cpu = os.cpu_count() or 2
        safe_max = min(cpu // 2, 4)
        safe_max = max(safe_max, 1)
        result = resolve_max_workers(100)
        self.assertLessEqual(result, safe_max)

    def test_respects_requested_when_below_safe_max(self):
        """If requested <= safe_max, return exactly requested (>=1)."""
        from parallel_runner import resolve_max_workers
        cpu = os.cpu_count() or 4
        safe_max = min(cpu // 2, 4)
        safe_max = max(safe_max, 1)
        if safe_max >= 2:
            self.assertEqual(resolve_max_workers(1), 1)


class TestMemoryGuard(unittest.TestCase):
    """Tests for parallel_runner.check_memory_ok() and get_free_ram_mb()."""

    def test_check_memory_ok_high_threshold_false(self):
        """check_memory_ok with impossibly high threshold must return False."""
        from parallel_runner import check_memory_ok, _PSUTIL_AVAILABLE
        if not _PSUTIL_AVAILABLE:
            self.skipTest("psutil not installed -- memory guard tests skipped")
        # 999 TB threshold => must fail on any real machine
        self.assertFalse(check_memory_ok(999_000_000))

    def test_check_memory_ok_zero_threshold_true(self):
        """check_memory_ok with 0 MB threshold must always return True."""
        from parallel_runner import check_memory_ok
        self.assertTrue(check_memory_ok(0))

    def test_get_free_ram_positive_or_none(self):
        """get_free_ram_mb must return a positive int or None."""
        from parallel_runner import get_free_ram_mb, _PSUTIL_AVAILABLE
        result = get_free_ram_mb()
        if _PSUTIL_AVAILABLE:
            self.assertIsInstance(result, int)
            self.assertGreater(result, 0)
        else:
            self.assertIsNone(result)

    def test_check_memory_ok_without_psutil(self):
        """check_memory_ok must return True if psutil not available."""
        with patch("parallel_runner._PSUTIL_AVAILABLE", False):
            from parallel_runner import check_memory_ok
            self.assertTrue(check_memory_ok(threshold_mb=999_000_000))


class TestRunParallelBatchSequential(unittest.TestCase):
    """Tests for run_parallel_batch() with workers=1 (sequential path)."""

    def _make_process_video(self, frames_count=3):
        """Return a mock process_video function that returns fake stats."""
        def _pv(video_path, output_dir, cfg, batch_index=0):
            return {
                "total_raw":         frames_count,
                "removed_blur":      0,
                "removed_duplicate": 0,
                "final_count":       frames_count,
                "final_paths":       [],
                "output_dir":        output_dir,
            }
        return _pv

    def test_empty_list_returns_empty(self):
        """run_parallel_batch on empty file list must return []."""
        from parallel_runner import run_parallel_batch
        result = run_parallel_batch([], "/out", {}, workers=1)
        self.assertEqual(result, [])

    def test_single_video_sequential_success(self):
        """run_parallel_batch(workers=1) must process 1 video successfully."""
        from parallel_runner import run_parallel_batch
        with tempfile.TemporaryDirectory() as d:
            video = os.path.join(d, "clip.mp4")
            Path(video).write_bytes(b"FAKE")
            fake_stats = {"final_count": 5, "output_dir": d}
            with patch("parallel_runner._run_sequential") as mock_seq:
                mock_seq.return_value = [{
                    "video": video, "video_name": "clip.mp4",
                    "stats": fake_stats, "status": "success", "error": None,
                }]
                result = run_parallel_batch([video], d, {}, workers=1)
                mock_seq.assert_called_once()
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["status"], "success")

    def test_process_video_exception_gives_error_status(self):
        """If process_video raises, result status must be 'error'."""
        import parallel_runner as pr
        import main as m_mod
        with tempfile.TemporaryDirectory() as d:
            video = os.path.join(d, "bad.mp4")
            Path(video).write_bytes(b"FAKE")
            orig_pv = m_mod.process_video
            try:
                m_mod.process_video = MagicMock(side_effect=RuntimeError("boom"))
                results = pr._run_sequential([video], d, {})
                self.assertEqual(results[0]["status"], "error")
                self.assertIn("boom", results[0]["error"])
            finally:
                m_mod.process_video = orig_pv

    def test_process_video_empty_stats_gives_skipped_status(self):
        """If process_video returns {}, status must be 'skipped'."""
        import parallel_runner as pr
        import main as m_mod
        with tempfile.TemporaryDirectory() as d:
            video = os.path.join(d, "empty.mp4")
            Path(video).write_bytes(b"FAKE")
            orig_pv = m_mod.process_video
            try:
                m_mod.process_video = MagicMock(return_value={})
                results = pr._run_sequential([video], d, {})
                self.assertEqual(results[0]["status"], "skipped")
            finally:
                m_mod.process_video = orig_pv

    def test_progress_callback_called_per_video(self):
        """progress_callback must be called once per video result."""
        import parallel_runner as pr
        import main as m_mod
        with tempfile.TemporaryDirectory() as d:
            v1 = os.path.join(d, "a.mp4")
            v2 = os.path.join(d, "b.mp4")
            Path(v1).write_bytes(b"F")
            Path(v2).write_bytes(b"F")
            orig_pv = m_mod.process_video
            callbacks = []
            try:
                m_mod.process_video = MagicMock(return_value={"final_count": 1})
                pr._run_sequential([v1, v2], d, {}, progress_callback=callbacks.append)
                self.assertEqual(len(callbacks), 2)
            finally:
                m_mod.process_video = orig_pv

    def test_results_in_original_order(self):
        """Results must be in the same order as input video_files."""
        import parallel_runner as pr
        import main as m_mod
        with tempfile.TemporaryDirectory() as d:
            videos = []
            for name in ["z.mp4", "a.mp4", "m.mp4"]:
                p = os.path.join(d, name)
                Path(p).write_bytes(b"F")
                videos.append(p)
            orig_pv = m_mod.process_video
            try:
                m_mod.process_video = MagicMock(return_value={"final_count": 1})
                results = pr._run_sequential(videos, d, {})
                result_videos = [r["video"] for r in results]
                self.assertEqual(result_videos, videos)
            finally:
                m_mod.process_video = orig_pv


class TestParallelRunnerWorkerFunction(unittest.TestCase):
    """Tests for the _worker() top-level function."""

    def test_worker_returns_error_dict_on_exception(self):
        """_worker must catch exceptions and return an error dict."""
        from parallel_runner import _worker
        with tempfile.TemporaryDirectory() as d:
            # Pass a path that won't exist as video -- process_video will raise
            with patch("main.process_video", side_effect=ValueError("test error")):
                result = _worker("/nonexistent/video.mp4", d, {}, 1)
                self.assertIsInstance(result, dict)
                self.assertIn("error", result)
                self.assertIsNotNone(result["error"])

    def test_worker_structure_on_success(self):
        """_worker must return {video, stats, error} on success."""
        from parallel_runner import _worker
        fake_stats = {"final_count": 3}
        with tempfile.TemporaryDirectory() as d:
            video = os.path.join(d, "clip.mp4")
            Path(video).write_bytes(b"FAKE")
            with patch("main.process_video", return_value=fake_stats):
                result = _worker(video, d, {}, 1)
                self.assertEqual(result["video"], video)
                self.assertEqual(result["stats"], fake_stats)
                self.assertIsNone(result["error"])


class TestRunParallelFolder(unittest.TestCase):
    """Tests for run_parallel_folder() convenience function."""

    def test_empty_folder_returns_empty(self):
        """run_parallel_folder on folder with no videos must return []."""
        from parallel_runner import run_parallel_folder
        with tempfile.TemporaryDirectory() as d:
            # Put only non-video files
            Path(os.path.join(d, "readme.txt")).write_bytes(b"hi")
            result = run_parallel_folder(d, d, {}, workers=1)
            self.assertEqual(result, [])

    def test_picks_up_video_extensions(self):
        """run_parallel_folder must find .mp4, .mov, etc."""
        from parallel_runner import run_parallel_folder
        import main as m_mod
        with tempfile.TemporaryDirectory() as d:
            for name in ["a.mp4", "b.mov", "c.mkv", "ignore.txt"]:
                Path(os.path.join(d, name)).write_bytes(b"FAKE")
            orig_pv = m_mod.process_video
            try:
                m_mod.process_video = MagicMock(return_value={"final_count": 1})
                results = run_parallel_folder(d, d, {}, workers=1)
                self.assertEqual(len(results), 3)  # 3 video exts, not .txt
            finally:
                m_mod.process_video = orig_pv


# =================================================================
# 2. PARALLEL BATCH API TESTS
# =================================================================

class TestParallelBatchAPI(unittest.TestCase):
    """Tests for POST /api/batch/run in app.py."""

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()
        self._tmpdir = tempfile.mkdtemp()
        self._app = flask_app

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_missing_input_returns_400(self):
        """POST /api/batch/run without input must return 400."""
        resp = self.client.post(
            "/api/batch/run",
            json={"output": self._tmpdir},
        )
        self.assertEqual(resp.status_code, 400)

    def test_missing_output_returns_400(self):
        """POST /api/batch/run without output must return 400."""
        resp = self.client.post(
            "/api/batch/run",
            json={"input": self._tmpdir},
        )
        self.assertEqual(resp.status_code, 400)

    def test_nonexistent_input_folder_returns_400(self):
        """POST /api/batch/run with non-existent input folder must return 400."""
        resp = self.client.post(
            "/api/batch/run",
            json={"input": "/nonexistent_folder_xyz999", "output": self._tmpdir},
        )
        self.assertEqual(resp.status_code, 400)

    def test_empty_folder_returns_no_started(self):
        """POST /api/batch/run on folder with no videos must report no videos."""
        resp = self.client.post(
            "/api/batch/run",
            json={"input": self._tmpdir, "output": self._tmpdir},
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertFalse(data.get("started", True))
        self.assertEqual(data.get("video_count", 0), 0)

    def test_valid_folder_starts_processing(self):
        """POST /api/batch/run with valid folder must return started=true."""
        # Create dummy .mp4 files
        for n in ["a.mp4", "b.mp4"]:
            Path(os.path.join(self._tmpdir, n)).write_bytes(b"FAKE")

        with patch("parallel_runner.run_parallel_batch") as mock_prb:
            mock_prb.return_value = []
            resp = self.client.post(
                "/api/batch/run",
                json={"input": self._tmpdir, "output": self._tmpdir, "workers": 1},
            )
            self.assertEqual(resp.status_code, 200)
            data = json.loads(resp.data)
            self.assertTrue(data.get("started"))
            self.assertEqual(data.get("video_count"), 2)

    def test_parallel_flag_sets_workers(self):
        """POST /api/batch/run with parallel=true must report parallel=true."""
        for n in ["x.mp4"]:
            Path(os.path.join(self._tmpdir, n)).write_bytes(b"FAKE")

        with patch("parallel_runner.run_parallel_batch") as mock_prb:
            mock_prb.return_value = []
            resp = self.client.post(
                "/api/batch/run",
                json={
                    "input": self._tmpdir, "output": self._tmpdir,
                    "parallel": True, "workers": 2,
                },
            )
            self.assertEqual(resp.status_code, 200)
            data = json.loads(resp.data)
            self.assertTrue(data.get("parallel"))

    def test_no_json_body_returns_400_or_415(self):
        """POST /api/batch/run with no JSON body must return 400 or 415."""
        resp = self.client.post("/api/batch/run", data="not json",
                                content_type="text/plain")
        self.assertIn(resp.status_code, [400, 415])


# =================================================================
# 3. DAG RUNNER UNIT TESTS
# =================================================================

class TestLoadDagSpec(unittest.TestCase):
    """Tests for dag_runner.load_dag_spec() and validate_dag_spec()."""

    def _valid_spec(self, source_path):
        return {
            "source": source_path,
            "output": "/out",
            "branches": [
                {"preset": "tiktok_pack"},
                {"preset": "youtube_shorts"},
            ],
        }

    def test_load_from_file(self):
        """load_dag_spec must load a valid JSON file."""
        from dag_runner import load_dag_spec
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "clip.mp4")
            Path(src).touch()
            spec = self._valid_spec(src)
            spec_file = os.path.join(d, "dag.json")
            with open(spec_file, "w") as f:
                json.dump(spec, f)
            result = load_dag_spec(spec_file)
            self.assertEqual(result["source"], src)
            self.assertEqual(len(result["branches"]), 2)

    def test_load_from_json_string(self):
        """load_dag_spec must parse a raw JSON string."""
        from dag_runner import load_dag_spec
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "clip.mp4")
            Path(src).touch()
            spec = self._valid_spec(src)
            result = load_dag_spec(json.dumps(spec))
            self.assertEqual(result["source"], src)

    def test_missing_file_raises(self):
        """load_dag_spec on non-existent file must raise FileNotFoundError."""
        from dag_runner import load_dag_spec
        with self.assertRaises(FileNotFoundError):
            load_dag_spec("/nonexistent/dag_spec.json")

    def test_missing_source_raises(self):
        """validate_dag_spec without source must raise ValueError."""
        from dag_runner import validate_dag_spec
        with self.assertRaises(ValueError):
            validate_dag_spec({"branches": [{"preset": "tiktok_pack"}]})

    def test_empty_source_raises(self):
        """validate_dag_spec with empty source must raise ValueError."""
        from dag_runner import validate_dag_spec
        with self.assertRaises(ValueError):
            validate_dag_spec({"source": "", "branches": [{"preset": "tiktok_pack"}]})

    def test_missing_branches_raises(self):
        """validate_dag_spec without branches must raise ValueError."""
        from dag_runner import validate_dag_spec
        with self.assertRaises(ValueError):
            validate_dag_spec({"source": "clip.mp4"})

    def test_empty_branches_raises(self):
        """validate_dag_spec with empty branches list must raise ValueError."""
        from dag_runner import validate_dag_spec
        with self.assertRaises(ValueError):
            validate_dag_spec({"source": "clip.mp4", "branches": []})

    def test_branch_without_preset_raises(self):
        """validate_dag_spec with branch missing preset must raise ValueError."""
        from dag_runner import validate_dag_spec
        with self.assertRaises(ValueError):
            validate_dag_spec({
                "source": "clip.mp4",
                "branches": [{"no_preset": "oops"}],
            })

    def test_valid_spec_returns_defaults(self):
        """validate_dag_spec on valid spec must return normalized dict."""
        from dag_runner import validate_dag_spec
        spec = {
            "source": "clip.mp4",
            "branches": [{"preset": "tiktok_pack"}],
        }
        result = validate_dag_spec(spec)
        self.assertEqual(result["source"], "clip.mp4")
        self.assertIsNone(result["output"])
        self.assertEqual(len(result["branches"]), 1)

    def test_non_dict_raises(self):
        """validate_dag_spec on non-dict must raise ValueError."""
        from dag_runner import validate_dag_spec
        with self.assertRaises(ValueError):
            validate_dag_spec([{"source": "clip.mp4"}])


class TestRunDag(unittest.TestCase):
    """Tests for dag_runner.run_dag()."""

    def _base_cfg(self):
        return {
            "extraction": {"mode": "fps", "fps": 5, "scene_threshold": 27.0,
                           "jpeg_quality": 2, "_hwaccel_args": []},
            "filter": {"blur_threshold": 80.0, "similarity_threshold": 0.70,
                       "dedup_method": "phash", "phash_size": 16},
            "scorer": {"enabled": False, "top_n": 30, "save_score_report": True},
            "output": {"keep_raw": False, "generate_html_preview": False,
                       "preview_columns": 5, "report_json": False,
                       "naming_pattern": "{video_name}",
                       "campaign": "", "lang": "", "ratio": ""},
            "hardware": {"encoder": "auto", "enable_hwaccel": False,
                         "_resolved_key": "cpu"},
            "batch": {"probe_before_run": False},
            "normalize": {"enabled": False},
            "no_cache": True,
        }

    def test_nonexistent_source_raises(self):
        """run_dag must raise FileNotFoundError for missing source."""
        from dag_runner import run_dag
        spec = {
            "source": "/nonexistent_source_xyz.mp4",
            "output": "/out",
            "branches": [{"preset": "tiktok_pack"}],
        }
        with self.assertRaises(FileNotFoundError):
            run_dag(spec, self._base_cfg(), default_output="/out")

    def test_branch_without_output_raises(self):
        """run_dag with no output anywhere must raise ValueError."""
        from dag_runner import run_dag, validate_dag_spec
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "clip.mp4")
            Path(src).write_bytes(b"FAKE")
            spec = validate_dag_spec({
                "source": src,
                "branches": [{"preset": "tiktok_pack"}],
            })
            # No output in spec, no default_output
            with self.assertRaises(ValueError):
                run_dag(spec, self._base_cfg(), default_output="")

    def test_single_branch_sequential_success(self):
        """run_dag with 1 branch must report success when process_video succeeds."""
        from dag_runner import run_dag, validate_dag_spec
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "clip.mp4")
            Path(src).write_bytes(b"FAKE")
            spec = validate_dag_spec({
                "source": src,
                "output": d,
                "branches": [{"preset": "tiktok_pack"}],
            })
            fake_stats = {"final_count": 3}
            with patch("main.process_video", return_value=fake_stats):
                with patch("preset_loader.apply_preset", return_value=self._base_cfg()):
                    result = run_dag(spec, self._base_cfg(), default_output=d)
            self.assertEqual(result["success"], 1)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(len(result["branches"]), 1)

    def test_multi_branch_all_success(self):
        """run_dag with multiple branches must process all and report success."""
        from dag_runner import run_dag, validate_dag_spec
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "clip.mp4")
            Path(src).write_bytes(b"FAKE")
            spec = validate_dag_spec({
                "source": src,
                "output": d,
                "branches": [
                    {"preset": "tiktok_pack"},
                    {"preset": "youtube_shorts"},
                    {"preset": "draft_preview"},
                ],
            })
            fake_stats = {"final_count": 2}
            with patch("main.process_video", return_value=fake_stats):
                with patch("preset_loader.apply_preset", return_value=self._base_cfg()):
                    result = run_dag(spec, self._base_cfg(), default_output=d)
            self.assertEqual(result["success"], 3)
            self.assertEqual(result["failed"], 0)

    def test_branch_crash_isolated(self):
        """1 branch crash must not abort other branches."""
        from dag_runner import run_dag, validate_dag_spec
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "clip.mp4")
            Path(src).write_bytes(b"FAKE")
            spec = validate_dag_spec({
                "source": src,
                "output": d,
                "branches": [
                    {"preset": "tiktok_pack"},
                    {"preset": "youtube_shorts"},
                ],
            })
            call_count = [0]

            def fake_pv(video_path, out, cfg, batch_index=0):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("first branch crash")
                return {"final_count": 2}

            with patch("main.process_video", side_effect=fake_pv):
                with patch("preset_loader.apply_preset", return_value=self._base_cfg()):
                    result = run_dag(spec, self._base_cfg(), default_output=d)

            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["success"], 1)

    def test_preset_load_failure_marks_branch_error(self):
        """If apply_preset raises, that branch must be marked 'error'."""
        from dag_runner import run_dag, validate_dag_spec
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "clip.mp4")
            Path(src).write_bytes(b"FAKE")
            spec = validate_dag_spec({
                "source": src,
                "output": d,
                "branches": [{"preset": "bad_preset__nonexistent"}],
            })
            with patch("preset_loader.apply_preset",
                       side_effect=ValueError("preset not found")):
                result = run_dag(spec, self._base_cfg(), default_output=d)

            self.assertEqual(result["failed"], 1)
            # Error message must contain something meaningful
            self.assertIsNotNone(result["branches"][0]["error"])
            self.assertGreater(len(result["branches"][0]["error"]), 0)

    def test_result_structure(self):
        """run_dag result must contain source, branches, success, skipped, failed."""
        from dag_runner import run_dag, validate_dag_spec
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "clip.mp4")
            Path(src).write_bytes(b"FAKE")
            spec = validate_dag_spec({
                "source": src,
                "output": d,
                "branches": [{"preset": "tiktok_pack"}],
            })
            with patch("main.process_video", return_value={"final_count": 1}):
                with patch("preset_loader.apply_preset", return_value=self._base_cfg()):
                    result = run_dag(spec, self._base_cfg(), default_output=d)
            for key in ("source", "branches", "success", "skipped", "failed"):
                self.assertIn(key, result)


# =================================================================
# 4. DAG API TESTS
# =================================================================

class TestDagAPIRoutes(unittest.TestCase):
    """Tests for POST /api/dag/run in app.py."""

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_no_body_returns_400_or_415(self):
        """POST /api/dag/run with no body must return 400 or 415."""
        resp = self.client.post("/api/dag/run", data="", content_type="text/plain")
        self.assertIn(resp.status_code, [400, 415])

    def test_invalid_spec_missing_source_returns_400(self):
        """POST /api/dag/run without source must return 400."""
        resp = self.client.post(
            "/api/dag/run",
            json={"branches": [{"preset": "tiktok_pack"}]},
        )
        self.assertEqual(resp.status_code, 400)

    def test_invalid_spec_missing_branches_returns_400(self):
        """POST /api/dag/run without branches must return 400."""
        resp = self.client.post(
            "/api/dag/run",
            json={"source": "clip.mp4"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_nonexistent_source_returns_404(self):
        """POST /api/dag/run with missing source file must return 404."""
        resp = self.client.post(
            "/api/dag/run",
            json={
                "source": "/nonexistent_forever_xyz.mp4",
                "output": self._tmpdir,
                "branches": [{"preset": "tiktok_pack"}],
            },
        )
        self.assertEqual(resp.status_code, 404)

    def test_spec_file_not_found_returns_400(self):
        """POST /api/dag/run with spec_file pointing to missing file must return 400."""
        resp = self.client.post(
            "/api/dag/run",
            json={"spec_file": "/nonexistent/dag.json"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_valid_spec_returns_200(self):
        """POST /api/dag/run with valid spec must return 200 with result."""
        src = os.path.join(self._tmpdir, "clip.mp4")
        Path(src).write_bytes(b"FAKE")
        base_cfg = {
            "extraction": {"mode": "fps", "fps": 5, "scene_threshold": 27.0,
                           "jpeg_quality": 2, "_hwaccel_args": []},
            "filter": {"blur_threshold": 80.0, "similarity_threshold": 0.70,
                       "dedup_method": "phash", "phash_size": 16},
            "scorer": {"enabled": False, "top_n": 30, "save_score_report": True},
            "output": {"keep_raw": False, "generate_html_preview": False,
                       "preview_columns": 5, "report_json": False,
                       "naming_pattern": "{video_name}",
                       "campaign": "", "lang": "", "ratio": ""},
            "hardware": {"encoder": "auto", "enable_hwaccel": False,
                         "_resolved_key": "cpu"},
            "batch": {"probe_before_run": False},
            "normalize": {"enabled": False},
            "no_cache": True,
        }
        with patch("dag_runner.run_dag", return_value={
            "source": src,
            "branches": [{"preset": "tiktok_pack", "output": self._tmpdir,
                          "status": "success", "stats": {}, "error": None}],
            "success": 1, "skipped": 0, "failed": 0,
        }):
            with patch("main.apply_defaults", return_value=base_cfg):
                with patch("main.load_config", return_value=base_cfg):
                    resp = self.client.post(
                        "/api/dag/run",
                        json={
                            "source": src,
                            "output": self._tmpdir,
                            "branches": [{"preset": "tiktok_pack"}],
                        },
                    )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("branches", data)
        self.assertIn("success", data)

    def test_spec_file_mode(self):
        """POST /api/dag/run with spec_file must load and process spec."""
        src = os.path.join(self._tmpdir, "clip.mp4")
        Path(src).write_bytes(b"FAKE")
        spec = {
            "source": src,
            "output": self._tmpdir,
            "branches": [{"preset": "tiktok_pack"}],
        }
        spec_file = os.path.join(self._tmpdir, "dag.json")
        with open(spec_file, "w") as f:
            json.dump(spec, f)

        with patch("dag_runner.run_dag", return_value={
            "source": src, "branches": [], "success": 0, "skipped": 0, "failed": 0,
        }):
            resp = self.client.post(
                "/api/dag/run",
                json={"spec_file": spec_file},
            )
        self.assertEqual(resp.status_code, 200)


# =================================================================
# 5. SYSTEM STATUS API TESTS
# =================================================================

class TestSystemStatusAPI(unittest.TestCase):
    """Tests for GET /api/system/status in app.py."""

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()

    def test_returns_200(self):
        """GET /api/system/status must return 200."""
        resp = self.client.get("/api/system/status")
        self.assertEqual(resp.status_code, 200)

    def test_returns_expected_fields(self):
        """GET /api/system/status must return all required fields."""
        resp = self.client.get("/api/system/status")
        data = json.loads(resp.data)
        for key in ("cpu_count", "max_safe_workers", "psutil_available"):
            self.assertIn(key, data, f"Missing field: {key}")

    def test_cpu_count_positive(self):
        """cpu_count must be a positive integer."""
        resp = self.client.get("/api/system/status")
        data = json.loads(resp.data)
        self.assertIsInstance(data["cpu_count"], int)
        self.assertGreater(data["cpu_count"], 0)

    def test_max_safe_workers_positive(self):
        """max_safe_workers must be at least 1."""
        resp = self.client.get("/api/system/status")
        data = json.loads(resp.data)
        self.assertGreaterEqual(data["max_safe_workers"], 1)

    def test_psutil_available_is_bool(self):
        """psutil_available must be a boolean."""
        resp = self.client.get("/api/system/status")
        data = json.loads(resp.data)
        self.assertIsInstance(data["psutil_available"], bool)

    def test_free_ram_mb_is_int_or_null(self):
        """free_ram_mb must be an int or null."""
        resp = self.client.get("/api/system/status")
        data = json.loads(resp.data)
        val = data.get("free_ram_mb")
        self.assertTrue(val is None or isinstance(val, int))


# =================================================================
# 6. CLI INTEGRATION TESTS
# =================================================================

class TestCLIWorkersFlag(unittest.TestCase):
    """Tests for --workers flag in main.py build_parser()."""

    def _env(self):
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        return env

    def test_workers_default_is_1(self):
        """--workers default must be 1 (backward compat)."""
        from main import build_parser
        p = build_parser()
        args = p.parse_args(["--list-presets"])
        self.assertEqual(args.workers, 1)

    def test_workers_accepts_integer(self):
        """--workers N must store N as integer."""
        from main import build_parser
        p = build_parser()
        args = p.parse_args(["--list-presets", "--workers", "3"])
        self.assertEqual(args.workers, 3)

    def test_list_presets_still_works_with_workers(self):
        """--list-presets combined with --workers must not fail."""
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "main.py"),
             "--list-presets", "--workers", "2"],
            capture_output=True, text=True, cwd=REPO_ROOT,
            env=self._env(),
        )
        self.assertEqual(result.returncode, 0)


class TestCLIDagFlag(unittest.TestCase):
    """Tests for --dag flag in main.py CLI."""

    def _env(self):
        return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

    def test_dag_arg_in_parser(self):
        """--dag must be a recognized flag."""
        from main import build_parser
        p = build_parser()
        args = p.parse_args(["--list-presets", "--dag", "dag.json"])
        self.assertEqual(args.dag, "dag.json")

    def test_dag_workers_default_is_1(self):
        """--dag-workers default must be 1."""
        from main import build_parser
        p = build_parser()
        args = p.parse_args(["--list-presets"])
        self.assertEqual(args.dag_workers, 1)

    def test_dag_nonexistent_file_exits_1(self):
        """--dag with nonexistent file must exit with code 1."""
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "main.py"),
             "--dag", "/nonexistent/dag_spec.json",
             "--output", "/tmp/out_xyz_test"],
            capture_output=True, text=True, cwd=REPO_ROOT,
            env=self._env(),
        )
        self.assertNotEqual(result.returncode, 0)

    def test_dag_invalid_json_exits_1(self):
        """--dag with invalid JSON content must exit with code 1."""
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            bad_file = os.path.join(d, "bad.json")
            with open(bad_file, "w") as f:
                f.write("{NOT VALID JSON")
            result = subprocess.run(
                [sys.executable, os.path.join(REPO_ROOT, "main.py"),
                 "--dag", bad_file, "--output", d],
                capture_output=True, text=True, cwd=REPO_ROOT,
                env=self._env(),
            )
            self.assertNotEqual(result.returncode, 0)


class TestCLIListPresetsStillWorks(unittest.TestCase):
    """Regression: existing flags must still function after Phase 4 changes."""

    def _env(self):
        return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

    def test_list_presets_exits_0(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "main.py"), "--list-presets"],
            capture_output=True, text=True, cwd=REPO_ROOT,
            env=self._env(),
        )
        self.assertEqual(result.returncode, 0)

    def test_hw_report_exits_0(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "main.py"), "--hw-report"],
            capture_output=True, text=True, cwd=REPO_ROOT,
            env=self._env(),
        )
        self.assertEqual(result.returncode, 0)

    def test_clear_cache_exits_0(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "main.py"), "--clear-cache"],
            capture_output=True, text=True, cwd=REPO_ROOT,
            env=self._env(),
        )
        self.assertEqual(result.returncode, 0)


# =================================================================
# 7. EXISTING API ROUTES STILL WORK (Regression)
# =================================================================

class TestExistingAPIRegressions(unittest.TestCase):
    """Ensure Phase 1-3 routes are not broken by Phase 4 changes."""

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()

    def test_presets_route_still_200(self):
        resp = self.client.get("/api/presets")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("presets", data)

    def test_queue_list_route_still_200(self):
        resp = self.client.get("/api/queue")
        self.assertEqual(resp.status_code, 200)

    def test_cache_stats_route_still_200(self):
        resp = self.client.get("/api/cache/stats")
        self.assertEqual(resp.status_code, 200)

    def test_watch_status_route_still_200(self):
        resp = self.client.get("/api/watch/status")
        self.assertEqual(resp.status_code, 200)

    def test_index_route_still_200(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)


# =================================================================
# 8. PSUTIL OPTIONAL DEPENDENCY HANDLING
# =================================================================

class TestPsutilOptional(unittest.TestCase):
    """Ensure parallel_runner degrades gracefully without psutil."""

    def test_check_memory_ok_without_psutil_is_true(self):
        """check_memory_ok always returns True if psutil unavailable."""
        import parallel_runner as pr
        orig = pr._PSUTIL_AVAILABLE
        try:
            pr._PSUTIL_AVAILABLE = False
            self.assertTrue(pr.check_memory_ok(threshold_mb=999_999_999))
        finally:
            pr._PSUTIL_AVAILABLE = orig

    def test_get_free_ram_mb_without_psutil_is_none(self):
        """get_free_ram_mb returns None if psutil unavailable."""
        import parallel_runner as pr
        orig = pr._PSUTIL_AVAILABLE
        try:
            pr._PSUTIL_AVAILABLE = False
            self.assertIsNone(pr.get_free_ram_mb())
        finally:
            pr._PSUTIL_AVAILABLE = orig


# =================================================================
# 9. DAG SPEC JSON ROUND-TRIP
# =================================================================

class TestDagSpecFileIO(unittest.TestCase):
    """Tests for DAG spec file I/O (load + validate cycle)."""

    def test_roundtrip_file_load(self):
        """Write a spec file, load it, validate structure."""
        from dag_runner import load_dag_spec
        with tempfile.TemporaryDirectory() as d:
            spec = {
                "source": os.path.join(d, "clip.mp4"),
                "output": d,
                "branches": [
                    {"preset": "tiktok_pack"},
                    {"preset": "draft_preview"},
                ],
            }
            Path(spec["source"]).touch()
            spec_file = os.path.join(d, "dag.json")
            with open(spec_file, "w") as f:
                json.dump(spec, f)
            result = load_dag_spec(spec_file)
            self.assertEqual(len(result["branches"]), 2)
            self.assertEqual(result["branches"][0]["preset"], "tiktok_pack")
            self.assertEqual(result["branches"][1]["preset"], "draft_preview")

    def test_output_is_optional_in_spec(self):
        """Spec without top-level output must have output=None."""
        from dag_runner import validate_dag_spec
        spec = {
            "source": "clip.mp4",
            "branches": [{"preset": "tiktok_pack"}],
        }
        result = validate_dag_spec(spec)
        self.assertIsNone(result["output"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
