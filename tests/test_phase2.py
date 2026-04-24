"""
tests/test_phase2.py
====================
Unit tests for FfmpegTool Phase 2 features:
  1. QueueManager       (queue_manager.py)
  2. Draft/Preview Mode (extractor.py + build_command)
  3. Error Parser       (error_parser.py)
  4. Normalizer         (normalizer.py)
  5. Queue API Routes   (app.py)

Run from repo root:
    python -m pytest tests/test_phase2.py -v
    python -m pytest tests/ -v --tb=short   (all phases)
"""

import os
import sys
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# ── Add repo root to path ──
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# ═══════════════════════════════════════════════════════════════════
# 1. QUEUE MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestQueueManager(unittest.TestCase):
    """Tests for queue_manager.py — QueueManager class."""

    def _make_qm(self, tmpdir):
        """Create a QueueManager backed by a temp JSON file."""
        from queue_manager import QueueManager
        qfile = os.path.join(tmpdir, "test_queue.json")
        return QueueManager(queue_file=qfile), qfile

    def test_add_creates_pending_item(self):
        """add() must create an item with status='pending'."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out")
            item = qm.get(item_id)
            self.assertIsNotNone(item)
            self.assertEqual(item.status, "pending")
            self.assertEqual(item.input, "/v/clip.mp4")
            self.assertEqual(item.output, "/out")

    def test_add_returns_8char_id(self):
        """add() must return an 8-character ID string."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/a.mp4", "/out")
            self.assertIsInstance(item_id, str)
            self.assertEqual(len(item_id), 8)

    def test_add_persists_to_json(self):
        """add() must write the item to the JSON file immediately."""
        with tempfile.TemporaryDirectory() as d:
            qm, qfile = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out")
            self.assertTrue(os.path.exists(qfile), "queue JSON file not created")
            with open(qfile, "r", encoding="utf-8") as f:
                data = json.load(f)
            ids = [i["id"] for i in data["items"]]
            self.assertIn(item_id, ids)

    def test_list_items_returns_all(self):
        """list_items() must return all added items."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            id1 = qm.add("/v/a.mp4", "/out")
            id2 = qm.add("/v/b.mp4", "/out")
            items = qm.list_items()
            ids = {i.id for i in items}
            self.assertIn(id1, ids)
            self.assertIn(id2, ids)

    def test_get_returns_item(self):
        """get(id) must return the correct QueueItem."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out", cfg_overrides={"extraction": {"fps": 3}})
            item = qm.get(item_id)
            self.assertIsNotNone(item)
            self.assertEqual(item.id, item_id)
            self.assertEqual(item.cfg_overrides["extraction"]["fps"], 3)

    def test_get_nonexistent_returns_none(self):
        """get() with unknown ID must return None."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            self.assertIsNone(qm.get("deadbeef"))

    def test_mark_running_changes_status(self):
        """mark_running() must set status to 'running' and set started_at."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out")
            qm.mark_running(item_id)
            item = qm.get(item_id)
            self.assertEqual(item.status, "running")
            self.assertIsNotNone(item.started_at)

    def test_mark_done_sets_status_and_stats(self):
        """mark_done() must set status='done' and store stats dict."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out")
            qm.mark_running(item_id)
            qm.mark_done(item_id, stats={"final_count": 42})
            item = qm.get(item_id)
            self.assertEqual(item.status, "done")
            self.assertEqual(item.stats["final_count"], 42)
            self.assertIsNotNone(item.finished_at)

    def test_mark_failed_sets_status_and_error(self):
        """mark_failed() must set status='failed' and error string."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out")
            qm.mark_failed(item_id, error="FFmpeg crashed")
            item = qm.get(item_id)
            self.assertEqual(item.status, "failed")
            self.assertEqual(item.error, "FFmpeg crashed")

    def test_retry_resets_failed_to_pending(self):
        """retry() on a failed item must reset it to pending."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out")
            qm.mark_failed(item_id, error="Some error")
            qm.retry(item_id)
            item = qm.get(item_id)
            self.assertEqual(item.status, "pending")
            self.assertIsNone(item.error)
            self.assertIsNone(item.started_at)

    def test_retry_resets_skipped_to_pending(self):
        """retry() on a skipped item must reset it to pending."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out")
            qm.mark_skipped(item_id, reason="No frames")
            qm.retry(item_id)
            self.assertEqual(qm.get(item_id).status, "pending")

    def test_retry_on_done_raises_valueerror(self):
        """retry() on a done item must raise ValueError."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out")
            qm.mark_done(item_id, stats={})
            with self.assertRaises(ValueError):
                qm.retry(item_id)

    def test_retry_nonexistent_raises_keyerror(self):
        """retry() with unknown ID must raise KeyError."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            with self.assertRaises(KeyError):
                qm.retry("deadbeef")

    def test_remove_pending_item(self):
        """remove() on a pending item must return True and delete it."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out")
            result = qm.remove(item_id)
            self.assertTrue(result)
            self.assertIsNone(qm.get(item_id))

    def test_remove_nonexistent_returns_false(self):
        """remove() with unknown ID must return False."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            self.assertFalse(qm.remove("deadbeef"))

    def test_remove_running_raises_valueerror(self):
        """remove() on a running item must raise ValueError."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out")
            qm.mark_running(item_id)
            with self.assertRaises(ValueError):
                qm.remove(item_id)

    def test_persists_across_reinstantiation(self):
        """Queue must reload correctly from JSON on a fresh QueueManager instance."""
        from queue_manager import QueueManager
        with tempfile.TemporaryDirectory() as d:
            qfile = os.path.join(d, "q.json")
            qm1 = QueueManager(queue_file=qfile)
            item_id = qm1.add("/v/clip.mp4", "/out")
            qm1.mark_failed(item_id, error="Oops")

            qm2 = QueueManager(queue_file=qfile)
            item = qm2.get(item_id)
            self.assertIsNotNone(item)
            self.assertEqual(item.status, "failed")
            self.assertEqual(item.error, "Oops")

    def test_running_items_reset_to_pending_on_load(self):
        """Items left as 'running' when the app crashed must be reset to 'pending' on reload."""
        from queue_manager import QueueManager
        with tempfile.TemporaryDirectory() as d:
            qfile = os.path.join(d, "q.json")
            qm1 = QueueManager(queue_file=qfile)
            item_id = qm1.add("/v/clip.mp4", "/out")
            qm1.mark_running(item_id)   # simulate crash mid-processing

            qm2 = QueueManager(queue_file=qfile)
            item = qm2.get(item_id)
            # Should be reset to pending, not stuck in running
            self.assertEqual(item.status, "pending")

    def test_pending_count(self):
        """pending_count() must return correct count of pending items."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            self.assertEqual(qm.pending_count(), 0)
            qm.add("/v/a.mp4", "/out")
            qm.add("/v/b.mp4", "/out")
            self.assertEqual(qm.pending_count(), 2)

    def test_run_next_calls_process_fn(self):
        """run_next() must call the process_video_fn with correct args."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out")

            mock_fn = MagicMock(return_value={"final_count": 5})
            base_cfg = {"extraction": {}, "filter": {}, "scorer": {}, "output": {},
                        "hardware": {}, "batch": {}, "normalize": {}}

            result_id = qm.run_next(base_cfg, mock_fn)

            self.assertEqual(result_id, item_id)
            mock_fn.assert_called_once()
            call_args = mock_fn.call_args
            self.assertEqual(call_args[0][0], "/v/clip.mp4")   # input
            self.assertEqual(call_args[0][1], "/out")           # output

    def test_run_next_marks_done_on_success(self):
        """run_next() must set status='done' when process_fn returns non-empty dict."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out")

            base_cfg = {"extraction": {}, "filter": {}, "scorer": {}, "output": {},
                        "hardware": {}, "batch": {}, "normalize": {}}
            qm.run_next(base_cfg, lambda *a, **kw: {"final_count": 10})

            self.assertEqual(qm.get(item_id).status, "done")

    def test_run_next_marks_failed_on_exception(self):
        """run_next() must set status='failed' when process_fn raises an exception."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            item_id = qm.add("/v/clip.mp4", "/out")

            base_cfg = {"extraction": {}, "filter": {}, "scorer": {}, "output": {},
                        "hardware": {}, "batch": {}, "normalize": {}}

            def boom(*a, **kw):
                raise RuntimeError("FFmpeg exploded")

            qm.run_next(base_cfg, boom)
            item = qm.get(item_id)
            self.assertEqual(item.status, "failed")
            self.assertIn("FFmpeg exploded", item.error)

    def test_run_next_on_empty_queue_returns_none(self):
        """run_next() on an empty queue must return None."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            base_cfg = {"extraction": {}, "filter": {}, "scorer": {}, "output": {},
                        "hardware": {}, "batch": {}, "normalize": {}}
            result = qm.run_next(base_cfg, MagicMock())
            self.assertIsNone(result)

    def test_thread_safety_concurrent_adds(self):
        """Concurrent add() calls must not corrupt the JSON file."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            errors = []

            def add_item(n):
                try:
                    qm.add(f"/v/clip_{n}.mp4", "/out")
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=add_item, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
            self.assertEqual(len(qm.list_items()), 20)

    def test_summary_counts_all_statuses(self):
        """summary() must return correct counts per status."""
        with tempfile.TemporaryDirectory() as d:
            qm, _ = self._make_qm(d)
            id1 = qm.add("/v/a.mp4", "/out")
            id2 = qm.add("/v/b.mp4", "/out")
            id3 = qm.add("/v/c.mp4", "/out")
            qm.mark_done(id1, stats={})
            qm.mark_failed(id2, error="err")
            # id3 stays pending
            s = qm.summary()
            self.assertEqual(s["pending"], 1)
            self.assertEqual(s["done"],    1)
            self.assertEqual(s["failed"],  1)
            self.assertEqual(s["total"],   3)


# ═══════════════════════════════════════════════════════════════════
# 2. DRAFT MODE TESTS
# ═══════════════════════════════════════════════════════════════════

class TestDraftMode(unittest.TestCase):
    """Tests for draft mode in extractor.py and build_command()."""

    @patch("extractor.subprocess.run")
    def test_draft_mode_injects_scale_filter(self, mock_run):
        """extract_by_fps with draft=True must include scale=-2:360 in vf filter."""
        from extractor import extract_by_fps
        mock_run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as d:
            extract_by_fps("/v/clip.mp4", d, fps=5, jpeg_quality=2, draft=True)

        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        self.assertIn("scale=-2:360", vf_value,
                      f"Expected scale=-2:360 in vf filter, got: {vf_value}")
        self.assertIn("fps=5", vf_value)

    @patch("extractor.subprocess.run")
    def test_no_draft_mode_uses_fps_only(self, mock_run):
        """extract_by_fps with draft=False must use plain fps= vf filter."""
        from extractor import extract_by_fps
        mock_run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as d:
            extract_by_fps("/v/clip.mp4", d, fps=5, jpeg_quality=2, draft=False)

        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        self.assertNotIn("scale", vf_value)
        self.assertEqual(vf_value, "fps=5")

    @patch("extractor.subprocess.run")
    def test_extract_frames_passes_draft_flag(self, mock_run):
        """extract_frames() must pass cfg['draft']=True down to extract_by_fps."""
        from extractor import extract_frames
        mock_run.return_value = MagicMock(returncode=0)

        cfg = {"mode": "fps", "fps": 3, "jpeg_quality": 2, "draft": True,
               "_hwaccel_args": []}

        with tempfile.TemporaryDirectory() as d:
            extract_frames("/v/clip.mp4", d, cfg)

        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        self.assertIn("scale=-2:360", vf_value)

    def test_build_command_includes_draft_flag(self):
        """build_command with draft=True must include --draft in command."""
        from app import build_command
        data = {
            "input": "/v/clip.mp4", "output": "/out",
            "mode": "fps", "fps": 5, "blur": 80, "sim": 0.70, "method": "phash",
            "draft": True,
        }
        cmd = build_command(data)
        self.assertIn("--draft", cmd)

    def test_build_command_no_draft_by_default(self):
        """build_command without draft key must NOT include --draft."""
        from app import build_command
        data = {
            "input": "/v/clip.mp4", "output": "/out",
            "mode": "fps", "fps": 5, "blur": 80, "sim": 0.70, "method": "phash",
        }
        cmd = build_command(data)
        self.assertNotIn("--draft", cmd)

    def test_build_command_no_normalize_flag(self):
        """build_command with no_normalize=True must include --no-normalize."""
        from app import build_command
        data = {
            "input": "/v/clip.mp4", "output": "/out",
            "mode": "fps", "fps": 5, "blur": 80, "sim": 0.70, "method": "phash",
            "no_normalize": True,
        }
        cmd = build_command(data)
        self.assertIn("--no-normalize", cmd)


# ═══════════════════════════════════════════════════════════════════
# 3. ERROR PARSER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestErrorParser(unittest.TestCase):
    """Tests for error_parser.py."""

    def _parse(self, stderr):
        from error_parser import parse_ffmpeg_error
        return parse_ffmpeg_error(stderr)

    def test_empty_stderr_returns_unknown(self):
        """Empty stderr string must return UNKNOWN category."""
        from error_parser import ErrorCategory
        result = self._parse("")
        self.assertEqual(result.category, ErrorCategory.UNKNOWN)

    def test_permission_denied(self):
        """'Permission denied' string must map to PERMISSION category."""
        from error_parser import ErrorCategory
        result = self._parse("ffmpeg: error: Permission denied opening /secret/file.mp4")
        self.assertEqual(result.category, ErrorCategory.PERMISSION)

    def test_codec_not_found(self):
        """'Decoder ... not found' must map to CODEC_UNSUPPORTED."""
        from error_parser import ErrorCategory
        result = self._parse("Decoder hevc_cuvid not found in libavcodec.")
        self.assertEqual(result.category, ErrorCategory.CODEC_UNSUPPORTED)

    def test_file_corrupt(self):
        """'moov atom not found' must map to FILE_CORRUPT."""
        from error_parser import ErrorCategory
        result = self._parse("moov atom not found")
        self.assertEqual(result.category, ErrorCategory.FILE_CORRUPT)

    def test_invalid_data(self):
        """'Invalid data found when processing input' must map to FILE_CORRUPT."""
        from error_parser import ErrorCategory
        result = self._parse("Invalid data found when processing input")
        self.assertEqual(result.category, ErrorCategory.FILE_CORRUPT)

    def test_gpu_oom(self):
        """'CUDA_ERROR_OUT_OF_MEMORY' must map to GPU_OOM."""
        from error_parser import ErrorCategory
        result = self._parse("Error code CUDA_ERROR_OUT_OF_MEMORY at line 42")
        self.assertEqual(result.category, ErrorCategory.GPU_OOM)

    def test_disk_full(self):
        """'No space left on device' must map to DISK_FULL."""
        from error_parser import ErrorCategory
        result = self._parse("write error: No space left on device")
        self.assertEqual(result.category, ErrorCategory.DISK_FULL)

    def test_ffmpeg_missing(self):
        """'ffmpeg not found' must map to FFMPEG_MISSING."""
        from error_parser import ErrorCategory
        result = self._parse("ffmpeg not found in PATH")
        self.assertEqual(result.category, ErrorCategory.FFMPEG_MISSING)

    def test_filter_error(self):
        """'Error while filtering' must map to FILTER_ERROR."""
        from error_parser import ErrorCategory
        result = self._parse("Error while filtering: cannot initialize output format")
        self.assertEqual(result.category, ErrorCategory.FILTER_ERROR)

    def test_unknown_error(self):
        """Unrecognized error text must map to UNKNOWN with last line as message."""
        from error_parser import ErrorCategory
        result = self._parse("some completely random unrecognized diagnostic message xyz")
        self.assertEqual(result.category, ErrorCategory.UNKNOWN)

    def test_format_error_returns_nonempty_string(self):
        """format_error() must return a non-empty string for all categories."""
        from error_parser import ErrorCategory, ParsedError, format_error
        for cat in ErrorCategory:
            parsed = ParsedError(category=cat, message="Test message", raw="raw stderr")
            result = format_error(parsed)
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

    def test_format_error_short_format(self):
        """format_error_short() must return single-line string with category value."""
        from error_parser import ErrorCategory, ParsedError, format_error_short
        parsed = ParsedError(
            category=ErrorCategory.CODEC_UNSUPPORTED,
            message="A required codec was not found.",
            raw="stderr"
        )
        result = format_error_short(parsed)
        self.assertIn("codec_unsupported", result)
        self.assertNotIn("\n", result)

    @patch("extractor.subprocess.run")
    def test_extractor_raises_human_readable_on_failure(self, mock_run):
        """extract_by_fps must raise RuntimeError with human-readable message on ffmpeg failure."""
        from extractor import extract_by_fps
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Decoder hevc_cuvid not found in libavcodec."
        )
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError) as ctx:
                extract_by_fps("/v/clip.mp4", d, fps=5, jpeg_quality=2)
        # Must mention codec in error message
        self.assertIn("codec", str(ctx.exception).lower())


# ═══════════════════════════════════════════════════════════════════
# 4. NORMALIZER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestNormalizer(unittest.TestCase):
    """Tests for normalizer.py."""

    def _make_probe(self, codec="h264", pix_fmt="yuv420p", ok=True):
        from probe_first import ProbeResult
        return ProbeResult(
            path="/v/clip.mp4",
            ok=ok,
            width=1920, height=1080, fps=30.0, duration=10.0,
            video_codec=codec,
            pixel_format=pix_fmt,
        )

    def test_needs_normalization_false_for_h264_yuv420p(self):
        """H.264/yuv420p must not require normalization."""
        from normalizer import needs_normalization
        probe = self._make_probe("h264", "yuv420p")
        needed, reason = needs_normalization(probe)
        self.assertFalse(needed)
        self.assertEqual(reason, "")

    def test_needs_normalization_false_for_hevc(self):
        """HEVC/yuv420p must not require normalization."""
        from normalizer import needs_normalization
        probe = self._make_probe("hevc", "yuv420p")
        needed, _ = needs_normalization(probe)
        self.assertFalse(needed)

    def test_needs_normalization_true_for_unknown_codec(self):
        """Unknown codec must require normalization."""
        from normalizer import needs_normalization
        probe = self._make_probe("prores", "yuv420p")
        needed, reason = needs_normalization(probe)
        self.assertTrue(needed)
        self.assertIn("prores", reason.lower())

    def test_needs_normalization_true_for_yuv422p(self):
        """yuv422p pixel format must trigger normalization."""
        from normalizer import needs_normalization
        probe = self._make_probe("h264", "yuv422p")
        needed, reason = needs_normalization(probe)
        self.assertTrue(needed)
        self.assertIn("yuv422p", reason)

    def test_needs_normalization_false_for_bad_probe(self):
        """Probe with ok=False must NOT trigger normalization (let pipeline handle it)."""
        from normalizer import needs_normalization
        probe = self._make_probe(ok=False)
        probe.ok = False
        needed, _ = needs_normalization(probe)
        self.assertFalse(needed)

    @patch("normalizer.subprocess.run")
    def test_normalize_video_transcodes_when_needed(self, mock_run):
        """normalize_video() must call ffmpeg and return a NormalizeResult with was_transcoded=True."""
        mock_run.return_value = MagicMock(returncode=0)
        from normalizer import normalize_video
        probe = self._make_probe("prores", "yuv422p")

        with tempfile.TemporaryDirectory() as d:
            result = normalize_video("/v/clip.mp4", d, cfg={"normalize": {"enabled": True}},
                                     probe=probe)

        self.assertTrue(result.was_transcoded)
        self.assertNotEqual(result.path, "/v/clip.mp4")
        self.assertTrue(result.path.endswith(".mp4"))
        mock_run.assert_called_once()

    @patch("normalizer.subprocess.run")
    def test_normalize_video_no_transcode_for_compatible(self, mock_run):
        """normalize_video() must NOT call ffmpeg if video is already compatible."""
        from normalizer import normalize_video
        probe = self._make_probe("h264", "yuv420p")

        with tempfile.TemporaryDirectory() as d:
            result = normalize_video("/v/clip.mp4", d, cfg={"normalize": {"enabled": True}},
                                     probe=probe)

        self.assertFalse(result.was_transcoded)
        self.assertEqual(result.path, "/v/clip.mp4")
        mock_run.assert_not_called()

    @patch("normalizer.subprocess.run")
    def test_normalize_video_disabled_by_cfg(self, mock_run):
        """normalize_video() with enabled=False must skip normalization entirely."""
        from normalizer import normalize_video
        probe = self._make_probe("prores", "yuv422p")

        with tempfile.TemporaryDirectory() as d:
            result = normalize_video("/v/clip.mp4", d,
                                     cfg={"normalize": {"enabled": False}},
                                     probe=probe)

        self.assertFalse(result.was_transcoded)
        mock_run.assert_not_called()

    def test_normalize_result_cleanup_removes_tmp(self):
        """NormalizeResult.cleanup() must delete the tmp_path file."""
        from normalizer import NormalizeResult
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        try:
            result = NormalizeResult(
                path=tmp_path, was_transcoded=True,
                reason="test", original_path="/v/orig.mp4",
                tmp_path=tmp_path,
            )
            self.assertTrue(os.path.exists(tmp_path))
            result.cleanup()
            self.assertFalse(os.path.exists(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @patch("normalizer.subprocess.run")
    def test_normalize_video_raises_on_ffmpeg_failure(self, mock_run):
        """normalize_video() must raise RuntimeError if ffmpeg fails."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Some ffmpeg encode error"
        )
        from normalizer import normalize_video
        probe = self._make_probe("prores", "yuv422p")

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                normalize_video("/v/clip.mp4", d,
                                cfg={"normalize": {"enabled": True}},
                                probe=probe)


# ═══════════════════════════════════════════════════════════════════
# 5. QUEUE API ROUTE TESTS
# ═══════════════════════════════════════════════════════════════════

class TestQueueAPIRoutes(unittest.TestCase):
    """Tests for queue-related Flask API routes in app.py."""

    def setUp(self):
        """Create a Flask test client with a fresh temp QueueManager."""
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()

        # Replace QUEUE_MGR with a fresh tmpdir-backed instance for isolation
        import tempfile as _tempfile
        self._tmpdir = _tempfile.mkdtemp()
        from queue_manager import QueueManager
        self._orig_qm = flask_app.QUEUE_MGR
        flask_app.QUEUE_MGR = QueueManager(
            queue_file=os.path.join(self._tmpdir, "q.json")
        )
        self._app_module = flask_app

    def tearDown(self):
        self._app_module.QUEUE_MGR = self._orig_qm
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_queue_add_returns_id(self):
        """POST /api/queue/add must return 200 with an item ID."""
        response = self.client.post(
            "/api/queue/add",
            json={"input": "/v/clip.mp4", "output": "/out"},
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("id", data)
        self.assertEqual(data["status"], "pending")

    def test_queue_add_missing_input_returns_400(self):
        """POST /api/queue/add without input must return 400."""
        response = self.client.post(
            "/api/queue/add",
            json={"output": "/out"},
        )
        self.assertEqual(response.status_code, 400)

    def test_queue_add_missing_output_returns_400(self):
        """POST /api/queue/add without output must return 400."""
        response = self.client.post(
            "/api/queue/add",
            json={"input": "/v/clip.mp4"},
        )
        self.assertEqual(response.status_code, 400)

    def test_queue_list_returns_items(self):
        """GET /api/queue must return items list and summary."""
        self.client.post("/api/queue/add", json={"input": "/v/a.mp4", "output": "/out"})
        self.client.post("/api/queue/add", json={"input": "/v/b.mp4", "output": "/out"})

        response = self.client.get("/api/queue")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("items", data)
        self.assertIn("summary", data)
        self.assertEqual(len(data["items"]), 2)

    def test_queue_retry_valid_item(self):
        """POST /api/queue/retry/<id> on failed item must return 200 with pending status."""
        add_resp = self.client.post(
            "/api/queue/add",
            json={"input": "/v/clip.mp4", "output": "/out"},
        )
        item_id = json.loads(add_resp.data)["id"]

        # Manually mark as failed
        self._app_module.QUEUE_MGR.mark_failed(item_id, error="Test failure")

        response = self.client.post(f"/api/queue/retry/{item_id}")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "pending")

    def test_queue_retry_invalid_id_returns_404(self):
        """POST /api/queue/retry/<unknown_id> must return 404."""
        response = self.client.post("/api/queue/retry/deadbeef")
        self.assertEqual(response.status_code, 404)

    def test_queue_delete_item(self):
        """DELETE /api/queue/<id> on pending item must return 200."""
        add_resp = self.client.post(
            "/api/queue/add",
            json={"input": "/v/clip.mp4", "output": "/out"},
        )
        item_id = json.loads(add_resp.data)["id"]

        response = self.client.delete(f"/api/queue/{item_id}")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data["ok"])

    def test_queue_delete_nonexistent_returns_404(self):
        """DELETE /api/queue/<unknown_id> must return 404."""
        response = self.client.delete("/api/queue/deadbeef")
        self.assertEqual(response.status_code, 404)


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
