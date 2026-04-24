"""
tests/test_phase1.py
====================
Unit & integration tests for FfmpegTool Phase 1 features:
  1. Preset System  (preset_loader.py)
  2. HW Detection   (hw_detect.py)
  3. Probe-First    (probe_first.py)
  4. Naming Pattern (main.py: resolve_video_output_name)
  5. CLI smoke test (main.py: --list-presets, --help)
  6. app.py build_command with preset/no_probe flags

Run from repo root:
    python -m pytest tests/test_phase1.py -v
  or without pytest:
    python tests/test_phase1.py
"""

import os
import sys
import json
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── Add repo root to path ──
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# ═══════════════════════════════════════════════════════════════════
# 1. PRESET LOADER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestPresetLoader(unittest.TestCase):
    """Tests for preset_loader.py"""

    def setUp(self):
        from preset_loader import PRESETS_DIR
        self.presets_dir = PRESETS_DIR

    def test_list_presets_returns_list(self):
        """list_presets() must return a list (even if empty)."""
        from preset_loader import list_presets
        result = list_presets()
        self.assertIsInstance(result, list)

    def test_list_presets_finds_shipped_presets(self):
        """All 4 shipped preset files must be discovered."""
        from preset_loader import list_presets
        presets = list_presets()
        stems = {p["file"] for p in presets}
        expected = {"tiktok_pack", "youtube_shorts", "draft_preview", "full_render_hq"}
        self.assertTrue(
            expected.issubset(stems),
            f"Missing presets: {expected - stems}. Found: {stems}"
        )

    def test_list_presets_has_name_and_description(self):
        """Each preset entry must have file, name, description keys."""
        from preset_loader import list_presets
        for preset in list_presets():
            self.assertIn("file",        preset)
            self.assertIn("name",        preset)
            self.assertIn("description", preset)
            self.assertIsInstance(preset["name"], str)
            self.assertTrue(preset["name"], "Preset name must not be empty")

    def test_load_preset_valid(self):
        """load_preset() returns dict for a valid preset name."""
        from preset_loader import load_preset
        data = load_preset("tiktok_pack")
        self.assertIsInstance(data, dict)
        self.assertIn("extraction", data)

    def test_load_preset_not_found_raises(self):
        """load_preset() raises FileNotFoundError for unknown preset."""
        from preset_loader import load_preset
        with self.assertRaises(FileNotFoundError):
            load_preset("nonexistent_preset_xyz")

    def test_load_preset_malformed_json_raises(self):
        """load_preset() raises ValueError for a preset with bad JSON."""
        from preset_loader import load_preset
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily patch PRESETS_DIR
            import preset_loader as pl
            original_dir = pl.PRESETS_DIR
            try:
                pl.PRESETS_DIR = tmpdir
                bad_file = os.path.join(tmpdir, "bad_preset.json")
                with open(bad_file, "w") as f:
                    f.write("{not valid json")
                with self.assertRaises(ValueError):
                    pl.load_preset("bad_preset")
            finally:
                pl.PRESETS_DIR = original_dir

    def test_apply_preset_overrides_cfg(self):
        """apply_preset() merges preset extraction values into base cfg."""
        from preset_loader import apply_preset
        base_cfg = {
            "extraction": {"mode": "fps", "fps": 5, "jpeg_quality": 2},
            "filter":     {"blur_threshold": 80.0},
            "scorer":     {"enabled": False},
            "output":     {},
            "hardware":   {},
            "batch":      {},
        }
        updated = apply_preset(base_cfg, "tiktok_pack")
        # TikTok pack uses scene mode — must override fps default
        self.assertEqual(updated["extraction"]["mode"], "scene")
        # base keys not in preset must still be present
        self.assertIn("jpeg_quality", updated["extraction"])

    def test_apply_preset_bad_name_returns_cfg_unchanged(self):
        """apply_preset() with bad name logs warning and returns base cfg intact."""
        from preset_loader import apply_preset
        base_cfg = {
            "extraction": {"fps": 5},
            "filter": {}, "scorer": {}, "output": {}, "hardware": {}, "batch": {}
        }
        result = apply_preset(base_cfg, "i_do_not_exist")
        self.assertEqual(result["extraction"]["fps"], 5)  # unchanged

    def test_apply_preset_ignores_comment_keys(self):
        """Preset _comment keys must not be applied to cfg sections."""
        from preset_loader import apply_preset
        base_cfg = {
            "extraction": {}, "filter": {}, "scorer": {},
            "output": {}, "hardware": {}, "batch": {}
        }
        # All shipped presets have _comment keys at root — should not set cfg["_comment"]
        apply_preset(base_cfg, "full_render_hq")
        self.assertNotIn("_comment", base_cfg)

    def test_all_shipped_presets_are_valid_json(self):
        """All shipped preset files must parse without error."""
        from preset_loader import load_preset, list_presets
        for p in list_presets():
            try:
                data = load_preset(p["file"])
                self.assertIsInstance(data, dict, f"{p['file']} should be a dict")
            except Exception as e:
                self.fail(f"Preset '{p['file']}' failed to load: {e}")


# ═══════════════════════════════════════════════════════════════════
# 2. HW DETECT TESTS
# ═══════════════════════════════════════════════════════════════════

class TestHwDetect(unittest.TestCase):
    """Tests for hw_detect.py"""

    def test_get_hw_profile_cpu_always_available(self):
        """CPU profile must always be returned and have no hwaccel."""
        from hw_detect import get_hw_profile
        profile = get_hw_profile("cpu")
        self.assertEqual(profile["encoder"], "libx264")
        self.assertIsNone(profile["hwaccel"])

    def test_get_hw_profile_nvenc(self):
        """NVENC profile must specify cuda hwaccel and h264_nvenc encoder."""
        from hw_detect import get_hw_profile
        profile = get_hw_profile("nvenc")
        self.assertEqual(profile["hwaccel"], "cuda")
        self.assertEqual(profile["encoder"], "h264_nvenc")

    def test_get_hw_profile_unknown_falls_back_to_cpu(self):
        """Unknown encoder key must fall back to CPU profile."""
        from hw_detect import get_hw_profile
        profile = get_hw_profile("unknown_encoder_xyz")
        self.assertEqual(profile["encoder"], "libx264")

    def test_get_ffmpeg_hwaccel_args_cpu_is_empty(self):
        """CPU encoder must return empty hwaccel args list."""
        from hw_detect import get_ffmpeg_hwaccel_args
        args = get_ffmpeg_hwaccel_args("cpu")
        self.assertEqual(args, [])

    def test_get_ffmpeg_hwaccel_args_nvenc(self):
        """NVENC must return ['-hwaccel', 'cuda'] args."""
        from hw_detect import get_ffmpeg_hwaccel_args
        args = get_ffmpeg_hwaccel_args("nvenc")
        self.assertEqual(args, ["-hwaccel", "cuda"])

    def test_get_ffmpeg_hwaccel_args_qsv(self):
        """QSV must return ['-hwaccel', 'qsv'] args."""
        from hw_detect import get_ffmpeg_hwaccel_args
        args = get_ffmpeg_hwaccel_args("qsv")
        self.assertEqual(args, ["-hwaccel", "qsv"])

    def test_probe_encoder_cpu_not_in_probe_list(self):
        """probe_encoder('cpu') returns False (cpu not in PROBE_CMDS, not probed)."""
        from hw_detect import probe_encoder
        # 'cpu' is not in _PROBE_CMDS so returns False — that's correct behaviour
        self.assertFalse(probe_encoder("cpu"))

    @patch("hw_detect.probe_encoder")
    def test_detect_best_encoder_falls_back_to_cpu(self, mock_probe):
        """When no HW encoder available, detect_best_encoder returns 'cpu'."""
        mock_probe.return_value = False
        from hw_detect import detect_best_encoder
        result = detect_best_encoder()
        self.assertEqual(result, "cpu")

    @patch("hw_detect.probe_encoder")
    def test_detect_best_encoder_prefers_nvenc(self, mock_probe):
        """When nvenc available, detect_best_encoder returns 'nvenc' first."""
        mock_probe.side_effect = lambda key: key == "nvenc"
        from hw_detect import detect_best_encoder
        result = detect_best_encoder()
        self.assertEqual(result, "nvenc")

    @patch("hw_detect.probe_encoder")
    def test_detect_best_encoder_qsv_when_no_nvenc(self, mock_probe):
        """When nvenc unavailable but qsv available, returns 'qsv'."""
        mock_probe.side_effect = lambda key: key == "qsv"
        from hw_detect import detect_best_encoder
        result = detect_best_encoder()
        self.assertEqual(result, "qsv")

    @patch("hw_detect.detect_best_encoder")
    def test_resolve_encoder_auto(self, mock_detect):
        """resolve_encoder with 'auto' calls detect_best_encoder."""
        mock_detect.return_value = "cpu"
        from hw_detect import resolve_encoder
        result = resolve_encoder({"encoder": "auto"})
        self.assertEqual(result, "cpu")
        mock_detect.assert_called_once()

    @patch("hw_detect.probe_encoder")
    def test_resolve_encoder_explicit_cpu(self, mock_probe):
        """resolve_encoder with explicit 'cpu' skips probe and returns 'cpu'."""
        from hw_detect import resolve_encoder
        result = resolve_encoder({"encoder": "cpu"})
        self.assertEqual(result, "cpu")
        mock_probe.assert_not_called()

    @patch("hw_detect.probe_encoder")
    def test_resolve_encoder_explicit_nvenc_available(self, mock_probe):
        """resolve_encoder with 'nvenc' when probe succeeds returns 'nvenc'."""
        mock_probe.return_value = True
        from hw_detect import resolve_encoder
        result = resolve_encoder({"encoder": "nvenc"})
        self.assertEqual(result, "nvenc")

    @patch("hw_detect.probe_encoder")
    def test_resolve_encoder_explicit_nvenc_unavailable(self, mock_probe):
        """resolve_encoder with 'nvenc' when probe fails falls back to 'cpu'."""
        mock_probe.return_value = False
        from hw_detect import resolve_encoder
        result = resolve_encoder({"encoder": "nvenc"})
        self.assertEqual(result, "cpu")

    def test_resolve_encoder_unknown_key_defaults_cpu(self):
        """resolve_encoder with unknown key falls back to 'cpu'."""
        from hw_detect import resolve_encoder
        result = resolve_encoder({"encoder": "unsupported_encoder"})
        self.assertEqual(result, "cpu")

    @patch("hw_detect.probe_encoder")
    def test_print_hw_report_is_ascii_safe(self, mock_probe):
        """print_hw_report() should print ASCII-safe console output."""
        mock_probe.side_effect = lambda key: key == "qsv"
        from hw_detect import print_hw_report

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_hw_report()

        out = buf.getvalue()
        self.assertIn("Hardware Encoder Availability Report", out)
        self.assertIn("Intel Quick Sync Video", out)
        self.assertIn("Available", out)
        self.assertNotRegex(out, r"[✓✗⚠─—]")


# ═══════════════════════════════════════════════════════════════════
# 3. PROBE FIRST TESTS
# ═══════════════════════════════════════════════════════════════════

class TestProbeFirst(unittest.TestCase):
    """Tests for probe_first.py"""

    def test_probe_nonexistent_file_fails(self):
        """probe_video() on a non-existent file returns ok=False with issue."""
        from probe_first import probe_video
        result = probe_video("/nonexistent/path/video.mp4")
        self.assertFalse(result.ok)
        self.assertTrue(len(result.issues) > 0)
        self.assertIn("not found", result.issues[0].lower())

    def test_probe_result_has_all_fields(self):
        """ProbeResult dataclass must expose all expected fields."""
        from probe_first import ProbeResult
        r = ProbeResult(path="/tmp/test.mp4")
        self.assertFalse(r.ok)
        self.assertIsNone(r.width)
        self.assertIsNone(r.height)
        self.assertIsNone(r.fps)
        self.assertIsNone(r.duration)
        self.assertIsNone(r.video_codec)
        self.assertIsNone(r.audio_codec)
        self.assertEqual(r.issues, [])

    def test_probe_result_to_dict(self):
        """ProbeResult.to_dict() must return a serialisable dict."""
        from probe_first import ProbeResult
        r = ProbeResult(path="/tmp/x.mp4", ok=True, width=1920, height=1080, fps=30.0)
        d = r.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["width"], 1920)
        self.assertEqual(d["fps"],   30.0)
        # must be JSON-serialisable
        json.dumps(d)

    @patch("probe_first.subprocess.run")
    def test_probe_video_valid_json(self, mock_run):
        """probe_video() with valid ffprobe JSON returns ok=True."""
        from probe_first import probe_video
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920, "height": 1080,
                    "r_frame_rate": "30/1",
                    "pix_fmt": "yuv420p",
                }],
                "format": {"duration": "30.5"},
            }),
            stderr="",
        )
        # Use a real temp file; mock stat() to avoid Windows 0-byte flush race
        mock_stat = MagicMock()
        mock_stat.return_value.st_size = 50_000  # pretend 50KB
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        try:
            with patch("probe_first.Path.stat", mock_stat):
                result = probe_video(tmp_path)
            self.assertTrue(result.ok, f"Expected ok=True, issues: {result.issues}")
            self.assertEqual(result.width, 1920)
            self.assertEqual(result.height, 1080)
            self.assertAlmostEqual(result.fps, 30.0)
            self.assertAlmostEqual(result.duration, 30.5)
            self.assertEqual(result.video_codec, "h264")
        finally:
            os.unlink(tmp_path)

    @patch("probe_first.subprocess.run")
    def test_probe_video_no_video_stream(self, mock_run):
        """probe_video() with audio-only file returns ok=False."""
        from probe_first import probe_video
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{"codec_type": "audio", "codec_name": "aac"}],
                "format":  {"duration": "60.0"},
            }),
            stderr="",
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        try:
            result = probe_video(tmp_path)
            self.assertFalse(result.ok)
            self.assertTrue(any("video stream" in i.lower() for i in result.issues))
        finally:
            os.unlink(tmp_path)

    @patch("probe_first.subprocess.run")
    def test_probe_video_very_short_duration(self, mock_run):
        """probe_video() on very short video flags a warning issue."""
        from probe_first import probe_video
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "codec_type": "video", "codec_name": "h264",
                    "width": 640, "height": 480, "r_frame_rate": "30/1",
                    "pix_fmt": "yuv420p",
                }],
                "format": {"duration": "0.1"},
            }),
            stderr="",
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        try:
            result = probe_video(tmp_path)
            self.assertFalse(result.ok)
            self.assertTrue(any("short" in i.lower() or "duration" in i.lower()
                                for i in result.issues))
        finally:
            os.unlink(tmp_path)

    @patch("probe_first.subprocess.run")
    def test_probe_video_ffprobe_timeout(self, mock_run):
        """probe_video() handles TimeoutExpired gracefully."""
        import subprocess
        from probe_first import probe_video
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)
        # Write dummy bytes so the file-size early check passes
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"dummy")
            tmp_path = f.name
        try:
            result = probe_video(tmp_path)
            self.assertFalse(result.ok)
            # Issue message should mention timeout or >30
            self.assertTrue(
                any("timeout" in i.lower() or ">30" in i or "30s" in i
                    for i in result.issues),
                f"Expected timeout message, got: {result.issues}"
            )
        finally:
            os.unlink(tmp_path)

    @patch("probe_first.probe_video")
    def test_scan_batch_separates_ok_and_bad(self, mock_probe):
        """scan_batch() correctly separates ok_files from bad_results."""
        from probe_first import scan_batch, ProbeResult
        mock_probe.side_effect = lambda path, **kw: (
            ProbeResult(path=path, ok=True,  width=1920, height=1080, fps=30.0, duration=10.0)
            if "good" in path else
            ProbeResult(path=path, ok=False, issues=["No video stream found"])
        )
        files = ["/videos/good_a.mp4", "/videos/bad_b.mp4", "/videos/good_c.mp4"]
        ok_files, bad_results = scan_batch(files, verbose=False)

        self.assertEqual(ok_files,    ["/videos/good_a.mp4", "/videos/good_c.mp4"])
        self.assertEqual(len(bad_results), 1)
        self.assertEqual(bad_results[0].path, "/videos/bad_b.mp4")

    @patch("probe_first.probe_video")
    def test_scan_batch_all_ok(self, mock_probe):
        """scan_batch() with all valid files returns empty bad_results."""
        from probe_first import scan_batch, ProbeResult
        mock_probe.return_value = ProbeResult(
            path="x", ok=True, width=1920, height=1080, fps=30.0, duration=10.0
        )
        ok_files, bad_results = scan_batch(["/a.mp4", "/b.mp4"], verbose=False)
        self.assertEqual(len(ok_files),     2)
        self.assertEqual(len(bad_results),  0)

    @patch("probe_first.probe_video")
    def test_scan_batch_all_bad(self, mock_probe):
        """scan_batch() with all bad files returns empty ok_files."""
        from probe_first import scan_batch, ProbeResult
        mock_probe.return_value = ProbeResult(path="x", ok=False, issues=["File not found"])
        ok_files, bad_results = scan_batch(["/a.mp4", "/b.mp4"], verbose=False)
        self.assertEqual(len(ok_files),    0)
        self.assertEqual(len(bad_results), 2)

    @patch("probe_first.probe_video")
    def test_scan_batch_verbose_output_is_ascii_safe(self, mock_probe):
        """scan_batch(verbose=True) should print ASCII-safe console output."""
        from probe_first import scan_batch, ProbeResult
        mock_probe.side_effect = lambda path, **kw: (
            ProbeResult(path=path, ok=True, width=1920, height=1080, fps=30.0, duration=10.0)
            if "good" in path else
            ProbeResult(path=path, ok=False, issues=["No video stream found"])
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            scan_batch(["/videos/good_a.mp4", "/videos/bad_b.mp4"], verbose=True)

        out = buf.getvalue()
        self.assertIn("[PROBE] Pre-flight scan - 2 file(s)", out)
        self.assertIn("OK   good_a.mp4", out)
        self.assertIn("BAD  bad_b.mp4", out)
        self.assertNotRegex(out, r"[✓✗⚠─—]")


# ═══════════════════════════════════════════════════════════════════
# 4. NAMING PATTERN TESTS
# ═══════════════════════════════════════════════════════════════════

class TestNamingPattern(unittest.TestCase):
    """Tests for resolve_video_output_name() in main.py"""

    def _make_cfg(self, pattern="{video_name}", campaign="", lang="", ratio=""):
        return {
            "output": {
                "naming_pattern": pattern,
                "campaign": campaign,
                "lang":     lang,
                "ratio":    ratio,
            }
        }

    def _resolve(self, video_path, cfg, batch_index=0):
        from main import resolve_video_output_name
        return resolve_video_output_name(video_path, cfg, batch_index)

    def test_default_pattern_returns_stem(self):
        """Default pattern '{video_name}' must equal the file stem."""
        cfg = self._make_cfg()
        result = self._resolve("G:/Videos/my_clip.mp4", cfg)
        self.assertEqual(result, "my_clip")

    def test_index_token(self):
        """'{video_name}_{index}' must produce zero-padded index."""
        cfg = self._make_cfg("{video_name}_{index}")
        result = self._resolve("clip.mp4", cfg, batch_index=7)
        self.assertEqual(result, "clip_007")

    def test_campaign_token(self):
        """'{campaign}_{video_name}' uses cfg output.campaign value."""
        cfg = self._make_cfg("{campaign}_{video_name}", campaign="summer_sale")
        result = self._resolve("clip.mp4", cfg)
        self.assertEqual(result, "summer_sale_clip")

    def test_lang_and_ratio_tokens(self):
        """{lang}_{ratio}_{video_name} produces correct compound name."""
        cfg = self._make_cfg("{lang}_{ratio}_{video_name}", lang="vi", ratio="9x16")
        result = self._resolve("ad_clip.mp4", cfg)
        self.assertEqual(result, "vi_9x16_ad_clip")

    def test_date_token_format(self):
        """{date} token must be 8 digits in YYYYMMDD format."""
        import re
        cfg = self._make_cfg("{video_name}_{date}")
        result = self._resolve("clip.mp4", cfg)
        self.assertTrue(re.match(r"clip_\d{8}$", result),
                        f"Expected 'clip_YYYYMMDD', got '{result}'")

    def test_empty_tokens_collapse_underscores(self):
        """Empty campaign token {campaign} must not leave double underscores."""
        cfg = self._make_cfg("{campaign}_{video_name}", campaign="")
        result = self._resolve("clip.mp4", cfg)
        # Should be "clip" not "_clip" or "clip" with leading underscore
        self.assertFalse(result.startswith("_"), f"Unexpected leading underscore: '{result}'")
        self.assertNotIn("__", result)

    def test_fallback_to_stem_on_empty_result(self):
        """If pattern resolves to empty string, fall back to video stem."""
        cfg = self._make_cfg("{campaign}", campaign="")
        result = self._resolve("myclip.mp4", cfg)
        self.assertEqual(result, "myclip")

    def test_windows_path_stem_extraction(self):
        """Works correctly with Windows-style paths."""
        cfg = self._make_cfg()
        result = self._resolve(r"G:\Videos\campaign\final_cut.mp4", cfg)
        self.assertEqual(result, "final_cut")


# ═══════════════════════════════════════════════════════════════════
# 5. CLI SMOKE TESTS
# ═══════════════════════════════════════════════════════════════════

class TestCLISmoke(unittest.TestCase):
    """Smoke tests for new CLI flags — does not call ffmpeg."""

    def test_help_flag(self):
        """python main.py --help must exit 0 and mention --preset."""
        import subprocess
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"  # force UTF-8 so Windows cp1252 doesn't choke on help text
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "main.py"), "--help"],
            capture_output=True, encoding="utf-8", env=env, cwd=REPO_ROOT
        )
        self.assertEqual(result.returncode, 0)
        combined = result.stdout + result.stderr
        self.assertIn("--preset", combined)
        self.assertIn("--list-presets", combined)
        self.assertIn("--no-probe", combined)
        self.assertIn("--hw-report", combined)

    def test_list_presets_flag(self):
        """python main.py --list-presets must exit 0 and show preset names."""
        import subprocess
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "main.py"), "--list-presets"],
            capture_output=True, encoding="utf-8", env=env, cwd=REPO_ROOT
        )
        self.assertEqual(result.returncode, 0, f"STDERR: {result.stderr}")
        self.assertIn("TikTok", result.stdout)
        self.assertIn("Draft", result.stdout)


# ═══════════════════════════════════════════════════════════════════
# 6. APP.PY build_command TESTS
# ═══════════════════════════════════════════════════════════════════

class TestAppBuildCommand(unittest.TestCase):
    """Tests for app.py build_command() preset & no_probe integration."""

    def _build(self, data):
        # Import here to avoid Flask startup overhead
        from app import build_command
        return build_command(data)

    def _base_data(self, **kwargs):
        base = {
            "input": "G:/Videos/clip.mp4",
            "output": "G:/Frames",
            "mode": "fps",
            "fps": 5,
            "blur": 80,
            "sim": 0.70,
            "method": "phash",
        }
        base.update(kwargs)
        return base

    def test_preset_added_to_command(self):
        """build_command with preset key should include --preset flag."""
        cmd = self._build(self._base_data(preset="tiktok_pack"))
        self.assertIn("--preset", cmd)
        self.assertIn("tiktok_pack", cmd)

    def test_no_preset_not_added(self):
        """build_command without preset key must not include --preset."""
        cmd = self._build(self._base_data())
        self.assertNotIn("--preset", cmd)

    def test_empty_preset_not_added(self):
        """build_command with empty string preset must not include --preset."""
        cmd = self._build(self._base_data(preset=""))
        self.assertNotIn("--preset", cmd)

    def test_no_probe_flag(self):
        """build_command with no_probe=True should include --no-probe."""
        cmd = self._build(self._base_data(no_probe=True))
        self.assertIn("--no-probe", cmd)

    def test_no_probe_false_not_added(self):
        """build_command with no_probe=False must not include --no-probe."""
        cmd = self._build(self._base_data(no_probe=False))
        self.assertNotIn("--no-probe", cmd)

    def test_preset_placed_before_input(self):
        """--preset flag must appear before --input in command."""
        cmd = self._build(self._base_data(preset="draft_preview"))
        preset_idx = cmd.index("--preset")
        input_idx  = cmd.index("--input")
        self.assertLess(preset_idx, input_idx,
                        "--preset should come before --input in command list")

    def test_api_presets_endpoint(self):
        """GET /api/presets must return list of presets with 'file' key."""
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        client = flask_app.app.test_client()
        response = client.get("/api/presets")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("presets", data)
        self.assertIsInstance(data["presets"], list)
        self.assertTrue(len(data["presets"]) > 0)
        for p in data["presets"]:
            self.assertIn("file", p)
            self.assertIn("name", p)


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
