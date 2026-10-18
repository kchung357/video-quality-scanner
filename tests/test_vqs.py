"""Unit and integration tests for Video Quality Scanner v2."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vqs.calibrate import calibration_checks, evaluate_labels, run_calibration
from vqs.config import ScanConfig, load_thresholds, thresholds_from_mapping
from vqs.decide.pairs import pair_verdict, plan_pairs
from vqs.decide.solo import decide_solo
from vqs.discover import find_video_files
from vqs.health import check_video_health, health_timeout_for
from vqs.metrics.classical import (
    blockiness_score,
    blur_profile,
    consensus_letterbox,
    measure_frame,
)
from vqs.metrics.nr_iqa import NrIqaScorer
from vqs.report.html import write_report
from vqs.store import Store
from vqs.util import looks_processed, normalized_stem, parse_simple_yaml, stems_related


FFMPEG = shutil.which("ffmpeg")


def encode_clip(path: Path, pattern: str = "testsrc", seconds: float = 1.0, size: str = "320x240"):
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"{pattern}=size={size}:rate=24",
        "-t",
        str(seconds),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True)


class MetricTests(unittest.TestCase):
    def test_blur_falls_with_gaussian(self):
        rng = np.random.default_rng(0)
        gray = rng.integers(0, 255, (180, 240), dtype=np.uint8)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        sharp, _ = blur_profile(frame)
        soft, _ = blur_profile(cv2.GaussianBlur(frame, (0, 0), 3))
        self.assertGreater(sharp, soft)

    def test_blockiness_on_blocks(self):
        block = np.zeros((128, 128), np.uint8)
        for y in range(0, 128, 8):
            for x in range(0, 128, 8):
                block[y : y + 8, x : x + 8] = 30 if ((x // 8 + y // 8) % 2) else 220
        score = blockiness_score(cv2.cvtColor(block, cv2.COLOR_GRAY2BGR))
        self.assertGreater(score, 1.2)

    def test_letterbox_consensus(self):
        pic = np.full((200, 320, 3), 120, np.uint8)
        letter = np.zeros((240, 320, 3), np.uint8)
        letter[20:220, :, :] = pic
        crop = consensus_letterbox([letter, letter, letter])
        self.assertGreaterEqual(crop[1], 16)
        self.assertLessEqual(crop[1] + crop[3], 224)

    def test_measure_excludes_dark(self):
        dark = np.zeros((120, 160, 3), np.uint8)
        measured = measure_frame(dark)
        self.assertTrue(measured["exclude_dark"])


class DecisionTests(unittest.TestCase):
    def test_soft_blocky_is_candidate(self):
        result = decide_solo(
            content_width=720,
            content_height=480,
            display_width=720,
            display_height=480,
            metrics={
                "avg_blur": 45.0,
                "avg_block": 1.55,
                "blur_scale_ratio": 1.1,
                "median_noise": 1.0,
                "avg_halo": 1.0,
                "combing": 0.0,
                "blur_cv": 0.2,
                "mostly_dark": False,
                "frames_used": 12,
            },
            nr_iqa_median=0.30,
            health_status="ok",
            is_processed=False,
            is_vr=False,
        )
        self.assertEqual(result["decision"], "topaz_candidate")

    def test_sharp_hd_skipped(self):
        result = decide_solo(
            content_width=1920,
            content_height=1080,
            display_width=1920,
            display_height=1080,
            metrics={
                "avg_blur": 400.0,
                "avg_block": 1.05,
                "blur_scale_ratio": 1.1,
                "median_noise": 1.0,
                "avg_halo": 1.0,
                "combing": 0.0,
                "blur_cv": 0.1,
                "mostly_dark": False,
                "frames_used": 12,
            },
            nr_iqa_median=0.78,
            health_status="ok",
            is_processed=False,
            is_vr=False,
        )
        self.assertEqual(result["decision"], "skip_or_ok")

    def test_health_failed_routes_to_repair(self):
        result = decide_solo(
            content_width=720,
            content_height=480,
            display_width=720,
            display_height=480,
            metrics={"frames_used": 0, "mostly_dark": True},
            nr_iqa_median=None,
            health_status="failed",
            is_processed=False,
            is_vr=False,
        )
        self.assertEqual(result["decision"], "repair_or_remux")

    def test_pair_improved_and_regressed(self):
        verdict, action, _, _ = pair_verdict(
            {
                "blur_native": 1.05,
                "blur_vs_base": 1.30,
                "block_native": 0.0,
                "noise_native": 1.0,
                "halo_vs_base": 1.0,
                "halo_delta": 0.0,
                "lpips_native": 0.1,
                "nr_uplift": 0.05,
            }
        )
        self.assertEqual(verdict, "improved")
        self.assertEqual(action, "keep_processed")

        verdict, action, _, _ = pair_verdict(
            {
                "blur_native": 0.5,
                "blur_vs_base": 0.8,
                "block_native": 0.2,
                "noise_native": 1.0,
                "halo_vs_base": 1.0,
                "halo_delta": 0.0,
                "lpips_native": 0.1,
                "nr_uplift": 0.0,
            }
        )
        self.assertTrue(str(verdict).startswith("regressed"))
        self.assertEqual(action, "keep_original")


class UtilTests(unittest.TestCase):
    def test_processed_markers(self):
        self.assertTrue(looks_processed("movie_topaz.mp4"))
        self.assertTrue(looks_processed("x.mkv", encoder="Topaz Video AI"))
        self.assertFalse(looks_processed("movie.mp4"))

    def test_stem_related(self):
        self.assertEqual(normalized_stem("DVDMS-388_topaz.mp4"), "dvdms-388")
        self.assertEqual(normalized_stem("DVDMS-388_3_prob4.mp4"), "dvdms-388_3")
        self.assertTrue(stems_related("DVDMS-388.mp4", "DVDMS-388_topaz.mp4"))
        self.assertTrue(stems_related("DVDMS-388.mp4", "DVDMS-388_3_prob4.mp4"))

    def test_yaml_and_thresholds(self):
        mapping = parse_simple_yaml("blur_soft: 90\ncodec_efficiency:\n  hevc: 1.6\n")
        thr = thresholds_from_mapping(mapping)
        self.assertEqual(thr.blur_soft, 90.0)
        self.assertAlmostEqual(thr.codec_efficiency["hevc"], 1.6)

    def test_plan_pairs(self):
        rows = [
            {
                "path": "/a/orig.mp4",
                "filename": "show.mp4",
                "folder": "/a",
                "duration": 10.0,
                "decision": "topaz_candidate",
                "is_probably_processed": False,
                "health_status": "ok",
                "error": "",
                "sample_aspect_ratio": "1:1",
                "size": 100,
            },
            {
                "path": "/a/proc.mp4",
                "filename": "show_topaz.mp4",
                "folder": "/a",
                "duration": 10.1,
                "decision": "already_processed",
                "is_probably_processed": True,
                "health_status": "ok",
                "error": "",
                "sample_aspect_ratio": "1:1",
                "size": 200,
            },
        ]
        plans = plan_pairs(rows, 1.0)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0][0]["filename"], "show.mp4")


class StoreReportTests(unittest.TestCase):
    def test_store_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.db"
            store = Store(db)
            store.upsert_video(
                {
                    "path": "/videos/a.mp4",
                    "filename": "a.mp4",
                    "folder": "/videos",
                    "size": 10,
                    "mtime": 1,
                    "decision": "topaz_candidate",
                    "confidence": 0.9,
                    "reason": "soft",
                    "health_status": "ok",
                    "avg_blur": 40,
                    "avg_block": 1.5,
                }
            )
            store.upsert_pair(
                {
                    "group_id": "abc",
                    "folder": "/videos",
                    "original_path": "/videos/a.mp4",
                    "other_path": "/videos/a_topaz.mp4",
                    "original_filename": "a.mp4",
                    "other_filename": "a_topaz.mp4",
                    "verdict": "improved",
                    "action": "keep_processed",
                    "confidence": 0.8,
                }
            )
            out = Path(tmp) / "report"
            index = write_report(store, out)
            self.assertTrue(index.exists())
            text = index.read_text(encoding="utf-8")
            self.assertIn("topaz_candidate", text)
            self.assertTrue((out / "videos.csv").exists())


class NrIqaFallbackTests(unittest.TestCase):
    def test_unavailable_scorer_returns_message(self):
        scorer = NrIqaScorer(model_name="definitely_missing_model_xyz", device="cpu")
        # Force failed load without requiring pyiqa by stubbing
        scorer._tried = True
        scorer._metric = None
        scorer._error = "missing"
        frame = np.zeros((64, 64, 3), np.uint8)
        result = scorer.score_bgr_frames([frame])
        self.assertFalse(result.available)
        self.assertIsNone(result.median)


class CalibrationTests(unittest.TestCase):
    def test_calibration_passes(self):
        self.assertEqual(run_calibration(), 0)
        checks = calibration_checks()
        self.assertTrue(all(ok for _name, ok, _detail in checks))

    def test_labels_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.db"
            store = Store(db)
            store.upsert_video(
                {
                    "path": "/v/good_upscale_src.mp4",
                    "filename": "good_upscale_src.mp4",
                    "folder": "/v",
                    "decision": "topaz_candidate",
                    "health_status": "ok",
                }
            )
            store.upsert_video(
                {
                    "path": "/v/sharp.mp4",
                    "filename": "sharp.mp4",
                    "folder": "/v",
                    "decision": "skip_or_ok",
                    "health_status": "ok",
                }
            )
            labels = Path(tmp) / "labels.csv"
            labels.write_text(
                "path,label\n"
                "/v/good_upscale_src.mp4,worth_topaz\n"
                "/v/sharp.mp4,not_worth\n",
                encoding="utf-8",
            )
            self.assertEqual(evaluate_labels(str(db), str(labels)), 0)


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class IntegrationTests(unittest.TestCase):
    def test_health_and_discover(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "clip.mp4"
            encode_clip(clip, seconds=0.5)
            files = find_video_files(root)
            self.assertEqual(len(files), 1)
            result = check_video_health(clip, duration=0.5, timeout=30)
            self.assertEqual(result.status, "ok")
            self.assertGreater(health_timeout_for(100), 100)

    def test_scan_smoke(self):
        from vqs.pipeline import scan_library

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "videos"
            root.mkdir()
            encode_clip(root / "a.mp4", seconds=0.8)
            encode_clip(root / "a_topaz.mp4", seconds=0.8)
            cfg = ScanConfig(
                root_folder=str(root),
                db_path=str(Path(tmp) / "vqs.db"),
                report_dir=str(Path(tmp) / "report"),
                workers=1,
                device="cpu",
                create_contact_sheets=True,
                run_health=True,
                run_solo=True,
                run_pairs=True,
                run_report=True,
                samples_explicit=True,
                sample_count=4,
                resume=False,
            )
            store = scan_library(cfg)
            videos = store.list_videos()
            self.assertGreaterEqual(len(videos), 2)
            self.assertTrue((Path(tmp) / "report" / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
