"""Unit and integration tests for Video Quality Scanner v2."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vqs.analyze import analyze_video, finish_nr_and_decision, load_spilled_frames, spill_frames
from vqs.calibrate import calibration_checks, evaluate_labels, run_calibration
from vqs.config import ScanConfig, load_thresholds, thresholds_from_mapping
from vqs.decide.pairs import pair_verdict, plan_pairs
from vqs.decide.solo import decide_solo
from vqs.discover import find_video_files
from vqs.health import build_health_command, check_video_health, health_timeout_for
from vqs.metrics.classical import (
    blockiness_score,
    blur_profile,
    consensus_letterbox,
    measure_frame,
)
from vqs.metrics.nr_iqa import NrIqaResult, NrIqaScorer
from vqs.sample import build_frame_command, ffmpeg_thread_count
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


class _ParentScorer:
    def __init__(self):
        self.calls = 0

    def score_bgr_frames(self, frames):
        self.calls += 1
        if not frames:
            return NrIqaResult(None, None, "fake", "parent-gpu", False, "no frames")
        values = [float(np.mean(frame)) / 255.0 for frame in frames]
        arr = np.asarray(values, dtype=np.float64)
        return NrIqaResult(float(arr.mean()), float(np.median(arr)), "fake", "parent-gpu", True)


def _comparable_row(row):
    return {
        "decision": row.get("decision"),
        "health_status": row.get("health_status"),
        "reason": row.get("reason"),
        "avg_blur": row.get("avg_blur"),
        "avg_block": row.get("avg_block"),
        "nr_iqa_median": row.get("nr_iqa_median"),
        "nr_iqa_device": row.get("nr_iqa_device"),
    }


class SchedulingTests(unittest.TestCase):
    def test_ffmpeg_threads_follow_the_pool(self):
        self.assertEqual(ffmpeg_thread_count(ScanConfig(workers=1)), 0)
        pooled = ScanConfig(workers=4, ffmpeg_threads=2)
        self.assertEqual(ffmpeg_thread_count(pooled), 2)
        cmd = build_frame_command("clip.mp4", 3.0, threads=ffmpeg_thread_count(pooled))
        self.assertEqual(cmd[cmd.index("-threads") + 1], "2")
        self.assertLess(cmd.index("-threads"), cmd.index("-i"))
        health = build_health_command("clip.mp4", threads=2)
        self.assertEqual(health[health.index("-threads") + 1], "2")
        self.assertLess(health.index("-threads"), health.index("-i"))
        self.assertNotIn("-threads", build_health_command("clip.mp4", threads=0))

    def test_spill_roundtrip(self):
        frame = np.zeros((8, 8, 3), np.uint8)
        frame[0, 0] = (1, 2, 3)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "frames.npz"
            spill_frames([(1.25, frame)], dest)
            loaded = load_spilled_frames(dest)
        self.assertEqual(loaded[0][0], 1.25)
        self.assertTrue(np.array_equal(loaded[0][1], frame))

    @unittest.skipUnless(FFMPEG, "ffmpeg not installed")
    def test_deferred_nr_matches_inline_and_skips_worker_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            encode_clip(clip, seconds=0.4)
            cfg = ScanConfig(
                root_folder=tmp,
                report_dir=str(Path(tmp) / "report"),
                workers=4,
                device="cuda",
                create_contact_sheets=False,
                samples_explicit=True,
                sample_count=2,
                resume=False,
            )
            scorer = _ParentScorer()
            with patch("vqs.analyze.get_scorer", side_effect=AssertionError("worker loaded a model")):
                deferred = analyze_video(str(clip), cfg, defer_nr=True)
            self.assertFalse(deferred.get("decision"))
            self.assertIn("_content_frames", deferred)
            finished = finish_nr_and_decision(deferred, cfg, scorer)
            direct = analyze_video(str(clip), cfg, nr_scorer=scorer)
            self.assertEqual(_comparable_row(direct), _comparable_row(finished))
            self.assertGreater(scorer.calls, 0)

    @unittest.skipUnless(FFMPEG, "ffmpeg not installed")
    def test_cuda_pool_keeps_parent_scores(self):
        from vqs.pipeline import scan_library

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "videos"
            root.mkdir()
            encode_clip(root / "a.mp4", seconds=0.4)
            encode_clip(root / "a_topaz.mp4", seconds=0.4)
            encode_clip(root / "b.mp4", pattern="smptebars", seconds=0.4)
            encode_clip(root / "b_topaz.mp4", pattern="smptebars", seconds=0.4)
            scorer = _ParentScorer()

            def make(workers, db_name):
                return ScanConfig(
                    root_folder=str(root),
                    db_path=str(Path(tmp) / db_name),
                    report_dir=str(Path(tmp) / f"report-{workers}"),
                    workers=workers,
                    device="cuda",
                    create_contact_sheets=False,
                    run_health=True,
                    run_solo=True,
                    run_pairs=True,
                    run_report=False,
                    samples_explicit=True,
                    sample_count=4,
                    resume=False,
                )

            with (
                patch("vqs.pipeline.resolve_device", return_value="cuda"),
                patch("vqs.pipeline.get_scorer", return_value=scorer),
                patch("vqs.pairs_run.resolve_device", return_value="cpu"),
            ):
                serial = scan_library(make(1, "one.db"))
                calls_after_serial = scorer.calls
                pooled = scan_library(make(2, "two.db"))
            self.assertGreater(scorer.calls, calls_after_serial)
            serial_rows = {row["filename"]: _comparable_row(row) for row in serial.list_videos()}
            pooled_rows = {row["filename"]: _comparable_row(row) for row in pooled.list_videos()}
            self.assertEqual(serial_rows, pooled_rows)
            for row in pooled_rows.values():
                self.assertEqual(row["nr_iqa_device"], "parent-gpu")
                self.assertEqual(row["health_status"], "ok")
            serial_pairs = {
                (row["original_filename"], row["other_filename"]): (row["action"], row["verdict"])
                for row in serial.list_pairs()
            }
            pooled_pairs = {
                (row["original_filename"], row["other_filename"]): (row["action"], row["verdict"])
                for row in pooled.list_pairs()
            }
            self.assertEqual(serial_pairs, pooled_pairs)
            self.assertGreaterEqual(len(serial_pairs), 2)

    @unittest.skipUnless(FFMPEG, "ffmpeg not installed")
    def test_pool_contact_sheets_and_health_command(self):
        from vqs.cli import cmd_health
        from vqs.pipeline import scan_library

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "videos"
            root.mkdir()
            encode_clip(root / "a.mp4", seconds=0.4)
            encode_clip(root / "b.mp4", pattern="smptebars", seconds=0.4)
            report = Path(tmp) / "report"
            cfg = ScanConfig(
                root_folder=str(root),
                db_path=str(Path(tmp) / "vqs.db"),
                report_dir=str(report),
                workers=2,
                device="cuda",
                create_contact_sheets=True,
                run_health=True,
                run_pairs=False,
                run_report=False,
                samples_explicit=True,
                sample_count=2,
                resume=False,
            )
            with (
                patch("vqs.pipeline.resolve_device", return_value="cuda"),
                patch("vqs.pipeline.get_scorer", return_value=_ParentScorer()),
            ):
                store = scan_library(cfg)
            sheets = list((report / "sheets").glob("*.jpg"))
            spill = list((report / "sheets" / "_spill").glob("*.npz"))
            self.assertGreaterEqual(len(sheets), 1)
            self.assertEqual(spill, [])
            self.assertTrue(any(row.get("contact_sheet") for row in store.list_videos()))

            args = type(
                "Args",
                (),
                {
                    "db": str(Path(tmp) / "health.db"),
                    "root": str(root),
                    "workers": 2,
                    "probe_timeout": 30.0,
                    "health_timeout": 30.0,
                },
            )()
            self.assertEqual(cmd_health(args), 0)
            from vqs.store import Store

            health_rows = Store(args.db).list_videos()
            self.assertEqual(len(health_rows), 2)
            self.assertTrue(all(row["health_status"] == "ok" for row in health_rows))


if __name__ == "__main__":
    unittest.main()
