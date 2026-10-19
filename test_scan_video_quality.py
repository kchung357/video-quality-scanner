import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import scan_video_quality as scan


class MetricTests(unittest.TestCase):
    def test_blur_and_blockiness_separate_simple_patterns(self):
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 255, (80, 80), dtype=np.uint8)
        blurry = cv2.GaussianBlur(sharp, (21, 21), 0)
        self.assertGreater(
            scan.blur_score_laplacian(sharp),
            scan.blur_score_laplacian(blurry),
        )

        block = np.zeros((128, 128), dtype=np.uint8)
        for y in range(0, 128, 8):
            for x in range(0, 128, 8):
                block[y:y + 8, x:x + 8] = int(rng.integers(0, 255))
        smooth = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
        self.assertGreater(scan.blockiness_score(block), scan.blockiness_score(smooth))
        self.assertGreater(scan.blockiness_score(block), scan.BLOCK_VERY)
        self.assertLess(scan.blockiness_score(smooth), scan.BLOCK_SLIGHT)

    def test_dark_and_flat_frames_are_left_out_of_the_average(self):
        black = np.zeros((80, 80, 3), dtype=np.uint8)
        detail = np.full((80, 80, 3), 180, dtype=np.uint8)
        cv2.rectangle(detail, (8, 8), (72, 72), (20, 20, 20), 2)
        mixed = scan.aggregate_metrics([
            scan.measure_frame(black),
            scan.measure_frame(detail),
        ])
        self.assertEqual(mixed["frames_used"], 1)
        self.assertEqual(mixed["frames_excluded"], 1)
        self.assertFalse(mixed["mostly_dark"])
        self.assertGreater(mixed["avg_brightness"], 0)

        only_dark = scan.aggregate_metrics([
            scan.measure_frame(black),
            scan.measure_frame(black),
        ])
        self.assertTrue(only_dark["mostly_dark"])
        self.assertEqual(only_dark["frames_excluded"], 2)
        self.assertEqual(
            scan.assign_recommendation(
                240,
                only_dark["avg_blur"],
                only_dark["avg_block"],
                score=80,
                mostly_dark=True,
            ),
            "manual_review",
        )
        # A black frame must not collect the full blur penalty.
        self.assertLess(
            scan.calculate_issue_score(240, None, None, None),
            scan.blur_points(1),
        )

    def test_grain_is_not_treated_as_detail(self):
        rng = np.random.default_rng(4)
        source = rng.integers(0, 255, (480, 640), dtype=np.uint8)
        soft = cv2.GaussianBlur(source, (0, 0), 6)
        grain = np.clip(
            soft.astype(np.float32) + rng.normal(0, 8, soft.shape),
            0,
            255,
        ).astype(np.uint8)
        blur, ratio = scan.blur_profile(grain, 640)
        self.assertGreaterEqual(blur, scan.BLUR_SLIGHTLY_SOFT)
        self.assertGreaterEqual(ratio, scan.BLUR_GRAIN_RATIO)
        self.assertFalse(scan.is_sharpish(blur, ratio, 1.0))
        self.assertEqual(
            scan.assign_recommendation(480, blur, 1.05, 9, blur_scale_ratio=ratio, median_noise=1),
            "manual_review",
        )

        edges = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(edges, (40, 40), (500, 360), 220, -1)
        cv2.putText(
            edges, "Hello", (60, 220),
            cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 4, cv2.LINE_AA,
        )
        edge_blur, edge_ratio = scan.blur_profile(edges, 640)
        self.assertGreaterEqual(edge_blur, scan.BLUR_SLIGHTLY_SOFT)
        self.assertLess(edge_ratio, scan.BLUR_GRAIN_RATIO)
        self.assertTrue(scan.is_sharpish(edge_blur, edge_ratio, 0.0))

    def test_bitrate_accounts_for_frame_rate(self):
        self.assertAlmostEqual(
            scan.bits_per_pixel_per_frame(8_000_000, 1920, 1080, 30),
            8_000_000 / (1920 * 1080 * 30),
        )
        self.assertIsNone(scan.bits_per_pixel_per_frame(8_000_000, 1920, 1080, 0))
        # 4 Mbps at 60 fps is sparse per frame. The same rate at 30 fps is not.
        bpp_60 = scan.bits_per_pixel_per_frame(4_000_000, 1920, 1080, 60)
        bpp_30 = scan.bits_per_pixel_per_frame(4_000_000, 1920, 1080, 30)
        self.assertEqual(scan.bitrate_points(1080, bpp_60), 15)
        self.assertEqual(scan.bitrate_points(1080, bpp_30), 8)

    def test_score_bands_include_ruined_hd_and_cap_at_100(self):
        ruined_hd = scan.calculate_issue_score(1080, 20, 1.8, 0.03)
        self.assertEqual(ruined_hd, 95)
        self.assertGreaterEqual(ruined_hd, 75)
        self.assertEqual(scan.calculate_issue_score(1080, 400, 1.1, 0.2), 0)
        self.assertEqual(scan.calculate_issue_score(200, 10, 3.0, 0.01), 100)

    def test_recommendations_cover_the_previous_gaps(self):
        cases = [
            (320, 400, 1.05, 12, 0.8, 0.2, False, False, "upscale_candidate_very_low_res"),
            (480, 220, 1.05, 9, 0.8, 0.2, False, False, "upscale_only_gentle"),
            (480, 120, 1.05, 17, 1.0, 0.0, False, False, "manual_review"),
            (320, 400, 1.05, 12, 3.5, 1.0, False, False, "manual_review"),
            (480, 400, 1.05, 9, 0.8, 6.0, False, False, "manual_review"),
            (1080, 400, 1.5, 22, 0.8, 0.2, False, False, "artifact_reduction_review"),
            (1080, 20, 1.8, 95, 1.0, 0.2, False, False, "top_priority_restore"),
            (480, 30, 1.7, 90, 1.0, 0.2, False, False, "restore_deblur_artifact_reduce"),
            (480, 80, 1.1, 30, 1.0, 0.2, False, False, "restore_or_deblur_then_upscale"),
            (240, 1, 1.0, 80, 1.0, 0.0, True, False, "manual_review"),
            (480, 220, 1.05, 9, 0.8, 0.2, False, True, "already_processed_review_only+upscale_only_gentle"),
            (1080, 400, 1.05, 0, 0.8, 0.2, False, False, "low_priority_or_ok"),
        ]
        for side, blur, block, score, ratio, noise, dark, processed, expected in cases:
            with self.subTest(expected=expected):
                got = scan.assign_recommendation(
                    side,
                    blur,
                    block,
                    score,
                    processed=processed,
                    median_noise=noise,
                    blur_scale_ratio=ratio,
                    mostly_dark=dark,
                )
                self.assertEqual(got, expected)

    def test_low_bitrate_flags_use_bits_per_pixel_per_frame(self):
        hd_flags = scan.build_issue_flags(1080, 400, 1.1, 0.03)
        self.assertIn("low_bitrate_for_hd", hd_flags)
        self.assertNotIn("low_bitrate_for_sd", hd_flags)
        sd_flags = scan.build_issue_flags(480, 400, 1.1, 0.08)
        self.assertIn("low_bitrate_for_sd", sd_flags)
        self.assertIn("low_resolution", sd_flags)


class GeometryTests(unittest.TestCase):
    def test_display_size_uses_sar_and_rotation(self):
        width, height = scan.display_dimensions(720, 480, 32 / 27, 0)
        self.assertEqual((width, height), (853, 480))
        self.assertEqual(scan.classify_resolution(min(width, height)), "low_resolution")

        portrait = scan.display_dimensions(1920, 1080, 1, -90)
        self.assertEqual(portrait, (1080, 1920))
        self.assertEqual(scan.classify_resolution(min(portrait)), "full_hd_1080")

        stored_portrait = scan.display_dimensions(1080, 1920, 1, 0)
        self.assertEqual(scan.classify_resolution(min(stored_portrait)), "full_hd_1080")
        self.assertEqual(scan.display_dimensions(720, 480, scan.parse_sar("0:1"), 0), (720, 480))
        self.assertEqual(scan.parse_sar("32:27"), 32 / 27)
        self.assertEqual(scan.parse_fraction("30000/1001"), 30000 / 1001)
        self.assertEqual(scan.format_duration(3661), "01:01:01")
        self.assertTrue(scan.is_probably_processed("movie_topaz_v2.mkv"))
        self.assertFalse(scan.is_probably_processed("movie.mkv"))

    def test_timestamps_and_accurate_seek(self):
        stamps = scan.sample_timestamps(10, 4)
        self.assertEqual(len(stamps), 4)
        self.assertGreaterEqual(min(stamps), 0.5)
        self.assertLessEqual(max(stamps), 9.96)
        self.assertEqual(scan.sample_timestamps(0, 8), [])
        self.assertEqual(scan.seek_points(10), (9.0, 1.0))
        pre, fine = scan.seek_points(0.2)
        self.assertEqual(pre, 0.0)
        self.assertAlmostEqual(fine, 0.2)

    def test_frame_command_is_raw_and_accurate(self):
        cmd = scan.build_frame_command("clip.mp4", 12.5, threads=1)
        joined = " ".join(cmd)
        first_ss = cmd.index("-ss")
        second_ss = cmd.index("-ss", first_ss + 1)
        input_at = cmd.index("-i")
        self.assertLess(first_ss, input_at)
        self.assertLess(input_at, second_ss)
        self.assertIn("-noautorotate", cmd)
        self.assertIn("rawvideo", cmd)
        self.assertIn("-threads", cmd)
        self.assertNotIn("scale", joined)
        self.assertNotIn(".jpg", joined)
        self.assertNotIn("png", joined)
        self.assertNotIn("-threads", " ".join(scan.build_frame_command("clip.mp4", 12.5, threads=0)))


class ReportTests(unittest.TestCase):
    def test_contact_sheet_names_include_the_full_path(self):
        first = scan.contact_sheet_filename("/library/a/folder/clip.mp4")
        second = scan.contact_sheet_filename("/library/b/folder/clip.mp4")
        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith(".jpg"))
        self.assertNotIn("/", first)
        self.assertNotIn(":", first)

        with tempfile.TemporaryDirectory() as tmp:
            frame = np.full((90, 160, 3), 40, dtype=np.uint8)
            cv2.rectangle(frame, (10, 10), (140, 70), (220, 220, 220), 2)
            flags = "very_blurry,slightly_blocky," + ("x" * 400)
            result = scan.result_row(
                path="/library/a/folder/clip.mp4",
                filename="clip.mp4",
                quality_issue_score=80,
                recommendation="top_priority_restore",
                width=1920,
                height=1080,
                display_width=1920,
                display_height=1080,
                avg_blur_score=12.5,
                avg_blockiness_score=1.7,
                avg_noise_score=1.2,
                issue_flags=flags,
            )
            out = scan.create_contact_sheet(
                "/library/a/folder/clip.mp4",
                [(1.5, frame), (3.0, frame)],
                result,
                tmp,
                160,
            )
            image = cv2.imread(out)
            self.assertIsNotNone(image)
            self.assertGreater(image.shape[0], 40)
            self.assertTrue(Path(out).name.endswith(scan.contact_sheet_filename("/library/a/folder/clip.mp4")))

    def test_errors_sort_after_scored_rows(self):
        rows = [
            {"path": "a", "quality_issue_score": None, "avg_blur_score": None, "error": "boom"},
            {"path": "b", "quality_issue_score": 10, "avg_blur_score": 80, "error": ""},
            {"path": "c", "quality_issue_score": 80, "avg_blur_score": 20, "error": ""},
            {"path": "d", "quality_issue_score": 10, "avg_blur_score": 30, "error": ""},
        ]
        frame = scan.sorted_dataframe(rows)
        self.assertEqual(list(frame["path"]), ["c", "d", "b", "a"])

    def test_legacy_sentinel_score_is_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.csv"
            pd.DataFrame([{
                "path": "a.mp4",
                "quality_issue_score": 999,
                "error": "boom",
            }]).to_csv(path, index=False)
            rows = scan.read_existing_rows(path)
        self.assertIsNone(rows[0]["quality_issue_score"])

    def test_missing_and_skipped_files_are_not_scored(self):
        missing = scan.analyze_video(
            "/tmp/does-not-exist-video-quality.mp4",
            scan.ScanConfig(),
        )
        self.assertIsNone(missing["quality_issue_score"])
        self.assertEqual(missing["recommendation"], "fix_path_or_missing_file")

        skipped = scan.analyze_video(
            "clip_topaz.mp4",
            scan.ScanConfig(skip_already_processed=True),
        )
        self.assertIsNone(skipped["quality_issue_score"])
        self.assertEqual(skipped["recommendation"], "skipped")

    def test_hidden_and_contact_sheet_dirs_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "clip.MP4").write_bytes(b"")
            (root / "notes.txt").write_bytes(b"")
            hidden = root / ".hidden"
            hidden.mkdir()
            (hidden / "secret.mp4").write_bytes(b"")
            sheets = root / "contact_sheets"
            sheets.mkdir()
            (sheets / "preview.mp4").write_bytes(b"")
            sub = root / "sub"
            sub.mkdir()
            (sub / "movie.mkv").write_bytes(b"")
            found = {path.name for path in scan.find_video_files(root, {"contact_sheets"})}
        self.assertEqual(found, {"clip.MP4", "movie.mkv"})

    def test_cli_overrides_defaults(self):
        defaults = scan.build_parser().parse_args([])
        self.assertTrue(defaults.resume)
        self.assertTrue(defaults.pairs)
        self.assertIsNone(defaults.samples)
        args = scan.build_parser().parse_args([
            "--root", "lib",
            "--no-resume",
            "--workers", "2",
            "--contact-sheets",
            "--samples", "4",
            "--no-pairs",
        ])
        config = scan.config_from_args(args)
        self.assertEqual(config.root_folder, "lib")
        self.assertFalse(config.resume)
        self.assertTrue(config.create_contact_sheets)
        self.assertEqual(config.workers, 2)
        self.assertEqual(config.sample_count, 4)
        self.assertTrue(config.samples_explicit)
        self.assertFalse(config.pairs)

    def test_wrap_text_fits_the_requested_width(self):
        lines = scan.wrap_text("word " * 40 + "Supercalifragilistic", 80, 0.5, 1)
        self.assertGreater(len(lines), 1)
        for line in lines:
            width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            self.assertLessEqual(width, 80)


def encode_clip(path, lavfi, *extra):
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", lavfi,
        *extra,
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        str(path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace")[-1000:]
        raise RuntimeError(detail)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is not installed")
class ScanIntegrationTests(unittest.TestCase):
    def test_scan_resume_contact_sheet_and_real_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encode_clip(root / "color.mp4", "testsrc=size=640x480:rate=25:duration=1")
            encode_clip(
                root / "wide.mp4",
                "testsrc=size=720x480:rate=25:duration=1",
                "-vf", "setsar=32/27",
            )
            encode_clip(root / "black.mp4", "color=c=black:size=320x240:rate=25:duration=1")
            report = root / "report.csv"
            sheets = root / "sheets"
            argv = [
                "--root", str(root),
                "--output", str(report),
                "--workers", "2",
                "--samples", "2",
                "--checkpoint-every", "1",
                "--contact-sheets",
                "--contact-sheet-folder", str(sheets),
                "--contact-sheet-min-score", "0",
                "--frame-timeout", "30",
            ]
            scan.main(argv)
            frame = pd.read_csv(report)
            self.assertEqual(len(frame), 3)
            self.assertEqual(set(frame["path"]), {str(root / name) for name in ("color.mp4", "wide.mp4", "black.mp4")})

            wide = frame.loc[frame["filename"] == "wide.mp4"].iloc[0]
            self.assertEqual(int(wide["width"]), 720)
            self.assertEqual(int(wide["height"]), 480)
            self.assertAlmostEqual(float(wide["display_width"]), 853, delta=1)
            self.assertEqual(int(wide["display_height"]), 480)
            self.assertEqual(wide["resolution_class"], "low_resolution")
            self.assertGreater(float(wide["bits_per_pixel_frame"]), 0)
            self.assertGreaterEqual(int(wide["sampled_frames"]), 1)
            wide_error = "" if pd.isna(wide["error"]) else str(wide["error"]).strip()
            self.assertEqual(wide_error, "")

            black = frame.loc[frame["filename"] == "black.mp4"].iloc[0]
            self.assertIn("mostly_dark_or_flat", str(black["issue_flags"]))
            self.assertEqual(black["recommendation"], "manual_review")
            self.assertLess(float(black["quality_issue_score"]), 40)

            for name in ("color.mp4", "wide.mp4", "black.mp4"):
                sheet = sheets / scan.contact_sheet_filename(root / name)
                self.assertTrue(sheet.exists(), sheet)
                image = cv2.imread(str(sheet))
                self.assertIsNotNone(image)
                self.assertGreater(image.shape[1], 0)

            scan.main(argv)
            resumed = pd.read_csv(report)
            self.assertEqual(len(resumed), 3)
            self.assertEqual(set(resumed["path"]), set(frame["path"]))

    def test_pair_mode_compares_the_same_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encode_clip(root / "CLIP-101.mp4", "testsrc=size=320x240:rate=25:duration=1")
            encode_clip(
                root / "CLIP-101_3_prob4.mp4",
                "testsrc=size=320x240:rate=25:duration=1",
                "-vf", "gblur=sigma=2",
            )
            report = root / "report.csv"
            scan.main([
                "--root", str(root),
                "--output", str(report),
                "--samples", "2",
                "--workers", "1",
                "--no-contact-sheets",
                "--frame-timeout", "30",
            ])
            groups = pd.read_csv(root / "report_groups.csv")
            self.assertEqual(len(groups), 1)
            row = groups.iloc[0]
            self.assertIn("lost_detail", str(row["verdict"]))
            self.assertEqual(row["action"], "keep_original")
            self.assertFalse(bool(row["geometry_mismatch"]))
            self.assertIn("CLIP-101.mp4", str(row["best_version"]))
            self.assertLess(float(row["blur_native"]), 0.85)
            frame = pd.read_csv(report)
            processed = frame.loc[frame["filename"] == "CLIP-101_3_prob4.mp4"].iloc[0]
            self.assertTrue(str(processed["recommendation"]).startswith("already_processed_review_only+"))
            self.assertIn("paired_with_original", str(processed["issue_flags"]))

    def test_pair_comparisons_use_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder, title in (("a", "ONE-1"), ("b", "TWO-2")):
                directory = root / folder
                directory.mkdir()
                encode_clip(directory / f"{title}.mp4", "testsrc=size=160x120:rate=25:duration=1")
                encode_clip(
                    directory / f"{title}_prob4.mp4",
                    "testsrc=size=160x120:rate=25:duration=1",
                    "-vf", "gblur=sigma=2",
                )
            report = root / "report.csv"
            scan.main([
                "--root", str(root),
                "--output", str(report),
                "--samples", "2",
                "--workers", "2",
                "--no-contact-sheets",
                "--frame-timeout", "30",
            ])
            groups = pd.read_csv(root / "report_groups.csv")
            self.assertEqual(len(groups), 2)
            self.assertEqual(set(groups["action"]), {"keep_original"})
            frame = pd.read_csv(report)
            processed = frame.loc[frame["filename"].astype(str).str.contains("_prob4")]
            self.assertEqual(len(processed), 2)
            for recommendation in processed["recommendation"]:
                self.assertTrue(str(recommendation).startswith("already_processed_review_only+"))


class MetricExtensionTests(unittest.TestCase):
    def test_noise_sigma_is_not_the_zero_median(self):
        rng = np.random.default_rng(0)
        noisy = np.clip(128 + rng.normal(0, 8, (180, 240)), 0, 255).astype(np.uint8)
        flat = np.full((180, 240), 128, np.uint8)
        edges = np.full((180, 240), 40, np.uint8)
        cv2.rectangle(edges, (20, 20), (200, 140), 220, -1)
        checker = (np.indices((180, 240)).sum(axis=0) % 2 * 255).astype(np.uint8)
        self.assertGreater(scan.noise_sigma(noisy), 5)
        self.assertLess(scan.noise_sigma(flat), 0.5)
        self.assertLess(scan.noise_sigma(edges), 1)
        self.assertLess(scan.noise_sigma(checker), 1)
        stats = scan.aggregate_metrics([scan.measure_frame(noisy) for _ in range(3)])
        self.assertGreater(stats["median_noise"], 1)

    def test_blur_class_splits_sharp_and_grain(self):
        self.assertEqual(scan.classify_blur(400, 1.1), "sharp")
        self.assertEqual(scan.classify_blur(400, 3.0), "grain_dominated")

    def test_score_parts_add_up_and_codec_adjusts_bitrate(self):
        parts = scan.issue_score_parts(1080, 20, 1.8, 0.03)
        self.assertEqual(parts["score_resolution"], 0)
        self.assertEqual(parts["score_blur"], 45)
        self.assertEqual(parts["score_blockiness"], 35)
        self.assertEqual(parts["score_bitrate"], 15)
        self.assertEqual(sum(parts.values()), 95)
        bpp = scan.bits_per_pixel_per_frame(2_000_000, 1920, 1080, 30)
        self.assertEqual(scan.bitrate_points(1080, bpp), 15)
        self.assertEqual(scan.bitrate_points(1080, bpp, codec="hevc"), 8)
        self.assertEqual(scan.bitrate_points(1080, bpp, vr=True), 0)

    def test_blurry_hd_and_upscale_and_vr_recommendations(self):
        self.assertEqual(
            scan.assign_recommendation(1080, 50, 1.1, 32),
            "check_source_upscaled_or_denoise",
        )
        self.assertEqual(
            scan.assign_recommendation(1920, 400, 1.05, 0, vr=True),
            "vr_equirect_review",
        )
        flags = scan.build_issue_flags(1920, 400, 1.1, 0.03, vr=True)
        self.assertIn("vr_equirect", flags)
        self.assertNotIn("low_bitrate_for_hd", flags)
        self.assertTrue(scan.is_vr_equirect(3840, 1920))
        self.assertFalse(scan.is_vr_equirect(1920, 1080))

    def test_duplicate_frames_are_excluded(self):
        detail = np.full((80, 80, 3), 180, dtype=np.uint8)
        cv2.rectangle(detail, (8, 8), (72, 72), (20, 20, 20), 2)
        measured = [scan.measure_frame(detail), scan.measure_frame(detail)]
        scan.mark_duplicate_frames(measured, [(0.0, detail), (1.0, detail)])
        stats = scan.aggregate_metrics(measured)
        self.assertEqual(stats["frames_used"], 1)
        self.assertEqual(stats["frames_excluded"], 1)
        self.assertFalse(stats["mostly_dark"])

    def test_letterbox_crop_and_metric_direction(self):
        checks = scan.calibration_checks()
        failed = [name for name, ok, _detail in checks if not ok]
        self.assertEqual(failed, [])

    def test_adaptive_sample_count_and_jitter(self):
        adaptive = scan.ScanConfig(samples_explicit=False)
        self.assertEqual(scan.resolved_sample_count(7200, adaptive), 24)
        self.assertEqual(scan.resolved_sample_count(10 * 3600, adaptive), 40)
        self.assertEqual(scan.resolved_sample_count(30, adaptive), 16)
        forced = scan.ScanConfig(sample_count=2, samples_explicit=True)
        self.assertEqual(scan.resolved_sample_count(7200, forced), 2)
        first = scan.sample_timestamps(3600, 8, seed=3)
        self.assertEqual(first, scan.sample_timestamps(3600, 8, seed=3))
        self.assertNotEqual(first, scan.sample_timestamps(3600, 8, seed=9))
        self.assertGreaterEqual(min(first), 0)
        self.assertLess(max(first), 3600)

    def test_config_file_overrides_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "blur_very: 10\ncodec_efficiency:\n  hevc: 2.0\n",
                encoding="utf-8",
            )
            loaded = scan.load_thresholds(path)
        self.assertEqual(loaded.blur_very, 10)
        self.assertEqual(loaded.codec_efficiency["hevc"], 2.0)
        self.assertEqual(loaded.codec_efficiency["h264"], 1.0)
        self.assertEqual(loaded.blur_soft, scan.BLUR_SOFT)
        with self.assertRaises(ValueError):
            scan.thresholds_from_mapping({"not_a_knob": 1})

    def test_resume_requires_schema_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            path.write_bytes(b"hello")
            size, mtime = scan.file_identity(path)
            current = {
                "schema_version": scan.SCHEMA_VERSION,
                "path": str(path),
                "size": size,
                "mtime": mtime,
            }
            self.assertTrue(scan.row_is_current(current))
            stale = dict(current)
            stale["schema_version"] = 1
            self.assertFalse(scan.row_is_current(stale))
            changed = dict(current)
            changed["size"] = size + 1
            self.assertFalse(scan.row_is_current(changed))

    def test_pair_matching_and_verdict(self):
        self.assertTrue(scan.stems_related("CLIP-101.mp4", "CLIP-101_3_prob4.mp4"))
        self.assertFalse(scan.stems_related("color.mp4", "wide.mp4"))
        self.assertEqual(scan.normalized_stem("movie_topaz.mkv"), "movie")

        def scored(path, **extra):
            row = {
                "path": path,
                "filename": Path(path).name,
                "folder": str(Path(path).parent),
                "duration": 90,
                "quality_issue_score": 10,
                "error": "",
                "is_probably_processed": False,
                "display_width": 1280,
                "display_height": 720,
                "sample_aspect_ratio": "1:1",
                "size": 10,
            }
            row.update(extra)
            return row

        original = scored(
            "/library/title/CLIP-101.mp4",
            size=50,
            content_width=1280,
            content_height=720,
        )
        processed = scored(
            "/library/title/CLIP-101_3_prob4.mp4",
            is_probably_processed=True,
            display_width=1552,
            display_height=720,
            content_width=1280,
            content_height=720,
            sample_aspect_ratio="97:80",
            size=20,
        )
        cropped = scored(
            "/library/title/CLIP-101_badcrop.mp4",
            is_probably_processed=True,
            content_width=1113,
            content_height=550,
            size=20,
        )
        unrelated = scored("/library/title/other-show.mp4", duration=90)
        plans = scan.plan_pairs([original, processed, unrelated], tolerance=1.0)
        paired = [plan for plan in plans if plan[2] != "ambiguous_group"]
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0][0]["filename"], "CLIP-101.mp4")
        self.assertEqual(paired[0][1]["filename"], "CLIP-101_3_prob4.mp4")
        self.assertFalse(scan.aspects_mismatch(original, processed))
        self.assertTrue(scan.aspects_mismatch(original, cropped))

        def deltas(**extra):
            base = {
                "blur_native": 1.0,
                "blur_vs_base": 1.0,
                "block_native": 0.0,
                "noise_native": 1.0,
                "halo_vs_base": 1.0,
                "halo_delta": 0.0,
                "pair_misaligned": False,
                "geometry_mismatch": False,
            }
            base.update(extra)
            return base

        self.assertEqual(scan.pair_verdict(deltas(blur_native=0.4)), "regressed:lost_detail")
        self.assertEqual(scan.pair_verdict(deltas(block_native=0.4)), "regressed:more_blocky")
        self.assertEqual(
            scan.pair_verdict(deltas(halo_vs_base=2.0, halo_delta=8.0)),
            "regressed:oversharpened",
        )
        self.assertEqual(scan.pair_verdict(deltas(blur_vs_base=1.4)), "improved")
        self.assertEqual(scan.pair_verdict(deltas()), "neutral")
        self.assertEqual(scan.pair_verdict(deltas(pair_misaligned=True)), "pair_misaligned")
        self.assertEqual(scan.pair_action("improved"), "keep_processed")
        self.assertEqual(scan.pair_action("neutral"), "keep_original")
        self.assertEqual(scan.pair_action("regressed:lost_detail"), "keep_original")
        self.assertEqual(scan.pair_action("pair_misaligned"), "manual_review")

    def test_retired_upscale_config_keys_are_ignored(self):
        loaded = scan.thresholds_from_mapping({
            "upscale_bandwidth_max": 0.5,
            "bandwidth_drop_db": 10,
            "blur_very": 12,
        })
        self.assertEqual(loaded.blur_very, 12)
        self.assertFalse(hasattr(loaded, "upscale_bandwidth_max"))


class GeometryAndPairTests(unittest.TestCase):
    def test_geometry_plan_covers_bars_sar_and_aspect(self):
        self.assertEqual(
            scan.geometry_plan(1280, 720, (112, 0, 1056, 720), sar=97 / 80),
            "crop_bars+reset_sar:1280x720",
        )
        self.assertEqual(
            scan.geometry_plan(720, 480, (0, 0, 720, 480), sar=1),
            "pad_to_16x9:854x480",
        )
        self.assertEqual(
            scan.geometry_plan(1920, 816, (0, 0, 1920, 816), sar=1),
            "pad_to_16x9:1920x1080",
        )
        self.assertEqual(
            scan.geometry_plan(1920, 1038, (0, 0, 1920, 1038), sar=1),
            "crop_to_16x9:1920x1080",
        )
        self.assertEqual(scan.geometry_plan(1920, 1080, (0, 0, 1920, 1080), sar=1), "")
        self.assertEqual(
            scan.geometry_plan(720, 480, (0, 0, 720, 480), sar=32 / 27),
            "reset_sar:854x480",
        )
        self.assertEqual(
            scan.geometry_plan(3840, 1920, (0, 0, 3840, 1920), sar=1, vr=True),
            "",
        )

    def test_letterbox_consensus_ignores_dark_and_minority_crops(self):
        bright = np.full((100, 160, 3), 140, np.uint8)
        bars = np.zeros((100, 160, 3), np.uint8)
        bars[20:80, :] = 140
        dark = np.zeros((100, 160, 3), np.uint8)
        self.assertEqual(scan.consensus_letterbox([bars] * 6 + [bright] * 2), (0, 20, 160, 60))
        self.assertEqual(scan.consensus_letterbox([bright] * 6 + [dark] * 4), (0, 0, 160, 100))
        self.assertEqual(scan.consensus_letterbox([bars] * 3 + [bright] * 5), (0, 0, 160, 100))

    def _show(self, metrics):
        summary = scan.summarize_pair_metrics([metrics])
        return scan.pair_verdict(summary), summary

    def test_pair_frames_distinguish_upscale_blur_blocks_halo_and_levels(self):
        original = cv2.cvtColor(scan._natural_frame(180, 320, 7), cv2.COLOR_GRAY2BGR)
        cubic = cv2.resize(original, (640, 360), interpolation=cv2.INTER_CUBIC)
        verdict, summary = self._show(scan.evaluate_pair_frames(original, cubic))
        self.assertEqual(verdict, "neutral")
        self.assertGreater(summary["ssim"], 0.9)

        blurred = cv2.GaussianBlur(original, (0, 0), 2.0)
        blurred_up = cv2.resize(blurred, (640, 360), interpolation=cv2.INTER_CUBIC)
        verdict, summary = self._show(scan.evaluate_pair_frames(original, blurred_up))
        self.assertEqual(verdict, "regressed:lost_detail")
        self.assertLess(summary["blur_native"], 0.65)

        yy, xx = np.mgrid[0:180, 0:320]
        photo = np.clip(128 + 40 * np.sin(xx / 3.0) + 25 * np.sin(yy / 4.0), 0, 255).astype(np.uint8)
        photo = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
        base = cv2.resize(photo, (640, 360), interpolation=cv2.INTER_CUBIC)
        base_f = base.astype(np.float32)
        boosted = np.clip(
            base_f + 0.25 * (base_f - cv2.GaussianBlur(base_f, (0, 0), 2.0)),
            0,
            255,
        ).astype(np.uint8)
        verdict, summary = self._show(scan.evaluate_pair_frames(photo, boosted))
        self.assertEqual(verdict, "improved")
        self.assertGreater(summary["blur_vs_base"], 1.15)

        edge = np.full((180, 320, 3), 50, np.uint8)
        cv2.rectangle(edge, (40, 30), (200, 140), (190, 190, 190), -1)
        cv2.circle(edge, (240, 90), 40, (210, 210, 210), -1)
        edge_base = cv2.resize(edge, (640, 360), interpolation=cv2.INTER_CUBIC)
        halo = np.clip(
            edge_base.astype(np.float32)
            + 1.2 * (edge_base.astype(np.float32) - cv2.GaussianBlur(edge_base, (0, 0), 1.0)),
            0,
            255,
        ).astype(np.uint8)
        verdict, _summary = self._show(scan.evaluate_pair_frames(edge, halo))
        self.assertEqual(verdict, "regressed:oversharpened")

        scene = np.full((240, 424), 128, np.float32)
        yy, xx = np.mgrid[0:240, 0:424]
        scene += 30 * np.sin(xx / 5.0) + 20 * np.sin(yy / 7.0)
        scene = np.clip(scene, 0, 255).astype(np.uint8)
        coded = scene.astype(np.float32).copy()
        for y in range(0, 240, 8):
            for x in range(0, 424, 8):
                coded[y:y + 8, x:x + 8] = coded[y:y + 8, x:x + 8].mean()
        upscaled = cv2.resize(coded.astype(np.uint8), (636, 360), interpolation=cv2.INTER_CUBIC)
        self.assertGreater(
            scan.blockiness_scaled(upscaled, 636 / 424, 360 / 240),
            scan.blockiness_score(upscaled),
        )
        verdict, summary = self._show(scan.evaluate_pair_frames(scene, upscaled))
        self.assertIn("more_blocky", verdict)
        self.assertGreater(summary["block_native"], 0.10)

        shifted = np.clip(photo.astype(np.int16) - 16, 0, 255).astype(np.uint8)
        verdict, summary = self._show(scan.evaluate_pair_frames(photo, shifted))
        self.assertEqual(verdict, "neutral")
        self.assertTrue(summary["range_shift"])
        self.assertGreater(summary["ssim"], 0.9)
        self.assertAlmostEqual(summary["luma_delta"], 16.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
