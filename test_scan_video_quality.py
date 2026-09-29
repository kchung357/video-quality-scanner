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
        self.assertEqual(defaults.samples, scan.SAMPLE_COUNT)
        args = scan.build_parser().parse_args([
            "--root", "lib",
            "--no-resume",
            "--workers", "2",
            "--contact-sheets",
            "--samples", "4",
        ])
        config = scan.config_from_args(args)
        self.assertEqual(config.root_folder, "lib")
        self.assertFalse(config.resume)
        self.assertTrue(config.create_contact_sheets)
        self.assertEqual(config.workers, 2)
        self.assertEqual(config.sample_count, 4)

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


if __name__ == "__main__":
    unittest.main()
