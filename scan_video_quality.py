"""Scan a video folder and write a CSV of heuristic quality issues.

Frame samples are full-resolution raw frames. Blur is measured after scaling
to the analysis width; blockiness and noise stay on the decoded pixels.
Run ``python scan_video_quality.py --help`` for the command-line options.
Defaults below are used when the matching flag is omitted.
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

ROOT_FOLDER = r"./videos"
OUTPUT_CSV = "video_quality_report.csv"

# Contact sheets are optional JPEGs for manual review.
CREATE_CONTACT_SHEETS = False
CONTACT_SHEET_MIN_SCORE = 55
CONTACT_SHEET_FOLDER = "contact_sheets"

# Number of frames to sample per video.
SAMPLE_COUNT = 8

# Laplacian blur is resolution-dependent. Frames are scaled to this width
# before the blur score so the thresholds below stay meaningful.
# Blockiness and noise are measured at the decoded resolution.
ANALYSIS_FRAME_MAX_WIDTH = 640
CONTACT_THUMB_WIDTH = 320

# Skip filenames that already look processed or upscaled.
SKIP_ALREADY_PROCESSED = False

# Parallel ffmpeg jobs. Each worker decodes its own video.
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".mpg", ".mpeg",
    ".m4v", ".ts", ".m2ts", ".webm", ".flv", ".3gp",
}

PROCESSED_MARKERS = [
    "_prob",
    "_pnat",
    "_proteus",
    "_gaia",
    "_artemis",
    "_iris",
    "_topaz",
    "_upscale",
    "_enhanced",
    "_denoise",
    "_deblur",
]

# ------------------------------------------------------------
# Thresholds. Classification, flags, score, and recommendations
# all read these values.
# Blur numbers apply to frames scaled to ANALYSIS_FRAME_MAX_WIDTH.
# ------------------------------------------------------------

BLUR_VERY = 35.0
BLUR_BLURRY = 60.0
BLUR_SOFT = 100.0
BLUR_SLIGHTLY_SOFT = 150.0
BLUR_SHARP = 300.0

# Laplacian at analysis width divided by Laplacian at half that width.
# Real edges survive the extra downscale (ratio near 1). Grain and noise
# do not (ratio near 4), so a high blur score is not real sharpness.
BLUR_GRAIN_RATIO = 2.0

BLOCK_SLIGHT = 1.30
BLOCK_BLOCKY = 1.45
BLOCK_VERY = 1.60

# Median of the median-filter residual, on the full decoded frame.
# A high median means the grain is everywhere. Sparse edges leave the
# median near zero, so subtitles and line art are not flagged as grain.
MEDIAN_NOISE_GRAINY = 4.0

# Bits per pixel per frame. About the previous per-second thresholds
# (HD 1.8, SD 2.5) divided by 30 fps, with a stricter "very low" band.
HD_BPP_VERY_LOW = 0.04
HD_BPP_LOW = 0.07
SD_BPP_VERY_LOW = 0.05
SD_BPP_LOW = 0.09

DARK_PIXEL_THRESHOLD = 16
DARK_FRAME_RATIO = 0.80
DARK_MEAN_BRIGHTNESS = 12.0
FLAT_STD_MAX = 4.0

SCORE_CAP = 100.0

CSV_COLUMNS = [
    "path",
    "filename",
    "folder",
    "width",
    "height",
    "display_width",
    "display_height",
    "sample_aspect_ratio",
    "rotation",
    "codec",
    "pix_fmt",
    "duration",
    "duration_hms",
    "frame_rate",
    "bit_rate",
    "size",
    "is_probably_processed",
    "sampled_frames",
    "frames_used",
    "frames_excluded",
    "avg_blur_score",
    "min_blur_score",
    "max_blur_score",
    "blur_scale_ratio",
    "avg_blockiness_score",
    "avg_noise_score",
    "median_noise_score",
    "avg_brightness",
    "avg_black_frame_ratio",
    "bitrate_per_pixel",
    "bits_per_pixel_frame",
    "quality_issue_score",
    "resolution_class",
    "blur_class",
    "blockiness_class",
    "issue_flags",
    "recommendation",
    "contact_sheet",
    "contact_sheet_error",
    "error",
]


@dataclass
class ScanConfig:
    root_folder: str = ROOT_FOLDER
    output_csv: str = OUTPUT_CSV
    create_contact_sheets: bool = CREATE_CONTACT_SHEETS
    contact_sheet_min_score: float = CONTACT_SHEET_MIN_SCORE
    contact_sheet_folder: str = CONTACT_SHEET_FOLDER
    sample_count: int = SAMPLE_COUNT
    analysis_frame_max_width: int = ANALYSIS_FRAME_MAX_WIDTH
    contact_thumb_width: int = CONTACT_THUMB_WIDTH
    skip_already_processed: bool = SKIP_ALREADY_PROCESSED
    workers: int = DEFAULT_WORKERS
    resume: bool = True
    frame_timeout: float = 60.0
    probe_timeout: float = 30.0
    checkpoint_every: int = 5


# ============================================================
# Utility functions
# ============================================================

def check_external_tools():
    """Make sure ffmpeg and ffprobe are available."""
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    if shutil.which("ffprobe") is None:
        missing.append("ffprobe")
    if missing:
        raise RuntimeError(
            "Missing required tool(s): "
            + ", ".join(missing)
            + ". Install FFmpeg and make sure ffmpeg and ffprobe are on PATH."
        )


def run_text(cmd, timeout):
    """Run a command and return stdout text."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out after {timeout:.0f}s: {cmd[0]}") from exc

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(err or f"Command failed with exit code {result.returncode}")
    return result.stdout.decode("utf-8", "replace")


def clip_error(message):
    text = " ".join(str(message).split())
    if len(text) > 500:
        return text[:500] + "..."
    return text


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def round_or_none(value, digits):
    if value is None:
        return None
    return round(float(value), digits)


def parse_fraction(frac_text):
    """Parse an ffprobe frame rate like '30000/1001'."""
    try:
        if frac_text is None or frac_text == "" or frac_text == "0/0":
            return 0.0
        text = str(frac_text)
        if "/" in text:
            num, den = text.split("/", 1)
            den = float(den)
            if den == 0:
                return 0.0
            return float(num) / den
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def parse_sar(text):
    """Return a sample-aspect-ratio as a float. Unknown values become 1."""
    if text is None:
        return 1.0
    raw = str(text).strip()
    if raw in ("", "0:1", "0:0", "N/A", "unknown"):
        return 1.0
    try:
        if ":" in raw:
            num, den = raw.split(":", 1)
            den_f = float(den)
            ratio = float(num) / den_f if den_f else 1.0
        elif "/" in raw:
            num, den = raw.split("/", 1)
            den_f = float(den)
            ratio = float(num) / den_f if den_f else 1.0
        else:
            ratio = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if ratio < 0.125 or ratio > 8:
        return 1.0
    return ratio


def is_probably_processed(path):
    """Detect whether the filename looks processed or upscaled."""
    name = Path(path).stem.lower()
    return any(marker in name for marker in PROCESSED_MARKERS)


def format_duration(seconds):
    """Convert seconds to HH:MM:SS."""
    seconds = safe_float(seconds, 0)
    if seconds <= 0:
        return ""
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def stream_rotation(stream):
    """Read rotation from display-matrix side data, then stream tags."""
    for item in stream.get("side_data_list") or []:
        if isinstance(item, dict) and item.get("rotation") is not None:
            return safe_float(item.get("rotation"), 0.0)
    tags = stream.get("tags") or {}
    for key in ("rotate", "rotation"):
        if tags.get(key) not in (None, ""):
            return safe_float(tags.get(key), 0.0)
    return 0.0


def display_dimensions(width, height, sar, rotation):
    """Pixel size after sample aspect ratio and quarter-turn rotation."""
    if width <= 0 or height <= 0:
        return 0, 0
    if sar <= 0:
        sar = 1.0
    display_w = max(1, int(round(width * float(sar))))
    display_h = int(height)
    turns = int(round(abs(rotation))) % 360
    if turns in (90, 270):
        return display_h, display_w
    return display_w, display_h


def short_side(width, height, display_width=0, display_height=0):
    """Shorter display side. Portrait and anamorphic files use this."""
    if display_width and display_height:
        return min(int(display_width), int(display_height))
    if width and height:
        return min(int(width), int(height))
    return 0


def result_row(**kwargs):
    row = {column: None for column in CSV_COLUMNS}
    unknown = set(kwargs) - set(row)
    if unknown:
        raise KeyError("Unknown result columns: " + ", ".join(sorted(unknown)))
    row.update(kwargs)
    return row


def base_row(video_path, meta=None, processed=False):
    path = Path(video_path)
    row = result_row(
        path=str(path),
        filename=path.name,
        folder=str(path.parent),
        is_probably_processed=bool(processed),
        sampled_frames=0,
        frames_used=0,
        frames_excluded=0,
        issue_flags="",
        recommendation="",
        contact_sheet="",
        contact_sheet_error="",
        error="",
    )
    if meta:
        for key in (
            "width",
            "height",
            "display_width",
            "display_height",
            "sample_aspect_ratio",
            "rotation",
            "codec",
            "pix_fmt",
            "duration",
            "duration_hms",
            "frame_rate",
            "bit_rate",
            "size",
        ):
            row[key] = meta.get(key)
    return row


# ============================================================
# FFprobe metadata
# ============================================================

def ffprobe_video(path, timeout):
    """Extract video metadata with one ffprobe call."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_streams",
        "-show_format",
        "-of", "json",
        str(path),
    ]
    output = run_text(cmd, timeout)
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON: {exc}") from exc

    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("No video stream found")

    stream = streams[0]
    fmt = data.get("format") or {}
    width = safe_int(stream.get("width"))
    height = safe_int(stream.get("height"))
    raw_sar = stream.get("sample_aspect_ratio") or ""
    sar = parse_sar(raw_sar)
    rotation = stream_rotation(stream)
    display_w, display_h = display_dimensions(width, height, sar, rotation)

    duration = safe_float(stream.get("duration"), 0.0)
    if duration <= 0:
        duration = safe_float(fmt.get("duration"), 0.0)

    frame_rate = parse_fraction(stream.get("avg_frame_rate"))
    if frame_rate <= 0:
        frame_rate = parse_fraction(stream.get("r_frame_rate"))
    if duration <= 0:
        nb_frames = safe_int(stream.get("nb_frames"), 0)
        if nb_frames > 0 and frame_rate > 0:
            duration = nb_frames / frame_rate

    bit_rate = safe_int(stream.get("bit_rate"), 0)
    if bit_rate <= 0:
        bit_rate = safe_int(fmt.get("bit_rate"), 0)

    size = safe_int(fmt.get("size"), 0)
    if size <= 0:
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = 0

    return {
        "width": width,
        "height": height,
        "display_width": display_w,
        "display_height": display_h,
        "sample_aspect_ratio": raw_sar or "1:1",
        "rotation": rotation,
        "codec": stream.get("codec_name") or "",
        "pix_fmt": stream.get("pix_fmt") or "",
        "duration": duration,
        "duration_hms": format_duration(duration),
        "frame_rate": round(frame_rate, 3) if frame_rate > 0 else 0.0,
        "bit_rate": bit_rate,
        "size": size,
    }


# ============================================================
# Frame extraction
# ============================================================

def sample_timestamps(duration, sample_count):
    """Timestamps spread across the middle of the file."""
    if duration <= 0 or sample_count <= 0:
        return []
    last_ok = max(float(duration) - 0.04, 0.0)
    if int(sample_count) == 1:
        return [min(float(duration) * 0.5, last_ok)]

    if duration < 60:
        start_ratio, end_ratio = 0.05, 0.95
    else:
        start_ratio, end_ratio = 0.10, 0.90
    start = float(duration) * start_ratio
    end = float(duration) * end_ratio
    if end < start:
        end = start
    stamps = np.linspace(start, end, int(sample_count))
    return [float(min(max(float(stamp), 0.0), last_ok)) for stamp in stamps]


def seek_points(timestamp):
    """Fast seek to 1s before the target, then an accurate decode."""
    timestamp = max(0.0, float(timestamp))
    pre = max(0.0, timestamp - 1.0)
    fine = timestamp - pre
    return pre, fine


def build_frame_command(video_path, timestamp, threads=0):
    """ffmpeg command for one full-resolution raw frame.

    ``-ss`` before ``-i`` jumps near the timestamp. ``-ss`` after ``-i``
    decodes to the exact time so the sample is not just the nearest
    keyframe. ``-noautorotate`` keeps the coded pixel grid, which is the
    grid blockiness is measured on. There is no scale and no JPEG.
    """
    pre, fine = seek_points(timestamp)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
    ]
    if threads and int(threads) > 0:
        cmd.extend(["-threads", str(int(threads))])
    cmd.extend([
        "-noautorotate",
        "-ss", f"{pre:.3f}",
        "-i", str(video_path),
        "-ss", f"{fine:.3f}",
        "-map", "0:v:0",
        "-frames:v", "1",
        "-an",
        "-sn",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ])
    return cmd


def decode_frame(video_path, timestamp, width, height, timeout, threads):
    """Return (bgr_frame, error). error is empty on success."""
    if width <= 0 or height <= 0 or width > 16384 or height > 16384:
        return None, f"Refusing to decode frame size {width}x{height}"

    cmd = build_frame_command(video_path, timestamp, threads=threads)
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"Timed out decoding frame at {float(timestamp):.2f}s"

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", "replace").strip()
        return None, err or f"ffmpeg failed at {float(timestamp):.2f}s"

    expected = int(width) * int(height) * 3
    if len(result.stdout) != expected:
        return None, (
            f"Unexpected frame size {len(result.stdout)} "
            f"(expected {expected}) at {float(timestamp):.2f}s"
        )
    frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape((int(height), int(width), 3))
    return frame.copy(), ""


def extract_sample_frames(video_path, meta, config):
    """Decode sample frames. One bad timestamp does not drop the file.

    A single select-filter pass would decode from the start of the file
    through the last sample. Separate accurate seeks stay fast on long
    videos and still land off the keyframe.
    """
    timestamps = sample_timestamps(meta["duration"], config.sample_count)
    threads = 1 if config.workers > 1 else 0
    frames = []
    errors = []
    for timestamp in timestamps:
        frame, error = decode_frame(
            video_path,
            timestamp,
            meta["width"],
            meta["height"],
            config.frame_timeout,
            threads,
        )
        if frame is None:
            if error:
                errors.append(error)
            continue
        frames.append((float(timestamp), frame))
    return frames, errors


# ============================================================
# Image metrics
# ============================================================

def to_gray(frame):
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def scale_gray(gray, max_width):
    height, width = gray.shape[:2]
    if width <= max_width or width <= 0:
        return gray
    new_width = int(max_width)
    new_height = max(1, int(round(height * (new_width / float(width)))))
    return cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)


def laplacian_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def blur_profile(frame, max_width=ANALYSIS_FRAME_MAX_WIDTH):
    """Return (laplacian variance, full/half ratio) at the analysis width."""
    gray = scale_gray(to_gray(frame), max_width)
    blur = laplacian_var(gray)
    height, width = gray.shape[:2]
    if width < 32 or height < 32:
        return blur, None
    half_width = max(16, width // 2)
    half_height = max(16, int(round(height * (half_width / float(width)))))
    half = cv2.resize(gray, (half_width, half_height), interpolation=cv2.INTER_AREA)
    half_blur = laplacian_var(half)
    return blur, blur / max(half_blur, 1e-6)


def blur_score_laplacian(frame, max_width=ANALYSIS_FRAME_MAX_WIDTH):
    """Higher value means sharper. Measured after scaling to max_width."""
    blur, _ratio = blur_profile(frame, max_width)
    return blur


def blockiness_score(frame):
    """Estimate 8x8 blocking on the frame as decoded, with no rescale.

    Higher means stronger block edges than the pixels inside the blocks.
    Hard edges, subtitles, animation, and interlacing can raise it.
    """
    gray = to_gray(frame).astype(np.float32)
    height, width = gray.shape[:2]
    if height < 16 or width < 16:
        return 0.0

    diff_x = np.abs(gray[:, 1:] - gray[:, :-1])
    diff_y = np.abs(gray[1:, :] - gray[:-1, :])
    boundary_cols = np.arange(7, width - 1, 8)
    boundary_rows = np.arange(7, height - 1, 8)
    if len(boundary_cols) == 0 or len(boundary_rows) == 0:
        return 0.0

    boundary_x = diff_x[:, boundary_cols].mean()
    boundary_y = diff_y[boundary_rows, :].mean()
    non_boundary_cols = np.setdiff1d(np.arange(diff_x.shape[1]), boundary_cols)
    non_boundary_rows = np.setdiff1d(np.arange(diff_y.shape[0]), boundary_rows)
    if len(non_boundary_cols) == 0 or len(non_boundary_rows) == 0:
        return 0.0

    non_boundary_x = diff_x[:, non_boundary_cols].mean()
    non_boundary_y = diff_y[non_boundary_rows, :].mean()
    boundary_mean = (boundary_x + boundary_y) / 2.0
    non_boundary_mean = (non_boundary_x + non_boundary_y) / 2.0
    return float(boundary_mean / max(non_boundary_mean, 1e-6))


def noise_profile(frame):
    """Return (mean residual, median residual) after a 3x3 median blur."""
    gray = to_gray(frame)
    residual = cv2.absdiff(gray, cv2.medianBlur(gray, 3))
    return float(np.mean(residual)), float(np.median(residual))


def noise_score(frame):
    """Mean high-frequency residual. Grain is not always a defect."""
    mean_residual, _median_residual = noise_profile(frame)
    return mean_residual


def black_frame_ratio(frame, threshold=DARK_PIXEL_THRESHOLD):
    gray = to_gray(frame)
    return float(np.mean(gray < threshold))


def brightness_score(frame):
    gray = to_gray(frame)
    return float(np.mean(gray))


def measure_frame(frame, max_width=ANALYSIS_FRAME_MAX_WIDTH):
    """Metrics for one decoded frame. Blur is scaled; blockiness is not."""
    gray = to_gray(frame)
    brightness = float(np.mean(gray))
    black = float(np.mean(gray < DARK_PIXEL_THRESHOLD))
    std = float(np.std(gray))
    exclude = (
        black >= DARK_FRAME_RATIO
        or brightness <= DARK_MEAN_BRIGHTNESS
        or std < FLAT_STD_MAX
    )
    blur, ratio = blur_profile(gray, max_width)
    mean_noise, median_noise = noise_profile(gray)
    return {
        "blur": blur,
        "blur_scale_ratio": ratio,
        "block": blockiness_score(gray),
        "noise": mean_noise,
        "median_noise": median_noise,
        "brightness": brightness,
        "black": black,
        "std": std,
        "exclude": exclude,
    }


def aggregate_metrics(measured):
    """Average blur, blockiness, and noise over frames that are not dark or flat.

    Brightness and the black-pixel ratio always use every sample. If every
    sample is dark or flat, those samples are used anyway and mostly_dark
    is set so the score does not treat a black frame as blur.
    """
    usable = [item for item in measured if not item["exclude"]]
    mostly_dark = not usable
    chosen = measured if mostly_dark else usable

    def mean(key):
        return float(np.mean([item[key] for item in chosen]))

    ratios = [
        item["blur_scale_ratio"]
        for item in chosen
        if item["blur_scale_ratio"] is not None
    ]
    return {
        "avg_blur": mean("blur"),
        "min_blur": float(np.min([item["blur"] for item in chosen])),
        "max_blur": float(np.max([item["blur"] for item in chosen])),
        "avg_block": mean("block"),
        "avg_noise": mean("noise"),
        "median_noise": mean("median_noise"),
        "blur_scale_ratio": float(np.mean(ratios)) if ratios else None,
        "avg_brightness": float(np.mean([item["brightness"] for item in measured])),
        "avg_black": float(np.mean([item["black"] for item in measured])),
        "frames_used": len(chosen),
        "frames_excluded": len(measured) - len(usable),
        "mostly_dark": mostly_dark,
    }


def bits_per_second_per_pixel(bit_rate, width, height):
    pixels = int(width) * int(height)
    if bit_rate <= 0 or pixels <= 0:
        return None
    return float(bit_rate) / pixels


def bits_per_pixel_per_frame(bit_rate, width, height, frame_rate):
    """Codec bits per displayed coded pixel per frame. Uses coded size."""
    pixels = int(width) * int(height)
    if bit_rate <= 0 or pixels <= 0 or frame_rate <= 0:
        return None
    return float(bit_rate) / (pixels * float(frame_rate))


# ============================================================
# Scoring and classification
# ============================================================

def classify_resolution(side):
    if side <= 0:
        return "unknown_resolution"
    if side <= 270:
        return "very_tiny"
    if side <= 360:
        return "very_low_resolution"
    if side <= 480:
        return "low_resolution"
    if side <= 576:
        return "sd_plus"
    if side <= 720:
        return "hd_720"
    if side <= 1080:
        return "full_hd_1080"
    return "above_1080"


def is_grain_dominated(avg_blur, blur_scale_ratio):
    if avg_blur is None or blur_scale_ratio is None:
        return False
    return avg_blur >= BLUR_SLIGHTLY_SOFT and blur_scale_ratio >= BLUR_GRAIN_RATIO


def is_high_grain(median_noise):
    return median_noise is not None and float(median_noise) >= MEDIAN_NOISE_GRAINY


def is_sharpish(avg_blur, blur_scale_ratio, median_noise):
    """True when the blur score is real detail, not grain."""
    if avg_blur is None or avg_blur < BLUR_SLIGHTLY_SOFT:
        return False
    if is_grain_dominated(avg_blur, blur_scale_ratio):
        return False
    if is_high_grain(median_noise):
        return False
    return True


def classify_blur(avg_blur, blur_scale_ratio=None):
    if avg_blur is None:
        return "unknown_blur"
    if avg_blur < BLUR_VERY:
        return "very_blurry"
    if avg_blur < BLUR_BLURRY:
        return "blurry"
    if avg_blur < BLUR_SOFT:
        return "soft"
    if avg_blur < BLUR_SLIGHTLY_SOFT:
        return "slightly_soft"
    if is_grain_dominated(avg_blur, blur_scale_ratio) or avg_blur >= BLUR_SHARP:
        return "sharp_or_noisy"
    return "acceptable_sharpness"


def classify_blockiness(avg_block):
    if avg_block is None:
        return "unknown_blockiness"
    if avg_block > BLOCK_VERY:
        return "very_blocky"
    if avg_block > BLOCK_BLOCKY:
        return "blocky"
    if avg_block > BLOCK_SLIGHT:
        return "slightly_blocky"
    return "not_obviously_blocky"


def resolution_points(side):
    if side <= 0:
        return 0
    if side <= 270:
        return 15
    if side <= 360:
        return 12
    if side <= 480:
        return 9
    if side <= 576:
        return 6
    if side <= 720:
        return 3
    return 0


def blur_points(avg_blur):
    if avg_blur is None:
        return 0
    if avg_blur < BLUR_VERY:
        return 45
    if avg_blur < BLUR_BLURRY:
        return 32
    if avg_blur < BLUR_SOFT:
        return 18
    if avg_blur < BLUR_SLIGHTLY_SOFT:
        return 8
    return 0


def block_points(avg_block):
    if avg_block is None:
        return 0
    if avg_block > BLOCK_VERY:
        return 35
    if avg_block > BLOCK_BLOCKY:
        return 22
    if avg_block > BLOCK_SLIGHT:
        return 10
    return 0


def bitrate_points(side, bits_per_pixel_frame):
    if not bits_per_pixel_frame or bits_per_pixel_frame <= 0 or side <= 0:
        return 0
    if side >= 720:
        if bits_per_pixel_frame < HD_BPP_VERY_LOW:
            return 15
        if bits_per_pixel_frame < HD_BPP_LOW:
            return 8
        return 0
    if side <= 480:
        if bits_per_pixel_frame < SD_BPP_VERY_LOW:
            return 15
        if bits_per_pixel_frame < SD_BPP_LOW:
            return 8
    return 0


def calculate_issue_score(side, avg_blur, avg_block, bits_per_pixel_frame):
    """0-100 issue score. Higher means a stronger restoration candidate.

    Each part has its own ceiling, so a ruined 1080p file can reach the
    high band without help from a low-resolution bonus. The total is capped.
    """
    total = (
        resolution_points(side)
        + blur_points(avg_blur)
        + block_points(avg_block)
        + bitrate_points(side, bits_per_pixel_frame)
    )
    return round(min(SCORE_CAP, float(total)), 2)


def build_issue_flags(
    short_side_px,
    avg_blur,
    avg_block,
    bits_per_pixel_frame,
    processed=False,
    median_noise=None,
    mostly_dark=False,
    blur_scale_ratio=None,
):
    flags = []
    if processed:
        flags.append("already_processed_name")

    resolution_class = classify_resolution(short_side_px)
    if resolution_class in {"very_tiny", "very_low_resolution", "low_resolution", "sd_plus"}:
        flags.append(resolution_class)

    if mostly_dark:
        flags.append("mostly_dark_or_flat")
    else:
        blur_class = classify_blur(avg_blur, blur_scale_ratio)
        block_class = classify_blockiness(avg_block)
        if blur_class in {"very_blurry", "blurry", "soft", "slightly_soft"}:
            flags.append(blur_class)
        if is_grain_dominated(avg_blur, blur_scale_ratio):
            flags.append("grain_dominated_sharpness")
        if is_high_grain(median_noise):
            flags.append("high_grain")
        if block_class in {"very_blocky", "blocky", "slightly_blocky"}:
            flags.append(block_class)

    if bits_per_pixel_frame and bits_per_pixel_frame > 0:
        if short_side_px >= 720 and bits_per_pixel_frame < HD_BPP_LOW:
            flags.append("low_bitrate_for_hd")
        elif 0 < short_side_px <= 480 and bits_per_pixel_frame < SD_BPP_LOW:
            flags.append("low_bitrate_for_sd")

    if not flags:
        flags.append("ok_or_uncertain")
    return flags


def assign_recommendation(
    side,
    avg_blur,
    avg_block,
    score,
    processed=False,
    median_noise=None,
    blur_scale_ratio=None,
    mostly_dark=False,
):
    """Pick a review label from the same thresholds as the score.

    Specific restoration labels win over the generic high-score label.
    A sharp low-resolution file is an upscale candidate even when the
    score is low. Grain is not treated as detail worth upscaling.
    """
    low_res = 0 < side <= 576
    very_low_res = 0 < side <= 360
    blurry = avg_blur is not None and avg_blur < BLUR_SOFT
    very_blurry = avg_blur is not None and avg_blur < BLUR_BLURRY
    slightly_soft = (
        avg_blur is not None
        and BLUR_SOFT <= avg_blur < BLUR_SLIGHTLY_SOFT
    )
    blocky = avg_block is not None and avg_block > BLOCK_BLOCKY
    sharpish = is_sharpish(avg_blur, blur_scale_ratio, median_noise)
    grain_dominated = is_grain_dominated(avg_blur, blur_scale_ratio)
    high_grain = is_high_grain(median_noise)

    if mostly_dark:
        base = "manual_review"
    elif low_res and very_blurry and blocky:
        base = "restore_deblur_artifact_reduce"
    elif low_res and blurry:
        base = "restore_or_deblur_then_upscale"
    elif score is not None and score >= 75:
        base = "top_priority_restore"
    elif low_res and sharpish and not blocky:
        base = "upscale_candidate_very_low_res" if very_low_res else "upscale_only_gentle"
    elif low_res and slightly_soft:
        base = "manual_review"
    elif blocky:
        base = "artifact_reduction_review"
    elif grain_dominated or (high_grain and low_res):
        base = "manual_review"
    elif score is not None and score >= 45:
        base = "manual_review"
    else:
        base = "low_priority_or_ok"

    if processed:
        return "already_processed_review_only+" + base
    return base


# ============================================================
# Contact sheet generation
# ============================================================

def resize_keep_aspect(frame, target_width):
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame.copy()
    if width == target_width:
        return frame.copy()
    scale = target_width / float(width)
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def wrap_text(text, max_width, font_scale, thickness):
    """Wrap text so each line fits in max_width pixels."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    raw = "" if text is None else str(text)
    if max_width <= 0:
        return [raw]

    lines = []
    current = ""
    for word in raw.split(" ") if raw else [""]:
        candidate = word if current == "" else current + " " + word
        if cv2.getTextSize(candidate, font, font_scale, thickness)[0][0] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        if cv2.getTextSize(word, font, font_scale, thickness)[0][0] <= max_width:
            current = word
            continue
        chunk = ""
        for char in word:
            trial = chunk + char
            if cv2.getTextSize(trial, font, font_scale, thickness)[0][0] <= max_width:
                chunk = trial
            else:
                if chunk:
                    lines.append(chunk)
                chunk = char
        current = chunk
    if current or not lines:
        lines.append(current)
    return lines


def draw_text_box(img, lines, x, y, font_scale=0.5, thickness=1):
    """Draw text on a dark box. Returns the box height."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not lines:
        return 0
    sample, baseline = cv2.getTextSize("Ag", font, font_scale, thickness)
    line_height = sample[1] + baseline + 4
    padding = 6
    max_width = 0
    for line in lines:
        size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, size[0])
    box_w = max_width + padding * 2
    box_h = line_height * len(lines) + padding * 2
    height, width = img.shape[:2]
    x2 = min(width - 1, x + box_w)
    y2 = min(height - 1, y + box_h)
    if x2 > x and y2 > y:
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    text_y = y + padding + sample[1]
    for line in lines:
        cv2.putText(
            img,
            line,
            (x + padding, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        text_y += line_height
    return box_h


def contact_sheet_filename(video_path):
    """Parent folder, stem, and a hash of the full path.

    The hash keeps two folders that share a name from overwriting each other.
    """
    path = Path(video_path)
    try:
        key = str(path.resolve())
    except OSError:
        key = str(path)
    digest = hashlib.sha1(key.encode("utf-8", "surrogateescape")).hexdigest()[:10]
    parent = path.parent.name or "root"
    raw = f"{parent}__{path.stem}"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")[:80] or "video"
    return f"{safe}__{digest}.jpg"


def create_contact_sheet(video_path, frames_with_ts, result, output_folder, thumb_width):
    """Write one JPEG contact sheet. Returns the output path."""
    if not frames_with_ts:
        return None

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for timestamp, frame in frames_with_ts:
        thumb = resize_keep_aspect(frame, thumb_width)
        label = format_duration(timestamp) or f"{timestamp:.1f}s"
        draw_text_box(thumb, [label], x=5, y=5, font_scale=0.45, thickness=1)
        thumbs.append(thumb)

    max_h = max(thumb.shape[0] for thumb in thumbs)
    normalized = []
    for thumb in thumbs:
        height, width = thumb.shape[:2]
        if height < max_h:
            pad = np.zeros((max_h - height, width, 3), dtype=np.uint8)
            thumb = np.vstack([thumb, pad])
        if thumb.shape[1] < thumb_width:
            pad = np.zeros((thumb.shape[0], thumb_width - thumb.shape[1], 3), dtype=np.uint8)
            thumb = np.hstack([thumb, pad])
        normalized.append(thumb)

    cols = min(4, len(normalized))
    grid_rows = int(math.ceil(len(normalized) / cols))
    sheet_w = cols * thumb_width
    raw_lines = [
        Path(video_path).name,
        (
            f"score={result.get('quality_issue_score')}  "
            f"recommendation={result.get('recommendation')}"
        ),
        (
            f"res={result.get('width')}x{result.get('height')}  "
            f"display={result.get('display_width')}x{result.get('display_height')}  "
            f"blur={result.get('avg_blur_score')}  "
            f"block={result.get('avg_blockiness_score')}  "
            f"noise={result.get('avg_noise_score')}"
        ),
        f"flags={result.get('issue_flags')}",
    ]
    wrapped = []
    for line in raw_lines:
        wrapped.extend(wrap_text(line, max(32, sheet_w - 24), 0.5, 1))
    if len(wrapped) > 14:
        wrapped = wrapped[:13] + ["..."]

    sample, baseline = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    line_height = sample[1] + baseline + 4
    header_h = line_height * len(wrapped) + 20
    sheet = np.zeros((header_h + grid_rows * max_h, sheet_w, 3), dtype=np.uint8)
    draw_text_box(sheet, wrapped, x=8, y=8, font_scale=0.5, thickness=1)

    for idx, thumb in enumerate(normalized):
        grid_r = idx // cols
        grid_c = idx % cols
        y1 = header_h + grid_r * max_h
        x1 = grid_c * thumb_width
        height, width = thumb.shape[:2]
        height = min(height, max_h)
        width = min(width, thumb_width)
        sheet[y1:y1 + height, x1:x1 + width] = thumb[:height, :width]

    out_path = output_folder / contact_sheet_filename(video_path)
    if not cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"Could not write contact sheet: {out_path}")
    return str(out_path)


# ============================================================
# Video analysis
# ============================================================

def analyze_video(video_path, config=None):
    """Analyze one video and return a report row."""
    config = config or ScanConfig()
    video_path = Path(video_path)
    processed = is_probably_processed(video_path)

    if config.skip_already_processed and processed:
        row = base_row(video_path, processed=processed)
        row["issue_flags"] = "skipped_already_processed_name"
        row["recommendation"] = "skipped"
        return row

    if not video_path.exists():
        row = base_row(video_path, processed=processed)
        row["issue_flags"] = "missing_file"
        row["recommendation"] = "fix_path_or_missing_file"
        row["error"] = "File does not exist"
        return row

    meta = ffprobe_video(video_path, config.probe_timeout)
    row = base_row(video_path, meta=meta, processed=processed)
    if meta["width"] <= 0 or meta["height"] <= 0:
        row["issue_flags"] = "could_not_sample"
        row["recommendation"] = "manual_check_could_not_sample"
        row["error"] = "Could not read frame size"
        return row
    if meta["duration"] <= 0:
        row["issue_flags"] = "could_not_sample"
        row["recommendation"] = "manual_check_could_not_sample"
        row["error"] = "Could not read a positive duration"
        return row

    frames, decode_errors = extract_sample_frames(video_path, meta, config)
    row["sampled_frames"] = len(frames)
    if not frames:
        detail = decode_errors[-1] if decode_errors else ""
        message = "Could not sample frames"
        if detail:
            message = message + ": " + detail
        row["issue_flags"] = "could_not_sample"
        row["recommendation"] = "manual_check_could_not_sample"
        row["error"] = clip_error(message)
        return row

    measured = [
        measure_frame(frame, config.analysis_frame_max_width)
        for _timestamp, frame in frames
    ]
    stats = aggregate_metrics(measured)
    side = short_side(
        meta["width"],
        meta["height"],
        meta["display_width"],
        meta["display_height"],
    )
    bpp_frame = bits_per_pixel_per_frame(
        meta["bit_rate"], meta["width"], meta["height"], meta["frame_rate"]
    )
    bpp_second = bits_per_second_per_pixel(
        meta["bit_rate"], meta["width"], meta["height"]
    )
    # Black and flat samples are not evidence of blur or blocking.
    blur_for_score = None if stats["mostly_dark"] else stats["avg_blur"]
    block_for_score = None if stats["mostly_dark"] else stats["avg_block"]
    score = calculate_issue_score(side, blur_for_score, block_for_score, bpp_frame)
    flags = build_issue_flags(
        side,
        stats["avg_blur"],
        stats["avg_block"],
        bpp_frame,
        processed=processed,
        median_noise=stats["median_noise"],
        mostly_dark=stats["mostly_dark"],
        blur_scale_ratio=stats["blur_scale_ratio"],
    )
    recommendation = assign_recommendation(
        side,
        stats["avg_blur"],
        stats["avg_block"],
        score,
        processed=processed,
        median_noise=stats["median_noise"],
        blur_scale_ratio=stats["blur_scale_ratio"],
        mostly_dark=stats["mostly_dark"],
    )
    row.update({
        "frames_used": stats["frames_used"],
        "frames_excluded": stats["frames_excluded"],
        "avg_blur_score": round(stats["avg_blur"], 3),
        "min_blur_score": round(stats["min_blur"], 3),
        "max_blur_score": round(stats["max_blur"], 3),
        "blur_scale_ratio": round_or_none(stats["blur_scale_ratio"], 3),
        "avg_blockiness_score": round(stats["avg_block"], 3),
        "avg_noise_score": round(stats["avg_noise"], 3),
        "median_noise_score": round(stats["median_noise"], 3),
        "avg_brightness": round(stats["avg_brightness"], 3),
        "avg_black_frame_ratio": round(stats["avg_black"], 4),
        "bitrate_per_pixel": round_or_none(bpp_second, 6),
        "bits_per_pixel_frame": round_or_none(bpp_frame, 6),
        "quality_issue_score": score,
        "resolution_class": classify_resolution(side),
        "blur_class": classify_blur(stats["avg_blur"], stats["blur_scale_ratio"]),
        "blockiness_class": classify_blockiness(stats["avg_block"]),
        "issue_flags": ",".join(flags),
        "recommendation": recommendation,
    })

    if config.create_contact_sheets and score >= config.contact_sheet_min_score:
        try:
            sheet_path = create_contact_sheet(
                video_path,
                frames,
                row,
                config.contact_sheet_folder,
                config.contact_thumb_width,
            )
            row["contact_sheet"] = sheet_path or ""
        except Exception as exc:
            row["contact_sheet"] = ""
            row["contact_sheet_error"] = clip_error(exc)
    return row


def analyze_one(video_path, config):
    """Worker entry. Failures become report rows instead of killing the scan."""
    try:
        return analyze_video(video_path, config)
    except Exception as exc:
        row = base_row(video_path)
        row["issue_flags"] = "error"
        row["recommendation"] = "manual_check_error"
        row["error"] = clip_error(exc)
        return row


# ============================================================
# File discovery and report IO
# ============================================================

def find_video_files(root_folder, skip_dir_names=None):
    root = Path(root_folder)
    skip = {name.lower() for name in (skip_dir_names or set())}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames
            if name.lower() not in skip and not name.startswith(".")
        ]
        for filename in filenames:
            if filename.startswith("."):
                continue
            path = Path(dirpath) / filename
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                yield path


def paths_already_scanned(rows):
    found = set()
    for row in rows:
        path = row.get("path") if isinstance(row, dict) else None
        if path:
            found.add(str(path))
    return found


def normalize_legacy_row(row):
    """Old reports used 999 as an error sentinel. That is not a quality score."""
    score = row.get("quality_issue_score")
    try:
        if score is not None and score != "" and float(score) >= 900:
            row["quality_issue_score"] = None
    except (TypeError, ValueError):
        pass
    return row


def read_existing_rows(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise RuntimeError(
            f"Could not read existing report {path}: {exc}. "
            "Pass --no-resume to replace it."
        ) from exc
    if "path" not in frame.columns:
        raise RuntimeError(
            f"Existing report {path} has no path column. "
            "Pass --no-resume to replace it."
        )
    records = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
    return [normalize_legacy_row(row) for row in records]


def rows_to_dataframe(rows):
    if not rows:
        return pd.DataFrame(columns=CSV_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in CSV_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, CSV_COLUMNS]


def sorted_dataframe(rows):
    frame = rows_to_dataframe(rows)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["_score"] = pd.to_numeric(frame["quality_issue_score"], errors="coerce")
    frame["_blur"] = pd.to_numeric(frame["avg_blur_score"], errors="coerce")
    frame = frame.sort_values(
        by=["_score", "_blur"],
        ascending=[False, True],
        na_position="last",
    )
    return frame.drop(columns=["_score", "_blur"])


def write_report(path, rows):
    """Atomically write a score-sorted CSV. Error rows sort last."""
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    frame = sorted_dataframe(rows)
    tmp = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return frame


def ranked_mask(frame):
    scores = pd.to_numeric(frame["quality_issue_score"], errors="coerce")
    errors = frame["error"].fillna("").astype(str).str.strip()
    return scores.notna() & (errors == "")


def print_summary(frame):
    print("\n================ Summary ================\n")
    print(f"Total rows: {len(frame)}")

    if "error" in frame.columns:
        error_count = int((frame["error"].fillna("").astype(str).str.strip() != "").sum())
        print(f"Rows with error: {error_count}")
    if "recommendation" in frame.columns:
        skipped = int((frame["recommendation"].fillna("") == "skipped").sum())
        if skipped:
            print(f"Skipped rows: {skipped}")

    ranked = frame.loc[ranked_mask(frame)] if "quality_issue_score" in frame.columns else frame.iloc[0:0]
    print(f"Ranked videos: {len(ranked)}")

    if not ranked.empty:
        print("\nScore distribution:")
        scores = pd.to_numeric(ranked["quality_issue_score"], errors="coerce")
        bins = [
            ("90+", scores >= 90),
            ("75-89", (scores >= 75) & (scores < 90)),
            ("55-74", (scores >= 55) & (scores < 75)),
            ("35-54", (scores >= 35) & (scores < 55)),
            ("0-34", (scores >= 0) & (scores < 35)),
        ]
        for label, mask in bins:
            print(f"  {label}: {int(mask.sum())}")

    if "recommendation" in frame.columns:
        print("\nRecommendation counts:")
        counts = frame["recommendation"].fillna("").value_counts()
        for rec, count in counts.items():
            print(f"  {rec}: {count}")

    print("\nTop 25 candidates:")
    columns_to_show = [
        "quality_issue_score",
        "recommendation",
        "width",
        "height",
        "display_width",
        "display_height",
        "avg_blur_score",
        "avg_blockiness_score",
        "bits_per_pixel_frame",
        "issue_flags",
        "path",
        "contact_sheet",
    ]
    if ranked.empty:
        print("  none")
        return
    existing_cols = [column for column in columns_to_show if column in ranked.columns]
    print(ranked[existing_cols].head(25).to_string(index=False))


# ============================================================
# Main
# ============================================================

def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def analysis_width_type(value):
    width = positive_int(value)
    if width < 32:
        raise argparse.ArgumentTypeError("must be at least 32")
    return width


def positive_float(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        description="Scan videos, estimate quality issues, and write a CSV report.",
        epilog=(
            "examples:\n"
            "  python scan_video_quality.py --root ./videos --output report.csv\n"
            "  python scan_video_quality.py --root ./videos --workers 4 --contact-sheets\n"
            "  python scan_video_quality.py --root ./videos --no-resume\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default=ROOT_FOLDER, help="Folder to scan recursively")
    parser.add_argument("--output", default=OUTPUT_CSV, help="CSV report path")
    parser.add_argument("--samples", type=positive_int, default=SAMPLE_COUNT, help="Frames per video")
    parser.add_argument(
        "--analysis-width",
        type=analysis_width_type,
        default=ANALYSIS_FRAME_MAX_WIDTH,
        help="Width used only for the blur score (default matches the blur thresholds)",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=DEFAULT_WORKERS,
        help="Videos to analyze at once (default: up to 4)",
    )
    parser.add_argument("--frame-timeout", type=positive_float, default=60.0, help="Seconds per sample frame")
    parser.add_argument("--probe-timeout", type=positive_float, default=30.0, help="Seconds per ffprobe call")
    parser.add_argument(
        "--checkpoint-every",
        type=positive_int,
        default=5,
        help="Rewrite the CSV after this many newly finished videos",
    )
    parser.add_argument("--contact-sheet-min-score", type=float, default=CONTACT_SHEET_MIN_SCORE)
    parser.add_argument("--contact-sheet-folder", default=CONTACT_SHEET_FOLDER)
    parser.add_argument("--contact-sheets", dest="create_contact_sheets", action="store_true")
    parser.add_argument("--no-contact-sheets", dest="create_contact_sheets", action="store_false")
    parser.add_argument("--skip-processed", dest="skip_already_processed", action="store_true")
    parser.add_argument("--no-skip-processed", dest="skip_already_processed", action="store_false")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(
        create_contact_sheets=CREATE_CONTACT_SHEETS,
        skip_already_processed=SKIP_ALREADY_PROCESSED,
        resume=True,
    )
    return parser


def config_from_args(args):
    return ScanConfig(
        root_folder=args.root,
        output_csv=args.output,
        create_contact_sheets=args.create_contact_sheets,
        contact_sheet_min_score=args.contact_sheet_min_score,
        contact_sheet_folder=args.contact_sheet_folder,
        sample_count=args.samples,
        analysis_frame_max_width=args.analysis_width,
        contact_thumb_width=CONTACT_THUMB_WIDTH,
        skip_already_processed=args.skip_already_processed,
        workers=args.workers,
        resume=args.resume,
        frame_timeout=args.frame_timeout,
        probe_timeout=args.probe_timeout,
        checkpoint_every=args.checkpoint_every,
    )


def scan_library(config):
    check_external_tools()
    root = Path(config.root_folder)
    if not root.exists():
        raise RuntimeError(f"Root folder does not exist: {root}")
    if not root.is_dir():
        raise RuntimeError(f"Root folder is not a directory: {root}")

    module_dir = str(Path(__file__).resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    skip_dirs = {"contact_sheets", Path(config.contact_sheet_folder).name}
    files = sorted(find_video_files(root, skip_dirs), key=lambda path: str(path).lower())
    print(f"Scanning root folder: {root}")
    print(f"Found {len(files)} video files.")

    existing = read_existing_rows(config.output_csv) if config.resume else []
    done = paths_already_scanned(existing)
    pending = [path for path in files if str(path) not in done]
    if config.resume and existing:
        print(
            f"Resume: skipping {len(files) - len(pending)} file(s) "
            f"already in {config.output_csv}."
        )

    worker_count = min(config.workers, len(pending)) if pending else 1
    print(f"Analyzing {len(pending)} video file(s) with {worker_count} worker(s).")
    rows = list(existing) if config.resume else []

    def checkpoint():
        write_report(config.output_csv, rows)

    if not pending:
        frame = checkpoint()
        print(f"\nDone. Report saved to: {config.output_csv}")
        print_summary(frame)
        return rows

    def store_result(row):
        rows.append(row)
        if row.get("error"):
            tqdm.write(f"Error: {row.get('path')}: {row.get('error')}")
        if len(rows) - len(existing if config.resume else []) >= 1:
            finished_new = len(rows) - (len(existing) if config.resume else 0)
            if finished_new % config.checkpoint_every == 0:
                checkpoint()

    try:
        if worker_count == 1:
            for video_path in tqdm(pending, desc="Analyzing videos"):
                store_result(analyze_one(video_path, config))
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
                futures = {
                    executor.submit(analyze_one, path, config): path
                    for path in pending
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Analyzing videos",
                ):
                    video_path = futures[future]
                    try:
                        store_result(future.result())
                    except Exception as exc:
                        failed = base_row(video_path)
                        failed["issue_flags"] = "error"
                        failed["recommendation"] = "manual_check_error"
                        failed["error"] = clip_error(exc)
                        store_result(failed)
    except KeyboardInterrupt:
        checkpoint()
        print(f"\nInterrupted. Partial report saved to: {config.output_csv}")
        raise

    frame = checkpoint()
    print(f"\nDone. Report saved to: {config.output_csv}")
    if config.create_contact_sheets:
        print(f"Contact sheets saved to: {config.contact_sheet_folder}")
    print_summary(frame)
    return rows


def main(argv=None):
    config = config_from_args(build_parser().parse_args(argv))
    scan_library(config)


if __name__ == "__main__":
    main()
