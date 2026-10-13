"""Scan a video folder and write a CSV of heuristic quality issues.

Frame samples are full-resolution raw frames. Blur is measured after a
letterbox crop and a scale to the analysis width. Blockiness stays on the
decoded pixel grid. Noise uses a float estimator. Pair mode asks whether a
processed file beat a plain upscale of the original. Run
``python scan_video_quality.py --help`` for the command-line options.
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
from dataclasses import dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

SCHEMA_VERSION = 3

ROOT_FOLDER = r"./videos"
OUTPUT_CSV = "video_quality_report.csv"

CREATE_CONTACT_SHEETS = False
CONTACT_SHEET_MIN_SCORE = 55
CONTACT_SHEET_FOLDER = "contact_sheets"

# Used only when --samples is passed. The default scan is adaptive.
SAMPLE_COUNT = 8
MIN_SAMPLE_COUNT = 16
MAX_SAMPLE_COUNT = 40
SAMPLE_EVERY_SECONDS = 300.0

ANALYSIS_FRAME_MAX_WIDTH = 640
CONTACT_THUMB_WIDTH = 320

SKIP_ALREADY_PROCESSED = False
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

ENCODER_MARKERS = ("topaz", "proteus", "artemis", "video2x", "realesrgan")

PAIR_MARKER_RE = re.compile(
    r"(?:[_\-. ](?:prob|pnat|proteus|gaia|artemis|iris|topaz|upscale|enhanced|denoise|deblur)\d*)+$",
    re.IGNORECASE,
)

INTERLACED_FIELD_ORDERS = {"tt", "bb", "tb", "bt"}

# Codec efficiency relative to H.264. The bitrate score uses
# bits_per_pixel_frame * this factor. HEVC at the same bitrate is not
# identical to H.264, but a 1.5x factor does not by itself clear a
# 2 Mbps 1080p stream from the low-bitrate band.
CODEC_EFFICIENCY = {
    "mpeg1video": 0.55,
    "mpeg2video": 0.6,
    "mpeg4": 0.7,
    "msmpeg4v2": 0.65,
    "msmpeg4v3": 0.7,
    "h263": 0.65,
    "h264": 1.0,
    "avc": 1.0,
    "hevc": 1.5,
    "h265": 1.5,
    "vp8": 0.9,
    "vp9": 1.4,
    "av1": 1.8,
}

CONTACT_REVIEW = {
    "manual_review",
    "check_source_upscaled_or_denoise",
    "vr_equirect_review",
}


@dataclass
class Thresholds:
    """Tunable cutoffs. A config file overrides these defaults."""

    blur_very: float = 35.0
    blur_blurry: float = 60.0
    blur_soft: float = 100.0
    blur_slightly_soft: float = 150.0
    blur_sharp: float = 300.0
    blur_grain_ratio: float = 2.0
    block_slight: float = 1.30
    block_blocky: float = 1.45
    block_very: float = 1.60
    median_noise_grainy: float = 4.0
    hd_bpp_very_low: float = 0.04
    hd_bpp_low: float = 0.07
    sd_bpp_very_low: float = 0.05
    sd_bpp_low: float = 0.09
    dark_pixel_threshold: float = 16.0
    dark_frame_ratio: float = 0.80
    dark_mean_brightness: float = 20.0
    flat_std_max: float = 8.0
    score_cap: float = 100.0
    combing_min: float = 0.12
    halo_flag: float = 8.0
    vr_min_width: int = 3840
    vr_aspect_tolerance: float = 0.02
    unstable_blur_cv: float = 1.0
    unstable_min_frames: int = 8
    pair_duration_seconds: float = 1.0
    pair_ssim_align: float = 0.60
    pair_align_fail_fraction: float = 0.34
    pair_min_frames: int = 4
    pair_blur_detail_ratio: float = 0.85
    pair_blur_severe_ratio: float = 0.65
    pair_noise_drop_ratio: float = 0.85
    pair_noise_ratio: float = 1.30
    pair_block_delta: float = 0.10
    pair_halo_ratio: float = 1.45
    pair_halo_delta: float = 4.0
    pair_blur_vs_base: float = 1.15
    pair_luma_shift: float = 10.0
    pair_chroma_shift: float = 8.0
    geometry_aspect_tolerance: float = 0.05
    geometry_fit_tolerance: float = 0.03
    geometry_crop_area_max: float = 0.10
    codec_efficiency: dict = field(default_factory=lambda: dict(CODEC_EFFICIENCY))


DEFAULT_THRESHOLDS = Thresholds()

# Names kept for existing tests and for callers that read the defaults.
BLUR_VERY = DEFAULT_THRESHOLDS.blur_very
BLUR_BLURRY = DEFAULT_THRESHOLDS.blur_blurry
BLUR_SOFT = DEFAULT_THRESHOLDS.blur_soft
BLUR_SLIGHTLY_SOFT = DEFAULT_THRESHOLDS.blur_slightly_soft
BLUR_SHARP = DEFAULT_THRESHOLDS.blur_sharp
BLUR_GRAIN_RATIO = DEFAULT_THRESHOLDS.blur_grain_ratio
BLOCK_SLIGHT = DEFAULT_THRESHOLDS.block_slight
BLOCK_BLOCKY = DEFAULT_THRESHOLDS.block_blocky
BLOCK_VERY = DEFAULT_THRESHOLDS.block_very
MEDIAN_NOISE_GRAINY = DEFAULT_THRESHOLDS.median_noise_grainy
HD_BPP_VERY_LOW = DEFAULT_THRESHOLDS.hd_bpp_very_low
HD_BPP_LOW = DEFAULT_THRESHOLDS.hd_bpp_low
SD_BPP_VERY_LOW = DEFAULT_THRESHOLDS.sd_bpp_very_low
SD_BPP_LOW = DEFAULT_THRESHOLDS.sd_bpp_low
DARK_PIXEL_THRESHOLD = DEFAULT_THRESHOLDS.dark_pixel_threshold
DARK_FRAME_RATIO = DEFAULT_THRESHOLDS.dark_frame_ratio
DARK_MEAN_BRIGHTNESS = DEFAULT_THRESHOLDS.dark_mean_brightness
FLAT_STD_MAX = DEFAULT_THRESHOLDS.flat_std_max
SCORE_CAP = DEFAULT_THRESHOLDS.score_cap

CSV_COLUMNS = [
    "schema_version",
    "path",
    "filename",
    "folder",
    "mtime",
    "width",
    "height",
    "display_width",
    "display_height",
    "content_width",
    "content_height",
    "content_aspect",
    "crop",
    "fix_geometry",
    "sample_aspect_ratio",
    "rotation",
    "codec",
    "pix_fmt",
    "encoder",
    "color_range",
    "color_space",
    "field_order",
    "duration",
    "duration_hms",
    "frame_rate",
    "bit_rate",
    "size",
    "is_probably_processed",
    "is_vr",
    "sampled_frames",
    "frames_used",
    "frames_excluded",
    "sample_errors",
    "avg_blur_score",
    "median_blur_score",
    "blur_p10",
    "blur_p90",
    "blur_iqr",
    "blur_cv",
    "min_blur_score",
    "max_blur_score",
    "blur_scale_ratio",
    "blur_best_tiles",
    "avg_blockiness_score",
    "avg_noise_score",
    "median_noise_score",
    "avg_halo_score",
    "combing_score",
    "avg_brightness",
    "avg_black_frame_ratio",
    "bitrate_per_pixel",
    "bits_per_pixel_frame",
    "bits_per_pixel_frame_eff",
    "score_resolution",
    "score_blur",
    "score_blockiness",
    "score_bitrate",
    "quality_issue_score",
    "resolution_class",
    "blur_class",
    "blockiness_class",
    "issue_flags",
    "recommendation",
    "contact_sheet",
    "contact_sheet_error",
    "sample_error",
    "error",
]

GROUP_COLUMNS = [
    "group_id",
    "folder",
    "original_path",
    "other_path",
    "original_filename",
    "other_filename",
    "duration_delta",
    "duration_mismatch",
    "geometry_mismatch",
    "original_display",
    "other_display",
    "original_content",
    "other_content",
    "blur_native",
    "blur_vs_base",
    "block_native",
    "noise_native",
    "halo_vs_base",
    "ssim",
    "luma_delta",
    "chroma_delta",
    "range_shift",
    "aligned_frames",
    "dropped_frames",
    "verdict",
    "action",
    "best_version",
    "note",
]


@dataclass
class ScanConfig:
    root_folder: str = ROOT_FOLDER
    output_csv: str = OUTPUT_CSV
    create_contact_sheets: bool = CREATE_CONTACT_SHEETS
    contact_sheet_min_score: float = CONTACT_SHEET_MIN_SCORE
    contact_sheet_folder: str = CONTACT_SHEET_FOLDER
    sample_count: int = SAMPLE_COUNT
    samples_explicit: bool = False
    analysis_frame_max_width: int = ANALYSIS_FRAME_MAX_WIDTH
    contact_thumb_width: int = CONTACT_THUMB_WIDTH
    skip_already_processed: bool = SKIP_ALREADY_PROCESSED
    workers: int = DEFAULT_WORKERS
    resume: bool = True
    frame_timeout: float = 60.0
    probe_timeout: float = 30.0
    checkpoint_every: int = 5
    pairs: bool = True
    groups_csv: str = ""
    thresholds: Thresholds = field(default_factory=Thresholds)


def active_thresholds(thresholds=None):
    return thresholds or DEFAULT_THRESHOLDS


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
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_bool(value):
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def round_or_none(value, digits):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


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


def looks_processed(path, encoder=""):
    """Filename markers, or an encoder tag from a known restoration tool."""
    if is_probably_processed(path):
        return True
    text = (encoder or "").lower()
    return any(marker in text for marker in ENCODER_MARKERS)


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


def file_identity(path):
    """Size and mtime, the resume identity together with the path."""
    st = Path(path).stat()
    return int(st.st_size), int(st.st_mtime)


def path_seed(path):
    digest = hashlib.sha1(str(path).encode("utf-8", "surrogateescape")).digest()
    return int.from_bytes(digest[:4], "little")


def codec_efficiency(codec, thresholds=None):
    table = active_thresholds(thresholds).codec_efficiency or {}
    if not codec:
        return 1.0
    try:
        return float(table.get(str(codec).lower(), 1.0))
    except (TypeError, ValueError):
        return 1.0


def recommendation_base(text):
    raw = "" if text is None else str(text)
    if "+" in raw:
        return raw.split("+", 1)[1]
    return raw


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
        schema_version=SCHEMA_VERSION,
        path=str(path),
        filename=path.name,
        folder=str(path.parent),
        is_probably_processed=bool(processed),
        is_vr=False,
        sampled_frames=0,
        frames_used=0,
        frames_excluded=0,
        sample_errors=0,
        issue_flags="",
        recommendation="",
        contact_sheet="",
        contact_sheet_error="",
        sample_error="",
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
            "encoder",
            "color_range",
            "color_space",
            "field_order",
            "duration",
            "duration_hms",
            "frame_rate",
            "bit_rate",
            "size",
        ):
            if key in meta:
                row[key] = meta.get(key)
    return row


def coerce_scalar(value):
    text = value.strip().strip("'\"")
    lower = text.lower()
    if lower in {"true", "yes"}:
        return True
    if lower in {"false", "no"}:
        return False
    try:
        if any(token in lower for token in (".", "e")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_simple_yaml(text):
    """Parse the flat config file this scanner writes and reads.

    Supports nested maps one level deep, comments, and scalars.
    """
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line: {raw.strip()}")
        key, _, value = line.strip().partition(":")
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid config indentation near {key}")
        parent = stack[-1][1]
        if not isinstance(parent, dict):
            raise ValueError(f"Cannot nest under {key}")
        if value == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = coerce_scalar(value)
    return root


# Removed in schema 3. Old config files may still name them.
_RETIRED_CONFIG_KEYS = {"upscale_bandwidth_max", "bandwidth_drop_db"}


def thresholds_from_mapping(mapping, base=None):
    """Overlay a config mapping on the built-in thresholds."""
    base = base or Thresholds()
    if not mapping:
        return base
    mapping = {key: value for key, value in mapping.items() if key not in _RETIRED_CONFIG_KEYS}
    unknown = sorted(key for key in mapping if key not in {item.name for item in fields(base)})
    if unknown:
        raise ValueError("Unknown config keys: " + ", ".join(unknown))
    values = {}
    for key, value in mapping.items():
        default = getattr(base, key)
        if key == "codec_efficiency":
            if not isinstance(value, dict):
                raise ValueError("codec_efficiency must be a map of codec names")
            merged = dict(base.codec_efficiency)
            for codec, factor in value.items():
                merged[str(codec).lower()] = float(factor)
            values[key] = merged
        elif isinstance(default, float):
            values[key] = float(value)
        elif isinstance(default, bool):
            values[key] = bool(value)
        elif isinstance(default, int):
            values[key] = int(value)
        else:
            values[key] = value
    data = {item.name: getattr(base, item.name) for item in fields(base)}
    data.update(values)
    return Thresholds(**data)


def load_thresholds(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return thresholds_from_mapping(parse_simple_yaml(text))


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
    tags = stream.get("tags") or {}
    fmt_tags = fmt.get("tags") or {}
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
        "encoder": tags.get("encoder") or fmt_tags.get("encoder") or "",
        "color_range": stream.get("color_range") or "",
        "color_space": stream.get("color_space") or "",
        "field_order": stream.get("field_order") or "",
        "duration": duration,
        "duration_hms": format_duration(duration),
        "frame_rate": round(frame_rate, 3) if frame_rate > 0 else 0.0,
        "bit_rate": bit_rate,
        "size": size,
    }


# ============================================================
# Frame extraction
# ============================================================

def resolved_sample_count(duration, config):
    """How many frames to take from one file.

    ``--samples`` forces a count. Otherwise the scan takes at least 16
    frames, one more for every five minutes, and never more than 40.
    """
    if config is not None and config.samples_explicit:
        return max(1, int(config.sample_count))
    adaptive = int(round(float(duration or 0) / SAMPLE_EVERY_SECONDS))
    return min(MAX_SAMPLE_COUNT, max(MIN_SAMPLE_COUNT, adaptive))


def sample_timestamps(duration, sample_count, seed=None):
    """Timestamps spread across the file, skipping the head and tail.

    ``seed`` adds a small repeatable jitter so two runs of the same file
    land on the same frames without always hitting the same scene beat.
    """
    if duration <= 0 or sample_count <= 0:
        return []
    last_ok = max(float(duration) - 0.04, 0.0)
    if int(sample_count) == 1:
        return [min(float(duration) * 0.5, last_ok)]

    edge = 0.05 if duration < 60 else 0.04
    start = float(duration) * edge
    end = float(duration) * (1.0 - edge)
    if end < start:
        end = start
    stamps = np.linspace(start, end, int(sample_count))
    if seed is not None and len(stamps) > 1 and end > start:
        gap = (end - start) / float(len(stamps))
        jitter = gap * 0.12
        rng = np.random.default_rng(int(seed) % (2**32))
        stamps = np.asarray(stamps, dtype=np.float64) + rng.uniform(-jitter, jitter, size=len(stamps))
    return [float(min(max(float(stamp), 0.0), last_ok)) for stamp in stamps]


def seek_points(timestamp):
    """Fast seek to 1s before the target, then an accurate decode."""
    timestamp = max(0.0, float(timestamp))
    pre = max(0.0, timestamp - 1.0)
    fine = timestamp - pre
    return pre, fine


def build_frame_command(video_path, timestamp, threads=0, frame_count=1):
    """ffmpeg command for one or more full-resolution raw frames.

    ``-ss`` before ``-i`` jumps near the timestamp. ``-ss`` after ``-i``
    decodes to the exact time so the sample is not just the nearest
    keyframe. ``-noautorotate`` keeps the coded pixel grid, which is the
    grid blockiness is measured on. There is no scale and no JPEG.
    """
    pre, fine = seek_points(timestamp)
    count = max(1, int(frame_count))
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
        "-frames:v", str(count),
        "-an",
        "-sn",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ])
    return cmd


def decode_frames(video_path, timestamp, width, height, timeout, threads, frame_count=1):
    """Return (frames, error). error is empty when at least one frame decoded."""
    if width <= 0 or height <= 0 or width > 16384 or height > 16384:
        return [], f"Refusing to decode frame size {width}x{height}"

    cmd = build_frame_command(
        video_path,
        timestamp,
        threads=threads,
        frame_count=frame_count,
    )
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], f"Timed out decoding frame at {float(timestamp):.2f}s"

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", "replace").strip()
        return [], err or f"ffmpeg failed at {float(timestamp):.2f}s"

    frame_bytes = int(width) * int(height) * 3
    if len(result.stdout) < frame_bytes:
        return [], (
            f"Unexpected frame size {len(result.stdout)} "
            f"(expected {frame_bytes}) at {float(timestamp):.2f}s"
        )
    got = len(result.stdout) // frame_bytes
    frames = []
    for index in range(got):
        start = index * frame_bytes
        raw = np.frombuffer(result.stdout[start:start + frame_bytes], dtype=np.uint8)
        frames.append(raw.reshape((int(height), int(width), 3)).copy())
    return frames, ""


def decode_frame(video_path, timestamp, width, height, timeout, threads):
    """Return (bgr_frame, error). error is empty on success."""
    frames, error = decode_frames(video_path, timestamp, width, height, timeout, threads, 1)
    if not frames:
        return None, error or f"ffmpeg failed at {float(timestamp):.2f}s"
    return frames[0], ""


def extract_sample_frames(video_path, meta, config, sample_count=None, seed=None):
    """Decode sample frames. One bad timestamp does not drop the file."""
    count = sample_count
    if count is None:
        count = resolved_sample_count(meta["duration"], config)
    if seed is None and config is not None:
        seed = path_seed(video_path)
    timestamps = sample_timestamps(meta["duration"], count, seed=seed)
    threads = 1 if config is not None and config.workers > 1 else 0
    timeout = 60.0 if config is None else config.frame_timeout
    frames = []
    errors = []
    for timestamp in timestamps:
        frame, error = decode_frame(
            video_path,
            timestamp,
            meta["width"],
            meta["height"],
            timeout,
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


def tile_sharpness(gray, tile=64):
    """Mean Laplacian variance of the sharpest fifth of tiles.

    This is recorded beside the full-frame score. It is not the number
    the blur thresholds were tuned on.
    """
    gray = to_gray(gray)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    height, width = lap.shape[:2]
    if height < tile or width < tile:
        return laplacian_var(gray)
    values = []
    for y in range(0, height - tile + 1, tile):
        for x in range(0, width - tile + 1, tile):
            values.append(float(lap[y:y + tile, x:x + tile].var()))
    if not values:
        return laplacian_var(gray)
    values.sort()
    keep = max(1, len(values) // 5)
    return float(np.mean(values[-keep:]))


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


def _block_period(scale):
    """8x8 codec grid after a resize. Near 1x stays on the original period."""
    if abs(float(scale) - 1.0) < 0.08:
        return 8
    return max(2, int(round(8.0 * float(scale))))


def blockiness_scaled(frame, scale_x=1.0, scale_y=1.0):
    """Blockiness on the codec grid after a resize, measured from frame origin.

    A 1.5x upscale turns 8x8 blocks into 12x12 blocks. Measuring that frame
    on an 8-pixel period misses them. The coded frame is not cropped first,
    so the grid still starts at pixel 0.
    """
    gray = to_gray(frame).astype(np.float32)
    height, width = gray.shape[:2]
    if height < 16 or width < 16:
        return 0.0
    diff_x = np.abs(gray[:, 1:] - gray[:, :-1])
    diff_y = np.abs(gray[1:, :] - gray[:-1, :])
    period_x = _block_period(scale_x)
    period_y = _block_period(scale_y)
    cols = np.arange(period_x - 1, diff_x.shape[1], period_x)
    rows = np.arange(period_y - 1, diff_y.shape[0], period_y)
    if cols.size < 2 or rows.size < 2:
        return 0.0
    boundary = (float(diff_x[:, cols].mean()) + float(diff_y[rows, :].mean())) / 2.0
    mask_x = np.ones(diff_x.shape[1], dtype=bool)
    mask_y = np.ones(diff_y.shape[0], dtype=bool)
    mask_x[cols] = False
    mask_y[rows] = False
    if not mask_x.any() or not mask_y.any():
        return 0.0
    interior = (float(diff_x[:, mask_x].mean()) + float(diff_y[mask_y, :].mean())) / 2.0
    return float(boundary / max(interior, 1e-6))


def noise_profile(frame):
    """Mean absolute residual after a 3x3 median blur, on 8-bit pixels."""
    gray = to_gray(frame)
    residual = cv2.absdiff(gray, cv2.medianBlur(gray, 3))
    return float(np.mean(residual))


def noise_sigma(gray, std_max=12.0):
    """Immerkaer noise sigma on locally flat pixels.

    The residual is float, and edges are left out. A checkerboard has no
    flat pixels, so it returns 0 instead of counting the pattern as grain.
    """
    g = np.asarray(to_gray(gray), np.float32)
    if g.shape[0] < 8 or g.shape[1] < 8:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    response = np.abs(cv2.filter2D(g, cv2.CV_32F, kernel)[1:-1, 1:-1])
    box = np.ones((5, 5), np.float32) / 25.0
    mean = cv2.filter2D(g, cv2.CV_32F, box)[1:-1, 1:-1]
    mean_sq = cv2.filter2D(g * g, cv2.CV_32F, box)[1:-1, 1:-1]
    local = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    flat = local <= float(std_max)
    if int(flat.sum()) < 64:
        return 0.0
    return float(response[flat].mean() * np.sqrt(0.5 * np.pi) / 6.0)


def noise_score(frame):
    """Mean high-frequency residual. Grain is not always a defect."""
    return noise_profile(frame)


def halo_score(gray):
    """Mean overshoot, in pixel levels, along strong edges.

    A clean step edge scores near 0. Unsharp masking pushes this up.
    """
    g = np.asarray(to_gray(gray), np.float32)
    if g.shape[0] < 8 or g.shape[1] < 8:
        return 0.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    threshold = max(float(np.percentile(mag, 92)), 8.0)
    ys, xs = np.nonzero(mag >= threshold)
    if ys.size < 40:
        return 0.0
    if ys.size > 2500:
        selected = np.linspace(0, ys.size - 1, 2500).astype(int)
        ys = ys[selected]
        xs = xs[selected]
    inv = 1.0 / (mag[ys, xs] + 1e-6)
    dx = gx[ys, xs] * inv
    dy = gy[ys, xs] * inv
    height, width = g.shape

    def sample(shift):
        yy = np.clip(np.rint(ys + dy * shift).astype(int), 0, height - 1)
        xx = np.clip(np.rint(xs + dx * shift).astype(int), 0, width - 1)
        return g[yy, xx]

    dark_side = sample(-2)
    before = sample(-1)
    after = sample(1)
    bright_side = sample(2)
    undershoot = np.maximum(0, dark_side - before)
    overshoot = np.maximum(0, after - bright_side)
    return float(np.mean(undershoot + overshoot))


def combing_score(gray):
    """Fraction of pixels whose neighbour row disagrees more than the same field.

    Progressive frames stay near 0. A woven pair of fields scores higher.
    """
    g = np.asarray(to_gray(gray), np.float32)
    if g.shape[0] < 8:
        return 0.0
    current = g[2:, :]
    adjacent = g[1:-1, :]
    same_field = g[:-2, :]
    inter = np.abs(current - adjacent)
    intra = np.abs(current - same_field)
    return float(np.mean(inter > (intra + 12.0)))


def letterbox_bounds(frame, thresh=16):
    """Crop rectangle (x, y, w, h) for black bars on the outer edges."""
    gray = to_gray(frame)
    height, width = gray.shape[:2]
    if height < 32 or width < 32:
        return (0, 0, width, height)
    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)
    top = 0
    while top < height - 1 and row_mean[top] <= thresh:
        top += 1
    bottom = height
    while bottom > top + 1 and row_mean[bottom - 1] <= thresh:
        bottom -= 1
    left = 0
    while left < width - 1 and col_mean[left] <= thresh:
        left += 1
    right = width
    while right > left + 1 and col_mean[right - 1] <= thresh:
        right -= 1
    if top < 4:
        top = 0
    if height - bottom < 4:
        bottom = height
    if left < 4:
        left = 0
    if width - right < 4:
        right = width
    crop_h = bottom - top
    crop_w = right - left
    if crop_h < height * 0.5 or crop_w < width * 0.5 or crop_h < 16 or crop_w < 16:
        return (0, 0, width, height)
    return (left, top, crop_w, crop_h)


def _boxes_near(left, right, tol=8):
    return all(abs(int(left[index]) - int(right[index])) <= tol for index in range(4))


def _median_box(boxes, width, height):
    left = int(np.median([box[0] for box in boxes]))
    top = int(np.median([box[1] for box in boxes]))
    right = int(np.median([box[0] + box[2] for box in boxes]))
    bottom = int(np.median([box[1] + box[3] for box in boxes]))
    left = max(0, min(left, width - 16))
    top = max(0, min(top, height - 16))
    right = max(left + 16, min(right, width))
    bottom = max(top + 16, min(bottom, height))
    if (bottom - top) < height * 0.5 or (right - left) < width * 0.5:
        return (0, 0, width, height)
    return (left, top, right - left, bottom - top)


def consensus_letterbox(frames, thresh=16, dark_mean=20.0):
    """One crop for the file, from non-dark frames that agree on the bars.

    A crop is kept only when it matches at least half of the frames that
    are bright enough to show a bar. One dark scene cannot set the crop.
    """
    if not frames:
        return (0, 0, 0, 0)
    gray = to_gray(frames[0])
    height, width = gray.shape[:2]
    full = (0, 0, width, height)
    usable = []
    for frame in frames:
        picture = to_gray(frame)
        if float(np.mean(picture)) <= float(dark_mean):
            continue
        usable.append(picture)
    if not usable:
        return full
    boxes = [letterbox_bounds(picture, thresh) for picture in usable]
    best = full
    best_count = sum(1 for box in boxes if _boxes_near(box, full))
    real = [box for box in boxes if not _boxes_near(box, full, tol=2)]
    if real:
        candidate = _median_box(real, width, height)
        count = sum(1 for box in boxes if _boxes_near(box, candidate))
        if count > best_count:
            best = candidate
            best_count = count
    if best_count * 2 < len(usable):
        return full
    return best


def even_dim(value):
    """Even pixel size, the size yuv420 encoders want."""
    number = max(2, int(round(float(value))))
    if number % 2:
        number += 1
    return number


def area_loss_to_16x9(aspect):
    """Fraction of the picture removed by a center crop to 16:9."""
    target = 16.0 / 9.0
    if aspect is None or aspect <= 0:
        return 1.0
    if aspect >= target:
        return 1.0 - (target / float(aspect))
    return 1.0 - (float(aspect) / target)


def geometry_canvas(display_w, display_h):
    """Standard 16:9 frame for HD content. SD stays at its own size."""
    short_side = min(display_w, display_h)
    long_side = max(display_w, display_h)
    if short_side >= 900 or long_side >= 1600:
        return (1920, 1080)
    if short_side >= 600 or long_side >= 1100:
        return (1280, 720)
    return None


def geometry_plan(frame_w, frame_h, crop, sar=1.0, rotation=0.0, vr=False, thresholds=None):
    """Label for a re-encode that clears bars, SAR, and a non-16:9 frame.

    The label is empty when the content is already square-pixel 16:9.
    SD is padded or cropped at its own display size. 720p and 1080p
    content use a 1280x720 or 1920x1080 canvas. The picture is not
    upscaled just to reach that canvas: a 4:3 SD frame is padded.
    """
    limits = active_thresholds(thresholds)
    if vr or frame_w <= 0 or frame_h <= 0 or not crop:
        return ""
    _x, _y, crop_w, crop_h = [int(value) for value in crop]
    display_w, display_h = display_dimensions(crop_w, crop_h, float(sar or 1.0), rotation)
    if display_w <= 0 or display_h <= 0:
        return ""
    aspect = float(display_w) / float(display_h)
    target = 16.0 / 9.0
    relative = abs(aspect - target) / target
    loss = area_loss_to_16x9(aspect)
    if relative <= limits.geometry_fit_tolerance:
        mode = "fit"
    elif loss <= limits.geometry_crop_area_max:
        mode = "crop"
    else:
        mode = "pad"

    canvas = geometry_canvas(display_w, display_h)
    if canvas is None:
        if mode == "pad":
            if aspect >= target:
                target_w = even_dim(display_w)
                target_h = even_dim(display_w / target)
            else:
                target_h = even_dim(display_h)
                target_w = even_dim(display_h * target)
        elif mode == "crop":
            if aspect >= target:
                target_h = even_dim(display_h)
                target_w = even_dim(display_h * target)
            else:
                target_w = even_dim(display_w)
                target_h = even_dim(display_w / target)
        else:
            target_w, target_h = even_dim(display_w), even_dim(display_h)
    else:
        target_w, target_h = canvas

    bars = crop_w <= int(frame_w) - 4 or crop_h <= int(frame_h) - 4
    sar_off = abs(float(sar or 1.0) - 1.0) > 0.01
    parts = []
    if bars:
        parts.append("crop_bars")
    if mode == "pad" and relative > limits.geometry_fit_tolerance:
        parts.append("pad_to_16x9")
    elif mode == "crop" and relative > limits.geometry_fit_tolerance:
        parts.append("crop_to_16x9")
    elif mode == "fit" and (
        abs(target_w - display_w) > 4 or abs(target_h - display_h) > 4
    ):
        parts.append("fit_16x9")
    if sar_off:
        parts.append("reset_sar")
    if not parts:
        return ""
    return "+".join(parts) + f":{target_w}x{target_h}"


def center_window(width, height, fraction=0.5):
    """8-pixel-aligned window in the middle of the frame."""
    crop_w = max(16, int(width * fraction))
    crop_h = max(16, int(height * fraction))
    crop_w -= crop_w % 8
    crop_h -= crop_h % 8
    crop_w = max(16, min(crop_w, width - (width % 8 if width >= 16 else 0)))
    crop_h = max(16, min(crop_h, height))
    x = max(0, ((width - crop_w) // 2) // 8 * 8)
    y = max(0, ((height - crop_h) // 2) // 8 * 8)
    if x + crop_w > width:
        crop_w = width - x
    if y + crop_h > height:
        crop_h = height - y
    return (int(x), int(y), int(crop_w), int(crop_h))


def apply_crop(gray, crop):
    if crop is None:
        return gray
    x, y, width, height = [int(value) for value in crop]
    x = max(0, x)
    y = max(0, y)
    if width <= 1 or height <= 1:
        return gray
    view = gray[y:y + height, x:x + width]
    if view.size == 0:
        return gray
    return view


def is_vr_equirect(width, height, thresholds=None):
    limits = active_thresholds(thresholds)
    if width < limits.vr_min_width or height <= 0:
        return False
    ratio = float(width) / float(height)
    return abs(ratio - 2.0) <= limits.vr_aspect_tolerance


def frame_fingerprint(gray):
    small = cv2.resize(to_gray(gray), (32, 32), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def fingerprints_match(first, second, max_mad=1.5):
    return float(np.mean(np.abs(first - second))) <= max_mad


def black_frame_ratio(frame, threshold=DARK_PIXEL_THRESHOLD):
    gray = to_gray(frame)
    return float(np.mean(gray < threshold))


def brightness_score(frame):
    gray = to_gray(frame)
    return float(np.mean(gray))


def measure_frame(frame, max_width=ANALYSIS_FRAME_MAX_WIDTH, crop=None, block_crop=None, thresholds=None):
    """Metrics for one decoded frame.

    Blur, noise, and halo use the active picture. Blockiness stays on the
    coded grid unless ``block_crop`` is set (VR center).
    """
    limits = active_thresholds(thresholds)
    gray_full = to_gray(frame)
    content = apply_crop(gray_full, crop)
    block_source = apply_crop(gray_full, block_crop) if block_crop is not None else gray_full
    brightness = float(np.mean(content))
    black = float(np.mean(content < limits.dark_pixel_threshold))
    std = float(np.std(content))
    exclude_dark = (
        black >= limits.dark_frame_ratio
        or brightness <= limits.dark_mean_brightness
        or std < limits.flat_std_max
    )
    blur, ratio = blur_profile(content, max_width)
    scaled = scale_gray(content, max_width)
    return {
        "blur": blur,
        "blur_scale_ratio": ratio,
        "blur_best_tiles": tile_sharpness(scaled),
        "block": blockiness_score(block_source),
        "noise": noise_profile(content),
        "noise_sigma": noise_sigma(content),
        "halo": halo_score(content),
        "combing": combing_score(gray_full),
        "brightness": brightness,
        "black": black,
        "std": std,
        "exclude_dark": exclude_dark,
        "exclude_duplicate": False,
        "exclude": exclude_dark,
    }


def mark_duplicate_frames(measured, frames, crop=None):
    """Drop a sample that repeats the previous kept frame."""
    previous = None
    for item, (_timestamp, frame) in zip(measured, frames):
        if item.get("exclude_dark"):
            continue
        fingerprint = frame_fingerprint(apply_crop(to_gray(frame), crop))
        if previous is not None and fingerprints_match(previous, fingerprint):
            item["exclude_duplicate"] = True
            item["exclude"] = True
        else:
            previous = fingerprint
    return measured


def _values(items, key):
    found = []
    for item in items:
        value = item.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number):
            continue
        found.append(number)
    return found


def aggregate_metrics(measured):
    """Robust blur stats, plus mean blockiness and a median noise sigma.

    Near-black, flat, and duplicate samples are left out of the picture
    metrics. If every sample is dark or flat, those samples are used
    anyway and mostly_dark is set so the score does not treat black as blur.
    """
    if not measured:
        raise ValueError("No measurements to aggregate")
    for item in measured:
        if "exclude_dark" not in item:
            item["exclude_dark"] = bool(item.get("exclude"))
        item.setdefault("exclude_duplicate", False)

    usable = [
        item for item in measured
        if not item.get("exclude_dark") and not item.get("exclude_duplicate")
    ]
    nondark = [item for item in measured if not item.get("exclude_dark")]
    if usable:
        chosen = usable
        mostly_dark = False
    elif nondark:
        chosen = nondark
        mostly_dark = False
    else:
        chosen = measured
        mostly_dark = True

    blur_vals = _values(chosen, "blur")
    blur_arr = np.asarray(blur_vals, dtype=np.float64) if blur_vals else np.asarray([], dtype=np.float64)

    def pct(percent):
        if blur_arr.size == 0:
            return None
        return float(np.percentile(blur_arr, percent))

    median_blur = float(np.median(blur_arr)) if blur_arr.size else None
    p10 = pct(10)
    p90 = pct(90)
    iqr = None
    if blur_arr.size:
        iqr = float(np.percentile(blur_arr, 75) - np.percentile(blur_arr, 25))
    blur_cv = None
    if median_blur is not None and iqr is not None and abs(median_blur) > 1e-6:
        blur_cv = float(iqr / median_blur)

    ratios = _values(chosen, "blur_scale_ratio")
    sigmas = _values(chosen, "noise_sigma")
    halos = _values(chosen, "halo")
    combing = _values(chosen, "combing")
    tiles = _values(chosen, "blur_best_tiles")
    blocks = _values(chosen, "block")
    noises = _values(chosen, "noise")

    dark_count = sum(1 for item in measured if item.get("exclude_dark"))
    duplicate_count = sum(
        1 for item in measured
        if item.get("exclude_duplicate") and not item.get("exclude_dark")
    )
    return {
        "avg_blur": float(np.mean(blur_arr)) if blur_arr.size else None,
        "median_blur": median_blur,
        "blur_p10": p10,
        "blur_p90": p90,
        "blur_iqr": iqr,
        "blur_cv": blur_cv,
        "min_blur": float(np.min(blur_arr)) if blur_arr.size else None,
        "max_blur": float(np.max(blur_arr)) if blur_arr.size else None,
        "avg_block": float(np.mean(blocks)) if blocks else None,
        "avg_noise": float(np.mean(noises)) if noises else None,
        "median_noise": float(np.median(sigmas)) if sigmas else None,
        "avg_halo": float(np.mean(halos)) if halos else None,
        "combing": float(np.median(combing)) if combing else None,
        "blur_best_tiles": float(np.median(tiles)) if tiles else None,
        "blur_scale_ratio": float(np.mean(ratios)) if ratios else None,
        "avg_brightness": float(np.mean(_values(measured, "brightness"))),
        "avg_black": float(np.mean(_values(measured, "black"))),
        "frames_used": len(chosen),
        "frames_excluded": dark_count + duplicate_count,
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


def ssim_gray(first, second):
    """Mean SSIM of two grayscale images with the same shape."""
    if first.shape != second.shape or min(first.shape) < 16:
        return None
    first_f = first.astype(np.float64)
    second_f = second.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(first_f, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(second_f, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(first_f * first_f, (11, 11), 1.5) - mu_a ** 2
    sigma_b = cv2.GaussianBlur(second_f * second_f, (11, 11), 1.5) - mu_b ** 2
    sigma_ab = cv2.GaussianBlur(first_f * second_f, (11, 11), 1.5) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(numerator / denominator))


# ============================================================
# Scoring and classification
# ============================================================

def classify_resolution(side, vr=False):
    if vr:
        return "vr_equirect"
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


def is_grain_dominated(avg_blur, blur_scale_ratio, thresholds=None):
    limits = active_thresholds(thresholds)
    if avg_blur is None or blur_scale_ratio is None:
        return False
    return avg_blur >= limits.blur_slightly_soft and blur_scale_ratio >= limits.blur_grain_ratio


def is_high_grain(median_noise, thresholds=None):
    limits = active_thresholds(thresholds)
    return median_noise is not None and float(median_noise) >= limits.median_noise_grainy


def is_sharpish(avg_blur, blur_scale_ratio, median_noise, thresholds=None):
    """True when the blur score is real detail, not grain."""
    limits = active_thresholds(thresholds)
    if avg_blur is None or avg_blur < limits.blur_slightly_soft:
        return False
    if is_grain_dominated(avg_blur, blur_scale_ratio, limits):
        return False
    if is_high_grain(median_noise, limits):
        return False
    return True


def classify_blur(avg_blur, blur_scale_ratio=None, thresholds=None):
    limits = active_thresholds(thresholds)
    if avg_blur is None:
        return "unknown_blur"
    if avg_blur < limits.blur_very:
        return "very_blurry"
    if avg_blur < limits.blur_blurry:
        return "blurry"
    if avg_blur < limits.blur_soft:
        return "soft"
    if avg_blur < limits.blur_slightly_soft:
        return "slightly_soft"
    if is_grain_dominated(avg_blur, blur_scale_ratio, limits):
        return "grain_dominated"
    if avg_blur >= limits.blur_sharp:
        return "sharp"
    return "acceptable_sharpness"


def classify_blockiness(avg_block, thresholds=None):
    limits = active_thresholds(thresholds)
    if avg_block is None:
        return "unknown_blockiness"
    if avg_block > limits.block_very:
        return "very_blocky"
    if avg_block > limits.block_blocky:
        return "blocky"
    if avg_block > limits.block_slight:
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


def blur_points(avg_blur, thresholds=None):
    limits = active_thresholds(thresholds)
    if avg_blur is None:
        return 0
    if avg_blur < limits.blur_very:
        return 45
    if avg_blur < limits.blur_blurry:
        return 32
    if avg_blur < limits.blur_soft:
        return 18
    if avg_blur < limits.blur_slightly_soft:
        return 8
    return 0


def block_points(avg_block, thresholds=None):
    limits = active_thresholds(thresholds)
    if avg_block is None:
        return 0
    if avg_block > limits.block_very:
        return 35
    if avg_block > limits.block_blocky:
        return 22
    if avg_block > limits.block_slight:
        return 10
    return 0


def bitrate_points(side, bits_per_pixel_frame, codec=None, vr=False, thresholds=None):
    limits = active_thresholds(thresholds)
    if vr or not bits_per_pixel_frame or bits_per_pixel_frame <= 0 or side <= 0:
        return 0
    adjusted = float(bits_per_pixel_frame) * codec_efficiency(codec, limits)
    if side >= 720:
        if adjusted < limits.hd_bpp_very_low:
            return 15
        if adjusted < limits.hd_bpp_low:
            return 8
        return 0
    if side <= 480:
        if adjusted < limits.sd_bpp_very_low:
            return 15
        if adjusted < limits.sd_bpp_low:
            return 8
    return 0


def issue_score_parts(side, avg_blur, avg_block, bits_per_pixel_frame, codec=None, vr=False, thresholds=None):
    """The four pieces of the issue score, before the cap."""
    return {
        "score_resolution": resolution_points(side),
        "score_blur": blur_points(avg_blur, thresholds),
        "score_blockiness": block_points(avg_block, thresholds),
        "score_bitrate": bitrate_points(
            side,
            bits_per_pixel_frame,
            codec=codec,
            vr=vr,
            thresholds=thresholds,
        ),
    }


def calculate_issue_score(
    side,
    avg_blur,
    avg_block,
    bits_per_pixel_frame,
    codec=None,
    vr=False,
    thresholds=None,
):
    """0-100 issue score. Higher means a stronger restoration candidate."""
    limits = active_thresholds(thresholds)
    total = sum(issue_score_parts(
        side,
        avg_blur,
        avg_block,
        bits_per_pixel_frame,
        codec=codec,
        vr=vr,
        thresholds=limits,
    ).values())
    return round(min(limits.score_cap, float(total)), 2)


def build_issue_flags(
    short_side_px,
    avg_blur,
    avg_block,
    bits_per_pixel_frame,
    processed=False,
    median_noise=None,
    mostly_dark=False,
    blur_scale_ratio=None,
    codec=None,
    vr=False,
    combed=False,
    halo=None,
    unstable=False,
    bitrate_side=None,
    thresholds=None,
):
    limits = active_thresholds(thresholds)
    flags = []
    if processed:
        flags.append("already_processed_name")
    if vr:
        flags.append("vr_equirect")

    resolution_class = classify_resolution(short_side_px, vr=vr)
    if resolution_class in {"very_tiny", "very_low_resolution", "low_resolution", "sd_plus"}:
        flags.append(resolution_class)

    if mostly_dark:
        flags.append("mostly_dark_or_flat")
    else:
        blur_class = classify_blur(avg_blur, blur_scale_ratio, limits)
        block_class = classify_blockiness(avg_block, limits)
        if blur_class in {"very_blurry", "blurry", "soft", "slightly_soft"}:
            flags.append(blur_class)
        if is_grain_dominated(avg_blur, blur_scale_ratio, limits):
            flags.append("grain_dominated_sharpness")
        if is_high_grain(median_noise, limits):
            flags.append("high_grain")
        if block_class in {"very_blocky", "blocky", "slightly_blocky"}:
            flags.append(block_class)
        if (
            halo is not None
            and avg_blur is not None
            and float(halo) >= limits.halo_flag
            and float(avg_blur) >= limits.blur_sharp
            and not is_high_grain(median_noise, limits)
        ):
            flags.append("possible_oversharpen")

    if combed:
        flags.append("interlaced_or_combed")
    if unstable:
        flags.append("unstable_blur")

    adjusted = None
    if bits_per_pixel_frame and bits_per_pixel_frame > 0:
        adjusted = float(bits_per_pixel_frame) * codec_efficiency(codec, limits)
    rate_side = short_side_px if bitrate_side is None else bitrate_side
    if adjusted and not vr:
        if rate_side >= 720 and adjusted < limits.hd_bpp_low:
            flags.append("low_bitrate_for_hd")
        elif 0 < rate_side <= 480 and adjusted < limits.sd_bpp_low:
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
    combed=False,
    vr=False,
    unstable=False,
    thresholds=None,
):
    """Pick a review label from the same thresholds as the score.

    Resolution uses the active picture. A blurry HD file is checked as a
    source or denoise problem. It is not sent through an upscale step.
    """
    limits = active_thresholds(thresholds)
    low_res = side is not None and 0 < side <= 576
    very_low_res = side is not None and 0 < side <= 360
    blurry = avg_blur is not None and avg_blur < limits.blur_soft
    very_blurry = avg_blur is not None and avg_blur < limits.blur_blurry
    slightly_soft = (
        avg_blur is not None
        and limits.blur_soft <= avg_blur < limits.blur_slightly_soft
    )
    blocky = avg_block is not None and avg_block > limits.block_blocky
    sharpish = is_sharpish(avg_blur, blur_scale_ratio, median_noise, limits)
    grain_dominated = is_grain_dominated(avg_blur, blur_scale_ratio, limits)
    high_grain = is_high_grain(median_noise, limits)

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
    elif side is not None and side >= 720 and blurry:
        base = "check_source_upscaled_or_denoise"
    elif score is not None and score >= 45:
        base = "manual_review"
    elif vr:
        base = "vr_equirect_review"
    else:
        base = "low_priority_or_ok"

    if combed and base == "artifact_reduction_review":
        base = "manual_review"
    if unstable and base == "low_priority_or_ok":
        base = "manual_review"
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


def wants_contact_sheet(result, config):
    if not config.create_contact_sheets:
        return False
    score = result.get("quality_issue_score")
    try:
        if score is not None and score != "" and float(score) >= config.contact_sheet_min_score:
            return True
    except (TypeError, ValueError):
        pass
    return recommendation_base(result.get("recommendation")) in CONTACT_REVIEW


# ============================================================
# Video analysis
# ============================================================

def _attach_identity(row, video_path):
    path = Path(video_path)
    if not path.exists():
        return row
    try:
        size, mtime = file_identity(path)
    except OSError:
        return row
    row["size"] = size
    row["mtime"] = mtime
    return row


def analyze_video(video_path, config=None):
    """Analyze one video and return a report row."""
    config = config or ScanConfig()
    limits = config.thresholds or DEFAULT_THRESHOLDS
    video_path = Path(video_path)
    processed = is_probably_processed(video_path)

    if config.skip_already_processed and processed:
        row = base_row(video_path, processed=processed)
        _attach_identity(row, video_path)
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
    processed = looks_processed(video_path, meta.get("encoder"))
    if config.skip_already_processed and processed:
        row = base_row(video_path, meta=meta, processed=processed)
        _attach_identity(row, video_path)
        row["issue_flags"] = "skipped_already_processed_name"
        row["recommendation"] = "skipped"
        return row

    row = base_row(video_path, meta=meta, processed=processed)
    _attach_identity(row, video_path)
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

    vr = is_vr_equirect(meta["display_width"], meta["display_height"], limits)
    row["is_vr"] = vr
    frames, decode_errors = extract_sample_frames(video_path, meta, config)
    row["sampled_frames"] = len(frames)
    row["sample_errors"] = len(decode_errors)
    if decode_errors:
        row["sample_error"] = clip_error(decode_errors[0])
    if not frames:
        detail = decode_errors[-1] if decode_errors else ""
        message = "Could not sample frames"
        if detail:
            message = message + ": " + detail
        row["issue_flags"] = "could_not_sample"
        row["recommendation"] = "manual_check_could_not_sample"
        row["error"] = clip_error(message)
        return row

    if vr:
        crop = center_window(meta["width"], meta["height"])
        block_crop = crop
    else:
        crop = consensus_letterbox(
            [frame for _timestamp, frame in frames],
            dark_mean=limits.dark_mean_brightness,
        )
        block_crop = None
    sar = parse_sar(meta["sample_aspect_ratio"])
    content_w, content_h = display_dimensions(crop[2], crop[3], sar, meta["rotation"])
    side = short_side(crop[2], crop[3], content_w, content_h)
    aspect = (float(content_w) / float(content_h)) if content_w and content_h else None
    measured = [
        measure_frame(
            frame,
            config.analysis_frame_max_width,
            crop=crop,
            block_crop=block_crop,
            thresholds=limits,
        )
        for _timestamp, frame in frames
    ]
    mark_duplicate_frames(measured, frames, crop)
    stats = aggregate_metrics(measured)
    container_side = short_side(
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
    bpp_eff = None if bpp_frame is None else bpp_frame * codec_efficiency(meta["codec"], limits)
    blur_for_score = None if stats["mostly_dark"] else stats["avg_blur"]
    block_for_score = None if stats["mostly_dark"] else stats["avg_block"]
    # Resolution follows the active picture. Bitrate keeps the container
    # side, because that is the grid the codec stored.
    parts = issue_score_parts(
        side,
        blur_for_score,
        block_for_score,
        bpp_frame,
        codec=meta["codec"],
        vr=vr,
        thresholds=limits,
    )
    parts["score_bitrate"] = bitrate_points(
        container_side,
        bpp_frame,
        codec=meta["codec"],
        vr=vr,
        thresholds=limits,
    )
    score = round(min(limits.score_cap, float(sum(parts.values()))), 2)
    combed = False
    field_order = str(meta.get("field_order") or "").lower()
    if field_order in INTERLACED_FIELD_ORDERS:
        combed = True
    if stats["combing"] is not None and stats["combing"] >= limits.combing_min:
        combed = True
    unstable = (
        stats["frames_used"] >= limits.unstable_min_frames
        and stats["blur_cv"] is not None
        and stats["blur_cv"] >= limits.unstable_blur_cv
    )
    flags = build_issue_flags(
        side,
        stats["avg_blur"],
        stats["avg_block"],
        bpp_frame,
        processed=processed,
        median_noise=stats["median_noise"],
        mostly_dark=stats["mostly_dark"],
        blur_scale_ratio=stats["blur_scale_ratio"],
        codec=meta["codec"],
        vr=vr,
        combed=combed,
        halo=stats["avg_halo"],
        unstable=unstable,
        bitrate_side=container_side,
        thresholds=limits,
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
        combed=combed,
        vr=vr,
        unstable=unstable,
        thresholds=limits,
    )
    row.update({
        "content_width": content_w,
        "content_height": content_h,
        "content_aspect": round_or_none(aspect, 4),
        "crop": ",".join(str(int(value)) for value in crop),
        "fix_geometry": geometry_plan(
            meta["width"],
            meta["height"],
            crop,
            sar=sar,
            rotation=meta["rotation"],
            vr=vr,
            thresholds=limits,
        ),
        "frames_used": stats["frames_used"],
        "frames_excluded": stats["frames_excluded"],
        "avg_blur_score": round_or_none(stats["avg_blur"], 3),
        "median_blur_score": round_or_none(stats["median_blur"], 3),
        "blur_p10": round_or_none(stats["blur_p10"], 3),
        "blur_p90": round_or_none(stats["blur_p90"], 3),
        "blur_iqr": round_or_none(stats["blur_iqr"], 3),
        "blur_cv": round_or_none(stats["blur_cv"], 3),
        "min_blur_score": round_or_none(stats["min_blur"], 3),
        "max_blur_score": round_or_none(stats["max_blur"], 3),
        "blur_scale_ratio": round_or_none(stats["blur_scale_ratio"], 3),
        "blur_best_tiles": round_or_none(stats["blur_best_tiles"], 3),
        "avg_blockiness_score": round_or_none(stats["avg_block"], 3),
        "avg_noise_score": round_or_none(stats["avg_noise"], 3),
        "median_noise_score": round_or_none(stats["median_noise"], 3),
        "avg_halo_score": round_or_none(stats["avg_halo"], 3),
        "combing_score": round_or_none(stats["combing"], 4),
        "avg_brightness": round_or_none(stats["avg_brightness"], 3),
        "avg_black_frame_ratio": round_or_none(stats["avg_black"], 4),
        "bitrate_per_pixel": round_or_none(bpp_second, 6),
        "bits_per_pixel_frame": round_or_none(bpp_frame, 6),
        "bits_per_pixel_frame_eff": round_or_none(bpp_eff, 6),
        "score_resolution": parts["score_resolution"],
        "score_blur": parts["score_blur"],
        "score_blockiness": parts["score_blockiness"],
        "score_bitrate": parts["score_bitrate"],
        "quality_issue_score": score,
        "resolution_class": classify_resolution(side, vr=vr),
        "blur_class": classify_blur(stats["avg_blur"], stats["blur_scale_ratio"], limits),
        "blockiness_class": classify_blockiness(stats["avg_block"], limits),
        "issue_flags": ",".join(flags),
        "recommendation": recommendation,
    })

    if wants_contact_sheet(row, config):
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
        _attach_identity(row, video_path)
        row["issue_flags"] = "error"
        row["recommendation"] = "manual_check_error"
        row["error"] = clip_error(exc)
        return row


# ============================================================
# Pair mode
# ============================================================

def normalized_stem(filename):
    stem = Path(str(filename)).stem.lower()
    previous = None
    while stem and previous != stem:
        previous = stem
        stem = PAIR_MARKER_RE.sub("", stem)
    return stem


def stems_related(name_a, name_b):
    """True when two filenames are the same title with a processing suffix."""
    left = normalized_stem(name_a)
    right = normalized_stem(name_b)
    if not left or not right:
        return False
    if left == right:
        return True
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    if len(short) < 4:
        return False
    if long.startswith(short) and (len(long) == len(short) or long[len(short)] in "_- ."):
        return True
    return False


def _union_find(size):
    parent = list(range(size))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        ra, rb = find(left), find(right)
        if ra != rb:
            parent[rb] = ra

    return find, union


def cluster_by_duration(rows, tolerance):
    find, union = _union_find(len(rows))
    durations = [safe_float(row.get("duration")) for row in rows]
    for i in range(len(rows)):
        if durations[i] <= 0:
            continue
        for j in range(i + 1, len(rows)):
            if durations[j] > 0 and abs(durations[i] - durations[j]) <= tolerance:
                union(i, j)
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(index), []).append(row)
    return list(groups.values())


def cluster_by_stem(rows):
    find, union = _union_find(len(rows))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if stems_related(rows[i].get("filename") or "", rows[j].get("filename") or ""):
                union(i, j)
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(index), []).append(row)
    return list(groups.values())


def row_has_score(row):
    if str(row.get("error") or "").strip():
        return False
    score = row.get("quality_issue_score")
    if score is None or score == "":
        return False
    try:
        return not math.isnan(float(score))
    except (TypeError, ValueError):
        return False


def display_aspect(row):
    width = safe_float(row.get("display_width"))
    height = safe_float(row.get("display_height"))
    if width <= 0 or height <= 0:
        return None
    return width / height


def row_content_aspect(row):
    """Aspect of the active picture. Display size is only a fallback."""
    width = safe_float(row.get("content_width"))
    height = safe_float(row.get("content_height"))
    if width > 0 and height > 0:
        return width / height
    return display_aspect(row)


def aspects_mismatch(left, right, tolerance=0.05):
    aspect_left = row_content_aspect(left)
    aspect_right = row_content_aspect(right)
    if aspect_left is None or aspect_right is None:
        return False
    return abs(aspect_left - aspect_right) / max(aspect_left, aspect_right) > tolerance


def choose_reference(left, right):
    """Prefer the square-pixel file, then the larger file, as the reference."""
    left_distance = abs(math.log(parse_sar(left.get("sample_aspect_ratio"))))
    right_distance = abs(math.log(parse_sar(right.get("sample_aspect_ratio"))))
    if abs(left_distance - right_distance) > 0.02:
        return (left, right) if left_distance < right_distance else (right, left)
    if safe_int(left.get("size")) >= safe_int(right.get("size")):
        return left, right
    return right, left


def plan_pairs(rows, tolerance):
    """Pair rows that share a folder, a duration, and a related filename."""
    by_folder = {}
    for row in rows:
        if not row_has_score(row):
            continue
        by_folder.setdefault(str(row.get("folder") or ""), []).append(row)
    plans = []
    for folder, items in by_folder.items():
        for duration_group in cluster_by_duration(items, tolerance):
            for component in cluster_by_stem(duration_group):
                if len(component) < 2:
                    continue
                processed = [row for row in component if as_bool(row.get("is_probably_processed"))]
                originals = [row for row in component if not as_bool(row.get("is_probably_processed"))]
                if len(originals) == 1 and processed:
                    for other in processed:
                        plans.append((originals[0], other, ""))
                    continue
                matched = False
                if processed and originals:
                    for other in processed:
                        matches = [
                            original for original in originals
                            if stems_related(original.get("filename") or "", other.get("filename") or "")
                        ]
                        if len(matches) == 1:
                            plans.append((matches[0], other, ""))
                            matched = True
                    if matched:
                        continue
                if len(component) == 2 and not processed:
                    reference, other = choose_reference(component[0], component[1])
                    plans.append((reference, other, "no_processed_marker"))
                    continue
                plans.append((None, component, "ambiguous_group"))
    return plans


def group_key(folder, duration):
    raw = f"{folder}|{round(safe_float(duration))}"
    return hashlib.sha1(raw.encode("utf-8", "surrogateescape")).hexdigest()[:12]


def to_bgr(frame):
    if frame.ndim == 2 or frame.shape[2] == 1:
        gray = frame if frame.ndim == 2 else frame[:, :, 0]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame


def rotate_bgr(image, rotation):
    turns = int(round(abs(float(rotation)))) % 360
    if turns == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if turns == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if turns == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def content_bgr(frame, crop, sar, rotation):
    """Active picture in square pixels, bars removed, then rotated."""
    image = to_bgr(frame)
    height, width = image.shape[:2]
    if crop is None:
        view = image
    else:
        x, y, crop_w, crop_h = [int(value) for value in crop]
        x = max(0, min(x, max(0, width - 1)))
        y = max(0, min(y, max(0, height - 1)))
        view = image[y:y + max(0, crop_h), x:x + max(0, crop_w)]
        if view.size == 0:
            view = image
    if abs(float(sar) - 1.0) > 0.01:
        view = cv2.resize(
            view,
            (max(1, int(round(view.shape[1] * float(sar)))), view.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return rotate_bgr(view, rotation)


def parse_crop(text, width, height):
    width = int(width or 0)
    height = int(height or 0)
    fallback = (0, 0, width, height)
    if text is None or str(text).strip() == "":
        return fallback
    parts = str(text).split(",")
    if len(parts) != 4:
        return fallback
    try:
        x, y, crop_w, crop_h = [int(float(part)) for part in parts]
    except ValueError:
        return fallback
    if crop_w <= 1 or crop_h <= 1:
        return fallback
    return (x, y, crop_w, crop_h)


def _row_meta(row):
    return {
        "width": safe_int(row.get("width")),
        "height": safe_int(row.get("height")),
        "duration": safe_float(row.get("duration")),
    }


def _resize_image(image, width, height, interpolation):
    width = max(1, int(width))
    height = max(1, int(height))
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=interpolation)


def safe_ratio(numerator, denominator, floor=1.0):
    """Ratio that stays 1 when both sides are below the noise floor."""
    num = float(numerator or 0.0)
    den = float(denominator or 0.0)
    if den < floor and num < floor:
        return 1.0
    return num / max(den, floor)


def mean_yuv(bgr):
    yuv = cv2.cvtColor(to_bgr(bgr), cv2.COLOR_BGR2YUV)
    return [float(yuv[:, :, channel].mean()) for channel in range(3)]


def shift_luma(bgr, delta):
    if abs(float(delta)) < 0.5:
        return to_bgr(bgr)
    yuv = cv2.cvtColor(to_bgr(bgr), cv2.COLOR_BGR2YUV).astype(np.int16)
    yuv[:, :, 0] = np.clip(yuv[:, :, 0] + int(round(float(delta))), 0, 255)
    return cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)


def align_ssim(first, second, width=160):
    """SSIM after both pictures are fit to a small common width."""
    left = to_gray(first)
    right = to_gray(second)
    if left.shape[1] < 16 or right.shape[1] < 16 or left.shape[0] < 16 or right.shape[0] < 16:
        return None
    left_h = max(16, int(round(left.shape[0] * (width / float(left.shape[1])))))
    right_h = max(16, int(round(right.shape[0] * (width / float(right.shape[1])))))
    left_small = cv2.resize(left, (width, left_h), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right, (width, right_h), interpolation=cv2.INTER_AREA)
    if right_small.shape != left_small.shape:
        right_small = cv2.resize(
            right_small,
            (left_small.shape[1], left_small.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return ssim_gray(left_small, right_small)


def evaluate_pair_frames(
    orig_frame,
    proc_frame,
    orig_crop=None,
    proc_crop=None,
    orig_sar=1.0,
    proc_sar=1.0,
    orig_rotation=0.0,
    proc_rotation=0.0,
):
    """One timestamp: native-grid damage test and bicubic baseline test.

    Native sharpness is the processed frame, downscaled to the original
    content size, divided by the same roundtrip of a bicubic upscale.
    Comparing with the original itself treats the resampling loss of a
    plain upscale as lost detail. The baseline test asks whether the
    processed frame is sharper than that bicubic upscale on the output canvas.
    Blockiness uses the scaled codec period on the coded frame, because an
    8x8 grid is no longer 8 pixels after an upscale.
    """
    orig_view = content_bgr(orig_frame, orig_crop, orig_sar, orig_rotation)
    proc_view = content_bgr(proc_frame, proc_crop, proc_sar, proc_rotation)
    if min(orig_view.shape[:2]) < 16 or min(proc_view.shape[:2]) < 16:
        return None
    orig_yuv = mean_yuv(orig_view)
    proc_yuv = mean_yuv(proc_view)
    luma_delta = orig_yuv[0] - proc_yuv[0]
    chroma_delta = max(abs(orig_yuv[1] - proc_yuv[1]), abs(orig_yuv[2] - proc_yuv[2]))
    if abs(luma_delta) >= 10.0:
        proc_view = shift_luma(proc_view, luma_delta)
    orig_h, orig_w = orig_view.shape[:2]
    proc_h, proc_w = proc_view.shape[:2]
    # Resize the color picture, then measure gray. A player upscale does
    # the same, so a plain bicubic file matches this baseline.
    base_view = _resize_image(orig_view, proc_w, proc_h, cv2.INTER_CUBIC)
    orig_gray = to_gray(orig_view)
    proc_gray = to_gray(proc_view)
    base = to_gray(base_view)
    proc_native = _resize_image(proc_gray, orig_w, orig_h, cv2.INTER_AREA)
    base_native = _resize_image(base, orig_w, orig_h, cv2.INTER_AREA)
    orig_coded_h, orig_coded_w = to_gray(orig_frame).shape[:2]
    proc_coded_h, proc_coded_w = to_gray(proc_frame).shape[:2]
    scale_x = (proc_coded_w / float(orig_coded_w)) if orig_coded_w else 1.0
    scale_y = (proc_coded_h / float(orig_coded_h)) if orig_coded_h else 1.0
    halo_proc = halo_score(proc_gray)
    halo_base = halo_score(base)
    return {
        "blur_native": safe_ratio(laplacian_var(proc_native), laplacian_var(base_native)),
        "blur_vs_base": safe_ratio(laplacian_var(proc_gray), laplacian_var(base)),
        "block_native": float(
            blockiness_scaled(proc_frame, scale_x, scale_y)
            - blockiness_scaled(orig_frame, 1.0, 1.0)
        ),
        "noise_native": safe_ratio(
            noise_sigma(proc_native),
            noise_sigma(base_native),
            floor=0.05,
        ),
        "halo_vs_base": float(halo_proc) / max(float(halo_base), 1.0),
        "halo_delta": float(halo_proc - halo_base),
        "ssim": ssim_gray(proc_native, orig_gray),
        "luma_delta": float(luma_delta),
        "chroma_delta": float(chroma_delta),
    }


def _median_of(items, key):
    values = []
    for item in items:
        value = item.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number):
            continue
        values.append(number)
    if not values:
        return None
    return float(np.median(values))


def summarize_pair_metrics(frames, thresholds=None):
    """Median of the per-frame deltas. Scene changes cancel out."""
    limits = active_thresholds(thresholds)
    summary = {
        key: _median_of(frames, key)
        for key in (
            "blur_native",
            "blur_vs_base",
            "block_native",
            "noise_native",
            "halo_vs_base",
            "halo_delta",
            "ssim",
            "luma_delta",
            "chroma_delta",
        )
    }
    luma = summary.get("luma_delta")
    summary["range_shift"] = luma is not None and abs(luma) >= limits.pair_luma_shift
    summary["pair_misaligned"] = False
    summary["geometry_mismatch"] = False
    return summary


def pair_verdict(deltas, thresholds=None):
    """Did the processed file beat a plain upscale, without damaging the source?

    SSIM is an alignment gate, not a quality failure. A level shift is
    recorded separately and removed before these ratios are measured.
    """
    limits = active_thresholds(thresholds)
    if not deltas:
        return "compare_failed"
    if deltas.get("geometry_mismatch"):
        return "geometry_mismatch"
    if deltas.get("pair_misaligned"):
        return "pair_misaligned"
    flags = []
    blur = deltas.get("blur_native")
    noise = deltas.get("noise_native")
    if blur is not None and (
        blur < limits.pair_blur_severe_ratio
        or (
            blur < limits.pair_blur_detail_ratio
            and (noise is None or noise >= limits.pair_noise_drop_ratio)
        )
    ):
        flags.append("lost_detail")
    block = deltas.get("block_native")
    if block is not None and block > limits.pair_block_delta:
        flags.append("more_blocky")
    if noise is not None and noise > limits.pair_noise_ratio:
        flags.append("more_noise")
    halo_ratio = deltas.get("halo_vs_base")
    halo_delta = deltas.get("halo_delta")
    if (
        halo_ratio is not None
        and halo_delta is not None
        and halo_ratio > limits.pair_halo_ratio
        and halo_delta > limits.pair_halo_delta
    ):
        flags.append("oversharpened")
    if flags:
        return "regressed:" + "+".join(flags)
    sharper = deltas.get("blur_vs_base")
    if sharper is not None and sharper > limits.pair_blur_vs_base:
        return "improved"
    return "neutral"


def pair_action(verdict):
    """Which file to keep. Neutral stays with the original."""
    if verdict == "improved":
        return "keep_processed"
    if verdict == "neutral" or str(verdict).startswith("regressed"):
        return "keep_original"
    return "manual_review"


def _sheet_pair(left_bgr, right_bgr, height=240):
    def fit(image):
        picture = to_bgr(image)
        frame_h, frame_w = picture.shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            return picture
        scale = height / float(frame_h)
        return cv2.resize(
            picture,
            (max(1, int(round(frame_w * scale))), height),
            interpolation=cv2.INTER_AREA,
        )

    left = fit(left_bgr)
    right = fit(right_bgr)
    divider = np.full((height, 4, 3), 255, np.uint8)
    return np.hstack([left, divider, right])


def compare_pair(original, other, config):
    """Sample both files at the same timestamps and score aligned frames.

    A timestamp whose pictures do not match is retried across two frames
    on either side. If too many timestamps still miss, the pair is
    misaligned and the quality verdict is not used.
    """
    limits = config.thresholds or DEFAULT_THRESHOLDS
    orig_meta = _row_meta(original)
    other_meta = _row_meta(other)
    duration = min(orig_meta["duration"], other_meta["duration"])
    count = min(8, resolved_sample_count(duration, config))
    timestamps = sample_timestamps(duration, count, seed=path_seed(original.get("path")))
    orig_crop = parse_crop(original.get("crop"), orig_meta["width"], orig_meta["height"])
    other_crop = parse_crop(other.get("crop"), other_meta["width"], other_meta["height"])
    orig_sar = parse_sar(original.get("sample_aspect_ratio"))
    other_sar = parse_sar(other.get("sample_aspect_ratio"))
    orig_rot = safe_float(original.get("rotation"))
    other_rot = safe_float(other.get("rotation"))
    fps = safe_float(other.get("frame_rate")) or safe_float(original.get("frame_rate")) or 30.0
    fps = min(max(fps, 1.0), 120.0)
    measured = []
    sheet_frames = []
    errors = []
    attempted = 0
    dropped = 0
    for timestamp in timestamps:
        orig_frame, orig_error = decode_frame(
            original.get("path"),
            timestamp,
            orig_meta["width"],
            orig_meta["height"],
            config.frame_timeout,
            1,
        )
        other_frame, other_error = decode_frame(
            other.get("path"),
            timestamp,
            other_meta["width"],
            other_meta["height"],
            config.frame_timeout,
            1,
        )
        if orig_frame is None or other_frame is None:
            if orig_error:
                errors.append(orig_error)
            if other_error:
                errors.append(other_error)
            continue
        attempted += 1
        orig_view = content_bgr(orig_frame, orig_crop, orig_sar, orig_rot)
        other_view = content_bgr(other_frame, other_crop, other_sar, other_rot)
        score = align_ssim(orig_view, other_view)
        chosen = other_frame
        if score is None or score < limits.pair_ssim_align:
            start = max(0.0, float(timestamp) - 2.0 / fps)
            window, window_error = decode_frames(
                other.get("path"),
                start,
                other_meta["width"],
                other_meta["height"],
                config.frame_timeout,
                1,
                5,
            )
            if window_error and not window:
                errors.append(window_error)
            best_frame = None
            best_score = -1.0
            for candidate in window:
                candidate_view = content_bgr(candidate, other_crop, other_sar, other_rot)
                candidate_score = align_ssim(orig_view, candidate_view)
                if candidate_score is not None and candidate_score > best_score:
                    best_score = candidate_score
                    best_frame = candidate
            if best_frame is None or best_score < limits.pair_ssim_align:
                dropped += 1
                continue
            chosen = best_frame
        metrics = evaluate_pair_frames(
            orig_frame,
            chosen,
            orig_crop,
            other_crop,
            orig_sar,
            other_sar,
            orig_rot,
            other_rot,
        )
        if not metrics:
            dropped += 1
            continue
        measured.append(metrics)
        if len(sheet_frames) < 4:
            chosen_view = content_bgr(chosen, other_crop, other_sar, other_rot)
            sheet_frames.append((timestamp, _sheet_pair(orig_view, chosen_view)))
    if attempted == 0:
        return None, sheet_frames, None, errors or ["Could not decode a shared timestamp"]
    needed = min(limits.pair_min_frames, attempted)
    fail_fraction = dropped / float(attempted)
    summary = summarize_pair_metrics(measured, limits) if measured else {
        "pair_misaligned": False,
        "geometry_mismatch": False,
        "range_shift": False,
    }
    summary["aligned_frames"] = len(measured)
    summary["dropped_frames"] = dropped
    if len(measured) < needed or fail_fraction > limits.pair_align_fail_fraction:
        summary["pair_misaligned"] = True
    return summary, sheet_frames, None, errors


def _empty_group(**kwargs):
    row = {column: None for column in GROUP_COLUMNS}
    row.update(kwargs)
    return row


def _mark_paired(row):
    row["is_probably_processed"] = True
    flags = [flag for flag in str(row.get("issue_flags") or "").split(",") if flag]
    if "paired_with_original" not in flags:
        flags.append("paired_with_original")
    if "already_processed_name" not in flags:
        flags.append("already_processed_name")
    row["issue_flags"] = ",".join(flags)
    recommendation = str(row.get("recommendation") or "")
    if recommendation and not recommendation.startswith("already_processed_review_only+"):
        row["recommendation"] = "already_processed_review_only+" + recommendation


def _field_interlaced(row):
    return str(row.get("field_order") or "").lower() in INTERLACED_FIELD_ORDERS


def compare_library(rows, config):
    """Write pair rows and mark processed siblings discovered by the match."""
    limits = config.thresholds or DEFAULT_THRESHOLDS
    plans = plan_pairs(rows, limits.pair_duration_seconds)
    group_rows = []
    for original, other, note in plans:
        if note == "ambiguous_group":
            component = other
            folder = str(component[0].get("folder") or "")
            duration = safe_float(component[0].get("duration"))
            group_rows.append(_empty_group(
                group_id=group_key(folder, duration),
                folder=folder,
                other_path=";".join(str(item.get("path") or "") for item in component),
                other_filename=";".join(str(item.get("filename") or "") for item in component),
                verdict="ambiguous_group",
                action="manual_review",
                note="More than one unmarked file shares this duration. Not compared.",
            ))
            continue
        folder = str(original.get("folder") or "")
        duration_delta = abs(safe_float(original.get("duration")) - safe_float(other.get("duration")))
        mismatch = aspects_mismatch(original, other, limits.geometry_aspect_tolerance)
        notes = [note] if note else []
        if _field_interlaced(original) or _field_interlaced(other):
            notes.append("interlaced")
        record = _empty_group(
            group_id=group_key(folder, original.get("duration")),
            folder=folder,
            original_path=str(original.get("path") or ""),
            other_path=str(other.get("path") or ""),
            original_filename=str(original.get("filename") or ""),
            other_filename=str(other.get("filename") or ""),
            duration_delta=round(duration_delta, 3),
            duration_mismatch=duration_delta > limits.pair_duration_seconds,
            geometry_mismatch=mismatch,
            original_display=f"{original.get('display_width')}x{original.get('display_height')}",
            other_display=f"{other.get('display_width')}x{other.get('display_height')}",
            original_content=f"{original.get('content_width')}x{original.get('content_height')}",
            other_content=f"{other.get('content_width')}x{other.get('content_height')}",
            note="; ".join(notes),
        )
        if mismatch:
            left_aspect = row_content_aspect(original)
            right_aspect = row_content_aspect(other)
            if left_aspect and right_aspect:
                notes.append(f"content aspect {left_aspect:.3f} vs {right_aspect:.3f}")
            record["note"] = "; ".join(part for part in notes if part)
            record["verdict"] = "geometry_mismatch"
            record["action"] = pair_action("geometry_mismatch")
            record["best_version"] = record["original_path"]
            if note != "no_processed_marker":
                _mark_paired(other)
            group_rows.append(record)
            continue
        try:
            summary, sheet_frames, _unused, errors = compare_pair(original, other, config)
        except Exception as exc:
            summary = None
            sheet_frames = None
            errors = [clip_error(exc)]
        if not summary:
            record["verdict"] = "compare_failed"
            record["action"] = "manual_review"
            record["best_version"] = record["original_path"]
            record["note"] = clip_error("; ".join(errors) if errors else "compare failed")
            group_rows.append(record)
            continue
        record.update({
            "blur_native": round_or_none(summary.get("blur_native"), 3),
            "blur_vs_base": round_or_none(summary.get("blur_vs_base"), 3),
            "block_native": round_or_none(summary.get("block_native"), 3),
            "noise_native": round_or_none(summary.get("noise_native"), 3),
            "halo_vs_base": round_or_none(summary.get("halo_vs_base"), 3),
            "ssim": round_or_none(summary.get("ssim"), 4),
            "luma_delta": round_or_none(summary.get("luma_delta"), 2),
            "chroma_delta": round_or_none(summary.get("chroma_delta"), 2),
            "range_shift": bool(summary.get("range_shift")),
            "aligned_frames": summary.get("aligned_frames"),
            "dropped_frames": summary.get("dropped_frames"),
        })
        if summary.get("range_shift"):
            luma = summary.get("luma_delta")
            notes.append("range_shift" if luma is None else f"range_shift luma_delta={luma:.1f}")
        chroma = summary.get("chroma_delta")
        if chroma is not None and chroma >= limits.pair_chroma_shift:
            notes.append(f"color_shift chroma_delta={chroma:.1f}")
        record["note"] = "; ".join(part for part in notes if part)
        verdict = pair_verdict(summary, limits)
        record["verdict"] = verdict
        record["action"] = pair_action(verdict)
        record["best_version"] = (
            record["other_path"] if verdict == "improved" else record["original_path"]
        )
        if note != "no_processed_marker":
            _mark_paired(other)
        if config.create_contact_sheets and verdict != "improved" and sheet_frames:
            try:
                create_contact_sheet(
                    other.get("path"),
                    sheet_frames,
                    {
                        "quality_issue_score": other.get("quality_issue_score"),
                        "recommendation": record["action"],
                        "width": other.get("width"),
                        "height": other.get("height"),
                        "display_width": other.get("display_width"),
                        "display_height": other.get("display_height"),
                        "avg_blur_score": record["blur_native"],
                        "avg_blockiness_score": record["block_native"],
                        "avg_noise_score": record["noise_native"],
                        "issue_flags": verdict,
                    },
                    config.contact_sheet_folder,
                    config.contact_thumb_width,
                )
            except Exception:
                pass
        group_rows.append(record)
    write_table(groups_output_path(config), group_rows, GROUP_COLUMNS)
    return group_rows


def groups_output_path(config):
    if config.groups_csv:
        return config.groups_csv
    output = Path(config.output_csv)
    return str(output.with_name(output.stem + "_groups" + output.suffix))


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


def row_is_current(row):
    """True when this row can be reused: same schema, size, and mtime."""
    if safe_int(row.get("schema_version")) != SCHEMA_VERSION:
        return False
    path = row.get("path")
    if not path or not Path(path).exists():
        return False
    try:
        size, mtime = file_identity(path)
    except OSError:
        return False
    return safe_int(row.get("size")) == size and safe_int(row.get("mtime")) == mtime


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


def rows_to_dataframe(rows, columns):
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, columns]


def sorted_dataframe(rows):
    frame = rows_to_dataframe(rows, CSV_COLUMNS)
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


def write_table(path, rows, columns, sort_scores=False):
    """Atomically replace a CSV."""
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    if sort_scores:
        frame = sorted_dataframe(rows)
    else:
        frame = rows_to_dataframe(rows, columns)
    tmp = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return frame


def write_report(path, rows):
    """Atomically write a score-sorted CSV. Error rows sort last."""
    return write_table(path, rows, CSV_COLUMNS, sort_scores=True)


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
        "fix_geometry",
        "avg_blur_score",
        "median_blur_score",
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


def print_group_summary(path):
    path = Path(path)
    if not path.exists():
        return
    frame = pd.read_csv(path)
    print(f"\nPair groups: {len(frame)} ({path})")
    if frame.empty or "verdict" not in frame.columns:
        return
    counts = frame["verdict"].fillna("").value_counts()
    for verdict, count in counts.items():
        print(f"  {verdict}: {count}")


# ============================================================
# Calibration
# ============================================================

def _natural_frame(height, width, seed):
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 120, np.float32)
    for _ in range(80):
        x = int(rng.integers(0, max(1, width - 8)))
        y = int(rng.integers(0, max(1, height - 8)))
        image[
            y:y + int(rng.integers(6, max(7, height // 6))),
            x:x + int(rng.integers(6, max(7, width // 6))),
        ] = rng.uniform(0, 255)
    image += rng.normal(0, 2, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def calibration_checks():
    """Direction checks for blur, noise, pairs, and blockiness.

    These are the in-memory half of a calibration set. They do not tune
    the numeric thresholds. They only fail when a metric moves backwards.
    """
    checks = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    source = _natural_frame(360, 640, 3)
    blur_scores = []
    for sigma in (0, 1, 2, 4):
        image = source if sigma == 0 else cv2.GaussianBlur(source, (0, 0), sigma)
        blur_scores.append(blur_score_laplacian(image, 640))
    add(
        "blur decreases as gaussian blur increases",
        all(earlier > later for earlier, later in zip(blur_scores, blur_scores[1:])),
        " ".join(f"{score:.1f}" for score in blur_scores),
    )

    rng = np.random.default_rng(5)
    noise = np.clip(128 + rng.normal(0, 8, (180, 240)), 0, 255).astype(np.uint8)
    flat = np.full((180, 240), 128, np.uint8)
    edges = np.full((180, 240), 40, np.uint8)
    cv2.rectangle(edges, (20, 20), (200, 140), 220, -1)
    checker = (np.indices((180, 240)).sum(axis=0) % 2 * 255).astype(np.uint8)
    add("noise sigma tracks grain", noise_sigma(noise) > 5, f"{noise_sigma(noise):.2f}")
    add("flat image noise is ~0", noise_sigma(flat) < 0.5, f"{noise_sigma(flat):.2f}")
    add("edges are not counted as noise", noise_sigma(edges) < 1.0, f"{noise_sigma(edges):.2f}")
    add("checkerboard is not counted as noise", noise_sigma(checker) < 1.0, f"{noise_sigma(checker):.2f}")
    add(
        "uint8 median residual is not the noise column",
        noise_sigma(noise) > 1.0,
        "sigma is float; the old median of an 8-bit residual stays at 0",
    )

    native = _natural_frame(180, 320, 7)
    cubic = cv2.resize(native, (640, 360), interpolation=cv2.INTER_CUBIC)
    cubic_metrics = evaluate_pair_frames(native, cubic)
    blurred = cv2.GaussianBlur(native, (0, 0), 2.0)
    blurred_up = cv2.resize(blurred, (640, 360), interpolation=cv2.INTER_CUBIC)
    blurred_metrics = evaluate_pair_frames(native, blurred_up)
    add(
        "bicubic upscale matches the bicubic baseline",
        cubic_metrics is not None
        and 0.9 <= cubic_metrics["blur_native"] <= 1.1
        and 0.9 <= cubic_metrics["blur_vs_base"] <= 1.1,
        str(cubic_metrics),
    )
    add(
        "a blurred upscale is softer than the bicubic roundtrip",
        blurred_metrics is not None
        and cubic_metrics is not None
        and blurred_metrics["blur_native"] < cubic_metrics["blur_native"] - 0.2,
        f"blurred={None if not blurred_metrics else round(blurred_metrics['blur_native'], 3)} "
        f"cubic={None if not cubic_metrics else round(cubic_metrics['blur_native'], 3)}",
    )

    content = _natural_frame(360, 720, 9)
    framed = np.zeros((404, 720), np.uint8)
    framed[22:382, :] = content
    bounds = letterbox_bounds(framed)
    cropped_blur = measure_frame(framed, crop=bounds)["blur"]
    content_blur = blur_score_laplacian(content, 640)
    add(
        "letterbox crop removes black bars",
        bounds[1] >= 16 and bounds[3] <= 370,
        str(bounds),
    )
    add(
        "blur after the letterbox crop matches the active picture",
        content_blur > 0 and abs(cropped_blur - content_blur) / content_blur < 0.02,
        f"content={content_blur:.1f} cropped={cropped_blur:.1f} bounds={bounds}",
    )

    block = np.zeros((128, 128), np.uint8)
    for y in range(0, 128, 8):
        for x in range(0, 128, 8):
            block[y:y + 8, x:x + 8] = int(rng.integers(0, 255))
    smooth = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
    add(
        "block grid outscores a smooth ramp",
        blockiness_score(block) > blockiness_score(smooth),
        f"block={blockiness_score(block):.2f} smooth={blockiness_score(smooth):.2f}",
    )
    upscaled_blocks = cv2.resize(block, (192, 192), interpolation=cv2.INTER_NEAREST)
    period_8 = blockiness_score(upscaled_blocks)
    period_scaled = blockiness_scaled(upscaled_blocks, 1.5, 1.5)
    add(
        "scaled block period sees a 1.5x upscale the 8px period misses",
        period_scaled > period_8 * 2,
        f"period8={period_8:.2f} scaled={period_scaled:.2f}",
    )
    return checks


def run_calibration():
    checks = calibration_checks()
    failures = 0
    for name, ok, detail in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            failures += 1
    if failures:
        raise SystemExit(f"{failures} calibration check(s) failed")
    print(f"\n{len(checks)} calibration checks passed.")


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
            "  python scan_video_quality.py --root ./videos --samples 8 --no-pairs\n"
            "  python scan_video_quality.py --calibrate\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default=ROOT_FOLDER, help="Folder to scan recursively")
    parser.add_argument("--output", default=OUTPUT_CSV, help="CSV report path")
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=None,
        help="Frames per video. Default: at least 16, plus one per 5 minutes, capped at 40",
    )
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
    parser.add_argument("--pairs", dest="pairs", action="store_true")
    parser.add_argument("--no-pairs", dest="pairs", action="store_false")
    parser.add_argument("--groups-output", default="", help="CSV for original/processed pairs")
    parser.add_argument("--config", default="", help="YAML file of threshold overrides")
    parser.add_argument("--calibrate", action="store_true", help="Run metric direction checks and exit")
    parser.set_defaults(
        create_contact_sheets=CREATE_CONTACT_SHEETS,
        skip_already_processed=SKIP_ALREADY_PROCESSED,
        resume=True,
        pairs=True,
    )
    return parser


def config_from_args(args):
    thresholds = load_thresholds(args.config) if args.config else Thresholds()
    return ScanConfig(
        root_folder=args.root,
        output_csv=args.output,
        create_contact_sheets=args.create_contact_sheets,
        contact_sheet_min_score=args.contact_sheet_min_score,
        contact_sheet_folder=args.contact_sheet_folder,
        sample_count=args.samples if args.samples is not None else SAMPLE_COUNT,
        samples_explicit=args.samples is not None,
        analysis_frame_max_width=args.analysis_width,
        contact_thumb_width=CONTACT_THUMB_WIDTH,
        skip_already_processed=args.skip_already_processed,
        workers=args.workers,
        resume=args.resume,
        frame_timeout=args.frame_timeout,
        probe_timeout=args.probe_timeout,
        checkpoint_every=args.checkpoint_every,
        pairs=args.pairs,
        groups_csv=args.groups_output,
        thresholds=thresholds,
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
    kept = []
    stale = 0
    if config.resume:
        for row in existing:
            if row_is_current(row):
                kept.append(row)
            else:
                stale += 1
        if stale:
            print(
                f"Resume: re-scanning {stale} file(s) whose report schema, "
                "size, or modification time does not match."
            )
    done = {str(row.get("path")) for row in kept if row.get("path")}
    pending = [path for path in files if str(path) not in done]
    if config.resume and kept:
        print(f"Resume: keeping {len(kept)} file(s) already in {config.output_csv}.")

    worker_count = min(config.workers, len(pending)) if pending else 1
    print(f"Analyzing {len(pending)} video file(s) with {worker_count} worker(s).")
    rows = list(kept)

    def checkpoint():
        return write_report(config.output_csv, rows)

    if not pending:
        if config.pairs:
            print("Comparing original and processed pairs...")
            compare_library(rows, config)
        frame = checkpoint()
        print(f"\nDone. Report saved to: {config.output_csv}")
        if config.pairs:
            print(f"Pair groups saved to: {groups_output_path(config)}")
            print_group_summary(groups_output_path(config))
        print_summary(frame)
        return rows

    def store_result(row):
        rows.append(row)
        if row.get("error"):
            tqdm.write(f"Error: {row.get('path')}: {row.get('error')}")
        finished_new = len(rows) - len(kept)
        if finished_new >= 1 and finished_new % config.checkpoint_every == 0:
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
                        _attach_identity(failed, video_path)
                        failed["issue_flags"] = "error"
                        failed["recommendation"] = "manual_check_error"
                        failed["error"] = clip_error(exc)
                        store_result(failed)
    except KeyboardInterrupt:
        checkpoint()
        print(f"\nInterrupted. Partial report saved to: {config.output_csv}")
        raise

    if config.pairs:
        print("Comparing original and processed pairs...")
        compare_library(rows, config)
    frame = checkpoint()
    print(f"\nDone. Report saved to: {config.output_csv}")
    if config.pairs:
        print(f"Pair groups saved to: {groups_output_path(config)}")
        print_group_summary(groups_output_path(config))
    if config.create_contact_sheets:
        print(f"Contact sheets saved to: {config.contact_sheet_folder}")
    print_summary(frame)
    return rows


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.calibrate:
        run_calibration()
        return
    config = config_from_args(args)
    scan_library(config)


if __name__ == "__main__":
    main()
