import os
import csv
import json
import math
import shutil
import tempfile
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

ROOT_FOLDER = r"./videos"
# Alternative, usually more stable on Windows:
# ROOT_FOLDER = r"Z:\videos"

OUTPUT_CSV = "video_quality_report.csv"

# Contact sheets are very useful for manual review.
CREATE_CONTACT_SHEETS = False

# Only create contact sheets for videos with score >= this.
CONTACT_SHEET_MIN_SCORE = 55

# Folder for contact-sheet JPGs.
CONTACT_SHEET_FOLDER = "contact_sheets"

# Number of frames to sample per video.
SAMPLE_COUNT = 8

# Extracted sample frame width for analysis.
# Keeping this moderate makes scanning much faster.
ANALYSIS_FRAME_MAX_WIDTH = 640

# Contact sheet thumbnail width.
CONTACT_THUMB_WIDTH = 320

# If True, skip files that look like existing processed/upscaled outputs.
# For now I recommend False, because you may want to compare originals vs _prob4/_pnat1.
SKIP_ALREADY_PROCESSED = False

# Extensions to scan.
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".mpg", ".mpeg",
    ".m4v", ".ts", ".m2ts", ".webm", ".flv", ".3gp"
}

# Filename markers that suggest the file is already processed/upscaled.
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


# ============================================================
# Utility functions
# ============================================================

def check_external_tools():
    """
    Make sure ffmpeg and ffprobe are available.
    """
    missing = []

    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")

    if shutil.which("ffprobe") is None:
        missing.append("ffprobe")

    if missing:
        raise RuntimeError(
            "Missing required tool(s): "
            + ", ".join(missing)
            + "\nPlease install FFmpeg and make sure ffmpeg.exe and ffprobe.exe are in PATH."
        )


def run_cmd(cmd):
    """
    Run a command and return stdout text.
    """
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    return result.stdout


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def parse_fraction(frac_text):
    """
    Parse ffprobe frame rate like '30000/1001'.
    """
    try:
        if not frac_text or frac_text == "0/0":
            return 0.0
        if "/" in frac_text:
            num, den = frac_text.split("/", 1)
            den = float(den)
            if den == 0:
                return 0.0
            return float(num) / den
        return float(frac_text)
    except Exception:
        return 0.0


def is_probably_processed(path):
    """
    Detect whether filename looks like an already processed/upscaled version.
    """
    name = Path(path).stem.lower()
    return any(marker in name for marker in PROCESSED_MARKERS)


def format_duration(seconds):
    """
    Convert seconds to HH:MM:SS string.
    """
    seconds = safe_float(seconds, 0)
    if seconds <= 0:
        return ""

    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================
# FFprobe metadata
# ============================================================

def ffprobe_video(path):
    """
    Extract video metadata using ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,codec_name,bit_rate,avg_frame_rate,r_frame_rate,duration,pix_fmt:format=duration,bit_rate,size",
        "-of", "json",
        str(path)
    ]

    output = run_cmd(cmd)
    data = json.loads(output)

    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No video stream found")

    stream = streams[0]
    fmt = data.get("format", {})

    width = safe_int(stream.get("width"))
    height = safe_int(stream.get("height"))
    codec = stream.get("codec_name", "")
    pix_fmt = stream.get("pix_fmt", "")

    duration = safe_float(stream.get("duration"), 0.0)
    if duration <= 0:
        duration = safe_float(fmt.get("duration"), 0.0)

    bit_rate = safe_int(stream.get("bit_rate"), 0)
    if bit_rate <= 0:
        bit_rate = safe_int(fmt.get("bit_rate"), 0)

    size = safe_int(fmt.get("size"), 0)

    avg_frame_rate = parse_fraction(stream.get("avg_frame_rate"))
    r_frame_rate = parse_fraction(stream.get("r_frame_rate"))
    frame_rate = avg_frame_rate if avg_frame_rate > 0 else r_frame_rate

    return {
        "width": width,
        "height": height,
        "codec": codec,
        "pix_fmt": pix_fmt,
        "duration": duration,
        "duration_hms": format_duration(duration),
        "frame_rate": round(frame_rate, 3),
        "bit_rate": bit_rate,
        "size": size,
    }


# ============================================================
# Frame extraction
# ============================================================

def extract_sample_frames(video_path, sample_count=8, max_width=640):
    """
    Extract sample frames from the video using ffmpeg.
    Returns list of tuples:
        [(timestamp_seconds, frame_bgr), ...]
    """
    meta = ffprobe_video(video_path)
    duration = meta["duration"]

    if duration <= 0:
        return []

    # Avoid very beginning and ending.
    start_ratio = 0.10
    end_ratio = 0.90

    if duration < 60:
        start_ratio = 0.05
        end_ratio = 0.95

    timestamps = np.linspace(
        duration * start_ratio,
        duration * end_ratio,
        sample_count
    )

    frames = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for i, ts in enumerate(timestamps):
            frame_path = tmpdir / f"frame_{i:03d}.jpg"

            vf = f"scale='min({max_width},iw)':-2"

            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-ss", str(float(ts)),
                "-i", str(video_path),
                "-frames:v", "1",
                "-vf", vf,
                "-q:v", "2",
                "-y",
                str(frame_path)
            ]

            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    check=True
                )

                img = cv2.imread(str(frame_path))
                if img is not None:
                    frames.append((float(ts), img))

            except subprocess.CalledProcessError:
                continue

    return frames


# ============================================================
# Image metrics
# ============================================================

def blur_score_laplacian(frame):
    """
    Higher value means sharper.
    Lower value means blurrier.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def blockiness_score(frame):
    """
    Estimate blockiness by comparing differences at 8x8 block boundaries
    against non-boundary regions.

    Higher value means stronger possible compression blocking.

    Important:
    This is a heuristic. It can be fooled by subtitles, hard edges, animation,
    interlacing, high-contrast patterns, and natural scene structure.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0

    diff_x = np.abs(gray[:, 1:] - gray[:, :-1])
    diff_y = np.abs(gray[1:, :] - gray[:-1, :])

    # Boundaries after every 8 pixels.
    boundary_cols = np.arange(7, w - 1, 8)
    boundary_rows = np.arange(7, h - 1, 8)

    if len(boundary_cols) == 0 or len(boundary_rows) == 0:
        return 0.0

    boundary_x = diff_x[:, boundary_cols].mean()
    boundary_y = diff_y[boundary_rows, :].mean()

    all_cols = np.arange(diff_x.shape[1])
    all_rows = np.arange(diff_y.shape[0])

    non_boundary_cols = np.setdiff1d(all_cols, boundary_cols)
    non_boundary_rows = np.setdiff1d(all_rows, boundary_rows)

    if len(non_boundary_cols) == 0 or len(non_boundary_rows) == 0:
        return 0.0

    non_boundary_x = diff_x[:, non_boundary_cols].mean()
    non_boundary_y = diff_y[non_boundary_rows, :].mean()

    boundary_mean = (boundary_x + boundary_y) / 2.0
    non_boundary_mean = (non_boundary_x + non_boundary_y) / 2.0

    ratio = boundary_mean / max(non_boundary_mean, 1e-6)
    return float(ratio)


def noise_score(frame):
    """
    Estimate high-frequency noise using difference between image and median blur.
    Higher value means noisier/grainier.

    Note:
    Grain is not always bad. Old analog video may naturally have grain.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    residual = cv2.absdiff(gray, denoised)
    return float(np.mean(residual))


def black_frame_ratio(frame, threshold=16):
    """
    Return approximate percentage of near-black pixels.
    Useful because very dark frames can skew blur detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray < threshold))


def brightness_score(frame):
    """
    Average brightness, 0 to 255.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


# ============================================================
# Improved scoring and classification
# ============================================================

def classify_resolution(width, height):
    if width <= 0 or height <= 0:
        return "unknown_resolution"

    if height <= 270:
        return "very_tiny"
    elif height <= 360:
        return "very_low_resolution"
    elif height <= 480:
        return "low_resolution"
    elif height <= 576:
        return "sd_plus"
    elif height <= 720:
        return "hd_720"
    elif height <= 1080:
        return "full_hd_1080"
    else:
        return "above_1080"


def classify_blur(avg_blur):
    if avg_blur is None:
        return "unknown_blur"

    if avg_blur < 35:
        return "very_blurry"
    elif avg_blur < 60:
        return "blurry"
    elif avg_blur < 100:
        return "soft"
    elif avg_blur < 150:
        return "slightly_soft"
    elif avg_blur < 300:
        return "acceptable_sharpness"
    else:
        return "sharp_or_noisy"


def classify_blockiness(avg_block):
    if avg_block is None:
        return "unknown_blockiness"

    # Less aggressive than original script.
    if avg_block > 1.60:
        return "very_blocky"
    elif avg_block > 1.45:
        return "blocky"
    elif avg_block > 1.30:
        return "slightly_blocky"
    else:
        return "not_obviously_blocky"


def calculate_improved_score(width, height, avg_blur, avg_block, bitrate_per_pixel):
    """
    Better scoring formula calibrated for your current library.

    Higher score = higher priority for Topaz/restoration review.
    """
    score = 0.0

    # Resolution score
    if height <= 0:
        score += 0
    elif height <= 270:
        score += 40
    elif height <= 360:
        score += 30
    elif height <= 480:
        score += 22
    elif height <= 576:
        score += 12
    elif height <= 720:
        score += 5

    # Blur score
    if avg_blur is not None:
        if avg_blur < 35:
            score += 40
        elif avg_blur < 60:
            score += 30
        elif avg_blur < 100:
            score += 15
        elif avg_blur < 150:
            score += 5

    # Blockiness score - less aggressive than before
    if avg_block is not None:
        if avg_block > 1.60:
            score += 25
        elif avg_block > 1.45:
            score += 18
        elif avg_block > 1.30:
            score += 8

    # Bitrate-per-pixel score
    # This is a rough heuristic, not a precise quality measure.
    if bitrate_per_pixel and bitrate_per_pixel > 0:
        if height >= 720 and bitrate_per_pixel < 1.8:
            score += 8
        elif height <= 480 and bitrate_per_pixel < 2.5:
            score += 8

    return round(score, 2)


def build_issue_flags(width, height, avg_blur, avg_block, bitrate_per_pixel, processed):
    flags = []

    resolution_class = classify_resolution(width, height)
    blur_class = classify_blur(avg_blur)
    block_class = classify_blockiness(avg_block)

    if processed:
        flags.append("already_processed_name")

    if resolution_class in ["very_tiny", "very_low_resolution", "low_resolution", "sd_plus"]:
        flags.append(resolution_class)

    if blur_class in ["very_blurry", "blurry", "soft", "slightly_soft"]:
        flags.append(blur_class)

    if block_class in ["very_blocky", "blocky", "slightly_blocky"]:
        flags.append(block_class)

    if bitrate_per_pixel and bitrate_per_pixel > 0:
        if height >= 720 and bitrate_per_pixel < 1.8:
            flags.append("low_bitrate_for_hd")
        elif height <= 480 and bitrate_per_pixel < 2.5:
            flags.append("low_bitrate_for_sd")

    if not flags:
        flags.append("ok_or_uncertain")

    return flags


def assign_recommendation(width, height, avg_blur, avg_block, score, processed):
    """
    Assign a practical Topaz workflow recommendation.
    """
    if processed:
        processed_note = "already_processed_review_only"
    else:
        processed_note = None

    low_res = height > 0 and height <= 576
    very_low_res = height > 0 and height <= 360
    blurry = avg_blur is not None and avg_blur < 100
    very_blurry = avg_blur is not None and avg_blur < 60
    blocky = avg_block is not None and avg_block > 1.45
    very_blocky = avg_block is not None and avg_block > 1.60
    sharpish = avg_blur is not None and avg_blur >= 150

    if score >= 75:
        base = "top_priority_restore"
    elif low_res and very_blurry and blocky:
        base = "restore_deblur_artifact_reduce"
    elif low_res and blurry:
        base = "restore_or_deblur_then_upscale"
    elif low_res and sharpish:
        base = "upscale_only_gentle"
    elif very_low_res:
        base = "upscale_candidate_very_low_res"
    elif blocky or very_blocky:
        base = "artifact_reduction_review"
    elif score >= 45:
        base = "manual_review"
    else:
        base = "low_priority_or_ok"

    if processed_note:
        return processed_note + "+" + base

    return base


# ============================================================
# Contact sheet generation
# ============================================================

def resize_keep_aspect(frame, target_width):
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame

    scale = target_width / float(w)
    target_height = max(1, int(round(h * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_text_box(img, lines, x, y, font_scale=0.5, thickness=1):
    """
    Draw text with a dark background for readability.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = int(22 * font_scale / 0.5)
    padding = 6

    max_width = 0
    for line in lines:
        size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, size[0])

    box_w = max_width + padding * 2
    box_h = line_height * len(lines) + padding * 2

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    text_y = y + padding + line_height - 6
    for line in lines:
        cv2.putText(
            img,
            line,
            (x + padding, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
        text_y += line_height


def create_contact_sheet(video_path, frames_with_ts, result, output_folder):
    """
    Create one JPG contact sheet for a video.
    """
    if not frames_with_ts:
        return None

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    thumbs = []

    for ts, frame in frames_with_ts:
        thumb = resize_keep_aspect(frame, CONTACT_THUMB_WIDTH)

        label = format_duration(ts)
        draw_text_box(
            thumb,
            [label],
            x=5,
            y=5,
            font_scale=0.45,
            thickness=1
        )

        thumbs.append(thumb)

    # Normalize thumb heights by padding.
    max_h = max(t.shape[0] for t in thumbs)
    normalized = []

    for t in thumbs:
        h, w = t.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
            t = np.vstack([t, pad])
        normalized.append(t)

    # Grid: 4 columns x 2 rows for 8 frames.
    cols = 4
    rows = int(math.ceil(len(normalized) / cols))

    cell_w = CONTACT_THUMB_WIDTH
    cell_h = max_h

    sheet_w = cols * cell_w
    header_h = 120
    sheet_h = header_h + rows * cell_h

    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    # Header
    name = Path(video_path).name
    info_lines = [
        name,
        f"score={result.get('quality_issue_score')}  recommendation={result.get('recommendation')}",
        f"res={result.get('width')}x{result.get('height')}  blur={result.get('avg_blur_score')}  block={result.get('avg_blockiness_score')}  noise={result.get('avg_noise_score')}",
        f"flags={result.get('issue_flags')}",
    ]

    draw_text_box(
        sheet,
        info_lines,
        x=8,
        y=8,
        font_scale=0.5,
        thickness=1
    )

    for idx, t in enumerate(normalized):
        r = idx // cols
        c = idx % cols

        y1 = header_h + r * cell_h
        x1 = c * cell_w

        h, w = t.shape[:2]
        sheet[y1:y1 + h, x1:x1 + w] = t

    # Safe filename
    stem = Path(video_path).stem
    parent = Path(video_path).parent.name
    safe_name = f"{parent}__{stem}".replace("\\", "_").replace("/", "_").replace(":", "_")
    out_path = output_folder / f"{safe_name}.jpg"

    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    return str(out_path)


# ============================================================
# Video analysis
# ============================================================

def analyze_video(video_path, sample_count=8):
    """
    Analyze one video and return metrics.
    """
    video_path = Path(video_path)
    processed = is_probably_processed(video_path)

    if SKIP_ALREADY_PROCESSED and processed:
        return {
            "path": str(video_path),
            "quality_issue_score": 0,
            "issue_flags": "skipped_already_processed_name",
            "recommendation": "skipped",
            "error": "",
        }

    if not video_path.exists():
        return {
            "path": str(video_path),
            "sampled_frames": 0,
            "quality_issue_score": 999,
            "issue_flags": "missing_file",
            "recommendation": "fix_path_or_missing_file",
            "error": "File does not exist when analyzing",
        }

    meta = ffprobe_video(video_path)

    frames_with_ts = extract_sample_frames(
        video_path,
        sample_count=sample_count,
        max_width=ANALYSIS_FRAME_MAX_WIDTH
    )

    if not frames_with_ts:
        return {
            **meta,
            "path": str(video_path),
            "is_probably_processed": processed,
            "sampled_frames": 0,
            "avg_blur_score": None,
            "avg_blockiness_score": None,
            "avg_noise_score": None,
            "avg_brightness": None,
            "avg_black_frame_ratio": None,
            "bitrate_per_pixel": None,
            "quality_issue_score": 999,
            "issue_flags": "could_not_sample",
            "recommendation": "manual_check_could_not_sample",
            "contact_sheet": "",
            "error": "Could not sample frames",
        }

    frames = [frame for ts, frame in frames_with_ts]

    blur_scores = [blur_score_laplacian(f) for f in frames]
    block_scores = [blockiness_score(f) for f in frames]
    noise_scores = [noise_score(f) for f in frames]
    brightness_scores = [brightness_score(f) for f in frames]
    black_ratios = [black_frame_ratio(f) for f in frames]

    avg_blur = float(np.mean(blur_scores))
    avg_block = float(np.mean(block_scores))
    avg_noise = float(np.mean(noise_scores))
    avg_brightness = float(np.mean(brightness_scores))
    avg_black_ratio = float(np.mean(black_ratios))

    width = meta["width"]
    height = meta["height"]
    bit_rate = meta["bit_rate"]

    pixels = max(width * height, 1)

    # This means bits per second per pixel.
    # It is not a perfect metric, but useful for comparing your own library.
    bitrate_per_pixel = bit_rate / pixels if bit_rate > 0 else 0

    score = calculate_improved_score(
        width=width,
        height=height,
        avg_blur=avg_blur,
        avg_block=avg_block,
        bitrate_per_pixel=bitrate_per_pixel
    )

    flags = build_issue_flags(
        width=width,
        height=height,
        avg_blur=avg_blur,
        avg_block=avg_block,
        bitrate_per_pixel=bitrate_per_pixel,
        processed=processed
    )

    recommendation = assign_recommendation(
        width=width,
        height=height,
        avg_blur=avg_blur,
        avg_block=avg_block,
        score=score,
        processed=processed
    )

    result = {
        **meta,
        "path": str(video_path),
        "filename": video_path.name,
        "folder": str(video_path.parent),
        "is_probably_processed": processed,
        "sampled_frames": len(frames),
        "avg_blur_score": round(avg_blur, 3),
        "min_blur_score": round(float(np.min(blur_scores)), 3),
        "max_blur_score": round(float(np.max(blur_scores)), 3),
        "avg_blockiness_score": round(avg_block, 3),
        "avg_noise_score": round(avg_noise, 3),
        "avg_brightness": round(avg_brightness, 3),
        "avg_black_frame_ratio": round(avg_black_ratio, 4),
        "bitrate_per_pixel": round(float(bitrate_per_pixel), 6),
        "quality_issue_score": score,
        "resolution_class": classify_resolution(width, height),
        "blur_class": classify_blur(avg_blur),
        "blockiness_class": classify_blockiness(avg_block),
        "issue_flags": ",".join(flags),
        "recommendation": recommendation,
        "contact_sheet": "",
        "error": "",
    }

    if CREATE_CONTACT_SHEETS and score >= CONTACT_SHEET_MIN_SCORE:
        try:
            sheet_path = create_contact_sheet(
                video_path=video_path,
                frames_with_ts=frames_with_ts,
                result=result,
                output_folder=CONTACT_SHEET_FOLDER
            )
            result["contact_sheet"] = sheet_path or ""
        except Exception as e:
            result["contact_sheet"] = ""
            result["contact_sheet_error"] = str(e)

    return result


# ============================================================
# File discovery
# ============================================================

def find_video_files(root_folder):
    root = Path(root_folder)

    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                yield path


# ============================================================
# Summary output
# ============================================================

def print_summary(df):
    print("\n================ Summary ================\n")

    print(f"Total rows: {len(df)}")

    if "error" in df.columns:
        error_count = int((df["error"].fillna("") != "").sum())
        print(f"Rows with error: {error_count}")

    if "quality_issue_score" in df.columns:
        print("\nScore distribution:")
        bins = [
            ("90+", df["quality_issue_score"] >= 90),
            ("75-89", (df["quality_issue_score"] >= 75) & (df["quality_issue_score"] < 90)),
            ("55-74", (df["quality_issue_score"] >= 55) & (df["quality_issue_score"] < 75)),
            ("35-54", (df["quality_issue_score"] >= 35) & (df["quality_issue_score"] < 55)),
            ("0-34", (df["quality_issue_score"] >= 0) & (df["quality_issue_score"] < 35)),
        ]

        for label, mask in bins:
            print(f"  {label}: {int(mask.sum())}")

    if "recommendation" in df.columns:
        print("\nRecommendation counts:")
        counts = df["recommendation"].fillna("").value_counts()
        for rec, count in counts.items():
            print(f"  {rec}: {count}")

    print("\nTop 25 candidates:")
    columns_to_show = [
        "quality_issue_score",
        "recommendation",
        "width",
        "height",
        "avg_blur_score",
        "avg_blockiness_score",
        "bit_rate",
        "issue_flags",
        "path",
        "contact_sheet",
    ]

    existing_cols = [c for c in columns_to_show if c in df.columns]
    print(df[existing_cols].head(25).to_string(index=False))


# ============================================================
# Main
# ============================================================

def main():
    check_external_tools()

    print(f"Scanning root folder: {ROOT_FOLDER}")

    video_files = list(find_video_files(ROOT_FOLDER))
    print(f"Found {len(video_files)} video files.")

    results = []

    for video_path in tqdm(video_files, desc="Analyzing videos"):
        try:
            result = analyze_video(video_path, sample_count=SAMPLE_COUNT)
            results.append(result)
        except Exception as e:
            results.append({
                "path": str(video_path),
                "filename": Path(video_path).name,
                "folder": str(Path(video_path).parent),
                "sampled_frames": 0,
                "quality_issue_score": 999,
                "issue_flags": "error",
                "recommendation": "manual_check_error",
                "error": str(e),
            })

    df = pd.DataFrame(results)

    if "quality_issue_score" in df.columns:
        df = df.sort_values(
            by=["quality_issue_score", "avg_blur_score"],
            ascending=[False, True],
            na_position="last"
        )

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nDone. Report saved to: {OUTPUT_CSV}")

    if CREATE_CONTACT_SHEETS:
        print(f"Contact sheets saved to: {CONTACT_SHEET_FOLDER}")

    print_summary(df)


if __name__ == "__main__":
    main()