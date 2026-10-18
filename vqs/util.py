"""Shared helpers for paths, numbers, and process helpers."""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".ts",
    ".m2ts",
    ".webm",
    ".flv",
    ".3gp",
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


def check_external_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "Missing required tool(s): "
            + ", ".join(missing)
            + ". Install FFmpeg and make sure ffmpeg and ffprobe are on PATH."
        )


def run_text(cmd, timeout: float) -> str:
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


def clip_error(message, limit: int = 500) -> str:
    text = " ".join(str(message).split())
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def round_or_none(value, digits: int):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def parse_fraction(frac_text) -> float:
    try:
        if frac_text is None or frac_text == "" or frac_text == "0/0":
            return 0.0
        text = str(frac_text)
        if "/" in text:
            num, den = text.split("/", 1)
            den_f = float(den)
            if den_f == 0:
                return 0.0
            return float(num) / den_f
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def parse_sar(text) -> float:
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


def is_probably_processed(path) -> bool:
    name = Path(path).stem.lower()
    return any(marker in name for marker in PROCESSED_MARKERS)


def looks_processed(path, encoder: str = "") -> bool:
    if is_probably_processed(path):
        return True
    text = (encoder or "").lower()
    return any(marker in text for marker in ENCODER_MARKERS)


def format_duration(seconds) -> str:
    seconds = safe_float(seconds, 0)
    if seconds <= 0:
        return ""
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def stream_rotation(stream) -> float:
    for item in stream.get("side_data_list") or []:
        if isinstance(item, dict) and item.get("rotation") is not None:
            return safe_float(item.get("rotation"), 0.0)
    tags = stream.get("tags") or {}
    for key in ("rotate", "rotation"):
        if tags.get(key) not in (None, ""):
            return safe_float(tags.get(key), 0.0)
    return 0.0


def display_dimensions(width, height, sar, rotation) -> tuple[int, int]:
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


def short_side(width, height, display_width=0, display_height=0) -> int:
    if display_width and display_height:
        return min(int(display_width), int(display_height))
    if width and height:
        return min(int(width), int(height))
    return 0


def file_identity(path) -> tuple[int, int]:
    st = Path(path).stat()
    return int(st.st_size), int(st.st_mtime)


def path_seed(path) -> int:
    digest = hashlib.sha1(str(path).encode("utf-8", "surrogateescape")).digest()
    return int.from_bytes(digest[:4], "little")


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


def parse_simple_yaml(text: str) -> dict:
    root: dict = {}
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
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = coerce_scalar(value)
    return root


def normalized_stem(filename: str) -> str:
    stem = Path(str(filename)).stem.lower()
    previous = None
    while stem and previous != stem:
        previous = stem
        stem = PAIR_MARKER_RE.sub("", stem)
    return stem


def stems_related(name_a: str, name_b: str) -> bool:
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


def group_key(folder: str, duration) -> str:
    raw = f"{folder}|{round(safe_float(duration))}"
    return hashlib.sha1(raw.encode("utf-8", "surrogateescape")).hexdigest()[:12]
