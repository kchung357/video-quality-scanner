"""Frame sampling and raw decode."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from vqs.config import ScanConfig
from vqs.util import path_seed

MIN_SAMPLE_COUNT = 16
MAX_SAMPLE_COUNT = 40
SAMPLE_EVERY_SECONDS = 300.0


def resolved_sample_count(duration, config: ScanConfig | None) -> int:
    if config is not None and config.samples_explicit:
        return max(1, int(config.sample_count))
    adaptive = int(round(float(duration or 0) / SAMPLE_EVERY_SECONDS))
    return min(MAX_SAMPLE_COUNT, max(MIN_SAMPLE_COUNT, adaptive))


def sample_timestamps(duration, sample_count, seed=None) -> list[float]:
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


def ffmpeg_thread_count(config: ScanConfig | None) -> int:
    """Threads for one FFmpeg process. 0 leaves the choice to FFmpeg."""
    if config is None:
        return 0
    return max(0, int(getattr(config, "ffmpeg_threads", 0) or 0))


def seek_points(timestamp) -> tuple[float, float]:
    timestamp = max(0.0, float(timestamp))
    pre = max(0.0, timestamp - 1.0)
    fine = timestamp - pre
    return pre, fine


def build_frame_command(video_path, timestamp, threads=0, frame_count=1) -> list[str]:
    pre, fine = seek_points(timestamp)
    count = max(1, int(frame_count))
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if threads and int(threads) > 0:
        cmd.extend(["-threads", str(int(threads))])
    cmd.extend(
        [
            "-noautorotate",
            "-ss",
            f"{pre:.3f}",
            "-i",
            str(video_path),
            "-ss",
            f"{fine:.3f}",
            "-map",
            "0:v:0",
            "-frames:v",
            str(count),
            "-an",
            "-sn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
    )
    return cmd


def decode_frames(video_path, timestamp, width, height, timeout, threads, frame_count=1):
    if width <= 0 or height <= 0 or width > 16384 or height > 16384:
        return [], f"Refusing to decode frame size {width}x{height}"

    cmd = build_frame_command(video_path, timestamp, threads=threads, frame_count=frame_count)
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
        raw = np.frombuffer(result.stdout[start : start + frame_bytes], dtype=np.uint8)
        frames.append(raw.reshape((int(height), int(width), 3)).copy())
    return frames, ""


def decode_frame(video_path, timestamp, width, height, timeout, threads):
    frames, error = decode_frames(video_path, timestamp, width, height, timeout, threads, 1)
    if not frames:
        return None, error or f"ffmpeg failed at {float(timestamp):.2f}s"
    return frames[0], ""


def maybe_cache_path(video_path, cache_dir: str) -> Path:
    """Return the source path. Cache dir reserved for future extract staging."""
    return Path(video_path)


def extract_sample_frames(video_path, meta, config: ScanConfig, sample_count=None, seed=None):
    count = sample_count
    if count is None:
        count = resolved_sample_count(meta["duration"], config)
    if seed is None:
        seed = path_seed(video_path)
    source = maybe_cache_path(video_path, config.cache_dir if config else "")
    timestamps = sample_timestamps(meta["duration"], count, seed=seed)
    threads = ffmpeg_thread_count(config)
    timeout = 60.0 if config is None else config.frame_timeout
    frames = []
    errors = []
    for timestamp in timestamps:
        frame, error = decode_frame(
            source,
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
