"""Full-file decode health check via FFmpeg."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from vqs.util import clip_error, safe_float


_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
_ERROR_HINTS = (
    "error",
    "invalid",
    "corrupt",
    "damaged",
    "missing picture",
    "concealing",
    "decode_slice",
    "ac-tex damaged",
    "overread",
    "truncated",
)
# Null-muxer timestamp complaints from ``-f null`` are not decode failures and
# do not affect typical playback; ``-xerror`` still aborts on them.
_BENIGN_HINTS = (
    "non monotonically increasing dts to muxer",
)


@dataclass
class HealthResult:
    status: str  # ok | errors | failed
    error_count: int = 0
    first_error: str = ""
    progress: float = 0.0  # 0..1 estimate from ffmpeg -progress time=
    message: str = ""


def health_timeout_for(duration: float, configured: float = 0.0) -> float:
    if configured and configured > 0:
        return float(configured)
    # Allow slow remote/storage seeks: 3x duration + two minutes, min 120s, max 6h.
    return min(6 * 3600.0, max(120.0, float(duration or 0) * 3.0 + 120.0))


def _parse_time(text: str) -> float | None:
    match = _TIME_RE.search(text)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _is_benign_line(line: str) -> bool:
    lower = line.lower()
    return any(token in lower for token in _BENIGN_HINTS)


def _is_error_line(line: str) -> bool:
    lower = line.lower()
    if not lower.strip() or _is_benign_line(line):
        return False
    return any(token in lower for token in _ERROR_HINTS)


def build_health_command(path, threads: int = 0) -> list[str]:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-xerror",
    ]
    if threads and int(threads) > 0:
        cmd.extend(["-threads", str(int(threads))])
    cmd.extend(
        [
            "-i",
            str(path),
            "-map",
            "0:v:0?",
            "-f",
            "null",
            "-",
        ]
    )
    return cmd


def check_video_health(path, duration: float = 0.0, timeout: float = 0.0, threads: int = 0) -> HealthResult:
    """Decode the whole file to null and collect stderr error lines.

    Uses ``-xerror`` so the first hard decode failure can abort early, while
    still recording soft error spam from libavcodec when the process continues.
    """
    path = Path(path)
    if not path.exists():
        return HealthResult(status="failed", first_error="file missing", message="file missing")

    limit = health_timeout_for(duration, timeout)
    cmd = build_health_command(path, threads=threads)
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=limit,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, (bytes, bytearray)) else ""
        progress = 0.0
        last_time = _parse_time(stderr)
        if last_time is not None and duration > 0:
            progress = min(1.0, last_time / float(duration))
        return HealthResult(
            status="failed",
            error_count=1,
            first_error=clip_error("Timed out during health decode"),
            progress=progress,
            message="timeout",
        )

    stderr = result.stderr.decode("utf-8", "replace")
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    substantive = [line for line in lines if not _is_benign_line(line)]
    errors = [line for line in substantive if _is_error_line(line)]
    first = clip_error(errors[0]) if errors else ""
    progress = 1.0
    last_time = None
    for line in lines:
        stamp = _parse_time(line)
        if stamp is not None:
            last_time = stamp
    if last_time is not None and duration > 0:
        progress = min(1.0, last_time / float(duration))

    if result.returncode != 0:
        # ``-xerror`` exits on the first error-level log, including benign
        # null-muxer DTS complaints. Treat those as healthy when nothing else
        # remains on stderr.
        if not substantive:
            return HealthResult(status="ok", error_count=0, first_error="", progress=1.0, message="ok")
        if not first:
            first = clip_error(substantive[-1] if substantive else f"ffmpeg exit {result.returncode}")
        return HealthResult(
            status="failed",
            error_count=max(1, len(errors)),
            first_error=first,
            progress=progress,
            message=f"exit {result.returncode}",
        )
    if errors:
        return HealthResult(
            status="errors",
            error_count=len(errors),
            first_error=first,
            progress=1.0,
            message="decode warnings",
        )
    return HealthResult(status="ok", error_count=0, first_error="", progress=1.0, message="ok")


def health_row_fields(result: HealthResult) -> dict:
    return {
        "health_status": result.status,
        "health_error_count": int(result.error_count),
        "health_first_error": result.first_error or "",
        "health_progress": safe_float(result.progress),
    }
