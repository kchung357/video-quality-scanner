"""FFprobe metadata."""

from __future__ import annotations

import json
from pathlib import Path

from vqs.util import (
    display_dimensions,
    format_duration,
    parse_fraction,
    parse_sar,
    run_text,
    safe_float,
    safe_int,
    stream_rotation,
)


def ffprobe_video(path, timeout: float) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
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
