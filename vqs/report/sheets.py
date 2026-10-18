"""Contact sheets for solo review and side-by-side pair compares."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import cv2
import numpy as np

from vqs.util import format_duration


def contact_sheet_filename(video_path) -> str:
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


def resize_keep_aspect(frame, target_width: int):
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


def create_contact_sheet(video_path, frames_with_ts, result, output_folder, thumb_width=320):
    """Write one JPEG contact sheet. Returns the output path string."""
    if not frames_with_ts:
        return ""

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
            f"decision={result.get('decision')}  "
            f"confidence={result.get('confidence')}  "
            f"health={result.get('health_status')}"
        ),
        (
            f"res={result.get('width')}x{result.get('height')}  "
            f"display={result.get('display_width')}x{result.get('display_height')}  "
            f"blur={result.get('avg_blur')}  "
            f"block={result.get('avg_block')}  "
            f"noise={result.get('avg_noise')}"
        ),
        f"reason={result.get('reason')}  flags={result.get('issue_flags')}",
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
        sheet[y1 : y1 + height, x1 : x1 + width] = thumb[:height, :width]

    out_path = output_folder / contact_sheet_filename(video_path)
    if not cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"Could not write contact sheet: {out_path}")
    return str(out_path)


def create_pair_sheet(left, right, out_path, thumb_width: int = 480):
    """Write a side-by-side JPEG for one original/processed frame pair.

    ``left`` / ``right`` are BGR frames (original and processed views).
    Returns the output path string, or empty string if frames are missing.
    """
    if left is None or right is None:
        return ""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    left_thumb = resize_keep_aspect(np.asarray(left), thumb_width)
    right_thumb = resize_keep_aspect(np.asarray(right), thumb_width)
    max_h = max(left_thumb.shape[0], right_thumb.shape[0])

    def _pad(thumb):
        height, width = thumb.shape[:2]
        if height < max_h:
            pad = np.zeros((max_h - height, width, 3), dtype=np.uint8)
            thumb = np.vstack([thumb, pad])
        if width < thumb_width:
            pad = np.zeros((thumb.shape[0], thumb_width - width, 3), dtype=np.uint8)
            thumb = np.hstack([thumb, pad])
        return thumb

    left_thumb = _pad(left_thumb)
    right_thumb = _pad(right_thumb)
    gap = 8
    header_h = 36
    sheet_w = thumb_width * 2 + gap
    sheet = np.zeros((header_h + max_h, sheet_w, 3), dtype=np.uint8)
    draw_text_box(sheet, ["original", "processed"], x=8, y=6, font_scale=0.55, thickness=1)
    sheet[header_h : header_h + max_h, 0:thumb_width] = left_thumb
    sheet[header_h : header_h + max_h, thumb_width + gap : thumb_width + gap + thumb_width] = right_thumb

    if not cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"Could not write pair sheet: {out_path}")
    return str(out_path)
