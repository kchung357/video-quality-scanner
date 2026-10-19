"""Per-video solo analysis pipeline."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vqs import SCHEMA_VERSION
from vqs.config import ScanConfig, resolve_device
from vqs.decide.solo import decide_solo
from vqs.health import check_video_health, health_row_fields
from vqs.metrics.classical import (
    ANALYSIS_FRAME_MAX_WIDTH,
    aggregate_metrics,
    center_window,
    consensus_letterbox,
    geometry_plan,
    is_vr_equirect,
    mark_duplicate_frames,
    measure_frame,
)
from vqs.metrics.nr_iqa import get_scorer
from vqs.probe import ffprobe_video
from vqs.sample import extract_sample_frames, ffmpeg_thread_count
from vqs.util import (
    clip_error,
    display_dimensions,
    file_identity,
    looks_processed,
    parse_sar,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def content_crops(measured, frames, crop) -> list:
    """BGR content crops in the same order NR-IQA scores them."""
    x, y, w, h = crop
    crops = []
    for item, (_ts, frame) in zip(measured, frames):
        if item.get("exclude"):
            continue
        crops.append(frame[y : y + h, x : x + w].copy())
    return crops


def spill_frames(frames, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dest,
        timestamps=np.asarray([ts for ts, _frame in frames], dtype=np.float64),
        frames=np.stack([frame for _ts, frame in frames], axis=0),
    )
    saved = dest if dest.suffix == ".npz" else Path(str(dest) + ".npz")
    return str(saved)


def load_spilled_frames(path) -> list:
    with np.load(path) as data:
        timestamps = data["timestamps"]
        frames = data["frames"]
        loaded = []
        for index in range(len(timestamps)):
            loaded.append((float(timestamps[index]), np.array(frames[index], copy=True)))
        return loaded


def _spill_path(config: ScanConfig, video_path: str) -> Path:
    digest = hashlib.sha1(str(video_path).encode("utf-8", "surrogateescape")).hexdigest()[:16]
    return Path(config.report_dir) / "sheets" / "_spill" / f"{digest}.npz"


def score_and_decide(row, config: ScanConfig, nr_scorer, content_frames, metrics) -> dict:
    scorer = nr_scorer
    if scorer is None:
        device = resolve_device(config.device)
        scorer = get_scorer(config.nr_iqa_model, device, config.nr_iqa_batch)
    nr = scorer.score_bgr_frames(content_frames[: min(12, len(content_frames))])
    row["nr_iqa_mean"] = nr.mean
    row["nr_iqa_median"] = nr.median
    row["nr_iqa_model"] = nr.model if nr.available else ""
    row["nr_iqa_device"] = nr.device if nr.available else "unavailable"
    if not nr.available and nr.message:
        row["issue_flags"] = (row.get("issue_flags") or "") + (
            ("," if row.get("issue_flags") else "") + "nr_iqa_unavailable"
        )
    decision = decide_solo(
        content_width=int(row.get("content_width") or 0),
        content_height=int(row.get("content_height") or 0),
        display_width=int(row.get("display_width") or 0),
        display_height=int(row.get("display_height") or 0),
        metrics=metrics,
        nr_iqa_median=nr.median,
        health_status=str(row.get("health_status") or "unknown"),
        is_processed=bool(row.get("is_probably_processed")),
        is_vr=bool(row.get("is_vr")),
        field_order=row.get("field_order") or "",
        thresholds=config.thresholds,
    )
    row.update(decision)
    return row


def finish_nr_and_decision(row, config: ScanConfig, nr_scorer) -> dict:
    """Score crops prepared by a CPU worker and apply the solo decision."""
    if row.get("decision"):
        row.pop("_content_frames", None)
        row.pop("_metrics", None)
        return row
    content_frames = row.pop("_content_frames", None) or []
    metrics = row.pop("_metrics", None) or {}
    return score_and_decide(row, config, nr_scorer, content_frames, metrics)


def _attach_review_frames(row, frames, crop, config: ScanConfig, spill: bool) -> None:
    row["_crop"] = crop
    if not frames or not config.create_contact_sheets:
        return
    if spill:
        row["_frames_path"] = spill_frames(frames, _spill_path(config, row["path"]))
        return
    row["_frames"] = frames


def analyze_video(
    video_path,
    config: ScanConfig,
    *,
    do_health: bool = True,
    existing: dict | None = None,
    nr_scorer=None,
    defer_nr: bool = False,
    spill_frames_to_disk: bool = False,
) -> dict:
    path = Path(video_path)
    row = {
        "path": str(path),
        "filename": path.name,
        "folder": str(path.parent),
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now(),
        "error": "",
        "sample_error": "",
        "contact_sheet": "",
        "issue_flags": "",
        "decision": "",
        "confidence": None,
        "reason": "",
    }
    try:
        size, mtime = file_identity(path)
    except OSError as exc:
        row.update(
            {
                "decision": "fix_path_or_missing_file",
                "confidence": 1.0,
                "reason": "file missing",
                "error": clip_error(exc),
                "health_status": "failed",
            }
        )
        return row

    row["size"] = size
    row["mtime"] = mtime

    try:
        meta = ffprobe_video(path, config.probe_timeout)
    except Exception as exc:  # noqa: BLE001
        row.update(
            {
                "decision": "manual_check_error",
                "confidence": 0.9,
                "reason": "ffprobe failed",
                "error": clip_error(exc),
                "health_status": "failed",
            }
        )
        return row

    row.update(meta)
    processed = looks_processed(path, meta.get("encoder") or "")
    row["is_probably_processed"] = processed
    vr = is_vr_equirect(meta["width"], meta["height"], config.thresholds)
    row["is_vr"] = vr

    if do_health:
        health = check_video_health(
            path,
            meta.get("duration") or 0.0,
            config.health_timeout,
            threads=ffmpeg_thread_count(config),
        )
        row.update(health_row_fields(health))
    elif existing:
        for key in (
            "health_status",
            "health_error_count",
            "health_first_error",
            "health_progress",
        ):
            if key in existing:
                row[key] = existing.get(key)
    else:
        row["health_status"] = "unknown"

    if config.skip_already_processed and processed:
        row.update(
            {
                "decision": "already_processed",
                "confidence": 0.8,
                "reason": "skipped by --skip-processed",
            }
        )
        return row

    if row.get("health_status") == "failed":
        decision = decide_solo(
            content_width=0,
            content_height=0,
            display_width=meta.get("display_width") or 0,
            display_height=meta.get("display_height") or 0,
            metrics={"frames_used": 0, "mostly_dark": True},
            nr_iqa_median=None,
            health_status="failed",
            is_processed=processed,
            is_vr=vr,
            field_order=meta.get("field_order") or "",
            thresholds=config.thresholds,
        )
        row.update(decision)
        return row

    frames, errors = extract_sample_frames(path, meta, config)
    row["sampled_frames"] = len(frames)
    row["sample_errors"] = len(errors)
    row["sample_error"] = clip_error(errors[0]) if errors else ""
    if not frames:
        row.update(
            {
                "decision": "manual_check_could_not_sample",
                "confidence": 0.9,
                "reason": "no frames decoded",
                "error": row["sample_error"] or "could not sample",
            }
        )
        return row

    only_frames = [frame for _ts, frame in frames]
    crop = consensus_letterbox(
        only_frames,
        dark_mean=config.thresholds.dark_mean_brightness,
    )
    block_crop = None
    if vr:
        block_crop = center_window(meta["width"], meta["height"], 0.5)

    measured = [
        measure_frame(
            frame,
            max_width=config.analysis_frame_max_width or ANALYSIS_FRAME_MAX_WIDTH,
            crop=crop,
            block_crop=block_crop,
            thresholds=config.thresholds,
        )
        for _ts, frame in frames
    ]
    mark_duplicate_frames(measured, frames, crop=crop)
    metrics = aggregate_metrics(measured)

    sar = parse_sar(meta.get("sample_aspect_ratio"))
    content_w, content_h = display_dimensions(
        crop[2], crop[3], sar, meta.get("rotation") or 0.0
    )
    row["crop"] = f"{crop[0]},{crop[1]},{crop[2]},{crop[3]}"
    row["content_width"] = content_w
    row["content_height"] = content_h
    row["content_aspect"] = (content_w / content_h) if content_h else None
    row["fix_geometry"] = geometry_plan(
        meta["width"],
        meta["height"],
        crop,
        sar=sar,
        rotation=meta.get("rotation") or 0.0,
        vr=vr,
        thresholds=config.thresholds,
    )

    for key in (
        "avg_blur",
        "median_blur",
        "blur_p10",
        "blur_p90",
        "blur_iqr",
        "blur_cv",
        "blur_scale_ratio",
        "blur_best_tiles",
        "avg_block",
        "avg_noise",
        "median_noise",
        "avg_halo",
        "combing",
        "avg_brightness",
        "avg_black",
        "frames_used",
        "frames_excluded",
        "mostly_dark",
    ):
        row[key] = metrics.get(key)

    content_frames = content_crops(measured, frames, crop)
    if defer_nr:
        # CPU worker stops before any model load. The parent scores these crops.
        row["_content_frames"] = content_frames
        row["_metrics"] = metrics
        _attach_review_frames(row, frames, crop, config, spill_frames_to_disk)
        return row

    score_and_decide(row, config, nr_scorer, content_frames, metrics)
    _attach_review_frames(row, frames, crop, config, spill_frames_to_disk)
    return row


def analyze_one(video_path, config: ScanConfig) -> dict:
    return analyze_video(video_path, config)
