"""Per-video solo analysis pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from vqs import SCHEMA_VERSION
from vqs.config import ScanConfig, resolve_device
from vqs.decide.solo import decide_solo
from vqs.health import check_video_health, health_row_fields
from vqs.metrics.classical import (
    ANALYSIS_FRAME_MAX_WIDTH,
    aggregate_metrics,
    apply_crop,
    center_window,
    consensus_letterbox,
    geometry_plan,
    is_vr_equirect,
    mark_duplicate_frames,
    measure_frame,
    to_gray,
)
from vqs.metrics.nr_iqa import get_scorer
from vqs.probe import ffprobe_video
from vqs.sample import extract_sample_frames
from vqs.util import (
    clip_error,
    display_dimensions,
    file_identity,
    looks_processed,
    parse_sar,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def analyze_video(
    video_path,
    config: ScanConfig,
    *,
    do_health: bool = True,
    existing: dict | None = None,
    nr_scorer=None,
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
        health = check_video_health(path, meta.get("duration") or 0.0, config.health_timeout)
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

    # NR-IQA on non-dark content crops
    device = resolve_device(config.device)
    scorer = nr_scorer or get_scorer(config.nr_iqa_model, device, config.nr_iqa_batch)
    content_frames = []
    for item, (_ts, frame) in zip(measured, frames):
        if item.get("exclude"):
            continue
        gray = apply_crop(to_gray(frame), crop)
        # Rebuild BGR crop from full frame
        x, y, w, h = crop
        content_frames.append(frame[y : y + h, x : x + w].copy())
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
        content_width=content_w,
        content_height=content_h,
        display_width=meta.get("display_width") or 0,
        display_height=meta.get("display_height") or 0,
        metrics=metrics,
        nr_iqa_median=nr.median,
        health_status=str(row.get("health_status") or "unknown"),
        is_processed=processed,
        is_vr=vr,
        field_order=meta.get("field_order") or "",
        thresholds=config.thresholds,
    )
    row.update(decision)

    # Attach frames for optional contact sheet generation by caller
    row["_frames"] = frames
    row["_crop"] = crop
    return row


def analyze_one(video_path, config: ScanConfig) -> dict:
    return analyze_video(video_path, config)
