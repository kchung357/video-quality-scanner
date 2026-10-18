"""Decide whether a video is worth sending to Topaz."""

from __future__ import annotations

from vqs.config import active_thresholds
from vqs.metrics.classical import (
    classify_blockiness,
    classify_blur,
    classify_resolution,
    is_grain_dominated,
    is_high_grain,
    is_sharpish,
)
from vqs.util import INTERLACED_FIELD_ORDERS, short_side


def decide_solo(
    *,
    content_width: int,
    content_height: int,
    display_width: int,
    display_height: int,
    metrics: dict,
    nr_iqa_median,
    health_status: str,
    is_processed: bool,
    is_vr: bool,
    field_order: str = "",
    thresholds=None,
) -> dict:
    """Return decision, confidence, reason, classes, and flags."""
    limits = active_thresholds(thresholds)
    side = short_side(content_width, content_height, content_width, content_height)
    if side <= 0:
        side = short_side(display_width, display_height, display_width, display_height)

    avg_blur = metrics.get("avg_blur")
    avg_block = metrics.get("avg_block")
    blur_scale_ratio = metrics.get("blur_scale_ratio")
    median_noise = metrics.get("median_noise")
    avg_halo = metrics.get("avg_halo")
    combing = metrics.get("combing")
    blur_cv = metrics.get("blur_cv")
    mostly_dark = bool(metrics.get("mostly_dark"))
    frames_used = int(metrics.get("frames_used") or 0)

    resolution_class = classify_resolution(side, vr=is_vr)
    blur_class = classify_blur(avg_blur, blur_scale_ratio, limits)
    block_class = classify_blockiness(avg_block, limits)

    flags: list[str] = []
    if health_status == "failed":
        return _result(
            "repair_or_remux",
            0.95,
            "decode health failed; fix file before Topaz",
            resolution_class,
            blur_class,
            block_class,
            ["health_failed"],
        )
    if health_status == "errors":
        flags.append("health_errors")

    if mostly_dark or frames_used <= 0:
        return _result(
            "manual_review",
            0.4,
            "samples mostly dark/flat; cannot judge Topaz worth",
            resolution_class,
            blur_class,
            block_class,
            flags + ["mostly_dark"],
        )

    if is_processed:
        return _result(
            "already_processed",
            0.7,
            "filename/encoder looks already processed; use pair mode to judge keep/delete",
            resolution_class,
            blur_class,
            block_class,
            flags + ["already_processed"],
        )

    if is_vr:
        return _result(
            "manual_review",
            0.5,
            "equirectangular VR; review manually (Topaz path differs)",
            resolution_class,
            blur_class,
            block_class,
            flags + ["vr"],
        )

    grainy = is_grain_dominated(avg_blur, blur_scale_ratio, limits) or is_high_grain(
        median_noise, limits
    )
    if grainy:
        flags.append("grain_dominated")

    soft = avg_blur is not None and avg_blur < limits.blur_soft
    very_soft = avg_blur is not None and avg_blur < limits.blur_blurry
    blocky = avg_block is not None and avg_block >= limits.topaz_block_min
    slightly_blocky = avg_block is not None and avg_block >= limits.block_slight
    sharp = is_sharpish(avg_blur, blur_scale_ratio, median_noise, limits)

    if avg_halo is not None and avg_halo >= limits.halo_flag and sharp and not grainy:
        flags.append("possible_oversharpen")
    if combing is not None and combing >= limits.combing_min:
        flags.append("combing")
    if str(field_order or "").lower() in INTERLACED_FIELD_ORDERS:
        flags.append("interlaced_field_order")
    if (
        blur_cv is not None
        and blur_cv >= limits.unstable_blur_cv
        and frames_used >= limits.unstable_min_frames
    ):
        flags.append("unstable_blur")

    nr = None if nr_iqa_median is None else float(nr_iqa_median)
    poor_nr = nr is not None and nr < limits.topaz_nr_iqa_max
    ok_nr = nr is not None and nr >= limits.topaz_nr_iqa_ok
    if nr is not None:
        flags.append(f"nr_iqa={nr:.3f}")

    # Room to improve: SD/720 always; HD only when actually soft (not
    # merely "blocky" from hard graphics / test patterns on a sharp frame).
    room = side <= limits.topaz_max_short_side
    soft_hd = (
        side > 720
        and side <= 1080
        and avg_blur is not None
        and avg_blur < limits.topaz_hd_blur_max
    )
    has_room = room and (side <= 720 or soft_hd or soft or very_soft)

    if "combing" in flags or "interlaced_field_order" in flags:
        return _result(
            "manual_review",
            0.55,
            "interlaced/combed; deinterlace before judging Topaz",
            resolution_class,
            blur_class,
            block_class,
            flags,
        )

    if "unstable_blur" in flags:
        return _result(
            "manual_review",
            0.5,
            "sharpness varies a lot across samples",
            resolution_class,
            blur_class,
            block_class,
            flags,
        )

    if grainy and not blocky and not very_soft:
        return _result(
            "skip_or_ok",
            0.65,
            "grain looks like sharpness; denoise first or skip Topaz",
            resolution_class,
            blur_class,
            block_class,
            flags,
        )

    if sharp and side > 720 and (ok_nr or nr is None):
        return _result(
            "skip_or_ok",
            0.75 if ok_nr else 0.6,
            "already sharp; Topaz unlikely to help"
            + (" (blocking alone is not enough on HD)" if blocky or slightly_blocky else ""),
            resolution_class,
            blur_class,
            block_class,
            flags,
        )

    if not has_room and not soft and not very_soft:
        return _result(
            "skip_or_ok",
            0.7,
            "no resolution/softness headroom for Topaz",
            resolution_class,
            blur_class,
            block_class,
            flags,
        )

    # Recoverable degradation evidence
    # Low-res: softness or blocking. HD: softness required (blockiness alone is weak).
    if side <= 720:
        recoverable = soft or blocky or slightly_blocky or very_soft
    else:
        recoverable = soft or very_soft or soft_hd
    if not recoverable:
        return _result(
            "skip_or_ok",
            0.55,
            "no clear recoverable softness/blocking for this resolution",
            resolution_class,
            blur_class,
            block_class,
            flags,
        )

    # Prefer NR-IQA confirmation when available
    confidence = 0.5
    reasons = []
    if very_soft:
        confidence += 0.2
        reasons.append("very soft")
    elif soft:
        confidence += 0.12
        reasons.append("soft")
    if blocky:
        confidence += 0.18
        reasons.append("blocky")
    elif slightly_blocky:
        confidence += 0.08
        reasons.append("slightly blocky")
    if side <= 480:
        confidence += 0.1
        reasons.append("low resolution")
    elif side <= 720:
        confidence += 0.05
        reasons.append("720p-class")

    if poor_nr:
        confidence += 0.2
        reasons.append("poor NR-IQA")
    elif nr is None:
        reasons.append("classical-only (no GPU IQA)")
        confidence -= 0.05
    elif ok_nr and not very_soft and not blocky:
        return _result(
            "skip_or_ok",
            0.6,
            "NR-IQA looks acceptable despite mild classical defects",
            resolution_class,
            blur_class,
            block_class,
            flags,
        )
    elif nr is not None and not poor_nr:
        confidence -= 0.05
        reasons.append("moderate NR-IQA")

    confidence = max(0.0, min(0.99, confidence))
    if confidence < limits.topaz_min_confidence:
        return _result(
            "manual_review",
            confidence,
            "borderline Topaz case: " + ", ".join(reasons),
            resolution_class,
            blur_class,
            block_class,
            flags,
        )

    return _result(
        "topaz_candidate",
        confidence,
        "worth Topaz: " + ", ".join(reasons),
        resolution_class,
        blur_class,
        block_class,
        flags + ["topaz_candidate"],
    )


def _result(decision, confidence, reason, resolution_class, blur_class, block_class, flags):
    return {
        "decision": decision,
        "confidence": round(float(confidence), 3),
        "reason": reason,
        "resolution_class": resolution_class,
        "blur_class": blur_class,
        "blockiness_class": block_class,
        "issue_flags": ",".join(flags),
    }
