"""Synthetic calibration checks and labeled-set precision/recall."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from vqs.decide.pairs import pair_verdict
from vqs.decide.solo import decide_solo
from vqs.metrics.classical import (
    blockiness_scaled,
    blur_profile,
    consensus_letterbox,
    laplacian_var,
    to_gray,
)
from vqs.metrics.pair_perceptual import summarize_pair_metrics
from vqs.store import Store


def _natural_frame(height, width, seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, size=(height, width), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    pattern = ((xx // 16 + yy // 16) % 2) * 40
    gray = np.clip(base.astype(np.int16) + pattern, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def calibration_checks() -> list[tuple[str, bool, str]]:
    results = []
    frame = _natural_frame(240, 320, 1)
    sharp, _ = blur_profile(frame)
    soft, _ = blur_profile(cv2.GaussianBlur(frame, (0, 0), 2.5))
    ok = soft < sharp
    results.append(("blur_falls_with_gaussian", ok, f"sharp={sharp:.1f} soft={soft:.1f}"))

    # Bicubic upscale then down loses some energy; a further blur must lose more.
    up = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
    down = cv2.resize(to_gray(up), (320, 240), interpolation=cv2.INTER_AREA)
    base_lap = laplacian_var(down)
    ratio = base_lap / max(laplacian_var(to_gray(frame)), 1e-6)
    results.append(("bicubic_roundtrip_finite", 0.05 < ratio < 2.0, f"ratio={ratio:.3f}"))

    blurred_up = cv2.GaussianBlur(up, (0, 0), 1.8)
    soft_down = cv2.resize(to_gray(blurred_up), (320, 240), interpolation=cv2.INTER_AREA)
    soft_ratio = laplacian_var(soft_down) / max(base_lap, 1e-6)
    results.append(("blurred_upscale_softer", soft_ratio < 0.95, f"ratio={soft_ratio:.3f}"))

    # 1.5x block grid visible at scaled period
    block = np.zeros((160, 160), np.uint8)
    for y in range(0, 160, 8):
        for x in range(0, 160, 8):
            block[y : y + 8, x : x + 8] = 40 if ((x // 8 + y // 8) % 2) else 200
    block_bgr = cv2.cvtColor(block, cv2.COLOR_GRAY2BGR)
    scaled = cv2.resize(block_bgr, (240, 240), interpolation=cv2.INTER_NEAREST)
    score = blockiness_scaled(scaled, 1.5, 1.5)
    results.append(("scaled_block_period", score > 1.2, f"block={score:.3f}"))

    # Letterbox consensus
    letter = np.zeros((240, 320, 3), np.uint8)
    letter[30:210, :, :] = frame[30:210, :, :]
    crop = consensus_letterbox([letter, letter, letter])
    results.append(("letterbox_crop", crop[1] >= 20 and crop[3] <= 200, f"crop={crop}"))

    # Solo decision sanity: soft low-res should be candidate (classical-only)
    soft_metrics = {
        "avg_blur": 40.0,
        "avg_block": 1.55,
        "blur_scale_ratio": 1.2,
        "median_noise": 1.0,
        "avg_halo": 1.0,
        "combing": 0.0,
        "blur_cv": 0.2,
        "mostly_dark": False,
        "frames_used": 10,
    }
    decision = decide_solo(
        content_width=640,
        content_height=360,
        display_width=640,
        display_height=360,
        metrics=soft_metrics,
        nr_iqa_median=0.30,
        health_status="ok",
        is_processed=False,
        is_vr=False,
    )
    results.append(
        (
            "soft_sd_is_candidate",
            decision["decision"] == "topaz_candidate",
            decision["decision"] + ": " + decision["reason"],
        )
    )

    sharp_metrics = dict(soft_metrics)
    sharp_metrics["avg_blur"] = 350.0
    sharp_metrics["avg_block"] = 1.1
    sharp_decision = decide_solo(
        content_width=1920,
        content_height=1080,
        display_width=1920,
        display_height=1080,
        metrics=sharp_metrics,
        nr_iqa_median=0.75,
        health_status="ok",
        is_processed=False,
        is_vr=False,
    )
    results.append(
        (
            "sharp_hd_skipped",
            sharp_decision["decision"] == "skip_or_ok",
            sharp_decision["decision"] + ": " + sharp_decision["reason"],
        )
    )

    # Pair: improved when blur_vs_base high and no damage
    verdict, action, _conf, _note = pair_verdict(
        {
            "blur_native": 1.05,
            "blur_vs_base": 1.30,
            "block_native": 0.0,
            "noise_native": 1.0,
            "halo_vs_base": 1.0,
            "halo_delta": 0.0,
            "lpips_native": 0.1,
            "nr_uplift": 0.05,
        }
    )
    results.append(("pair_improved", verdict == "improved" and action == "keep_processed", verdict))

    return results


def run_calibration(quick: bool = True) -> int:
    checks = calibration_checks()
    failed = 0
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        if not ok:
            failed += 1
    if failed:
        print(f"{failed} calibration check(s) failed")
        return 1
    print("All calibration checks passed")
    return 0


def evaluate_labels(db_path: str, labels_csv: str) -> int:
    """Compare store decisions against a labels.csv.

    Expected columns:
      path,label
    where label is one of:
      worth_topaz, not_worth, keep_processed, keep_original, manual_review, repair
    """
    store = Store(db_path)
    videos = {row["path"]: row for row in store.list_videos()}
    pairs = store.list_pairs()
    pair_by_other = {row.get("other_path"): row for row in pairs if row.get("other_path")}
    pair_by_orig = {row.get("original_path"): row for row in pairs if row.get("original_path")}

    with Path(labels_csv).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        print("No labels found")
        return 1

    tp = fp = fn = tn = 0
    pair_ok = pair_n = 0
    unknown = 0
    details = []

    for item in rows:
        path = (item.get("path") or "").strip()
        label = (item.get("label") or "").strip().lower()
        if not path or not label:
            continue

        if label in {"worth_topaz", "not_worth", "repair"}:
            row = videos.get(path)
            if not row:
                unknown += 1
                details.append((path, label, "missing_in_db"))
                continue
            pred = row.get("decision")
            if label == "worth_topaz":
                hit = pred == "topaz_candidate"
                if hit:
                    tp += 1
                else:
                    fn += 1
            elif label == "not_worth":
                hit = pred in {"skip_or_ok", "already_processed"}
                if hit:
                    tn += 1
                else:
                    fp += 1
            else:  # repair
                hit = pred == "repair_or_remux" or row.get("health_status") in {"failed", "errors"}
                if hit:
                    tp += 1
                else:
                    fn += 1
            details.append((path, label, pred if hit else f"pred={pred}"))
        elif label in {"keep_processed", "keep_original", "manual_review"}:
            pair = pair_by_other.get(path) or pair_by_orig.get(path)
            if not pair:
                unknown += 1
                details.append((path, label, "missing_pair"))
                continue
            pair_n += 1
            action = pair.get("action")
            hit = action == label or (
                label == "manual_review" and action == "manual_review"
            )
            if hit:
                pair_ok += 1
            details.append((path, label, action if hit else f"pred={action}"))
        else:
            unknown += 1
            details.append((path, label, "unknown_label"))

    print("Solo upscale detection (worth_topaz / not_worth / repair):")
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    print(f"  tp={tp} fp={fp} fn={fn} tn={tn}")
    if precision is not None:
        print(f"  precision={precision:.3f}")
    if recall is not None:
        print(f"  recall={recall:.3f}")
    if pair_n:
        print(f"Pair actions: {pair_ok}/{pair_n} exact matches ({pair_ok/pair_n:.3f})")
    if unknown:
        print(f"Unmatched/unknown labels: {unknown}")

    print("\nSuggested threshold tweaks:")
    if precision is not None and precision < 0.7 and (tp + fp) >= 5:
        print("  - Raise topaz_nr_iqa_max or topaz_block_min / lower blur softness (fewer false candidates)")
    if recall is not None and recall < 0.7 and (tp + fn) >= 5:
        print("  - Lower topaz_nr_iqa_max or topaz_min_confidence (catch more soft/blocky files)")
    if pair_n and pair_ok / pair_n < 0.7:
        print("  - Adjust pair_blur_vs_base / pair_lpips_max / pair_nr_uplift_min against labeled pairs")
    if precision is None and recall is None and pair_n == 0:
        print("  - Need more overlapping labels in the scanned library")

    for path, label, note in details[:30]:
        print(f"  {label:16} {note:24} {path}")
    return 0
