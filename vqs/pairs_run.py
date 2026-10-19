"""Run pair comparisons against the library store."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

from tqdm import tqdm

from vqs.config import ScanConfig, resolve_device
from vqs.decide.pairs import (
    aspects_mismatch,
    best_version_path,
    pair_verdict,
    plan_pairs,
)
from vqs.metrics.nr_iqa import get_scorer
from vqs.metrics.pair_perceptual import (
    LpipsScorer,
    align_ssim,
    apply_pair_model_scores,
    evaluate_pair_frames,
    parse_crop,
    summarize_pair_metrics,
)
from vqs.sample import decode_frame, ffmpeg_thread_count, resolved_sample_count, sample_timestamps
from vqs.store import Store
from vqs.util import (
    clip_error,
    group_key,
    parse_sar,
    path_seed,
    safe_float,
    safe_int,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_meta(row):
    return {
        "width": safe_int(row.get("width")),
        "height": safe_int(row.get("height")),
        "duration": safe_float(row.get("duration")),
    }


def _pair_decode_threads(config: ScanConfig) -> int:
    threads = ffmpeg_thread_count(config)
    # Single-process pairs historically pinned FFmpeg to one thread.
    return threads if threads > 0 else 1


def compare_pair(
    original: dict,
    other: dict,
    config: ScanConfig,
    lpips=None,
    nr=None,
    defer_models: bool = False,
) -> dict:
    limits = config.thresholds
    folder = str(original.get("folder") or "")
    gid = group_key(folder, original.get("duration"))
    base = {
        "group_id": gid,
        "folder": folder,
        "original_path": original.get("path"),
        "other_path": other.get("path"),
        "original_filename": original.get("filename"),
        "other_filename": other.get("filename"),
        "duration_delta": abs(
            safe_float(original.get("duration")) - safe_float(other.get("duration"))
        ),
        "geometry_mismatch": 0,
        "updated_at": _now(),
        "note": "",
        "contact_sheet": "",
    }

    if aspects_mismatch(original, other, limits.geometry_aspect_tolerance):
        base.update(
            {
                "geometry_mismatch": 1,
                "verdict": "geometry_mismatch",
                "action": "manual_review",
                "best_version": "",
                "confidence": 0.8,
                "note": "content aspect mismatch",
                "aligned_frames": 0,
                "dropped_frames": 0,
            }
        )
        return base

    duration = min(safe_float(original.get("duration")), safe_float(other.get("duration")))
    count = min(8, resolved_sample_count(duration, config))
    stamps = sample_timestamps(duration, count, seed=path_seed(original.get("path")))
    orig_meta = _row_meta(original)
    other_meta = _row_meta(other)
    orig_crop = parse_crop(original.get("crop"), orig_meta["width"], orig_meta["height"])
    other_crop = parse_crop(other.get("crop"), other_meta["width"], other_meta["height"])
    orig_sar = parse_sar(original.get("sample_aspect_ratio"))
    other_sar = parse_sar(other.get("sample_aspect_ratio"))
    orig_rot = safe_float(original.get("rotation"))
    other_rot = safe_float(other.get("rotation"))

    kept = []
    dropped = 0
    sheet_pair = None
    decode_threads = _pair_decode_threads(config)
    model_lpips = None if defer_models else lpips
    model_nr = None if defer_models else nr
    for stamp in stamps:
        orig_frame, orig_err = decode_frame(
            original["path"],
            stamp,
            orig_meta["width"],
            orig_meta["height"],
            config.frame_timeout,
            decode_threads,
        )
        if orig_frame is None:
            dropped += 1
            continue
        best = None
        best_score = -1.0
        for delta in (0.0, -2.0 / 30.0, 2.0 / 30.0, -4.0 / 30.0, 4.0 / 30.0):
            proc_frame, proc_err = decode_frame(
                other["path"],
                max(0.0, stamp + delta),
                other_meta["width"],
                other_meta["height"],
                config.frame_timeout,
                decode_threads,
            )
            if proc_frame is None:
                continue
            score = align_ssim(orig_frame, proc_frame)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best = proc_frame
            if score >= limits.pair_ssim_align:
                break
        if best is None or best_score < limits.pair_ssim_align:
            dropped += 1
            continue
        metrics = evaluate_pair_frames(
            orig_frame,
            best,
            orig_crop=orig_crop,
            proc_crop=other_crop,
            orig_sar=orig_sar,
            proc_sar=other_sar,
            orig_rotation=orig_rot,
            proc_rotation=other_rot,
            lpips_scorer=model_lpips,
            nr_scorer=model_nr,
        )
        if metrics is None:
            dropped += 1
            continue
        kept.append(metrics)
        if sheet_pair is None:
            sheet_pair = (metrics.get("orig_view"), metrics.get("proc_view"))

    aligned = len(kept)
    if aligned < limits.pair_min_frames or (
        aligned + dropped > 0 and dropped / float(aligned + dropped) > limits.pair_align_fail_fraction
    ):
        base.update(
            {
                "verdict": "misaligned",
                "action": "manual_review",
                "best_version": "",
                "confidence": 0.7,
                "note": "pair_misaligned",
                "aligned_frames": aligned,
                "dropped_frames": dropped,
            }
        )
        return base

    if defer_models:
        base["_kept"] = kept
        base["_sheet_pair"] = sheet_pair
        base["aligned_frames"] = aligned
        base["dropped_frames"] = dropped
        return base

    return _verdict_from_kept(base, kept, dropped, limits, original, other, sheet_pair)


def apply_deferred_pair_models(base: dict, lpips, nr, limits) -> dict:
    """Score views prepared by a CPU worker, then apply the pair verdict."""
    kept = base.pop("_kept", None)
    if not kept or base.get("verdict"):
        return base
    for metrics in kept:
        apply_pair_model_scores(metrics, lpips_scorer=lpips, nr_scorer=nr)
    sheet_pair = base.get("_sheet_pair")
    original = {"path": base.get("original_path")}
    other = {"path": base.get("other_path")}
    dropped = int(base.get("dropped_frames") or 0)
    return _verdict_from_kept(base, kept, dropped, limits, original, other, sheet_pair)


def _verdict_from_kept(base, kept, dropped, limits, original, other, sheet_pair):
    summary = summarize_pair_metrics(kept, limits)
    verdict, action, confidence, note = pair_verdict(summary, limits)
    base.update(
        {
            "blur_native": summary.get("blur_native"),
            "blur_vs_base": summary.get("blur_vs_base"),
            "block_native": summary.get("block_native"),
            "noise_native": summary.get("noise_native"),
            "halo_vs_base": summary.get("halo_vs_base"),
            "ssim": summary.get("ssim"),
            "lpips_native": summary.get("lpips_native"),
            "nr_uplift": summary.get("nr_uplift"),
            "luma_delta": summary.get("luma_delta"),
            "range_shift": int(bool(summary.get("range_shift"))),
            "aligned_frames": len(kept),
            "dropped_frames": dropped,
            "verdict": verdict,
            "action": action,
            "best_version": best_version_path(action, original, other),
            "confidence": confidence,
            "note": note,
            "_sheet_pair": sheet_pair,
        }
    )
    return base


def _pair_worker(original, other, config):
    return compare_pair(original, other, config, defer_models=True)


def _store_pair_result(store: Store, result: dict, note: str) -> dict:
    if note:
        result["note"] = (result.get("note") or "") + (f"; {note}" if result.get("note") else note)
    store_row = {k: v for k, v in result.items() if not k.startswith("_")}
    store.upsert_pair(store_row)
    return result


def _compare_failed(original, other, exc) -> dict:
    return {
        "group_id": group_key(str(original.get("folder") or ""), original.get("duration")),
        "folder": original.get("folder"),
        "original_path": original.get("path"),
        "other_path": other.get("path"),
        "original_filename": original.get("filename"),
        "other_filename": other.get("filename"),
        "duration_delta": abs(
            safe_float(original.get("duration")) - safe_float(other.get("duration"))
        ),
        "geometry_mismatch": 0,
        "verdict": "compare_failed",
        "action": "manual_review",
        "best_version": "",
        "confidence": 0.3,
        "note": clip_error(exc),
        "aligned_frames": 0,
        "dropped_frames": 0,
        "updated_at": _now(),
        "contact_sheet": "",
    }


def run_pairs(store: Store, config: ScanConfig) -> list[dict]:
    if config.workers <= 0:
        config.workers = max(1, min(8, os.cpu_count() or 1))
    rows = store.list_videos()
    plans = plan_pairs(rows, config.thresholds.pair_duration_seconds)
    comparable = [plan for plan in plans if plan[0] is not None]
    worker_count = min(config.workers, len(comparable)) if comparable else 1
    use_pool = worker_count > 1
    if use_pool:
        config.ffmpeg_threads = 2
    device = resolve_device(config.device)
    lpips = LpipsScorer(device=device)
    nr = get_scorer(config.nr_iqa_model, device, config.nr_iqa_batch)
    results = []
    store.clear_pairs()

    def finish_compare(original, other, note, result):
        if "_kept" in result:
            result = apply_deferred_pair_models(result, lpips, nr, config.thresholds)
        return _store_pair_result(store, result, note)

    ambiguous = [plan for plan in plans if plan[0] is None]
    jobs = [plan for plan in plans if plan[0] is not None]

    for original, other, note in ambiguous:
        for item in other:
            result = {
                "group_id": group_key(str(item.get("folder") or ""), item.get("duration")),
                "folder": item.get("folder"),
                "original_path": item.get("path"),
                "other_path": "",
                "original_filename": item.get("filename"),
                "other_filename": "",
                "duration_delta": 0.0,
                "geometry_mismatch": 0,
                "verdict": "ambiguous_group",
                "action": "manual_review",
                "best_version": "",
                "confidence": 0.5,
                "note": note or "ambiguous_group",
                "aligned_frames": 0,
                "dropped_frames": 0,
                "updated_at": _now(),
                "contact_sheet": "",
            }
            store.upsert_pair(result)
            results.append(result)

    if use_pool:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as pool:
            futures = {
                pool.submit(_pair_worker, original, other, config): (original, other, note)
                for original, other, note in jobs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="pairs", unit="pair"):
                original, other, note = futures[future]
                try:
                    result = finish_compare(original, other, note, future.result())
                except Exception as exc:  # noqa: BLE001
                    result = _compare_failed(original, other, exc)
                    store.upsert_pair(result)
                results.append(result)
        return results

    for original, other, note in tqdm(jobs, desc="pairs", unit="pair"):
        try:
            result = compare_pair(original, other, config, lpips=lpips, nr=nr)
            result = _store_pair_result(store, result, note)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            result = _compare_failed(original, other, exc)
            store.upsert_pair(result)
            results.append(result)
    return results
