"""Pair planning and keep/delete verdicts."""

from __future__ import annotations

import math
from pathlib import Path

from vqs.config import active_thresholds
from vqs.util import as_bool, parse_sar, safe_float, safe_int, stems_related


def _union_find(size):
    parent = list(range(size))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        ra, rb = find(left), find(right)
        if ra != rb:
            parent[rb] = ra

    return find, union


def cluster_by_duration(rows, tolerance):
    find, union = _union_find(len(rows))
    durations = [safe_float(row.get("duration")) for row in rows]
    for i in range(len(rows)):
        if durations[i] <= 0:
            continue
        for j in range(i + 1, len(rows)):
            if durations[j] > 0 and abs(durations[i] - durations[j]) <= tolerance:
                union(i, j)
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(index), []).append(row)
    return list(groups.values())


def cluster_by_stem(rows):
    find, union = _union_find(len(rows))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if stems_related(rows[i].get("filename") or "", rows[j].get("filename") or ""):
                union(i, j)
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(index), []).append(row)
    return list(groups.values())


def row_ready(row) -> bool:
    if str(row.get("error") or "").strip():
        return False
    if str(row.get("health_status") or "") == "failed":
        return False
    decision = str(row.get("decision") or "")
    return decision not in {"", "missing"}


def display_aspect(row):
    width = safe_float(row.get("display_width"))
    height = safe_float(row.get("display_height"))
    if width <= 0 or height <= 0:
        return None
    return width / height


def row_content_aspect(row):
    width = safe_float(row.get("content_width"))
    height = safe_float(row.get("content_height"))
    if width > 0 and height > 0:
        return width / height
    return display_aspect(row)


def aspects_mismatch(left, right, tolerance=0.05):
    aspect_left = row_content_aspect(left)
    aspect_right = row_content_aspect(right)
    if aspect_left is None or aspect_right is None:
        return False
    return abs(aspect_left - aspect_right) / max(aspect_left, aspect_right) > tolerance


def choose_reference(left, right):
    left_distance = abs(math.log(parse_sar(left.get("sample_aspect_ratio"))))
    right_distance = abs(math.log(parse_sar(right.get("sample_aspect_ratio"))))
    if abs(left_distance - right_distance) > 0.02:
        return (left, right) if left_distance < right_distance else (right, left)
    if safe_int(left.get("size")) >= safe_int(right.get("size")):
        return left, right
    return right, left


def plan_pairs(rows, tolerance):
    by_folder = {}
    for row in rows:
        if not row_ready(row):
            continue
        by_folder.setdefault(str(row.get("folder") or ""), []).append(row)
    plans = []
    for _folder, items in by_folder.items():
        for duration_group in cluster_by_duration(items, tolerance):
            for component in cluster_by_stem(duration_group):
                if len(component) < 2:
                    continue
                processed = [row for row in component if as_bool(row.get("is_probably_processed"))]
                originals = [
                    row for row in component if not as_bool(row.get("is_probably_processed"))
                ]
                if len(originals) == 1 and processed:
                    for other in processed:
                        plans.append((originals[0], other, ""))
                    continue
                matched = False
                if processed and originals:
                    for other in processed:
                        matches = [
                            original
                            for original in originals
                            if stems_related(
                                original.get("filename") or "", other.get("filename") or ""
                            )
                        ]
                        if len(matches) == 1:
                            plans.append((matches[0], other, ""))
                            matched = True
                    if matched:
                        continue
                if len(component) == 2 and not processed:
                    reference, other = choose_reference(component[0], component[1])
                    plans.append((reference, other, "no_processed_marker"))
                    continue
                plans.append((None, component, "ambiguous_group"))
    return plans


def pair_verdict(deltas, thresholds=None) -> tuple[str, str, float, str]:
    """Return (verdict, action, confidence, note)."""
    limits = active_thresholds(thresholds)
    if not deltas:
        return "compare_failed", "manual_review", 0.3, "no metrics"
    if deltas.get("geometry_mismatch"):
        return "geometry_mismatch", "manual_review", 0.8, "content aspect changed"
    if deltas.get("pair_misaligned"):
        return "misaligned", "manual_review", 0.7, "could not align enough frames"

    flags = []
    blur_native = deltas.get("blur_native")
    blur_vs_base = deltas.get("blur_vs_base")
    block_native = deltas.get("block_native")
    noise_native = deltas.get("noise_native")
    halo_vs_base = deltas.get("halo_vs_base")
    halo_delta = deltas.get("halo_delta")
    lpips_native = deltas.get("lpips_native")
    nr_uplift = deltas.get("nr_uplift")

    lost_detail = blur_native is not None and blur_native < limits.pair_blur_detail_ratio
    severe_soft = blur_native is not None and blur_native < limits.pair_blur_severe_ratio
    if lost_detail:
        # Softness with similar noise drop can be denoise, not damage.
        if (
            noise_native is not None
            and noise_native <= limits.pair_noise_drop_ratio
            and not severe_soft
        ):
            flags.append("denoise_softness")
        else:
            flags.append("lost_detail")
    if block_native is not None and block_native >= limits.pair_block_delta:
        flags.append("more_blocky")
    if noise_native is not None and noise_native >= limits.pair_noise_ratio:
        flags.append("more_noise")
    if (
        halo_vs_base is not None
        and halo_delta is not None
        and halo_vs_base >= limits.pair_halo_ratio
        and halo_delta >= limits.pair_halo_delta
    ):
        flags.append("oversharpened")
    if lpips_native is not None and lpips_native > limits.pair_lpips_max:
        flags.append("fidelity_loss")

    damage = [f for f in flags if f != "denoise_softness"]
    if damage:
        return (
            "regressed:" + ",".join(damage),
            "keep_original",
            0.8,
            "processed looks worse: " + ", ".join(damage),
        )

    improved_blur = blur_vs_base is not None and blur_vs_base >= limits.pair_blur_vs_base
    improved_nr = nr_uplift is not None and nr_uplift >= limits.pair_nr_uplift_min
    if improved_blur or improved_nr:
        conf = 0.7
        bits = []
        if improved_blur:
            conf += 0.1
            bits.append(f"sharper than bicubic ({blur_vs_base:.2f})")
        if improved_nr:
            conf += 0.15
            bits.append(f"NR uplift {nr_uplift:.3f}")
        if "denoise_softness" in flags:
            bits.append("denoise tradeoff")
        return "improved", "keep_processed", min(0.95, conf), "beats bicubic: " + ", ".join(bits)

    return (
        "neutral",
        "keep_original",
        0.6,
        "no clear gain over player-scale bicubic; keep original",
    )


def best_version_path(action: str, original: dict, other: dict) -> str:
    if action == "keep_processed":
        return str(other.get("path") or "")
    if action == "keep_original":
        return str(original.get("path") or "")
    return ""


def pair_basename(path: str) -> str:
    return Path(path).name
