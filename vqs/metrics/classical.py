"""Classical frame metrics ported from v1 (trimmed)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from vqs.config import active_thresholds
from vqs.util import display_dimensions


ANALYSIS_FRAME_MAX_WIDTH = 640


def to_gray(frame):
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def scale_gray(gray, max_width):
    height, width = gray.shape[:2]
    if width <= max_width or width <= 0:
        return gray
    new_width = int(max_width)
    new_height = max(1, int(round(height * (new_width / float(width)))))
    return cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)


def laplacian_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def blur_profile(frame, max_width=ANALYSIS_FRAME_MAX_WIDTH):
    gray = scale_gray(to_gray(frame), max_width)
    blur = laplacian_var(gray)
    height, width = gray.shape[:2]
    if width < 32 or height < 32:
        return blur, None
    half_width = max(16, width // 2)
    half_height = max(16, int(round(height * (half_width / float(width)))))
    half = cv2.resize(gray, (half_width, half_height), interpolation=cv2.INTER_AREA)
    half_blur = laplacian_var(half)
    return blur, blur / max(half_blur, 1e-6)


def tile_sharpness(gray, tile=64):
    gray = to_gray(gray)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    height, width = lap.shape[:2]
    if height < tile or width < tile:
        return laplacian_var(gray)
    values = []
    for y in range(0, height - tile + 1, tile):
        for x in range(0, width - tile + 1, tile):
            values.append(float(lap[y : y + tile, x : x + tile].var()))
    if not values:
        return laplacian_var(gray)
    values.sort()
    keep = max(1, len(values) // 5)
    return float(np.mean(values[-keep:]))


def blockiness_score(frame):
    gray = to_gray(frame).astype(np.float32)
    height, width = gray.shape[:2]
    if height < 16 or width < 16:
        return 0.0
    diff_x = np.abs(gray[:, 1:] - gray[:, :-1])
    diff_y = np.abs(gray[1:, :] - gray[:-1, :])
    boundary_cols = np.arange(7, width - 1, 8)
    boundary_rows = np.arange(7, height - 1, 8)
    if len(boundary_cols) == 0 or len(boundary_rows) == 0:
        return 0.0
    boundary_x = diff_x[:, boundary_cols].mean()
    boundary_y = diff_y[boundary_rows, :].mean()
    non_boundary_cols = np.setdiff1d(np.arange(diff_x.shape[1]), boundary_cols)
    non_boundary_rows = np.setdiff1d(np.arange(diff_y.shape[0]), boundary_rows)
    if len(non_boundary_cols) == 0 or len(non_boundary_rows) == 0:
        return 0.0
    non_boundary_x = diff_x[:, non_boundary_cols].mean()
    non_boundary_y = diff_y[non_boundary_rows, :].mean()
    boundary_mean = (boundary_x + boundary_y) / 2.0
    non_boundary_mean = (non_boundary_x + non_boundary_y) / 2.0
    return float(boundary_mean / max(non_boundary_mean, 1e-6))


def _block_period(scale):
    if abs(float(scale) - 1.0) < 0.08:
        return 8
    return max(2, int(round(8.0 * float(scale))))


def blockiness_scaled(frame, scale_x=1.0, scale_y=1.0):
    gray = to_gray(frame).astype(np.float32)
    height, width = gray.shape[:2]
    if height < 16 or width < 16:
        return 0.0
    diff_x = np.abs(gray[:, 1:] - gray[:, :-1])
    diff_y = np.abs(gray[1:, :] - gray[:-1, :])
    period_x = _block_period(scale_x)
    period_y = _block_period(scale_y)
    cols = np.arange(period_x - 1, diff_x.shape[1], period_x)
    rows = np.arange(period_y - 1, diff_y.shape[0], period_y)
    if cols.size < 2 or rows.size < 2:
        return 0.0
    boundary = (float(diff_x[:, cols].mean()) + float(diff_y[rows, :].mean())) / 2.0
    mask_x = np.ones(diff_x.shape[1], dtype=bool)
    mask_y = np.ones(diff_y.shape[0], dtype=bool)
    mask_x[cols] = False
    mask_y[rows] = False
    if not mask_x.any() or not mask_y.any():
        return 0.0
    interior = (float(diff_x[:, mask_x].mean()) + float(diff_y[mask_y, :].mean())) / 2.0
    return float(boundary / max(interior, 1e-6))


def noise_profile(frame):
    gray = to_gray(frame)
    residual = cv2.absdiff(gray, cv2.medianBlur(gray, 3))
    return float(np.mean(residual))


def noise_sigma(gray, std_max=12.0):
    g = np.asarray(to_gray(gray), np.float32)
    if g.shape[0] < 8 or g.shape[1] < 8:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    response = np.abs(cv2.filter2D(g, cv2.CV_32F, kernel)[1:-1, 1:-1])
    box = np.ones((5, 5), np.float32) / 25.0
    mean = cv2.filter2D(g, cv2.CV_32F, box)[1:-1, 1:-1]
    mean_sq = cv2.filter2D(g * g, cv2.CV_32F, box)[1:-1, 1:-1]
    local = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    flat = local <= float(std_max)
    if int(flat.sum()) < 64:
        return 0.0
    return float(response[flat].mean() * np.sqrt(0.5 * np.pi) / 6.0)


def halo_score(gray):
    g = np.asarray(to_gray(gray), np.float32)
    if g.shape[0] < 8 or g.shape[1] < 8:
        return 0.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    threshold = max(float(np.percentile(mag, 92)), 8.0)
    ys, xs = np.nonzero(mag >= threshold)
    if ys.size < 40:
        return 0.0
    if ys.size > 2500:
        selected = np.linspace(0, ys.size - 1, 2500).astype(int)
        ys = ys[selected]
        xs = xs[selected]
    inv = 1.0 / (mag[ys, xs] + 1e-6)
    dx = gx[ys, xs] * inv
    dy = gy[ys, xs] * inv
    height, width = g.shape

    def sample(shift):
        yy = np.clip(np.rint(ys + dy * shift).astype(int), 0, height - 1)
        xx = np.clip(np.rint(xs + dx * shift).astype(int), 0, width - 1)
        return g[yy, xx]

    dark_side = sample(-2)
    before = sample(-1)
    after = sample(1)
    bright_side = sample(2)
    undershoot = np.maximum(0, dark_side - before)
    overshoot = np.maximum(0, after - bright_side)
    return float(np.mean(undershoot + overshoot))


def combing_score(gray):
    g = np.asarray(to_gray(gray), np.float32)
    if g.shape[0] < 8:
        return 0.0
    current = g[2:, :]
    adjacent = g[1:-1, :]
    same_field = g[:-2, :]
    inter = np.abs(current - adjacent)
    intra = np.abs(current - same_field)
    return float(np.mean(inter > (intra + 12.0)))


def letterbox_bounds(frame, thresh=16):
    gray = to_gray(frame)
    height, width = gray.shape[:2]
    if height < 32 or width < 32:
        return (0, 0, width, height)
    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)
    top = 0
    while top < height - 1 and row_mean[top] <= thresh:
        top += 1
    bottom = height
    while bottom > top + 1 and row_mean[bottom - 1] <= thresh:
        bottom -= 1
    left = 0
    while left < width - 1 and col_mean[left] <= thresh:
        left += 1
    right = width
    while right > left + 1 and col_mean[right - 1] <= thresh:
        right -= 1
    if top < 4:
        top = 0
    if height - bottom < 4:
        bottom = height
    if left < 4:
        left = 0
    if width - right < 4:
        right = width
    crop_h = bottom - top
    crop_w = right - left
    if crop_h < height * 0.5 or crop_w < width * 0.5 or crop_h < 16 or crop_w < 16:
        return (0, 0, width, height)
    return (left, top, crop_w, crop_h)


def _boxes_near(left, right, tol=8):
    return all(abs(int(left[index]) - int(right[index])) <= tol for index in range(4))


def _median_box(boxes, width, height):
    left = int(np.median([box[0] for box in boxes]))
    top = int(np.median([box[1] for box in boxes]))
    right = int(np.median([box[0] + box[2] for box in boxes]))
    bottom = int(np.median([box[1] + box[3] for box in boxes]))
    left = max(0, min(left, width - 16))
    top = max(0, min(top, height - 16))
    right = max(left + 16, min(right, width))
    bottom = max(top + 16, min(bottom, height))
    if (bottom - top) < height * 0.5 or (right - left) < width * 0.5:
        return (0, 0, width, height)
    return (left, top, right - left, bottom - top)


def consensus_letterbox(frames, thresh=16, dark_mean=20.0):
    if not frames:
        return (0, 0, 0, 0)
    gray = to_gray(frames[0])
    height, width = gray.shape[:2]
    full = (0, 0, width, height)
    usable = []
    for frame in frames:
        picture = to_gray(frame)
        if float(np.mean(picture)) <= float(dark_mean):
            continue
        usable.append(picture)
    if not usable:
        return full
    boxes = [letterbox_bounds(picture, thresh) for picture in usable]
    best = full
    best_count = sum(1 for box in boxes if _boxes_near(box, full))
    real = [box for box in boxes if not _boxes_near(box, full, tol=2)]
    if real:
        candidate = _median_box(real, width, height)
        count = sum(1 for box in boxes if _boxes_near(box, candidate))
        if count > best_count:
            best = candidate
            best_count = count
    if best_count * 2 < len(usable):
        return full
    return best


def even_dim(value):
    number = max(2, int(round(float(value))))
    if number % 2:
        number += 1
    return number


def area_loss_to_16x9(aspect):
    target = 16.0 / 9.0
    if aspect is None or aspect <= 0:
        return 1.0
    if aspect >= target:
        return 1.0 - (target / float(aspect))
    return 1.0 - (float(aspect) / target)


def geometry_canvas(display_w, display_h):
    short = min(display_w, display_h)
    long_side = max(display_w, display_h)
    if short >= 900 or long_side >= 1600:
        return (1920, 1080)
    if short >= 600 or long_side >= 1100:
        return (1280, 720)
    return None


def geometry_plan(frame_w, frame_h, crop, sar=1.0, rotation=0.0, vr=False, thresholds=None):
    limits = active_thresholds(thresholds)
    if vr or frame_w <= 0 or frame_h <= 0 or not crop:
        return ""
    _x, _y, crop_w, crop_h = [int(value) for value in crop]
    display_w, display_h = display_dimensions(crop_w, crop_h, float(sar or 1.0), rotation)
    if display_w <= 0 or display_h <= 0:
        return ""
    aspect = float(display_w) / float(display_h)
    target = 16.0 / 9.0
    relative = abs(aspect - target) / target
    loss = area_loss_to_16x9(aspect)
    if relative <= limits.geometry_fit_tolerance:
        mode = "fit"
    elif loss <= limits.geometry_crop_area_max:
        mode = "crop"
    else:
        mode = "pad"

    canvas = geometry_canvas(display_w, display_h)
    if canvas is None:
        if mode == "pad":
            if aspect >= target:
                target_w = even_dim(display_w)
                target_h = even_dim(display_w / target)
            else:
                target_h = even_dim(display_h)
                target_w = even_dim(display_h * target)
        elif mode == "crop":
            if aspect >= target:
                target_h = even_dim(display_h)
                target_w = even_dim(display_h * target)
            else:
                target_w = even_dim(display_w)
                target_h = even_dim(display_w / target)
        else:
            target_w, target_h = even_dim(display_w), even_dim(display_h)
    else:
        target_w, target_h = canvas

    bars = crop_w <= int(frame_w) - 4 or crop_h <= int(frame_h) - 4
    sar_off = abs(float(sar or 1.0) - 1.0) > 0.01
    parts = []
    if bars:
        parts.append("crop_bars")
    if mode == "pad" and relative > limits.geometry_fit_tolerance:
        parts.append("pad_to_16x9")
    elif mode == "crop" and relative > limits.geometry_fit_tolerance:
        parts.append("crop_to_16x9")
    elif mode == "fit" and (abs(target_w - display_w) > 4 or abs(target_h - display_h) > 4):
        parts.append("fit_16x9")
    if sar_off:
        parts.append("reset_sar")
    if not parts:
        return ""
    return "+".join(parts) + f":{target_w}x{target_h}"


def center_window(width, height, fraction=0.5):
    crop_w = max(16, int(width * fraction))
    crop_h = max(16, int(height * fraction))
    crop_w -= crop_w % 8
    crop_h -= crop_h % 8
    crop_w = max(16, min(crop_w, width - (width % 8 if width >= 16 else 0)))
    crop_h = max(16, min(crop_h, height))
    x = max(0, ((width - crop_w) // 2) // 8 * 8)
    y = max(0, ((height - crop_h) // 2) // 8 * 8)
    if x + crop_w > width:
        crop_w = width - x
    if y + crop_h > height:
        crop_h = height - y
    return (int(x), int(y), int(crop_w), int(crop_h))


def apply_crop(gray, crop):
    if crop is None:
        return gray
    x, y, width, height = [int(value) for value in crop]
    x = max(0, x)
    y = max(0, y)
    if width <= 1 or height <= 1:
        return gray
    view = gray[y : y + height, x : x + width]
    if view.size == 0:
        return gray
    return view


def is_vr_equirect(width, height, thresholds=None):
    limits = active_thresholds(thresholds)
    if width < limits.vr_min_width or height <= 0:
        return False
    ratio = float(width) / float(height)
    return abs(ratio - 2.0) <= limits.vr_aspect_tolerance


def frame_fingerprint(gray):
    small = cv2.resize(to_gray(gray), (32, 32), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def fingerprints_match(first, second, max_mad=1.5):
    return float(np.mean(np.abs(first - second))) <= max_mad


def measure_frame(frame, max_width=ANALYSIS_FRAME_MAX_WIDTH, crop=None, block_crop=None, thresholds=None):
    limits = active_thresholds(thresholds)
    gray_full = to_gray(frame)
    content = apply_crop(gray_full, crop)
    block_source = apply_crop(gray_full, block_crop) if block_crop is not None else gray_full
    brightness = float(np.mean(content))
    black = float(np.mean(content < limits.dark_pixel_threshold))
    std = float(np.std(content))
    exclude_dark = (
        black >= limits.dark_frame_ratio
        or brightness <= limits.dark_mean_brightness
        or std < limits.flat_std_max
    )
    blur, ratio = blur_profile(content, max_width)
    scaled = scale_gray(content, max_width)
    return {
        "blur": blur,
        "blur_scale_ratio": ratio,
        "blur_best_tiles": tile_sharpness(scaled),
        "block": blockiness_score(block_source),
        "noise": noise_profile(content),
        "noise_sigma": noise_sigma(content),
        "halo": halo_score(content),
        "combing": combing_score(gray_full),
        "brightness": brightness,
        "black": black,
        "std": std,
        "exclude_dark": exclude_dark,
        "exclude_duplicate": False,
        "exclude": exclude_dark,
    }


def mark_duplicate_frames(measured, frames, crop=None):
    previous = None
    for item, (_timestamp, frame) in zip(measured, frames):
        if item.get("exclude_dark"):
            continue
        fingerprint = frame_fingerprint(apply_crop(to_gray(frame), crop))
        if previous is not None and fingerprints_match(previous, fingerprint):
            item["exclude_duplicate"] = True
            item["exclude"] = True
        else:
            previous = fingerprint
    return measured


def _values(items, key):
    found = []
    for item in items:
        value = item.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number):
            continue
        found.append(number)
    return found


def aggregate_metrics(measured):
    if not measured:
        raise ValueError("No measurements to aggregate")
    for item in measured:
        if "exclude_dark" not in item:
            item["exclude_dark"] = bool(item.get("exclude"))
        item.setdefault("exclude_duplicate", False)

    usable = [
        item
        for item in measured
        if not item.get("exclude_dark") and not item.get("exclude_duplicate")
    ]
    nondark = [item for item in measured if not item.get("exclude_dark")]
    if usable:
        chosen = usable
        mostly_dark = False
    elif nondark:
        chosen = nondark
        mostly_dark = False
    else:
        chosen = measured
        mostly_dark = True

    blur_vals = _values(chosen, "blur")
    blur_arr = np.asarray(blur_vals, dtype=np.float64) if blur_vals else np.asarray([], dtype=np.float64)

    def pct(percent):
        if blur_arr.size == 0:
            return None
        return float(np.percentile(blur_arr, percent))

    median_blur = float(np.median(blur_arr)) if blur_arr.size else None
    p10 = pct(10)
    p90 = pct(90)
    iqr = None
    if blur_arr.size:
        iqr = float(np.percentile(blur_arr, 75) - np.percentile(blur_arr, 25))
    blur_cv = None
    if median_blur is not None and iqr is not None and abs(median_blur) > 1e-6:
        blur_cv = float(iqr / median_blur)

    dark_count = sum(1 for item in measured if item.get("exclude_dark"))
    duplicate_count = sum(
        1 for item in measured if item.get("exclude_duplicate") and not item.get("exclude_dark")
    )
    return {
        "avg_blur": float(np.mean(blur_arr)) if blur_arr.size else None,
        "median_blur": median_blur,
        "blur_p10": p10,
        "blur_p90": p90,
        "blur_iqr": iqr,
        "blur_cv": blur_cv,
        "min_blur": float(np.min(blur_arr)) if blur_arr.size else None,
        "max_blur": float(np.max(blur_arr)) if blur_arr.size else None,
        "avg_block": float(np.mean(_values(chosen, "block"))) if _values(chosen, "block") else None,
        "avg_noise": float(np.mean(_values(chosen, "noise"))) if _values(chosen, "noise") else None,
        "median_noise": float(np.median(_values(chosen, "noise_sigma")))
        if _values(chosen, "noise_sigma")
        else None,
        "avg_halo": float(np.mean(_values(chosen, "halo"))) if _values(chosen, "halo") else None,
        "combing": float(np.median(_values(chosen, "combing")))
        if _values(chosen, "combing")
        else None,
        "blur_best_tiles": float(np.median(_values(chosen, "blur_best_tiles")))
        if _values(chosen, "blur_best_tiles")
        else None,
        "blur_scale_ratio": float(np.mean(_values(chosen, "blur_scale_ratio")))
        if _values(chosen, "blur_scale_ratio")
        else None,
        "avg_brightness": float(np.mean(_values(measured, "brightness"))),
        "avg_black": float(np.mean(_values(measured, "black"))),
        "frames_used": len(chosen),
        "frames_excluded": dark_count + duplicate_count,
        "mostly_dark": mostly_dark,
    }


def ssim_gray(first, second):
    if first.shape != second.shape or min(first.shape) < 16:
        return None
    first_f = first.astype(np.float64)
    second_f = second.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(first_f, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(second_f, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(first_f * first_f, (11, 11), 1.5) - mu_a**2
    sigma_b = cv2.GaussianBlur(second_f * second_f, (11, 11), 1.5) - mu_b**2
    sigma_ab = cv2.GaussianBlur(first_f * second_f, (11, 11), 1.5) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(numerator / denominator))


def classify_resolution(side, vr=False):
    if vr:
        return "vr_equirect"
    if side <= 0:
        return "unknown_resolution"
    if side <= 270:
        return "very_tiny"
    if side <= 360:
        return "very_low_resolution"
    if side <= 480:
        return "low_resolution"
    if side <= 576:
        return "sd_plus"
    if side <= 720:
        return "hd_720"
    if side <= 1080:
        return "full_hd_1080"
    return "above_1080"


def is_grain_dominated(avg_blur, blur_scale_ratio, thresholds=None):
    limits = active_thresholds(thresholds)
    if avg_blur is None or blur_scale_ratio is None:
        return False
    return avg_blur >= limits.blur_slightly_soft and blur_scale_ratio >= limits.blur_grain_ratio


def is_high_grain(median_noise, thresholds=None):
    limits = active_thresholds(thresholds)
    return median_noise is not None and float(median_noise) >= limits.median_noise_grainy


def is_sharpish(avg_blur, blur_scale_ratio, median_noise, thresholds=None):
    limits = active_thresholds(thresholds)
    if avg_blur is None:
        return False
    if is_grain_dominated(avg_blur, blur_scale_ratio, limits):
        return False
    if is_high_grain(median_noise, limits) and avg_blur < limits.blur_sharp:
        return False
    return avg_blur >= limits.blur_slightly_soft


def classify_blur(avg_blur, blur_scale_ratio=None, thresholds=None):
    limits = active_thresholds(thresholds)
    if avg_blur is None:
        return "unknown"
    if is_grain_dominated(avg_blur, blur_scale_ratio, limits):
        return "grain_dominated"
    if avg_blur < limits.blur_very:
        return "very_blurry"
    if avg_blur < limits.blur_blurry:
        return "blurry"
    if avg_blur < limits.blur_soft:
        return "soft"
    if avg_blur < limits.blur_slightly_soft:
        return "slightly_soft"
    if avg_blur >= limits.blur_sharp:
        return "sharp"
    return "acceptable_sharpness"


def classify_blockiness(avg_block, thresholds=None):
    limits = active_thresholds(thresholds)
    if avg_block is None:
        return "unknown"
    if avg_block >= limits.block_very:
        return "very_blocky"
    if avg_block >= limits.block_blocky:
        return "blocky"
    if avg_block >= limits.block_slight:
        return "slightly_blocky"
    return "clean"
