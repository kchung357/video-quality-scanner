"""Pair-frame perceptual comparison: classical + optional LPIPS / NR uplift."""

from __future__ import annotations

import math

import cv2
import numpy as np

from vqs.config import active_thresholds
from vqs.metrics.classical import (
    blockiness_scaled,
    blur_profile,
    halo_score,
    laplacian_var,
    noise_sigma,
    ssim_gray,
    to_gray,
)
from vqs.util import parse_sar, safe_float, safe_int


def to_bgr(frame):
    if frame.ndim == 2 or frame.shape[2] == 1:
        gray = frame if frame.ndim == 2 else frame[:, :, 0]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame


def rotate_bgr(image, rotation):
    turns = int(round(abs(float(rotation)))) % 360
    if turns == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if turns == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if turns == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def content_bgr(frame, crop, sar, rotation):
    image = to_bgr(frame)
    height, width = image.shape[:2]
    if crop is None:
        view = image
    else:
        x, y, crop_w, crop_h = [int(value) for value in crop]
        x = max(0, min(x, max(0, width - 1)))
        y = max(0, min(y, max(0, height - 1)))
        view = image[y : y + max(0, crop_h), x : x + max(0, crop_w)]
        if view.size == 0:
            view = image
    if abs(float(sar) - 1.0) > 0.01:
        view = cv2.resize(
            view,
            (max(1, int(round(view.shape[1] * float(sar)))), view.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return rotate_bgr(view, rotation)


def parse_crop(text, width, height):
    width = int(width or 0)
    height = int(height or 0)
    fallback = (0, 0, width, height)
    if text is None or str(text).strip() == "":
        return fallback
    parts = str(text).split(",")
    if len(parts) != 4:
        return fallback
    try:
        x, y, crop_w, crop_h = [int(float(part)) for part in parts]
    except ValueError:
        return fallback
    if crop_w <= 1 or crop_h <= 1:
        return fallback
    return (x, y, crop_w, crop_h)


def _resize_image(image, width, height, interpolation):
    width = max(1, int(width))
    height = max(1, int(height))
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=interpolation)


def safe_ratio(numerator, denominator, floor=1.0):
    num = float(numerator or 0.0)
    den = float(denominator or 0.0)
    if den < floor and num < floor:
        return 1.0
    return num / max(den, floor)


def mean_yuv(bgr):
    yuv = cv2.cvtColor(to_bgr(bgr), cv2.COLOR_BGR2YUV)
    return [float(yuv[:, :, channel].mean()) for channel in range(3)]


def shift_luma(bgr, delta):
    if abs(float(delta)) < 0.5:
        return to_bgr(bgr)
    yuv = cv2.cvtColor(to_bgr(bgr), cv2.COLOR_BGR2YUV).astype(np.int16)
    yuv[:, :, 0] = np.clip(yuv[:, :, 0] + int(round(float(delta))), 0, 255)
    return cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)


def align_ssim(first, second, width=160):
    left = to_gray(first)
    right = to_gray(second)
    if left.shape[1] < 16 or right.shape[1] < 16 or left.shape[0] < 16 or right.shape[0] < 16:
        return None
    left_h = max(16, int(round(left.shape[0] * (width / float(left.shape[1])))))
    right_h = max(16, int(round(right.shape[0] * (width / float(right.shape[1])))))
    left_small = cv2.resize(left, (width, left_h), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right, (width, right_h), interpolation=cv2.INTER_AREA)
    if right_small.shape != left_small.shape:
        right_small = cv2.resize(
            right_small,
            (left_small.shape[1], left_small.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return ssim_gray(left_small, right_small)


class LpipsScorer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._fn = None
        self._torch = None
        self._tried = False
        self._error = ""

    def _load(self) -> bool:
        if self._tried:
            return self._fn is not None
        self._tried = True
        try:
            import torch
            import lpips

            self._torch = torch
            self._fn = lpips.LPIPS(net="alex").to(device)
            self._fn.eval()
            return True
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            return False

    def distance(self, a_bgr, b_bgr) -> float | None:
        if not self._load():
            return None
        torch = self._torch
        a = cv2.cvtColor(to_bgr(a_bgr), cv2.COLOR_BGR2RGB)
        b = cv2.cvtColor(to_bgr(b_bgr), cv2.COLOR_BGR2RGB)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        side = 256
        scale = min(1.0, float(side) / float(max(a.shape[0], a.shape[1])))
        if scale < 1.0:
            size = (max(1, int(round(a.shape[1] * scale))), max(1, int(round(a.shape[0] * scale))))
            a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, size, interpolation=cv2.INTER_AREA)
        ta = torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        tb = torch.from_numpy(b).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        ta = ta.to(self.device)
        tb = tb.to(self.device)
        with torch.no_grad():
            value = self._fn(ta, tb)
        return float(value.detach().cpu().item())


def evaluate_pair_frames(
    orig_frame,
    proc_frame,
    orig_crop=None,
    proc_crop=None,
    orig_sar=1.0,
    proc_sar=1.0,
    orig_rotation=0.0,
    proc_rotation=0.0,
    lpips_scorer: LpipsScorer | None = None,
    nr_scorer=None,
):
    orig_view = content_bgr(orig_frame, orig_crop, orig_sar, orig_rotation)
    proc_view = content_bgr(proc_frame, proc_crop, proc_sar, proc_rotation)
    if min(orig_view.shape[:2]) < 16 or min(proc_view.shape[:2]) < 16:
        return None
    orig_yuv = mean_yuv(orig_view)
    proc_yuv = mean_yuv(proc_view)
    luma_delta = orig_yuv[0] - proc_yuv[0]
    chroma_delta = max(abs(orig_yuv[1] - proc_yuv[1]), abs(orig_yuv[2] - proc_yuv[2]))
    if abs(luma_delta) >= 10.0:
        proc_view = shift_luma(proc_view, luma_delta)
    orig_h, orig_w = orig_view.shape[:2]
    proc_h, proc_w = proc_view.shape[:2]
    base_view = _resize_image(orig_view, proc_w, proc_h, cv2.INTER_CUBIC)
    orig_gray = to_gray(orig_view)
    proc_gray = to_gray(proc_view)
    base = to_gray(base_view)
    proc_native = _resize_image(proc_gray, orig_w, orig_h, cv2.INTER_AREA)
    base_native = _resize_image(base, orig_w, orig_h, cv2.INTER_AREA)
    orig_coded_h, orig_coded_w = to_gray(orig_frame).shape[:2]
    proc_coded_h, proc_coded_w = to_gray(proc_frame).shape[:2]
    scale_x = (proc_coded_w / float(orig_coded_w)) if orig_coded_w else 1.0
    scale_y = (proc_coded_h / float(orig_coded_h)) if orig_coded_h else 1.0
    halo_proc = halo_score(proc_gray)
    halo_base = halo_score(base)

    return apply_pair_model_scores(
        {
            "blur_native": safe_ratio(laplacian_var(proc_native), laplacian_var(base_native)),
            "blur_vs_base": safe_ratio(laplacian_var(proc_gray), laplacian_var(base)),
            "block_native": float(
                blockiness_scaled(proc_frame, scale_x, scale_y) - blockiness_scaled(orig_frame, 1.0, 1.0)
            ),
            "noise_native": safe_ratio(noise_sigma(proc_native), noise_sigma(base_native), floor=0.05),
            "halo_vs_base": float(halo_proc) / max(float(halo_base), 1.0),
            "halo_delta": float(halo_proc - halo_base),
            "ssim": ssim_gray(proc_native, orig_gray),
            "lpips_native": None,
            "nr_uplift": None,
            "luma_delta": float(luma_delta),
            "chroma_delta": float(chroma_delta),
            "orig_view": orig_view,
            "proc_view": proc_view,
        },
        lpips_scorer=lpips_scorer,
        nr_scorer=nr_scorer,
    )


def apply_pair_model_scores(metrics, lpips_scorer=None, nr_scorer=None):
    """Fill LPIPS and NR uplift for one aligned pair of content views.

    Called from ``evaluate_pair_frames`` and again from the parent process when
    decode ran in a worker that must not load CUDA.
    """
    if lpips_scorer is None and nr_scorer is None:
        return metrics
    orig_view = metrics.get("orig_view")
    proc_view = metrics.get("proc_view")
    if orig_view is None or proc_view is None:
        return metrics
    orig_w = orig_view.shape[1]
    orig_h = orig_view.shape[0]
    proc_h, proc_w = proc_view.shape[:2]
    if lpips_scorer is not None:
        # Fidelity: processed downscaled to original vs original content.
        proc_native_bgr = _resize_image(proc_view, orig_w, orig_h, cv2.INTER_AREA)
        metrics["lpips_native"] = lpips_scorer.distance(orig_view, proc_native_bgr)
    if nr_scorer is not None and getattr(nr_scorer, "available", False):
        base_view = _resize_image(orig_view, proc_w, proc_h, cv2.INTER_CUBIC)
        proc_score = nr_scorer.score_bgr_frames([proc_view])
        base_score = nr_scorer.score_bgr_frames([base_view])
        if proc_score.median is not None and base_score.median is not None:
            metrics["nr_uplift"] = float(proc_score.median - base_score.median)
    return metrics


def _median_of(items, key):
    values = []
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
        values.append(number)
    if not values:
        return None
    return float(np.median(values))


def summarize_pair_metrics(frames, thresholds=None):
    limits = active_thresholds(thresholds)
    summary = {
        key: _median_of(frames, key)
        for key in (
            "blur_native",
            "blur_vs_base",
            "block_native",
            "noise_native",
            "halo_vs_base",
            "halo_delta",
            "ssim",
            "lpips_native",
            "nr_uplift",
            "luma_delta",
            "chroma_delta",
        )
    }
    luma = summary.get("luma_delta")
    summary["range_shift"] = luma is not None and abs(luma) >= limits.pair_luma_shift
    summary["pair_misaligned"] = False
    summary["geometry_mismatch"] = False
    return summary
