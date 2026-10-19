"""Configuration and tunable thresholds for v2."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

from vqs.util import CODEC_EFFICIENCY, parse_simple_yaml


@dataclass
class Thresholds:
    """Tunable cutoffs. A config file overlays these defaults."""

    blur_very: float = 35.0
    blur_blurry: float = 60.0
    blur_soft: float = 100.0
    blur_slightly_soft: float = 150.0
    blur_sharp: float = 300.0
    blur_grain_ratio: float = 2.0
    block_slight: float = 1.30
    block_blocky: float = 1.45
    block_very: float = 1.60
    median_noise_grainy: float = 4.0
    dark_pixel_threshold: float = 16.0
    dark_frame_ratio: float = 0.80
    dark_mean_brightness: float = 20.0
    flat_std_max: float = 8.0
    combing_min: float = 0.12
    halo_flag: float = 8.0
    vr_min_width: int = 3840
    vr_aspect_tolerance: float = 0.02
    unstable_blur_cv: float = 1.0
    unstable_min_frames: int = 8

    # Solo upscale-worthiness
    topaz_max_short_side: int = 1080
    topaz_hd_blur_max: float = 100.0
    topaz_block_min: float = 1.40
    topaz_nr_iqa_max: float = 0.45  # TOPIQ-NR: higher is better; below this is poor
    topaz_nr_iqa_ok: float = 0.62
    topaz_min_confidence: float = 0.45

    # Pair compare
    pair_duration_seconds: float = 1.0
    pair_ssim_align: float = 0.60
    pair_align_fail_fraction: float = 0.34
    pair_min_frames: int = 4
    pair_blur_detail_ratio: float = 0.85
    pair_blur_severe_ratio: float = 0.65
    pair_noise_drop_ratio: float = 0.85
    pair_noise_ratio: float = 1.30
    pair_block_delta: float = 0.10
    pair_halo_ratio: float = 1.45
    pair_halo_delta: float = 4.0
    pair_blur_vs_base: float = 1.15
    pair_lpips_max: float = 0.35  # fidelity loss vs original on native grid
    pair_nr_uplift_min: float = 0.03  # NR-IQA gain vs bicubic on output canvas
    pair_luma_shift: float = 10.0
    pair_chroma_shift: float = 8.0
    geometry_aspect_tolerance: float = 0.05
    geometry_fit_tolerance: float = 0.03
    geometry_crop_area_max: float = 0.10
    codec_efficiency: dict = field(default_factory=lambda: dict(CODEC_EFFICIENCY))


DEFAULT_THRESHOLDS = Thresholds()


@dataclass
class ScanConfig:
    root_folder: str = "./videos"
    db_path: str = "vqs.db"
    report_dir: str = "report"
    create_contact_sheets: bool = True
    contact_sheet_folder: str = "contact_sheets"
    sample_count: int = 16
    samples_explicit: bool = False
    analysis_frame_max_width: int = 640
    contact_thumb_width: int = 320
    skip_already_processed: bool = False
    workers: int = 4
    # 0 lets FFmpeg pick its own thread count. The scan sets 2 while a process
    # pool is running so several decodes do not each take every core.
    ffmpeg_threads: int = 0
    resume: bool = True
    frame_timeout: float = 60.0
    probe_timeout: float = 30.0
    health_timeout: float = 0.0  # 0 = duration * 3 + 120, capped later
    checkpoint_every: int = 5
    run_health: bool = True
    run_solo: bool = True
    run_pairs: bool = True
    run_report: bool = True
    device: str = "auto"  # auto | cuda | cpu
    nr_iqa_model: str = "topiq_nr"
    nr_iqa_batch: int = 8
    cache_dir: str = ""
    thresholds: Thresholds = field(default_factory=Thresholds)


def active_thresholds(thresholds=None) -> Thresholds:
    return thresholds or DEFAULT_THRESHOLDS


_RETIRED_CONFIG_KEYS = {
    "upscale_bandwidth_max",
    "bandwidth_drop_db",
    "hd_bpp_very_low",
    "hd_bpp_low",
    "sd_bpp_very_low",
    "sd_bpp_low",
}


def thresholds_from_mapping(mapping, base=None) -> Thresholds:
    base = base or Thresholds()
    if not mapping:
        return base
    mapping = {key: value for key, value in mapping.items() if key not in _RETIRED_CONFIG_KEYS}
    known = {item.name for item in fields(base)}
    unknown = sorted(key for key in mapping if key not in known)
    if unknown:
        raise ValueError("Unknown config keys: " + ", ".join(unknown))
    values = {}
    for key, value in mapping.items():
        default = getattr(base, key)
        if key == "codec_efficiency":
            if not isinstance(value, dict):
                raise ValueError("codec_efficiency must be a map of codec names")
            merged = dict(base.codec_efficiency)
            for codec, factor in value.items():
                merged[str(codec).lower()] = float(factor)
            values[key] = merged
        elif isinstance(default, float):
            values[key] = float(value)
        elif isinstance(default, bool):
            values[key] = bool(value)
        elif isinstance(default, int):
            values[key] = int(value)
        else:
            values[key] = value
    data = {item.name: getattr(base, item.name) for item in fields(base)}
    data.update(values)
    return Thresholds(**data)


def load_thresholds(path) -> Thresholds:
    text = Path(path).read_text(encoding="utf-8")
    return thresholds_from_mapping(parse_simple_yaml(text))


def resolve_device(requested: str) -> str:
    wanted = (requested or "auto").strip().lower()
    if wanted == "cpu":
        return "cpu"
    try:
        import torch

        if wanted in {"cuda", "auto"} and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if wanted == "cuda":
        return "cpu"
    return "cpu"
