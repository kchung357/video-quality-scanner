# Video Quality Scanner

A Python tool for scanning video files, estimating quality issues, and generating a CSV report for restoration review.

It samples frames from each video, calculates heuristic quality metrics, assigns a quality issue score, and saves the results to a CSV report. The report is a way to find videos that may need restoration, deblurring, artifact reduction, upscaling, or a manual look. Files that share a folder and a duration can also be compared as an original and a processed version.

## Features

- Recursively scans a video folder
- Supports common video formats such as `.mp4`, `.mkv`, `.avi`, `.mov`, `.wmv`, `.mpg`, `.mpeg`, `.m4v`, `.ts`, `.m2ts`, `.webm`, `.flv`, and `.3gp`
- Reads metadata once with FFprobe, including sample aspect ratio, rotation, encoder, color, and field order
- Decodes full-resolution sample frames with FFmpeg
- Crops black bars before blur, noise, and halo, and leaves blockiness on the coded 8×8 grid
- Measures blur with Laplacian variance after scaling to a fixed analysis width
- Records a tile sharpness score beside the full-frame blur score
- Estimates compression blockiness on the decoded pixels, without scaling or JPEG
- Suggests a 16:9 re-encode from the content box: crop bars, reset sample aspect ratio, or pad/crop to 16:9
- Estimates noise with a float sigma on flat pixels, and separates real detail from grain
- Skips near-black, nearly flat, and duplicate frames when averaging picture metrics
- Samples at least 16 frames, spread through the file, and reports median, percentiles, and spread
- Calculates bits per pixel per frame, with a codec-efficiency adjustment
- Gives VR equirectangular frames their own class and skips the HD bitrate rule
- Classifies resolution from the shorter display side of the active picture
- Assigns a 0–100 quality issue score and writes the four parts of that score
- Suggests a review or restoration recommendation
- Compares an original with a processed sibling at the same timestamps, against a bicubic upscale of the original
- Writes a CSV report and can resume a scan when the file has not changed
- Optional contact sheets for manual review

## Requirements

- Python 3.8 or newer
- FFmpeg
- FFprobe

FFmpeg and FFprobe must be installed and available on `PATH`.

Python packages:

- `opencv-python`
- `numpy`
- `pandas`
- `tqdm`

## Installation

Clone the repository:

```bash
git clone https://github.com/kchung357/video-quality-scanner.git
cd video-quality-scanner
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Check that FFmpeg is installed:

```bash
ffmpeg -version
ffprobe -version
```

## Usage

```bash
python scan_video_quality.py --root ./videos --output video_quality_report.csv
```

Useful options:

```bash
python scan_video_quality.py --root ./videos --workers 4
python scan_video_quality.py --root ./videos --samples 8 --contact-sheets
python scan_video_quality.py --root ./videos --config config.example.yaml
python scan_video_quality.py --root ./videos --no-resume
python scan_video_quality.py --calibrate
python scan_video_quality.py --help
```

| Option | Default | Meaning |
|---|---|---|
| `--root` | `./videos` | Folder to scan recursively |
| `--output` | `video_quality_report.csv` | CSV report path |
| `--samples` | adaptive | Frames per video. Default is at least 16, one per 5 minutes, capped at 40 |
| `--workers` | up to 4 | Videos, and pair comparisons, run at the same time |
| `--analysis-width` | `640` | Width used only for the blur score |
| `--frame-timeout` | `60` | Seconds allowed for one sample frame |
| `--probe-timeout` | `30` | Seconds allowed for one FFprobe call |
| `--checkpoint-every` | `5` | Rewrite the CSV after this many new videos |
| `--resume` / `--no-resume` | resume | Keep rows whose schema, size, and modification time still match |
| `--pairs` / `--no-pairs` | pairs on | Compare original and processed siblings |
| `--groups-output` | beside the report | Where the pair CSV is written |
| `--config` | none | YAML file of threshold overrides |
| `--contact-sheets` | off | Write a JPEG contact sheet for review candidates |
| `--contact-sheet-min-score` | `55` | Score that also earns a contact sheet |
| `--contact-sheet-folder` | `contact_sheets` | Where contact sheets are written |
| `--skip-processed` | off | Skip filenames and encoder tags that already look processed |
| `--calibrate` | off | Run metric direction checks and exit |

The same defaults live in `scan_video_quality.py`. Flags override them. `config.example.yaml` lists every threshold the config file can change.

A scan rewrites the CSV as it goes, so an interrupted run can be continued with `--resume`. A row is reused only when `schema_version` is current and the file size and modification time still match. Replacing a file, or opening a report from an older version of this script, scans that file again. `--no-resume` replaces the report.

Hidden directories and the contact-sheet folder are not scanned.

## How frames are sampled

Each video is probed once. The default sample count is at least 16 frames, with one additional frame for every five minutes, and never more than 40. `--samples` forces a fixed count. Timestamps skip the first and last 4% of a long file (5% of a short one) and add a small jitter from a seed of the path, so a later run of the same file hits the same frames.

For each timestamp, FFmpeg seeks to about one second earlier, then decodes forward to that time. The sample is a raw frame at the coded resolution, not a JPEG and not a pre-scaled image.

That matters for the metrics:

- Blockiness is measured on the decoded 8×8 grid. Scaling or JPEG would invent or destroy those blocks. Letterbox bars are not cropped out of this measurement, and a VR frame uses the center of the grid.
- Blur, noise, and halo are measured on the active picture after a letterbox crop. Blur is then scaled to 640 px wide, so the blur thresholds stay comparable across resolutions.
- Grain is separated from real edges by comparing the blur score with the same score at half that width. Edges survive the extra downscale. Grain does not.
- Noise is an Immerkaer sigma on locally flat pixels. The old median of an 8-bit residual was 0 on real pictures, so `median_noise_score` now comes from this float estimate, then the median across frames.
- Near-black frames (mean at or below 20), nearly solid frames (standard deviation below 8), and duplicates of the previous kept frame are left out of the picture averages. If every sample is dark or flat, the score ignores blur and blockiness and the recommendation is `manual_review`.
- The blur summary is the mean, the median, the 10th and 90th percentiles, the interquartile range, and `blur_cv` (that range divided by the median). A large `blur_cv` on at least 8 frames adds `unstable_blur` and can move an otherwise OK file to `manual_review`.

`--analysis-width` changes the blur numbers. Leave it at 640 unless you also retune the blur thresholds. The tile score in `blur_best_tiles` is recorded for later calibration. It is not the number those thresholds use.

## Output

By default the report is saved as:

```text
video_quality_report.csv
```

`schema_version` is `3`. Reports from older versions are not resumed in place.

| Column | Description |
|---|---|
| `path` | Path to the video file |
| `filename` | Video filename |
| `folder` | Parent folder |
| `mtime` | File modification time, in seconds. Used with `size` for resume |
| `width`, `height` | Coded frame size |
| `display_width`, `display_height` | Size after sample aspect ratio and rotation |
| `content_width`, `content_height` | Display size after the letterbox crop |
| `content_aspect` | `content_width / content_height` |
| `crop` | Coded crop as `x,y,width,height` |
| `fix_geometry` | Re-encode hint such as `crop_bars+reset_sar:1280x720`, or blank when the picture is already square-pixel 16:9 |
| `sample_aspect_ratio` | Sample aspect ratio reported by FFprobe |
| `rotation` | Display rotation in degrees |
| `codec` | Video codec |
| `pix_fmt` | Pixel format |
| `encoder` | Encoder tag from the stream or format |
| `color_range`, `color_space` | Color metadata from FFprobe |
| `field_order` | Field order from FFprobe |
| `duration` | Duration in seconds |
| `duration_hms` | Duration as `HH:MM:SS` |
| `frame_rate` | Video frame rate |
| `bit_rate` | Bitrate in bits per second |
| `size` | File size in bytes |
| `is_probably_processed` | Filename marker, known encoder tag, or a paired sibling |
| `is_vr` | 2:1 frame at least 3840 px wide |
| `sampled_frames` | Frames that decoded |
| `frames_used` | Frames included in the picture metrics |
| `frames_excluded` | Near-black, flat, or duplicate frames left out |
| `sample_errors` | Timestamps that failed to decode |
| `sample_error` | First decode error, when some timestamps failed and others succeeded |
| `avg_blur_score` | Mean blur/sharpness score. Higher means sharper |
| `median_blur_score`, `blur_p10`, `blur_p90`, `blur_iqr`, `blur_cv` | Robust blur summary |
| `min_blur_score`, `max_blur_score` | Blur range across the frames that were averaged |
| `blur_scale_ratio` | Blur at the analysis width divided by blur at half width. Values near 1 are real edges. Values near 4 are grain |
| `blur_best_tiles` | Sharpest fifth of 64×64 tiles. Not used by the score yet |
| `avg_blockiness_score` | Estimated compression blockiness |
| `avg_noise_score` | Mean high-frequency residual |
| `median_noise_score` | Median Immerkaer sigma on flat pixels. High values mean grain across the frame |
| `avg_halo_score` | Edge overshoot in pixel levels. High values with a very high blur score suggest oversharpening |
| `combing_score` | Share of pixels that look interlaced |
| `avg_brightness` | Average brightness of the active picture, 0 to 255 |
| `avg_black_frame_ratio` | Fraction of near-black pixels in the active picture |
| `bitrate_per_pixel` | Bits per second per coded pixel |
| `bits_per_pixel_frame` | Bits per coded pixel per frame |
| `bits_per_pixel_frame_eff` | That value times the codec efficiency factor |
| `score_resolution`, `score_blur`, `score_blockiness`, `score_bitrate` | The four parts of the issue score |
| `quality_issue_score` | Overall issue score from 0 to 100. Blank when the file could not be scored |
| `resolution_class` | Resolution classification from the shorter side of the active picture |
| `blur_class` | `very_blurry`, `blurry`, `soft`, `slightly_soft`, `acceptable_sharpness`, `sharp`, or `grain_dominated` |
| `blockiness_class` | Blockiness classification |
| `issue_flags` | Detected quality issue flags |
| `recommendation` | Suggested review or restoration action |
| `contact_sheet` | Path to the contact sheet, if one was written |
| `contact_sheet_error` | Contact-sheet error, if writing it failed |
| `error` | Error message, if analysis of the file failed |

`bitrate_per_pixel` and `bits_per_pixel_frame` are different measurements. So are `sampled_frames` and `frames_used`. Both pairs stay in the report.

Failed files have a blank `quality_issue_score`. They stay in the CSV and are listed apart from the ranked candidates. A timestamp that fails while other frames succeed is counted in `sample_errors` and does not blank the score. Older reports used `999` for a failed file. A resumed report clears those sentinel scores so they are not treated as the worst videos.

## Quality issue score

`quality_issue_score` is a heuristic from 0 to 100. Higher scores are stronger candidates for restoration or manual review. `score_resolution`, `score_blur`, `score_blockiness`, and `score_bitrate` are the parts, before the cap of 100.

| Part | Ceiling | What raises it |
|---|---:|---|
| Resolution | 15 | Shorter side of the active picture at or below 720, more as it gets smaller |
| Blur | 45 | Low Laplacian variance at the analysis width |
| Blockiness | 35 | Strong 8×8 block edges on the decoded frame |
| Bitrate | 15 | Low bits per pixel per frame for the container size, after the codec factor. VR scores 0 |

A ruined 1080p file can reach the high band from blur, blockiness, and bitrate alone. Resolution is only a small bonus, so SD footage is not automatically ranked above damaged HD.

The codec factor is `h264` 1.0, `hevc` 1.5, `av1` 1.8, with lower factors for MPEG-2 and MPEG-4. A 2 Mbps 1080p30 HEVC file is about 0.032 bits per pixel per frame. Times 1.5 that is still under the HD low-bitrate cutoff of 0.07, so it can keep `low_bitrate_for_hd`. The factor moves it out of the very-low band. Change the factors in the config file if that cutoff does not match the library.

| Score | Meaning |
|---:|---|
| `90+` | Very high priority review |
| `75-89` | High priority review |
| `55-74` | Medium priority review |
| `35-54` | Low to medium priority review |
| `0-34` | Low priority or probably acceptable |

Scores from older versions of this script are not comparable. The frame measurement and the point scale both changed.

## Recommendations

| Recommendation | Meaning |
|---|---|
| `top_priority_restore` | Score is at least 75 and a more specific label did not apply |
| `restore_deblur_artifact_reduce` | Low resolution, very blurry, and blocky |
| `restore_or_deblur_then_upscale` | Low resolution and blurry. A blurry HD file is not sent here |
| `upscale_only_gentle` | Low resolution with real detail and no obvious blocking |
| `upscale_candidate_very_low_res` | Very low resolution with real detail |
| `check_source_upscaled_or_denoise` | Blurry HD that does not look like a straight upscale |
| `artifact_reduction_review` | Blocky, without a stronger restoration label. Combing moves this to `manual_review` |
| `vr_equirect_review` | Equirectangular VR with no stronger defect label |
| `manual_review` | Slightly soft, dark, grainy, unstable, or otherwise uncertain |
| `low_priority_or_ok` | Low score and no specific defect label |
| `manual_check_error` | Analysis raised an unexpected error |
| `manual_check_could_not_sample` | Metadata or frame decoding failed |
| `fix_path_or_missing_file` | The file was missing |
| `skipped` | Skipped because the filename or encoder looks processed |

A blurry HD file is `check_source_upscaled_or_denoise`. The scanner does not guess that a soft HD container is upscaled SD, and it does not tell you to upscale a file that is already HD. When both the original and a processed file are present, pair mode answers whether that process helped.

Filenames that already look processed, or files paired with an original, get the same label with an `already_processed_review_only+` prefix, for example `already_processed_review_only+upscale_only_gentle`. Markers include `_topaz`, `_upscale`, `_proteus`, `_gaia`, `_prob`, and similar tokens. An encoder tag that contains Topaz, Proteus, Artemis, Video2X, or Real-ESRGAN counts as well.

A sharp low-resolution file is labeled as an upscale candidate even when its score is low. Grain and near-black frames are not treated as detail worth upscaling.

## Geometry

`fix_geometry` says what a re-encode should do. The content box is the median letterbox crop, and only when those bars show up on at least half of the frames that are not near-black. Resolution class uses that content size.

The display aspect is `content width × sample aspect ratio / content height`.

| Picture | What the label means |
|---|---|
| Within about 3% of 16:9 | Scale to fit the target frame |
| Cropping to 16:9 would drop at most about 10% of the area | `crop_to_16x9` |
| Anything wider or taller than that | `pad_to_16x9` |
| Black bars in the stored frame | `crop_bars` |
| Sample aspect ratio is not 1:1 | `reset_sar` |

720p-class content targets `1280×720`. 1080p-class content targets `1920×1080`. SD stays at its own display size, so a 4:3 DVD is padded rather than upscaled. A `1056×720` picture with sample aspect ratio `97:80` is exactly 16:9 once the side bars are cropped, so the label is `crop_bars+reset_sar:1280x720`.

## Pair groups

Pair mode is on unless you pass `--no-pairs`. It writes a second CSV next to the report. `video_quality_report.csv` produces `video_quality_report_groups.csv`.

Two scored files are compared when they share a folder, their durations are within about a second, and their filenames are the same title. Processing suffixes such as `_prob4` and `_topaz` are removed before that comparison, so `DVDMS-388_3_prob4` can match `DVDMS-388`. Files in the same folder with the same duration but unrelated names, such as different episodes, are not compared.

The reference is the file that does not look processed. Pair comparisons use the same `--workers` pool as the scan. Each pair still decodes with one ffmpeg thread, so several pairs do not each take every core. Both files are sampled at the same timestamps. If a pair of frames is not the same picture (SSIM below about 0.6), the scanner tries two frames on either side. If more than about a third of the timestamps still miss, the verdict is `pair_misaligned`.

Each kept timestamp is measured on its own, then the median of those deltas is the pair result. Two comparisons are made:

- Native grid. The processed frame is downscaled to the original content size. Sharpness is compared with a bicubic upscale of the original brought back the same way, so the loss from resampling itself is not called damage. Blockiness is measured on the processed file at the scaled codec period (`8 × scale`), because a 1.5× upscale turns 8×8 blocks into 12×12 blocks.
- Baseline. The original is bicubic-upscaled to the processed canvas. This is the "is it better than playback scaling?" test.

A luma shift of about 10 or more, the usual tv-versus-full range step, is corrected before SSIM and the other ratios. It is recorded as `range_shift` and does not by itself fail the pair.

| Column | Meaning |
|---|---|
| `blur_native` | Processed sharpness on the source grid, divided by the bicubic roundtrip. Below 1 means softer than a plain upscale |
| `blur_vs_base` | Processed sharpness on the output canvas, divided by the bicubic upscale |
| `block_native` | Scaled-period blockiness minus the original's 8-pixel blockiness |
| `noise_native` | Noise on the source grid, as a ratio |
| `halo_vs_base` | Edge overshoot on the output canvas, divided by the bicubic upscale |
| `ssim` | Fidelity on the source grid after the luma correction. This is not a quality score |
| `geometry_mismatch` | Content aspects differ by more than about 5%. Metrics are skipped |
| `luma_delta`, `range_shift` | Mean Y difference before the correction |
| `verdict` | `improved`, `neutral`, `regressed:<flags>`, `pair_misaligned`, `geometry_mismatch`, `ambiguous_group`, or `compare_failed` |
| `action` | `keep_processed`, `keep_original`, or `manual_review` |
| `best_version` | The file to keep |

`improved` means the processed file is clearly sharper than bicubic and none of the damage flags fired. `neutral` means no damage and no gain over playback scaling, so the action stays `keep_original`. Damage flags are `lost_detail`, `more_blocky`, `more_noise`, and `oversharpened`. A sharpness drop that comes with a similar noise drop is treated as denoise, unless the softness is severe.

`ambiguous_group` means several unmarked files share a duration and the scanner will not guess. `geometry_mismatch` means the content aspect changed, which is usually a bad crop, and the quality verdict would not be meaningful.

These ratio cutoffs are starting points. `config.example.yaml` lists them. Check them against pairs you have already looked at before deleting either file.

## Contact sheets

Contact sheets are off unless you pass `--contact-sheets`.

They are written for a video whose score is at least `--contact-sheet-min-score`, and also for `manual_review`, `check_source_upscaled_or_denoise`, and `vr_equirect_review`. A pair whose verdict is anything other than `improved` gets a sheet of the two frames side by side.

The file name includes the parent folder, the stem, and a short hash of the full path, so two folders with the same name do not overwrite each other.

## Calibration

```bash
python scan_video_quality.py --calibrate
```

This checks that blur falls as Gaussian blur increases, that a bicubic upscale matches its own baseline, that a blurred upscale is softer than that baseline, that a 1.5× block grid is visible at the scaled period, and that a letterbox crop measures the picture inside the bars. It does not choose new thresholds. Threshold changes belong after that check, and after a set of files you have labeled by eye.

```bash
python -m unittest test_scan_video_quality.py
```

The integration tests are skipped when FFmpeg is not installed.

## Important notes

These metrics are heuristics. They are not a full quality measurement.

Results can still be affected by:

- Subtitles and hard graphics
- Animation
- Interlacing, which is flagged from field order and from a combing score
- Intentional blur or artistic effects
- Scene content that is naturally soft or very contrasty

Manual review is still the right step before restoring, upscaling, deleting, replacing, or re-encoding a video.

## Privacy

Do not commit private videos, personal media, generated CSV reports, or contact sheets.

The CSV can contain private file paths. Contact sheets contain frames from the videos. This repository should contain the script, tests, documentation, and other safe project files.

## License

This project is licensed under the MIT License.
