# Video Quality Scanner

A Python tool for scanning video files, estimating quality issues, and generating a CSV report for restoration review.

It samples frames from each video, calculates heuristic quality metrics, assigns a quality issue score, and saves the results to a CSV report. The report is a way to find videos that may need restoration, deblurring, artifact reduction, upscaling, or a manual look.

## Features

- Recursively scans a video folder
- Supports common video formats such as `.mp4`, `.mkv`, `.avi`, `.mov`, `.wmv`, `.mpg`, `.mpeg`, `.m4v`, `.ts`, `.m2ts`, `.webm`, `.flv`, and `.3gp`
- Reads metadata once with FFprobe, including sample aspect ratio and rotation
- Decodes full-resolution sample frames with FFmpeg
- Measures blur with Laplacian variance, after scaling to a fixed analysis width
- Estimates compression blockiness on the decoded pixels, without scaling or JPEG
- Estimates noise, and separates real detail from grain
- Skips near-black and nearly flat frames when averaging blur and blockiness
- Calculates bits per pixel per frame, so frame rate is part of the bitrate check
- Classifies resolution from the shorter display side
- Assigns a 0–100 quality issue score
- Suggests a review or restoration recommendation
- Writes a CSV report and can resume a scan that was interrupted
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
python scan_video_quality.py --root ./videos --no-resume
python scan_video_quality.py --help
```

| Option | Default | Meaning |
|---|---|---|
| `--root` | `./videos` | Folder to scan recursively |
| `--output` | `video_quality_report.csv` | CSV report path |
| `--samples` | `8` | Frames sampled per video |
| `--workers` | up to 4 | Videos analyzed at the same time |
| `--analysis-width` | `640` | Width used only for the blur score |
| `--frame-timeout` | `60` | Seconds allowed for one sample frame |
| `--probe-timeout` | `30` | Seconds allowed for one FFprobe call |
| `--checkpoint-every` | `5` | Rewrite the CSV after this many new videos |
| `--resume` / `--no-resume` | resume | Skip paths already present in the CSV |
| `--contact-sheets` | off | Write a JPEG contact sheet for high scores |
| `--contact-sheet-min-score` | `55` | Score required for a contact sheet |
| `--contact-sheet-folder` | `contact_sheets` | Where contact sheets are written |
| `--skip-processed` | off | Skip filenames that already look processed |

The same defaults live at the top of `scan_video_quality.py`. Flags override them.

A scan rewrites the CSV as it goes, so an interrupted run can be continued with `--resume`. Paths already in the report are not analyzed again. `--no-resume` replaces the report.

Hidden directories and the contact-sheet folder are not scanned.

## How frames are sampled

Each video is probed once. Eight timestamps (by default) are spread across the middle of the file. For each timestamp, FFmpeg seeks to about one second earlier, then decodes forward to that time. The sample is a raw frame at the coded resolution, not a JPEG and not a pre-scaled image.

That matters for the metrics:

- Blockiness is measured on the decoded 8×8 grid. Scaling or JPEG would invent or destroy those blocks.
- Blur is measured after scaling the frame to 640 px wide, so the blur thresholds stay comparable across resolutions.
- Grain is separated from real edges by comparing the blur score with the same score at half that width. Edges survive the extra downscale. Grain does not.
- Near-black frames and nearly solid frames are left out of the blur, blockiness, and noise averages. If every sample is dark or flat, the score ignores blur and blockiness and the recommendation is `manual_review`.

`--analysis-width` changes the blur numbers. Leave it at 640 unless you also retune the blur thresholds.

## Output

By default the report is saved as:

```text
video_quality_report.csv
```

| Column | Description |
|---|---|
| `path` | Path to the video file |
| `filename` | Video filename |
| `folder` | Parent folder |
| `width`, `height` | Coded frame size |
| `display_width`, `display_height` | Size after sample aspect ratio and rotation |
| `sample_aspect_ratio` | Sample aspect ratio reported by FFprobe |
| `rotation` | Display rotation in degrees |
| `codec` | Video codec |
| `pix_fmt` | Pixel format |
| `duration` | Duration in seconds |
| `duration_hms` | Duration as `HH:MM:SS` |
| `frame_rate` | Video frame rate |
| `bit_rate` | Bitrate in bits per second |
| `size` | File size in bytes |
| `sampled_frames` | Frames that decoded |
| `frames_used` | Frames included in the blur and blockiness averages |
| `frames_excluded` | Near-black or flat frames left out of those averages |
| `avg_blur_score` | Average blur/sharpness score. Higher means sharper |
| `min_blur_score`, `max_blur_score` | Blur range across the frames that were averaged |
| `blur_scale_ratio` | Blur at the analysis width divided by blur at half width. Values near 1 are real edges. Values near 4 are grain |
| `avg_blockiness_score` | Estimated compression blockiness |
| `avg_noise_score` | Mean high-frequency residual |
| `median_noise_score` | Median residual. High values mean grain across the frame, not a few hard edges |
| `avg_brightness` | Average brightness, 0 to 255 |
| `avg_black_frame_ratio` | Fraction of near-black pixels |
| `bitrate_per_pixel` | Bits per second per coded pixel |
| `bits_per_pixel_frame` | Bits per coded pixel per frame. This is what the score uses |
| `quality_issue_score` | Overall issue score from 0 to 100. Blank when the file could not be scored |
| `resolution_class` | Resolution classification from the shorter display side |
| `blur_class` | Blur classification |
| `blockiness_class` | Blockiness classification |
| `issue_flags` | Detected quality issue flags |
| `recommendation` | Suggested review or restoration action |
| `contact_sheet` | Path to the contact sheet, if one was written |
| `contact_sheet_error` | Contact-sheet error, if writing it failed |
| `error` | Error message, if analysis failed |

Failed files have a blank `quality_issue_score`. They stay in the CSV and are listed apart from the ranked candidates. Older reports used `999` for that case. A resumed report clears those sentinel scores so they are not treated as the worst videos.

## Quality issue score

`quality_issue_score` is a heuristic from 0 to 100. Higher scores are stronger candidates for restoration or manual review.

The parts, and their ceilings, are:

| Part | Ceiling | What raises it |
|---|---:|---|
| Resolution | 15 | Shorter display side at or below 720, more as it gets smaller |
| Blur | 45 | Low Laplacian variance at the analysis width |
| Blockiness | 35 | Strong 8×8 block edges on the decoded frame |
| Bitrate | 15 | Low bits per pixel per frame for HD or SD |

A ruined 1080p file can reach the high band from blur, blockiness, and bitrate alone. Resolution is only a small bonus, so SD footage is not automatically ranked above damaged HD.

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
| `restore_or_deblur_then_upscale` | Low resolution and blurry |
| `upscale_only_gentle` | Low resolution with real detail and no obvious blocking |
| `upscale_candidate_very_low_res` | Very low resolution with real detail |
| `artifact_reduction_review` | Blocky, without a stronger restoration label |
| `manual_review` | Slightly soft, dark, grainy, or otherwise uncertain |
| `low_priority_or_ok` | Low score and no specific defect label |
| `manual_check_error` | Analysis raised an unexpected error |
| `manual_check_could_not_sample` | Metadata or frame decoding failed |
| `fix_path_or_missing_file` | The file was missing |
| `skipped` | Skipped because the filename looks processed |

Filenames that already look processed get the same label with an `already_processed_review_only+` prefix, for example `already_processed_review_only+upscale_only_gentle`. The `already_processed_name` flag is also set. Markers include `_topaz`, `_upscale`, `_proteus`, `_gaia`, `_deblur`, and similar tokens.

A sharp low-resolution file is labeled as an upscale candidate even when its score is low. Grain and near-black frames are not treated as detail worth upscaling.

## Contact sheets

Contact sheets are off unless you pass `--contact-sheets`.

They are written to `--contact-sheet-folder` for videos whose score is at least `--contact-sheet-min-score`. The file name includes the parent folder, the stem, and a short hash of the full path, so two folders with the same name do not overwrite each other. Long titles and flag lists wrap inside the header.

## Tests

```bash
python -m unittest test_scan_video_quality.py
```

The integration test is skipped when FFmpeg is not installed.

## Important notes

These metrics are heuristics. They are not a full quality measurement.

Results can still be affected by:

- Subtitles and hard graphics
- Animation
- Interlacing
- Intentional blur or artistic effects
- Scene content that is naturally soft or very contrasty

Manual review is still the right step before restoring, upscaling, deleting, replacing, or re-encoding a video.

## Privacy

Do not commit private videos, personal media, generated CSV reports, or contact sheets.

The CSV can contain private file paths. Contact sheets contain frames from the videos. This repository should contain the script, tests, documentation, and other safe project files.

## License

This project is licensed under the MIT License.
