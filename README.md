# Video Quality Scanner

A Python tool for scanning video files, estimating quality issues, and generating a CSV report for restoration review.

This script is useful for reviewing large video libraries and finding videos that may need restoration, deblurring, artifact reduction, upscaling, or manual quality checking.

It samples frames from each video, calculates several heuristic quality metrics, assigns a quality issue score, and saves the results to a CSV report.

## Features

- Recursively scans a video folder
- Supports common video formats such as `.mp4`, `.mkv`, `.avi`, `.mov`, `.wmv`, `.mpg`, `.mpeg`, `.m4v`, `.ts`, `.m2ts`, `.webm`, `.flv`, and `.3gp`
- Uses FFprobe to read video metadata
- Uses FFmpeg to extract sample frames
- Measures blur using Laplacian variance
- Estimates compression blockiness
- Estimates high-frequency noise or grain
- Measures brightness and near-black frame ratio
- Calculates bitrate per pixel
- Classifies resolution, blur, and blockiness
- Assigns a quality issue score
- Suggests practical review or restoration recommendations
- Outputs a CSV report
- Optional contact sheet generation for manual review

## Requirements

- Python 3.8 or newer
- FFmpeg
- FFprobe

FFmpeg and FFprobe must be installed and available in your system `PATH`.

Python packages:

- `opencv-python`
- `numpy`
- `pandas`
- `tqdm`

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/video-quality-scanner.git
cd video-quality-scanner
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Make sure FFmpeg is installed:

```bash
ffmpeg -version
ffprobe -version
```

## Usage

Edit the configuration section in `scan_video_quality.py` before running.

Set the folder you want to scan:

```python
ROOT_FOLDER = r"./videos"
```

Set the output CSV filename:

```python
OUTPUT_CSV = "video_quality_report.csv"
```

Run the scanner:

```bash
python scan_video_quality.py
```

The script will scan the selected folder, analyze video files, and generate a CSV report.

## Output

By default, the report is saved as:

```text
video_quality_report.csv
```

The CSV report may include columns such as:

| Column | Description |
|---|---|
| `path` | Full path to the video file |
| `filename` | Video filename |
| `folder` | Parent folder |
| `width` | Video width |
| `height` | Video height |
| `codec` | Video codec |
| `duration` | Duration in seconds |
| `duration_hms` | Duration formatted as `HH:MM:SS` |
| `frame_rate` | Video frame rate |
| `bit_rate` | Video bitrate |
| `size` | File size in bytes |
| `sampled_frames` | Number of frames sampled |
| `avg_blur_score` | Average blur/sharpness score |
| `avg_blockiness_score` | Estimated compression blockiness |
| `avg_noise_score` | Estimated noise or grain |
| `avg_brightness` | Average brightness |
| `avg_black_frame_ratio` | Percentage of near-black pixels |
| `bitrate_per_pixel` | Bitrate divided by pixel count |
| `quality_issue_score` | Overall issue score |
| `resolution_class` | Resolution classification |
| `blur_class` | Blur classification |
| `blockiness_class` | Blockiness classification |
| `issue_flags` | Detected quality issue flags |
| `recommendation` | Suggested review or restoration action |
| `contact_sheet` | Path to generated contact sheet, if enabled |
| `error` | Error message, if analysis failed |

## Quality Issue Score

The `quality_issue_score` is a heuristic score.

Higher scores usually mean the video may be a better candidate for restoration or manual review.

The score considers:

- Low resolution
- Blur or softness
- Compression blockiness
- Low bitrate per pixel

Example interpretation:

| Score Range | Meaning |
|---:|---|
| `90+` | Very high priority review |
| `75-89` | High priority review |
| `55-74` | Medium priority review |
| `35-54` | Low to medium priority review |
| `0-34` | Low priority or probably acceptable |

## Recommendations

The script may assign recommendations such as:

| Recommendation | Meaning |
|---|---|
| `top_priority_restore` | Strong restoration candidate |
| `restore_deblur_artifact_reduce` | Review for deblurring and artifact reduction |
| `restore_or_deblur_then_upscale` | Review for restoration before upscaling |
| `upscale_only_gentle` | Possible gentle upscale candidate |
| `upscale_candidate_very_low_res` | Very low resolution upscale candidate |
| `artifact_reduction_review` | Review for compression artifact reduction |
| `manual_review` | Needs manual checking |
| `low_priority_or_ok` | Low priority or probably acceptable |
| `manual_check_error` | Analysis failed and needs manual checking |

## Contact Sheets

Contact sheets are optional.

To enable them, edit:

```python
CREATE_CONTACT_SHEETS = True
```

Contact sheets are saved to:

```python
CONTACT_SHEET_FOLDER = "contact_sheets"
```

By default, contact sheets are only created for videos with a score greater than or equal to:

```python
CONTACT_SHEET_MIN_SCORE = 55
```

Contact sheets can be useful for quickly reviewing sampled frames without opening each video manually.

## Important Notes

This tool uses heuristic image and video metrics. It does not provide a perfect objective measurement of video quality.

Results can be affected by:

- Dark scenes
- Subtitles
- Animation
- Interlacing
- Film grain
- High-contrast patterns
- Scene changes
- Low-light footage
- Intentional blur or artistic effects

Manual review is recommended before deciding whether a video should be restored, upscaled, deleted, replaced, or re-encoded.

## Privacy Notice

Do not commit private videos, personal media, generated CSV reports, or contact sheets to this repository.

The CSV report may contain private file paths. Contact sheets may contain frames from private videos.

This repository should contain only the script, documentation, and safe project files.

## License

This project is licensed under the MIT License.
