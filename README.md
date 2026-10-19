# Video Quality Scanner v2

Scan a video library (including a Windows mapped SMB/NAS share), find files that are **worth sending to Topaz**, judge whether **already-upscaled siblings are worth keeping**, and run a **full decode health check**.

v2 replaces the old single-file heuristic CSV tool with:

- Modular Python package (`vqs`)
- SQLite database + HTML review report
- Classical artifact metrics (blur, blockiness, grain, halo, combing)
- Optional **GPU no-reference IQA** (TOPIQ-NR via `pyiqa`) for Topaz worthiness
- Optional **LPIPS** + bicubic baseline for keep/delete on processed pairs
- Dedicated FFmpeg health pass

The legacy script [`scan_video_quality.py`](scan_video_quality.py) remains for reference and is **deprecated**.

## Requirements

- Windows 11 (or Linux/macOS)
- Python 3.11+
- FFmpeg + FFprobe on `PATH`
- For GPU accuracy (recommended on RTX 5070 Ti): CUDA PyTorch + `pyiqa` + `lpips`

## Install (Windows)

```powershell
cd video-quality-scanner
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .

# GPU stack (install a CUDA wheel that matches your driver first):
# https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pyiqa lpips
```

If CUDA PyTorch fails on a 50-series card, v2 still runs on CPU classical metrics and records `nr_iqa_unavailable`. Pair LPIPS also falls back to classical-only verdicts.

Core-only:

```powershell
pip install -r requirements.txt
pip install -e .
```

## Quick start

```powershell
# Map your NAS, then:
python -m vqs scan --root Z:\Videos --db D:\vqs\vqs.db --out D:\vqs\report --device cuda --cache-dir D:\vqs-cache
```

Open `D:\vqs\report\index.html` in a browser. Use the tabs:

- **Topaz candidates** — ranked by confidence
- **Keep processed / Keep original** — pair verdicts
- **Corrupt / repair** — decode health failures
- **Export filtered CSV** — for building a Topaz queue

### Commands

```powershell
python -m vqs scan --root Z:\Videos --db vqs.db --device cuda
python -m vqs health --root Z:\Videos --db vqs.db
python -m vqs pairs --db vqs.db
python -m vqs report --db vqs.db --out report
python -m vqs calibrate --quick
python -m vqs calibrate --db vqs.db --labels labels.csv
```

| Flag | Meaning |
|---|---|
| `--root` | Library root (recursive) |
| `--db` | SQLite path (prefer local SSD, not the NAS) |
| `--out` | HTML/CSV report folder |
| `--device` | `auto` / `cuda` / `cpu` |
| `--workers` | Parallel workers for probe, health, sampling, and classical metrics. With CUDA, NR-IQA and LPIPS stay in the parent process |
| `--samples` | Fixed frame count (default adaptive 16–40) |
| `--cache-dir` | Reserved local cache directory for NAS-heavy runs |
| `--skip-health` / `--skip-pairs` / `--skip-report` | Run a subset |
| `--no-resume` | Ignore prior DB rows |
| `--config` | YAML threshold overrides ([config.example.yaml](config.example.yaml)) |

## What “accurate” means in v2

### Worth Topaz?

A file becomes `topaz_candidate` only when:

1. Health is not `failed`
2. There is room to improve (content short side ≤ 1080, or soft/blocky HD)
3. Classical metrics show recoverable softness and/or blocking
4. Optional NR-IQA is poor (or classical-only if GPU models are unavailable)
5. Grain-dominated “fake sharpness”, mostly-dark samples, VR, and interlacing are filtered or sent to `manual_review`

Each row has `decision`, `confidence`, and a short `reason` — not an opaque 0–100 soup.

### Keep the upscale?

Same-folder, duration-matched, stem-related pairs (e.g. `show.mp4` + `show_topaz.mp4`):

1. Align frames (SSIM gate + small search)
2. **Fidelity**: LPIPS of processed-downscaled vs original (when available)
3. **Uplift**: sharper / better NR-IQA than a **bicubic upscale of the original**
4. Damage flags: lost detail, more blocky, more noise, oversharpen, fidelity loss

Actions: `keep_processed`, `keep_original`, `manual_review`.

### Health

```text
ffmpeg -v error -xerror -i FILE -f null -
```

Statuses: `ok`, `errors`, `failed`. Corrupt files are never recommended for Topaz.

## Accuracy workflow (do this on your library)

Heuristics alone will never match your taste. After the first scan:

1. Label ~50–100 files in `labels.csv` (see [labels.example.csv](labels.example.csv)):
   - `worth_topaz` / `not_worth` / `repair`
   - `keep_processed` / `keep_original` / `manual_review`
2. Run:

```powershell
python -m vqs calibrate --db D:\vqs\vqs.db --labels labels.csv
```

3. Adjust thresholds in a YAML config from the printed suggestions, rescan changed rows with `--no-resume` or after touching files.

## NAS / SMB notes

- Keep `--db` and `--out` on the **local SSD** (`D:\…`), not the share.
- Antivirus real-time scanning of the mapped drive often dominates runtime.
- Prefer a mapped drive letter over a UNC path if tools mis-handle `\\server\share`.
- Full health decode reads the whole file once; quality sampling seeks ~16–40 times.

## Migrating from v1

| v1 | v2 |
|---|---|
| `scan_video_quality.py` | `python -m vqs scan` |
| CSV report | SQLite + `report/index.html` (+ CSV export) |
| `quality_issue_score` | `decision` + `confidence` + `reason` |
| Sample decode errors only | Dedicated `health` pass |
| Laplacian/block heuristics only | Classical + optional TOPIQ-NR / LPIPS |
| schema 3 CSV resume | schema 4 SQLite resume |

v1 reports are not resumed into v2. Start a fresh `--db`.

## Tests

```powershell
python -m vqs calibrate --quick
python -m unittest discover -s tests -v
```

Integration tests need FFmpeg.

## Privacy

Do not commit private videos, personal paths, generated databases, reports, or contact sheets.

## License

MIT
