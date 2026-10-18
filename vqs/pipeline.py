"""High-level scan orchestration."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from vqs.analyze import analyze_video
from vqs.config import ScanConfig, resolve_device
from vqs.discover import find_video_files
from vqs.metrics.nr_iqa import get_scorer
from vqs.pairs_run import run_pairs
from vqs.report.html import write_report
from vqs.report.sheets import contact_sheet_filename, create_contact_sheet, create_pair_sheet
from vqs.store import Store
from vqs.util import check_external_tools, file_identity


def _default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _analyze_worker(args):
    path, config = args
    return analyze_video(path, config, do_health=config.run_health)


def scan_library(config: ScanConfig) -> Store:
    check_external_tools()
    if config.workers <= 0:
        config.workers = _default_workers()

    store = Store(config.db_path)
    skip_dirs = {config.contact_sheet_folder, Path(config.report_dir).name, "sheets"}
    files = find_video_files(config.root_folder, skip_dir_names=skip_dirs)
    print(f"Found {len(files)} video(s) under {config.root_folder}")

    pending = []
    resumed = 0
    for path in files:
        try:
            size, mtime = file_identity(path)
        except OSError:
            pending.append(path)
            continue
        if config.resume and store.is_current(str(path), size, mtime):
            existing = store.get_video(str(path))
            # Resume skips fully analyzed rows that already have a decision
            if existing and existing.get("decision"):
                resumed += 1
                continue
        pending.append(path)

    print(f"Resume kept {resumed}; analyzing {len(pending)}")

    device = resolve_device(config.device)
    print(f"NR-IQA device: {device} (model={config.nr_iqa_model})")
    nr_scorer = get_scorer(config.nr_iqa_model, device, config.nr_iqa_batch)

    sheet_dir = Path(config.report_dir) / "sheets"
    if config.create_contact_sheets:
        sheet_dir.mkdir(parents=True, exist_ok=True)

    # ProcessPool for classical+health is expensive with CUDA; run sequential
    # when GPU IQA is wanted so one scorer stays warm. Workers>1 still helps
    # classical-only CPU scans.
    use_pool = config.workers > 1 and device == "cpu"

    done = 0
    if use_pool:
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            futures = {
                pool.submit(analyze_video, str(path), config, do_health=config.run_health): path
                for path in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="scan", unit="video"):
                row = future.result()
                _persist_row(store, row, config, sheet_dir, nr_scorer=None)
                done += 1
                if config.checkpoint_every and done % config.checkpoint_every == 0:
                    print(f"Checkpoint: {done}/{len(pending)}")
    else:
        for path in tqdm(pending, desc="scan", unit="video"):
            existing = store.get_video(str(path))
            row = analyze_video(
                str(path),
                config,
                do_health=config.run_health,
                existing=existing,
                nr_scorer=nr_scorer,
            )
            _persist_row(store, row, config, sheet_dir, nr_scorer=nr_scorer)
            done += 1
            if config.checkpoint_every and done % config.checkpoint_every == 0:
                print(f"Checkpoint: {done}/{len(pending)}")

    if config.run_pairs:
        print("Running pair comparisons…")
        pair_results = run_pairs(store, config)
        if config.create_contact_sheets:
            for result in pair_results:
                sheet_pair = result.pop("_sheet_pair", None) if isinstance(result, dict) else None
                if not sheet_pair:
                    continue
                if result.get("action") == "keep_processed" and result.get("verdict") == "improved":
                    continue
                left, right = sheet_pair
                name = contact_sheet_filename(result.get("other_path") or result.get("original_path"))
                out = sheet_dir / f"pair_{name}"
                path = create_pair_sheet(left, right, out)
                if path:
                    result["contact_sheet"] = path
                    store.upsert_pair({k: v for k, v in result.items() if not k.startswith("_")})

    if config.run_report:
        index = write_report(store, config.report_dir)
        print(f"Report written: {index}")

    _print_summary(store)
    return store


def _persist_row(store: Store, row: dict, config: ScanConfig, sheet_dir: Path, nr_scorer=None) -> None:
    frames = row.pop("_frames", None)
    row.pop("_crop", None)
    if (
        config.create_contact_sheets
        and frames
        and row.get("decision")
        in {
            "topaz_candidate",
            "manual_review",
            "repair_or_remux",
            "check_source_upscaled_or_denoise",
        }
    ):
        try:
            path = create_contact_sheet(
                row["path"],
                frames,
                row,
                sheet_dir,
                thumb_width=config.contact_thumb_width,
            )
            row["contact_sheet"] = path
        except Exception as exc:  # noqa: BLE001
            row["contact_sheet"] = ""
            row["error"] = (row.get("error") or "") or f"contact sheet: {exc}"
    store.upsert_video({k: v for k, v in row.items() if not k.startswith("_")})


def _print_summary(store: Store) -> None:
    videos = store.list_videos()
    pairs = store.list_pairs()
    from collections import Counter

    decisions = Counter(v.get("decision") for v in videos)
    health = Counter(v.get("health_status") for v in videos)
    actions = Counter(p.get("action") for p in pairs)
    print("\n=== Summary ===")
    print(f"Videos: {len(videos)}")
    for key, count in decisions.most_common():
        print(f"  decision {key}: {count}")
    for key, count in health.most_common():
        print(f"  health {key}: {count}")
    if pairs:
        print(f"Pairs: {len(pairs)}")
        for key, count in actions.most_common():
            print(f"  action {key}: {count}")
    top = [v for v in videos if v.get("decision") == "topaz_candidate"]
    top.sort(key=lambda item: float(item.get("confidence") or 0), reverse=True)
    if top:
        print("\nTop Topaz candidates:")
        for item in top[:15]:
            print(
                f"  {item.get('confidence', 0):.2f}  "
                f"{item.get('filename')}  ({item.get('reason')})"
            )
