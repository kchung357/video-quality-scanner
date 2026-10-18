"""Command-line interface for Video Quality Scanner v2."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from vqs import __version__
from vqs.calibrate import evaluate_labels, run_calibration
from vqs.config import ScanConfig, load_thresholds, resolve_device
from vqs.discover import find_video_files
from vqs.health import check_video_health, health_row_fields
from vqs.pipeline import scan_library
from vqs.probe import ffprobe_video
from vqs.report.html import write_report
from vqs.store import Store
from vqs.util import check_external_tools, clip_error, file_identity


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vqs",
        description="Video Quality Scanner v2 — Topaz worthiness, pair keep/delete, decode health",
    )
    parser.add_argument("--version", action="version", version=f"vqs {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Full library scan (health + solo + pairs + report)")
    _add_common(scan)
    scan.add_argument("--skip-health", action="store_true", help="Skip full decode health pass")
    scan.add_argument("--skip-solo", action="store_true", help="Skip quality / Topaz decisions")
    scan.add_argument("--skip-pairs", action="store_true", help="Skip original/processed compares")
    scan.add_argument("--skip-report", action="store_true", help="Skip HTML report generation")
    scan.add_argument("--no-contact-sheets", action="store_true")
    scan.add_argument("--skip-processed", action="store_true")
    scan.add_argument("--no-resume", action="store_true")

    health = sub.add_parser("health", help="Decode-health pass only")
    _add_common(health, need_root=True)

    pairs = sub.add_parser("pairs", help="Pair compare using existing DB rows")
    _add_common(pairs, need_root=False)

    report = sub.add_parser("report", help="Rebuild HTML/CSV report from DB")
    report.add_argument("--db", default="vqs.db")
    report.add_argument("--out", default="report", dest="report_dir")

    calibrate = sub.add_parser("calibrate", help="Synthetic checks and/or labels precision/recall")
    calibrate.add_argument("--quick", action="store_true", default=True)
    calibrate.add_argument("--labels", default="", help="labels.csv path")
    calibrate.add_argument("--db", default="vqs.db")

    return parser


def _add_common(parser: argparse.ArgumentParser, need_root: bool = True) -> None:
    if need_root:
        parser.add_argument("--root", default="./videos", help="Library root (mapped SMB drive OK)")
    parser.add_argument("--db", default="vqs.db", help="SQLite database path")
    parser.add_argument("--out", default="report", dest="report_dir", help="HTML report folder")
    parser.add_argument("--config", default="", help="YAML threshold overrides")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--nr-iqa-model", default="topiq_nr")
    parser.add_argument("--nr-iqa-batch", type=positive_int, default=8)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--samples", type=positive_int, default=None)
    parser.add_argument("--cache-dir", default="", help="Optional local SSD cache directory")
    parser.add_argument("--frame-timeout", type=float, default=60.0)
    parser.add_argument("--probe-timeout", type=float, default=30.0)
    parser.add_argument("--health-timeout", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=positive_int, default=5)


def config_from_args(args) -> ScanConfig:
    thresholds = load_thresholds(args.config) if getattr(args, "config", "") else None
    samples = getattr(args, "samples", None)
    cfg = ScanConfig(
        root_folder=getattr(args, "root", "./videos"),
        db_path=getattr(args, "db", "vqs.db"),
        report_dir=getattr(args, "report_dir", "report"),
        create_contact_sheets=not getattr(args, "no_contact_sheets", False),
        sample_count=samples or 16,
        samples_explicit=samples is not None,
        skip_already_processed=getattr(args, "skip_processed", False),
        workers=getattr(args, "workers", 4),
        resume=not getattr(args, "no_resume", False),
        frame_timeout=getattr(args, "frame_timeout", 60.0),
        probe_timeout=getattr(args, "probe_timeout", 30.0),
        health_timeout=getattr(args, "health_timeout", 0.0),
        checkpoint_every=getattr(args, "checkpoint_every", 5),
        run_health=not getattr(args, "skip_health", False),
        run_solo=not getattr(args, "skip_solo", False),
        run_pairs=not getattr(args, "skip_pairs", False),
        run_report=not getattr(args, "skip_report", False),
        device=getattr(args, "device", "auto"),
        nr_iqa_model=getattr(args, "nr_iqa_model", "topiq_nr"),
        nr_iqa_batch=getattr(args, "nr_iqa_batch", 8),
        cache_dir=getattr(args, "cache_dir", ""),
        thresholds=thresholds or ScanConfig().thresholds,
    )
    return cfg


def cmd_health(args) -> int:
    check_external_tools()
    store = Store(args.db)
    files = find_video_files(args.root)
    print(f"Health-checking {len(files)} file(s)…")
    from tqdm import tqdm

    for path in tqdm(files, unit="video"):
        try:
            size, mtime = file_identity(path)
            meta = ffprobe_video(path, args.probe_timeout)
            result = check_video_health(path, meta.get("duration") or 0.0, args.health_timeout)
            row = {
                "path": str(path),
                "filename": path.name,
                "folder": str(path.parent),
                "size": size,
                "mtime": mtime,
                **meta,
                **health_row_fields(result),
            }
            existing = store.get_video(str(path)) or {}
            existing.update(row)
            if not existing.get("decision"):
                if result.status == "failed":
                    existing["decision"] = "repair_or_remux"
                    existing["confidence"] = 0.95
                    existing["reason"] = "decode health failed"
                elif not existing.get("decision"):
                    existing["decision"] = existing.get("decision") or ""
            store.upsert_video(existing)
        except Exception as exc:  # noqa: BLE001
            store.upsert_video(
                {
                    "path": str(path),
                    "filename": path.name,
                    "folder": str(path.parent),
                    "health_status": "failed",
                    "health_first_error": clip_error(exc),
                    "decision": "repair_or_remux",
                    "confidence": 0.9,
                    "reason": "health/probe error",
                    "error": clip_error(exc),
                }
            )
    print(f"Done. Database: {args.db}")
    return 0


def cmd_pairs(args) -> int:
    from vqs.pairs_run import run_pairs

    check_external_tools()
    cfg = config_from_args(args)
    cfg.run_pairs = True
    store = Store(cfg.db_path)
    run_pairs(store, cfg)
    if not getattr(args, "skip_report", False):
        write_report(store, cfg.report_dir)
    return 0


def cmd_report(args) -> int:
    store = Store(args.db)
    index = write_report(store, args.report_dir)
    print(f"Wrote {index}")
    return 0


def cmd_calibrate(args) -> int:
    code = run_calibration(quick=True)
    if args.labels:
        code = evaluate_labels(args.db, args.labels) or code
    return code


def cmd_scan(args) -> int:
    cfg = config_from_args(args)
    print(f"vqs {__version__}  device={resolve_device(cfg.device)}")
    scan_library(cfg)
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "scan": cmd_scan,
        "health": cmd_health,
        "pairs": cmd_pairs,
        "report": cmd_report,
        "calibrate": cmd_calibrate,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
