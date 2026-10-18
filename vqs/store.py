"""SQLite persistence for resume, query, and HTML export."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from vqs import SCHEMA_VERSION

VIDEO_COLUMNS = [
    "path",
    "filename",
    "folder",
    "size",
    "mtime",
    "schema_version",
    "width",
    "height",
    "display_width",
    "display_height",
    "content_width",
    "content_height",
    "content_aspect",
    "crop",
    "fix_geometry",
    "sample_aspect_ratio",
    "rotation",
    "codec",
    "pix_fmt",
    "encoder",
    "color_range",
    "color_space",
    "field_order",
    "duration",
    "duration_hms",
    "frame_rate",
    "bit_rate",
    "is_probably_processed",
    "is_vr",
    "health_status",
    "health_error_count",
    "health_first_error",
    "health_progress",
    "sampled_frames",
    "frames_used",
    "frames_excluded",
    "sample_errors",
    "sample_error",
    "avg_blur",
    "median_blur",
    "blur_p10",
    "blur_p90",
    "blur_iqr",
    "blur_cv",
    "blur_scale_ratio",
    "blur_best_tiles",
    "avg_block",
    "avg_noise",
    "median_noise",
    "avg_halo",
    "combing",
    "avg_brightness",
    "avg_black",
    "mostly_dark",
    "nr_iqa_mean",
    "nr_iqa_median",
    "nr_iqa_model",
    "nr_iqa_device",
    "resolution_class",
    "blur_class",
    "blockiness_class",
    "decision",
    "confidence",
    "reason",
    "issue_flags",
    "contact_sheet",
    "error",
    "updated_at",
]

PAIR_COLUMNS = [
    "group_id",
    "folder",
    "original_path",
    "other_path",
    "original_filename",
    "other_filename",
    "duration_delta",
    "geometry_mismatch",
    "blur_native",
    "blur_vs_base",
    "block_native",
    "noise_native",
    "halo_vs_base",
    "ssim",
    "lpips_native",
    "nr_uplift",
    "luma_delta",
    "range_shift",
    "aligned_frames",
    "dropped_frames",
    "verdict",
    "action",
    "best_version",
    "confidence",
    "note",
    "contact_sheet",
    "updated_at",
]


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    path TEXT PRIMARY KEY,
                    filename TEXT,
                    folder TEXT,
                    size INTEGER,
                    mtime INTEGER,
                    schema_version INTEGER,
                    width INTEGER,
                    height INTEGER,
                    display_width INTEGER,
                    display_height INTEGER,
                    content_width INTEGER,
                    content_height INTEGER,
                    content_aspect REAL,
                    crop TEXT,
                    fix_geometry TEXT,
                    sample_aspect_ratio TEXT,
                    rotation REAL,
                    codec TEXT,
                    pix_fmt TEXT,
                    encoder TEXT,
                    color_range TEXT,
                    color_space TEXT,
                    field_order TEXT,
                    duration REAL,
                    duration_hms TEXT,
                    frame_rate REAL,
                    bit_rate INTEGER,
                    is_probably_processed INTEGER,
                    is_vr INTEGER,
                    health_status TEXT,
                    health_error_count INTEGER,
                    health_first_error TEXT,
                    health_progress REAL,
                    sampled_frames INTEGER,
                    frames_used INTEGER,
                    frames_excluded INTEGER,
                    sample_errors INTEGER,
                    sample_error TEXT,
                    avg_blur REAL,
                    median_blur REAL,
                    blur_p10 REAL,
                    blur_p90 REAL,
                    blur_iqr REAL,
                    blur_cv REAL,
                    blur_scale_ratio REAL,
                    blur_best_tiles REAL,
                    avg_block REAL,
                    avg_noise REAL,
                    median_noise REAL,
                    avg_halo REAL,
                    combing REAL,
                    avg_brightness REAL,
                    avg_black REAL,
                    mostly_dark INTEGER,
                    nr_iqa_mean REAL,
                    nr_iqa_median REAL,
                    nr_iqa_model TEXT,
                    nr_iqa_device TEXT,
                    resolution_class TEXT,
                    blur_class TEXT,
                    blockiness_class TEXT,
                    decision TEXT,
                    confidence REAL,
                    reason TEXT,
                    issue_flags TEXT,
                    contact_sheet TEXT,
                    error TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pairs (
                    group_id TEXT,
                    folder TEXT,
                    original_path TEXT,
                    other_path TEXT,
                    original_filename TEXT,
                    other_filename TEXT,
                    duration_delta REAL,
                    geometry_mismatch INTEGER,
                    blur_native REAL,
                    blur_vs_base REAL,
                    block_native REAL,
                    noise_native REAL,
                    halo_vs_base REAL,
                    ssim REAL,
                    lpips_native REAL,
                    nr_uplift REAL,
                    luma_delta REAL,
                    range_shift INTEGER,
                    aligned_frames INTEGER,
                    dropped_frames INTEGER,
                    verdict TEXT,
                    action TEXT,
                    best_version TEXT,
                    confidence REAL,
                    note TEXT,
                    contact_sheet TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (original_path, other_path)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

    def get_video(self, path: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM videos WHERE path = ?", (path,)).fetchone()
            return dict(row) if row else None

    def is_current(self, path: str, size: int, mtime: int) -> bool:
        row = self.get_video(path)
        if not row:
            return False
        return (
            int(row.get("schema_version") or 0) == SCHEMA_VERSION
            and int(row.get("size") or 0) == int(size)
            and int(row.get("mtime") or 0) == int(mtime)
            and not str(row.get("error") or "").strip()
        )

    def upsert_video(self, row: dict[str, Any]) -> None:
        data = {key: row.get(key) for key in VIDEO_COLUMNS}
        data["schema_version"] = data.get("schema_version") or SCHEMA_VERSION
        placeholders = ", ".join("?" for _ in VIDEO_COLUMNS)
        columns = ", ".join(VIDEO_COLUMNS)
        updates = ", ".join(f"{key}=excluded.{key}" for key in VIDEO_COLUMNS if key != "path")
        values = [_sqlite_value(data.get(key)) for key in VIDEO_COLUMNS]
        with self.connect() as conn:
            conn.execute(
                f"INSERT INTO videos ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(path) DO UPDATE SET {updates}",
                values,
            )

    def upsert_videos(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.upsert_video(row)

    def list_videos(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM videos ORDER BY folder, filename").fetchall()
            return [dict(row) for row in rows]

    def clear_pairs(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM pairs")

    def upsert_pair(self, row: dict[str, Any]) -> None:
        data = {key: row.get(key) for key in PAIR_COLUMNS}
        placeholders = ", ".join("?" for _ in PAIR_COLUMNS)
        columns = ", ".join(PAIR_COLUMNS)
        updates = ", ".join(
            f"{key}=excluded.{key}" for key in PAIR_COLUMNS if key not in {"original_path", "other_path"}
        )
        values = [_sqlite_value(data.get(key)) for key in PAIR_COLUMNS]
        with self.connect() as conn:
            conn.execute(
                f"INSERT INTO pairs ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(original_path, other_path) DO UPDATE SET {updates}",
                values,
            )

    def list_pairs(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM pairs ORDER BY folder, original_filename").fetchall()
            return [dict(row) for row in rows]

    def export_json(self, path: str | Path) -> Path:
        out = Path(path)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "videos": self.list_videos(),
            "pairs": self.list_pairs(),
        }
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return out

    def set_meta(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )


def _sqlite_value(value):
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return None
    return value
