"""Recursive video discovery."""

from __future__ import annotations

from pathlib import Path

from vqs.util import VIDEO_EXTENSIONS


def find_video_files(root_folder, skip_dir_names=None) -> list[Path]:
    root = Path(root_folder)
    skip = {name.lower() for name in (skip_dir_names or [])}
    found: list[Path] = []
    if not root.exists():
        return found
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        parts = {part.lower() for part in path.parts}
        if any(part.startswith(".") for part in path.parts):
            continue
        if skip and parts.intersection(skip):
            continue
        found.append(path)
    found.sort(key=lambda item: str(item).lower())
    return found
