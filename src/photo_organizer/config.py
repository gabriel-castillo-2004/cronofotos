"""Runtime configuration, Windows path validation, and OneDrive placeholder detection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
FILE_ATTRIBUTE_OFFLINE = 0x1000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
CLOUD_ONLY_MASK = (
    FILE_ATTRIBUTE_OFFLINE
    | FILE_ATTRIBUTE_RECALL_ON_OPEN
    | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".heic",
        ".heif",
        ".webp",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".dng",
        ".raw",
        ".cr2",
        ".nef",
        ".arw",
    }
)

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mov",
        ".m4v",
        ".avi",
        ".mkv",
        ".3gp",
        ".webm",
        ".mts",
        ".m2ts",
    }
)

MEDIA_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

IGNORED_SIDECAR_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".json",
        ".txt",
        ".xml",
        ".ini",
        ".db",
        ".tmp",
    }
)


def normalize_win_path(path: str | Path) -> Path:
    """Normalize a path, supporting Windows '\\\\?\\' extended-length prefixes."""
    raw = str(path)
    if os.name == "nt":
        # Strip extended-length prefix for standard Path operations, then re-resolve safely.
        if raw.startswith("\\\\?\\UNC\\"):
            raw = "\\\\" + raw[8:]
        elif raw.startswith("\\\\?\\"):
            raw = raw[4:]
    p = Path(raw).expanduser()
    try:
        return p.resolve(strict=False)
    except OSError:
        return Path(os.path.abspath(str(p)))


def to_win_long_path(path: str | Path) -> str:
    """Convert a Windows path to '\\\\?\\' form when length >= 240."""
    resolved = str(normalize_win_path(path))
    if os.name == "nt" and len(resolved) >= 240 and not resolved.startswith("\\\\?\\"):
        if resolved.startswith("\\\\"):
            return "\\\\?\\UNC\\" + resolved[2:]
        return "\\\\?\\" + resolved
    return resolved


def validate_source_dest_pair(source: str | Path, dest: str | Path) -> tuple[Path, Path]:
    """Reject equal, parent, child, or symlink-loop overlapping paths."""
    src_raw = Path(str(source)).expanduser()
    dst_raw = Path(str(dest)).expanduser()

    # Detect explicit symlink loops before resolving
    for candidate in (src_raw, dst_raw):
        if candidate.is_symlink():
            try:
                candidate.resolve(strict=True)
            except RuntimeError as exc:
                raise ValueError(f"Symlink loop detected in path: {candidate}") from exc

    src_res = normalize_win_path(src_raw)
    dst_res = normalize_win_path(dst_raw)

    src_cmp = Path(os.path.normcase(str(src_res)))
    dst_cmp = Path(os.path.normcase(str(dst_res)))

    if src_cmp == dst_cmp:
        raise ValueError("Source and destination cannot be the same path.")
    if src_cmp in dst_cmp.parents:
        raise ValueError("Destination cannot be inside source directory.")
    if dst_cmp in src_cmp.parents:
        raise ValueError("Source cannot be inside destination directory.")

    return src_res, dst_res


def is_onedrive_placeholder(
    path: str | Path,
    file_attributes: int | None = None,
) -> bool:
    """Detect OneDrive/cloud-only placeholder files without opening or hydrating bytes."""
    if file_attributes is None:
        try:
            st = os.lstat(path)
        except OSError:
            return False
        file_attributes = getattr(st, "st_file_attributes", 0)
    is_reparse = bool(file_attributes & FILE_ATTRIBUTE_REPARSE_POINT)
    is_cloud = bool(file_attributes & CLOUD_ONLY_MASK)
    return is_reparse and is_cloud


@dataclass
class OrganizerConfig:
    """Runtime configuration for local photo organizer."""

    db_path: Path = field(default_factory=lambda: Path("data/organizer.db"))
    thumb_cache_dir: Path = field(default_factory=lambda: Path("data/thumbs"))
    exiftool_path: str | None = None
    ffmpeg_path: str | None = None
    time_gap_hours: float = 4.0
    gps_distance_km: float = 25.0
    max_workers: int = 4
    gemini_model: str = "gemini-2.5-flash"
    host: str = "127.0.0.1"
    port: int = 8000
