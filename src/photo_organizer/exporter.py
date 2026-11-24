"""Idempotent crash-safe two-phase physical file exporter."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from photo_organizer.config import IGNORED_SIDECAR_EXTENSIONS, to_win_long_path
from photo_organizer.db import open_connection
from photo_organizer.media import MediaMetadataService
from photo_organizer.worker import compute_sha256_streaming


def cleanup_orphaned_tmp_files(dest_dir: Path) -> int:
    """Remove any leftover .tmp files from an interrupted export."""
    removed = 0
    if not dest_dir.exists():
        return 0
    for candidate in dest_dir.rglob("*.tmp"):
        if candidate.is_file():
            try:
                candidate.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def run_two_phase_export(
    db_path: Path,
    job_id: str,
    dest_dir: Path,
    media_service: MediaMetadataService,
    *,
    crash_after: int | None = None,
) -> dict[str, int]:
    """Export staged media_items: PENDING_EXPORT -> COPIED -> METADATA_EMBEDDED -> VERIFIED."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    cleaned = cleanup_orphaned_tmp_files(dest_dir)

    conn = open_connection(db_path)
    exported = 0
    duplicates = 0
    skipped = 0
    failed = 0

    try:
        rows = conn.execute(
            """
            SELECT id, source_path, ext, sha256, is_duplicate, is_placeholder,
                   target_folder, target_filename, metadata_json, export_state
            FROM media_items
            WHERE job_id = ?
            ORDER BY target_folder ASC, target_filename ASC
            """,
            (job_id,),
        ).fetchall()

        for idx, row in enumerate(rows):
            if row["export_state"] == "VERIFIED":
                skipped += 1
                continue
            if row["is_placeholder"]:
                skipped += 1
                continue
            if str(row["ext"]).lower() in IGNORED_SIDECAR_EXTENSIONS:
                skipped += 1
                continue

            src = Path(row["source_path"])
            if not src.is_file():
                conn.execute(
                    "UPDATE media_items SET export_state = 'FAILED' WHERE id = ?",
                    (row["id"],),
                )
                conn.commit()
                failed += 1
                continue

            target_dir = dest_dir / row["target_folder"]
            target_dir.mkdir(parents=True, exist_ok=True)
            final_path = target_dir / row["target_filename"]
            tmp_path = target_dir / f"{row['target_filename']}.tmp"

            shutil.copy2(to_win_long_path(src), to_win_long_path(tmp_path))

            # Verify copied bytes against source SHA-256
            expected_sha = row["sha256"] or compute_sha256_streaming(src)
            actual_sha = compute_sha256_streaming(tmp_path)
            if actual_sha != expected_sha:
                tmp_path.unlink(missing_ok=True)
                conn.execute(
                    "UPDATE media_items SET export_state = 'FAILED' WHERE id = ?",
                    (row["id"],),
                )
                conn.commit()
                failed += 1
                continue

            conn.execute(
                "UPDATE media_items SET export_state = 'COPIED' WHERE id = ?",
                (row["id"],),
            )
            conn.commit()

            # Embed metadata on destination .tmp copy only (source untouched)
            meta: dict[str, Any] = {}
            if row["metadata_json"]:
                try:
                    meta = json.loads(row["metadata_json"])
                except json.JSONDecodeError:
                    meta = {}
            media_service.embed_metadata(tmp_path, meta)

            conn.execute(
                "UPDATE media_items SET export_state = 'METADATA_EMBEDDED' WHERE id = ?",
                (row["id"],),
            )
            conn.commit()

            if crash_after is not None and idx == crash_after:
                raise RuntimeError("Simulated mid-export crash")

            os.replace(to_win_long_path(tmp_path), to_win_long_path(final_path))
            conn.execute(
                "UPDATE media_items SET export_state = 'VERIFIED' WHERE id = ?",
                (row["id"],),
            )
            conn.commit()

            exported += 1
            if row["is_duplicate"]:
                duplicates += 1
    finally:
        conn.close()

    return {
        "exported": exported,
        "duplicates": duplicates,
        "skipped": skipped,
        "failed": failed,
        "cleaned_tmp": cleaned,
    }
