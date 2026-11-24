"""Resumable async pipeline orchestrator with pause/resume and 4 Hz progress snapshots."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from photo_organizer.clustering import ClusterItem, cluster_and_stage_items
from photo_organizer.config import (
    IGNORED_SIDECAR_EXTENSIONS,
    MEDIA_EXTENSIONS,
    is_onedrive_placeholder,
    validate_source_dest_pair,
)
from photo_organizer.db import SingleWriterQueue, init_db, open_connection
from photo_organizer.media import MediaMetadataService
from photo_organizer.sidecar import build_json_index, merge_metadata, resolve_sidecar


def compute_sha256_streaming(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash using bounded streaming reads."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ProgressSnapshot:
    """Aggregate job progress emitted at most every 250ms (4 Hz)."""

    job_id: str
    status: str
    stage: str
    total_files: int = 0
    processed_files: int = 0
    duplicates_count: int = 0
    manual_review_count: int = 0
    items_per_sec: float = 0.0
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "duplicates_count": self.duplicates_count,
            "manual_review_count": self.manual_review_count,
            "items_per_sec": round(self.items_per_sec, 2),
            "updated_at": self.updated_at,
        }


class PipelineOrchestrator:
    """Runs Stage 1 (walk+SHA256), Stage 2 (Takeout+EXIF), and Stage 3 (cluster+stage)."""

    def __init__(
        self,
        db_path: Path,
        media_service: MediaMetadataService,
        max_workers: int = 4,
        on_progress: Callable[[ProgressSnapshot], None] | None = None,
    ) -> None:
        self.db_path = db_path
        self.media_service = media_service
        self.max_workers = max_workers
        self.on_progress = on_progress
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._last_emit_ts = 0.0
        self._current_snapshot = ProgressSnapshot(job_id="", status="idle", stage="init")
        init_db(self.db_path)

    @property
    def current_snapshot(self) -> ProgressSnapshot:
        return self._current_snapshot

    def pause(self) -> None:
        self._pause_event.clear()
        self._current_snapshot.status = "paused"
        self._emit(force=True)

    def resume(self) -> None:
        self._pause_event.set()
        if self._current_snapshot.status == "paused":
            self._current_snapshot.status = "scanning"
            self._emit(force=True)

    def _emit(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_emit_ts) < 0.25:
            return
        self._last_emit_ts = now
        self._current_snapshot.updated_at = datetime.now(timezone.utc).isoformat()
        if self.on_progress:
            self.on_progress(self._current_snapshot)

    async def run_job(
        self,
        source_dir: Path,
        dest_dir: Path,
        *,
        job_id: str | None = None,
        time_gap_hours: float = 4.0,
        gps_distance_km: float = 25.0,
    ) -> str:
        """Execute or resume a scan+stage job."""
        src, dst = validate_source_dest_pair(source_dir, dest_dir)
        jid = job_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        start_ts = time.monotonic()

        self._current_snapshot = ProgressSnapshot(
            job_id=jid,
            status="scanning",
            stage="discovery",
        )
        self._emit(force=True)

        writer = SingleWriterQueue(self.db_path, batch_size=80)
        writer.enqueue(
            """
            INSERT OR REPLACE INTO jobs
                (id, source_dir, dest_dir, status, stage, created_at, updated_at)
            VALUES (?, ?, ?, 'scanning', 'discovery', ?, ?)
            """,
            (jid, str(src), str(dst), now, now),
        )
        writer.flush()

        try:
            # Stage 1: Walk & Streaming SHA-256 deduplication
            media_paths: list[Path] = []
            for root, _dirs, files in os.walk(src):
                root_p = Path(root)
                for fname in sorted(files):
                    fpath = root_p / fname
                    ext = fpath.suffix.lower()
                    if ext in IGNORED_SIDECAR_EXTENSIONS:
                        continue
                    if ext in MEDIA_EXTENSIONS or ext:
                        media_paths.append(fpath)

            self._current_snapshot.total_files = len(media_paths)
            self._current_snapshot.stage = "hashing"
            self._emit(force=True)

            seen_hashes: set[str] = set()
            dir_json_cache: dict[Path, dict[str, Path]] = {}
            cluster_inputs: list[ClusterItem] = []

            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for idx, fpath in enumerate(media_paths, start=1):
                    await self._pause_event.wait()

                    item_id = uuid.uuid4().hex
                    ext = fpath.suffix.lower()
                    if is_onedrive_placeholder(fpath):
                        writer.enqueue(
                            """
                            INSERT OR REPLACE INTO media_items
                                (id, job_id, source_path, ext, size_bytes, is_placeholder,
                                 target_folder, target_filename, route_reason)
                            VALUES (?, ?, ?, ?, 0, 1, '_Revision_Manual', ?, 'onedrive_placeholder')
                            """,
                            (item_id, jid, str(fpath), ext, f"{item_id}{ext}"),
                        )
                        self._current_snapshot.manual_review_count += 1
                        self._current_snapshot.processed_files = idx
                        self._emit()
                        continue

                    size_bytes = fpath.stat().st_size
                    sha = await loop.run_in_executor(pool, compute_sha256_streaming, fpath)
                    is_dup = sha in seen_hashes
                    if not is_dup:
                        seen_hashes.add(sha)
                    else:
                        self._current_snapshot.duplicates_count += 1

                    # Stage 2: Sidecar resolution & EXIF/QuickTime extraction
                    self._current_snapshot.stage = "metadata"
                    parent = fpath.parent
                    if parent not in dir_json_cache:
                        dir_json_cache[parent] = build_json_index(parent)
                    sidecar_res = resolve_sidecar(fpath, dir_json_cache[parent])
                    embedded = await loop.run_in_executor(
                        pool,
                        self.media_service.read_metadata,
                        fpath,
                    )
                    merged = merge_metadata(embedded, sidecar_res.get("metadata", {}))

                    route_override = (
                        "_Revision_Manual"
                        if sidecar_res.get("route") == "_Revision_Manual"
                        else None
                    )
                    cluster_item = ClusterItem(
                        id=item_id,
                        source_path=str(fpath),
                        ext=ext,
                        sha256=sha,
                        is_duplicate=is_dup,
                        captured_at=merged.get("captured_at"),
                        latitude=merged.get("latitude"),
                        longitude=merged.get("longitude"),
                        route_reason=route_override,
                    )
                    cluster_inputs.append(cluster_item)

                    writer.enqueue(
                        """
                        INSERT OR REPLACE INTO media_items
                            (id, job_id, source_path, ext, size_bytes, sha256, is_duplicate,
                             captured_at, latitude, longitude, sidecar_path, metadata_json,
                             target_folder, target_filename, export_state)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '_Staging', ?, 'PENDING_EXPORT')
                        """,
                        (
                            item_id,
                            jid,
                            str(fpath),
                            ext,
                            size_bytes,
                            sha,
                            1 if is_dup else 0,
                            merged.get("captured_at"),
                            merged.get("latitude"),
                            merged.get("longitude"),
                            sidecar_res.get("sidecar_path"),
                            json.dumps(merged),
                            f"__staging_{item_id}{ext}",
                        ),
                    )
                    elapsed = max(0.001, time.monotonic() - start_ts)
                    self._current_snapshot.processed_files = idx
                    self._current_snapshot.items_per_sec = idx / elapsed
                    self._emit()

            writer.flush()

            # Stage 3: Temporal/spatial clustering + deterministic naming
            self._current_snapshot.stage = "clustering"
            self._emit(force=True)
            staged = cluster_and_stage_items(
                cluster_inputs,
                time_gap_hours=time_gap_hours,
                gps_distance_km=gps_distance_km,
            )
            manual_count = self._current_snapshot.manual_review_count
            for item_id, info in staged.items():
                if str(info["target_folder"]).startswith("_Revision_Manual"):
                    manual_count += 1
                writer.enqueue(
                    """
                    UPDATE media_items
                    SET target_folder = ?, target_filename = ?, location_label = ?, route_reason = ?
                    WHERE id = ?
                    """,
                    (
                        info["target_folder"],
                        info["target_filename"],
                        info["location_label"],
                        info["route_reason"],
                        item_id,
                    ),
                )

            self._current_snapshot.manual_review_count = manual_count
            self._current_snapshot.status = "completed"
            self._current_snapshot.stage = "completed"
            writer.enqueue(
                """
                UPDATE jobs
                SET status = 'completed', stage = 'completed', total_files = ?,
                    processed_files = ?, duplicates_count = ?, manual_review_count = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    self._current_snapshot.total_files,
                    self._current_snapshot.processed_files,
                    self._current_snapshot.duplicates_count,
                    self._current_snapshot.manual_review_count,
                    datetime.now(timezone.utc).isoformat(),
                    jid,
                ),
            )
            writer.flush()
            self._emit(force=True)
            return jid
        except Exception as exc:
            self._current_snapshot.status = "failed"
            conn = open_connection(self.db_path)
            try:
                conn.execute(
                    "UPDATE jobs SET status = 'failed', error_message = ?, updated_at = ? WHERE id = ?",
                    (str(exc), datetime.now(timezone.utc).isoformat(), jid),
                )
                conn.commit()
            finally:
                conn.close()
            self._emit(force=True)
            raise
        finally:
            writer.close()
