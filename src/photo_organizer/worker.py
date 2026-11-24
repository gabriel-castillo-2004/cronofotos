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


VALID_BATCH_SIZES = (50, 100, 250, 500)
DEFAULT_BATCH_SIZE = 100


def normalize_batch_size(batch_size: int | None) -> int:
    """Return a valid positive batch size, defaulting to 100."""
    if batch_size is None:
        return DEFAULT_BATCH_SIZE
    val = int(batch_size)
    return val if val > 0 else DEFAULT_BATCH_SIZE


def compute_sha256_streaming(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash using bounded streaming reads."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class _DiscoveredMedia:
    """Global read-only scan entry prior to batched metadata extraction."""

    item_id: str
    path: Path
    ext: str
    size_bytes: int
    is_placeholder: bool
    sha256: str | None = None
    is_duplicate: bool = False


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
    batch_size: int = DEFAULT_BATCH_SIZE
    solo_primer_lote: bool = False
    batches: list[dict[str, Any]] = field(default_factory=list)
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
            "batch_size": self.batch_size,
            "solo_primer_lote": self.solo_primer_lote,
            "batches": self.batches,
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
        batch_size: int = DEFAULT_BATCH_SIZE,
        solo_primer_lote: bool = False,
    ) -> str:
        """Execute or resume a scan+stage job with global size+SHA-256 dedup and batching."""
        src, dst = validate_source_dest_pair(source_dir, dest_dir)
        jid = job_id or uuid.uuid4().hex
        eff_batch_size = normalize_batch_size(batch_size)
        now = datetime.now(timezone.utc).isoformat()
        start_ts = time.monotonic()

        self._current_snapshot = ProgressSnapshot(
            job_id=jid,
            status="scanning",
            stage="discovery",
            batch_size=eff_batch_size,
            solo_primer_lote=solo_primer_lote,
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
            # Stage 1: Fast read-only global walk + size-first SHA-256 deduplication
            raw_paths: list[Path] = []
            for root, _dirs, files in os.walk(src):
                root_p = Path(root)
                for fname in sorted(files):
                    fpath = root_p / fname
                    ext = fpath.suffix.lower()
                    if ext in IGNORED_SIDECAR_EXTENSIONS:
                        continue
                    if ext in MEDIA_EXTENSIONS or ext:
                        raw_paths.append(fpath)

            self._current_snapshot.total_files = len(raw_paths)
            self._current_snapshot.stage = "hashing"
            self._emit(force=True)

            discovered: list[_DiscoveredMedia] = []
            size_groups: dict[int, list[_DiscoveredMedia]] = {}
            for fpath in raw_paths:
                item_id = uuid.uuid4().hex
                ext = fpath.suffix.lower()
                if is_onedrive_placeholder(fpath):
                    discovered.append(
                        _DiscoveredMedia(
                            item_id=item_id,
                            path=fpath,
                            ext=ext,
                            size_bytes=0,
                            is_placeholder=True,
                        )
                    )
                else:
                    sz = fpath.stat().st_size
                    entry = _DiscoveredMedia(
                        item_id=item_id,
                        path=fpath,
                        ext=ext,
                        size_bytes=sz,
                        is_placeholder=False,
                    )
                    discovered.append(entry)
                    size_groups.setdefault(sz, []).append(entry)

            loop = asyncio.get_running_loop()
            seen_hashes: set[str] = set()
            global_duplicates = 0

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                # Only hash files that share a file size with at least one other file
                for sz, group in size_groups.items():
                    if len(group) <= 1:
                        continue
                    for entry in group:
                        await self._pause_event.wait()
                        sha = await loop.run_in_executor(pool, compute_sha256_streaming, entry.path)
                        entry.sha256 = sha
                        if sha in seen_hashes:
                            entry.is_duplicate = True
                            global_duplicates += 1
                        else:
                            seen_hashes.add(sha)

                self._current_snapshot.duplicates_count = global_duplicates

                # Split discovered items into configurable batches
                chunks = [
                    discovered[i : i + eff_batch_size]
                    for i in range(0, len(discovered), eff_batch_size)
                ]
                total_batches = len(chunks)
                batches_info: list[dict[str, Any]] = [
                    {
                        "batch_index": idx + 1,
                        "total_batches": total_batches,
                        "label": f"Lote {idx + 1} de {total_batches}",
                        "status": "pendiente",
                        "total_items": len(chunk),
                        "processed_items": 0,
                        "duplicates_skipped": sum(1 for c in chunk if c.is_duplicate),
                        "errors": 0,
                        "elapsed_seconds": 0.0,
                        "error_message": None,
                    }
                    for idx, chunk in enumerate(chunks)
                ]
                self._current_snapshot.batches = batches_info
                self._emit(force=True)

                dir_json_cache: dict[Path, dict[str, Path]] = {}
                cluster_inputs: list[ClusterItem] = []
                processed_total = 0

                for b_idx, chunk in enumerate(chunks):
                    batch_info = batches_info[b_idx]
                    batch_info["status"] = "en_proceso"
                    batch_start = time.monotonic()
                    self._current_snapshot.stage = "metadata"
                    self._emit(force=True)

                    for entry in chunk:
                        await self._pause_event.wait()
                        fpath = entry.path
                        ext = entry.ext
                        item_id = entry.item_id

                        if entry.is_placeholder:
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
                            processed_total += 1
                            batch_info["processed_items"] += 1
                            batch_info["elapsed_seconds"] = round(time.monotonic() - batch_start, 3)
                            self._current_snapshot.processed_files = processed_total
                            self._emit()
                            continue

                        # Stage 2: Sidecar resolution & EXIF/QuickTime extraction
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
                            sha256=entry.sha256 or "",
                            is_duplicate=entry.is_duplicate,
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
                                entry.size_bytes,
                                entry.sha256,
                                1 if entry.is_duplicate else 0,
                                merged.get("captured_at"),
                                merged.get("latitude"),
                                merged.get("longitude"),
                                sidecar_res.get("sidecar_path"),
                                json.dumps(merged),
                                f"__staging_{item_id}{ext}",
                            ),
                        )
                        processed_total += 1
                        batch_info["processed_items"] += 1
                        batch_info["elapsed_seconds"] = round(time.monotonic() - batch_start, 3)
                        elapsed = max(0.001, time.monotonic() - start_ts)
                        self._current_snapshot.processed_files = processed_total
                        self._current_snapshot.items_per_sec = processed_total / elapsed
                        self._emit()

                    writer.flush()
                    batch_info["elapsed_seconds"] = round(time.monotonic() - batch_start, 3)
                    batch_info["status"] = "completado"

                    if solo_primer_lote and b_idx == 0 and total_batches > 1:
                        break

            writer.flush()

            # Stage 3: Temporal/spatial clustering + deterministic naming for processed items
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

            has_pending = any(b["status"] == "pendiente" for b in self._current_snapshot.batches)
            final_status = "partial" if (solo_primer_lote and has_pending) else "completed"
            self._current_snapshot.manual_review_count = manual_count
            self._current_snapshot.status = final_status
            self._current_snapshot.stage = final_status
            writer.enqueue(
                """
                UPDATE jobs
                SET status = ?, stage = ?, total_files = ?,
                    processed_files = ?, duplicates_count = ?, manual_review_count = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    final_status,
                    final_status,
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
