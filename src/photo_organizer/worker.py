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

from photo_organizer.ai_classifier import DEFAULT_AGY_MODEL, classify_with_agy
from photo_organizer.clustering import ClusterItem, cluster_and_stage_items, offline_reverse_geocode
from photo_organizer.config import (
    IGNORED_SIDECAR_EXTENSIONS,
    MEDIA_EXTENSIONS,
    is_onedrive_placeholder,
    validate_source_dest_pair,
)
from photo_organizer.db import SingleWriterQueue, init_db, open_connection, recompute_filenames
from photo_organizer.media import MediaMetadataService
from photo_organizer.sidecar import build_json_index, merge_metadata, resolve_sidecar


INDEX_FILENAME = "_cronofotos_index.json"
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


def save_library_index(dest_dir: Path, index: dict[str, dict[str, Any]]) -> None:
    """Persist _cronofotos_index.json atomically inside dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    index_path = dest_dir / INDEX_FILENAME
    tmp_path = dest_dir / f"{INDEX_FILENAME}.tmp"
    tmp_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, index_path)


def load_library_index(dest_dir: Path) -> dict[str, dict[str, Any]]:
    """Load _cronofotos_index.json and index any pre-existing files in dest_dir."""
    index: dict[str, dict[str, Any]] = {}
    if not dest_dir.exists():
        return index

    index_path = dest_dir / INDEX_FILENAME
    if index_path.is_file():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                index = loaded
        except (OSError, json.JSONDecodeError):
            index = {}

    known_paths = {
        str(v.get("rel_path")): (sha, int(v.get("size_bytes", -1)))
        for sha, v in index.items()
        if isinstance(v, dict) and v.get("rel_path")
    }
    dirty = False
    for existing in sorted(dest_dir.rglob("*")):
        if not existing.is_file():
            continue
        if existing.name in {INDEX_FILENAME, f"{INDEX_FILENAME}.tmp"} or existing.suffix == ".tmp":
            continue
        if "_Duplicados" in existing.parts:
            continue
        rel = existing.relative_to(dest_dir).as_posix()
        sz = existing.stat().st_size
        prev = known_paths.get(rel)
        if prev is not None and prev[1] == sz:
            continue
        sha = compute_sha256_streaming(existing)
        index[sha] = {
            "rel_path": rel,
            "size_bytes": sz,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        dirty = True

    if dirty or not index_path.is_file():
        save_library_index(dest_dir, index)
    return index


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
    already_in_dest: bool = False


@dataclass
class ProgressSnapshot:
    """Aggregate job progress emitted at most every 250ms (4 Hz)."""

    job_id: str
    status: str
    stage: str
    phase_label: str = "1/4 Depurando duplicados"
    total_files: int = 0
    processed_files: int = 0
    new_count: int = 0
    duplicates_count: int = 0
    screenshots_count: int = 0
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
            "phase_label": self.phase_label,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "new_count": self.new_count,
            "duplicates_count": self.duplicates_count,
            "screenshots_count": self.screenshots_count,
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
        antigravity_cli: str = "agy",
        ai_model: str = DEFAULT_AGY_MODEL,
    ) -> None:
        self.db_path = db_path
        self.media_service = media_service
        self.max_workers = max_workers
        self.on_progress = on_progress
        self.antigravity_cli = antigravity_cli
        self.ai_model = ai_model
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
            phase_label="1/4 Depurando duplicados",
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
            self._current_snapshot.phase_label = "1/4 Depurando duplicados"
            self._emit(force=True)

            dest_index = load_library_index(dst)
            existing_dest_hashes = set(dest_index.keys())
            existing_dest_sizes = {
                int(v.get("size_bytes", -1))
                for v in dest_index.values()
                if isinstance(v, dict)
            }

            discovered: list[_DiscoveredMedia] = []
            size_groups: dict[int, list[_DiscoveredMedia]] = {}
            total_raw = len(raw_paths)
            for idx, fpath in enumerate(raw_paths, start=1):
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
                if idx % 25 == 0 or idx == total_raw:
                    self._current_snapshot.processed_files = idx
                    elapsed = max(0.001, time.monotonic() - start_ts)
                    self._current_snapshot.items_per_sec = idx / elapsed
                    self._emit(force=True)

            loop = asyncio.get_running_loop()
            seen_hashes: set[str] = set()
            global_duplicates = 0

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                # Hash files that share a file size with another source file OR with destination library
                for sz, group in size_groups.items():
                    if len(group) <= 1 and sz not in existing_dest_sizes:
                        continue
                    for entry in group:
                        await self._pause_event.wait()
                        sha = await loop.run_in_executor(pool, compute_sha256_streaming, entry.path)
                        entry.sha256 = sha
                        if sha in existing_dest_hashes:
                            entry.already_in_dest = True
                            entry.is_duplicate = True
                            global_duplicates += 1
                        elif sha in seen_hashes:
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
                    self._current_snapshot.phase_label = "2/4 Clasificando fotos y capturas"
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
                        if entry.already_in_dest:
                            writer.enqueue(
                                """
                                INSERT OR REPLACE INTO media_items
                                    (id, job_id, source_path, ext, size_bytes, sha256, is_duplicate,
                                     captured_at, latitude, longitude, sidecar_path, metadata_json,
                                     target_folder, target_filename, route_reason, export_state)
                                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, '_Omitidos_Ya_En_Biblioteca', ?, 'already_in_library', 'PENDING_EXPORT')
                                """,
                                (
                                    item_id,
                                    jid,
                                    str(fpath),
                                    ext,
                                    entry.size_bytes,
                                    entry.sha256,
                                    merged.get("captured_at"),
                                    merged.get("latitude"),
                                    merged.get("longitude"),
                                    sidecar_res.get("sidecar_path"),
                                    json.dumps(merged),
                                    f"__skipped_{item_id}{ext}",
                                ),
                            )
                            processed_total += 1
                            batch_info["processed_items"] += 1
                            batch_info["elapsed_seconds"] = round(time.monotonic() - batch_start, 3)
                            elapsed = max(0.001, time.monotonic() - start_ts)
                            self._current_snapshot.processed_files = processed_total
                            self._current_snapshot.items_per_sec = processed_total / elapsed
                            self._emit()
                            continue

                        has_camera_exif = bool(embedded.get("make") or embedded.get("model"))
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
                            has_camera_exif=has_camera_exif,
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

            # Stage 3: Temporal/spatial clustering + AI folder naming + deterministic naming
            self._current_snapshot.stage = "clustering"
            self._current_snapshot.phase_label = "3/4 Nombrando carpetas con IA"
            self._emit(force=True)
            staged = cluster_and_stage_items(
                cluster_inputs,
                time_gap_hours=time_gap_hours,
                gps_distance_km=gps_distance_km,
            )

            # Group 01_Recuerdos/YYYY/YYYY-MM - ... folders and name them via agy / fallback
            folder_items: dict[str, list[ClusterItem]] = {}
            for ci in cluster_inputs:
                tf = staged[ci.id]["target_folder"]
                if tf.startswith("01_Recuerdos/"):
                    folder_items.setdefault(tf, []).append(ci)

            folder_renames: dict[str, str] = {}
            for old_folder, group_items in folder_items.items():
                parts = old_folder.split("/")
                if len(parts) < 3:
                    continue
                year_part = parts[1]
                ym_prefix = parts[2].split(" - ")[0]
                first_item = group_items[0]
                orig_parent = Path(first_item.source_path).parent.name
                loc_label = offline_reverse_geocode(first_item.latitude, first_item.longitude)
                gps_txt = (
                    f" | GPS: {loc_label}"
                    if loc_label and not loc_label.startswith("GPS ")
                    else ""
                )
                summary = f"{ym_prefix} | {len(group_items)} fotos | carpeta: {orig_parent}{gps_txt}"
                ai_name = classify_with_agy(
                    summary,
                    cli_command=self.antigravity_cli,
                    model=self.ai_model,
                )
                safe_ai = "".join("_" if c in '<>:"/\\|?*' else c for c in ai_name).strip()
                if safe_ai:
                    folder_renames[old_folder] = f"01_Recuerdos/{year_part}/{ym_prefix} - {safe_ai}"

            if folder_renames:
                for ci in cluster_inputs:
                    old_tf = staged[ci.id]["target_folder"]
                    if old_tf in folder_renames:
                        staged[ci.id]["target_folder"] = folder_renames[old_tf]

            manual_count = self._current_snapshot.manual_review_count
            screenshots_count = 0
            new_count = 0
            for ci in cluster_inputs:
                info = staged[ci.id]
                tf_str = str(info["target_folder"])
                if tf_str.startswith("_Revision_Manual"):
                    manual_count += 1
                elif tf_str.startswith("02_Capturas_y_WhatsApp"):
                    screenshots_count += 1
                if not ci.is_duplicate:
                    new_count += 1
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
                        ci.id,
                    ),
                )
            writer.flush()
            if folder_renames:
                conn_recomp = open_connection(self.db_path)
                try:
                    with conn_recomp:
                        for new_tf in set(folder_renames.values()):
                            recompute_filenames(conn_recomp, jid, new_tf)
                finally:
                    conn_recomp.close()

            has_pending = any(b["status"] == "pendiente" for b in self._current_snapshot.batches)
            final_status = "partial" if (solo_primer_lote and has_pending) else "completed"
            self._current_snapshot.manual_review_count = manual_count
            self._current_snapshot.screenshots_count = screenshots_count
            self._current_snapshot.new_count = new_count
            self._current_snapshot.status = final_status
            self._current_snapshot.stage = final_status
            self._current_snapshot.phase_label = "4/4 Listo"
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
