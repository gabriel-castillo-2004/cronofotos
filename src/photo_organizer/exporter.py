"""Idempotent crash-safe two-phase physical file exporter."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from photo_organizer.config import IGNORED_SIDECAR_EXTENSIONS, to_win_long_path
from photo_organizer.db import open_connection, recompute_filenames
from photo_organizer.media import MediaMetadataService
from photo_organizer.worker import (
    MASTER_INDEX_FILENAME,
    compute_sha256_streaming,
    load_index_file,
    load_library_index,
    save_index_file,
    save_library_index,
)

OMITTED_ROUTE_REASONS: frozenset[str] = frozenset(
    {"already_in_library", "ya respaldado en índice maestro"}
)

PORTABLE_PLAN_FILENAME = "_cronofotos_plan.json"


def save_portable_plan_atomic(plan_path: Path, plan_data: dict[str, Any]) -> Path:
    """Atomically write _cronofotos_plan.json via a .tmp file and os.replace."""
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = plan_path.parent / f"{plan_path.name}.tmp"
    payload = json.dumps(plan_data, ensure_ascii=False, indent=2)
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp_path), str(plan_path))
    return plan_path


def build_portable_plan(db_path: Path, job_id: str) -> dict[str, Any]:
    """Build the portable tree plan dictionary from SQLite without touching media files."""
    conn = open_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT sha256, target_folder, is_duplicate
            FROM media_items
            WHERE job_id = ? AND sha256 IS NOT NULL AND sha256 != ''
            ORDER BY is_duplicate ASC, target_folder ASC, target_filename ASC
            """,
            (job_id,),
        ).fetchall()

        files_map: dict[str, str] = {}
        all_folders: set[str] = set()
        for r in rows:
            sha = str(r["sha256"])
            folder = str(r["target_folder"])
            all_folders.add(folder)
            if sha not in files_map:
                files_map[sha] = folder

        debate_rows = conn.execute(
            """
            SELECT folder, status, history_json
            FROM folder_debate_state
            WHERE job_id = ?
            ORDER BY folder ASC
            """,
            (job_id,),
        ).fetchall()

        folders_state: dict[str, dict[str, Any]] = {}
        debate_history: list[dict[str, Any]] = []
        for dr in debate_rows:
            f_name = str(dr["folder"])
            st = str(dr["status"] or "pending")
            try:
                hist = json.loads(dr["history_json"] or "[]")
                if not isinstance(hist, list):
                    hist = []
            except json.JSONDecodeError:
                hist = []
            folders_state[f_name] = {
                "status": st,
                "state": st,
                "history": hist,
            }
            for entry in hist:
                if isinstance(entry, dict):
                    debate_history.append({"folder": f_name, **entry})

        for f_name in sorted(all_folders):
            if f_name not in folders_state:
                folders_state[f_name] = {
                    "status": "pending",
                    "state": "pending",
                    "history": [],
                }

        return {
            "version": 1,
            "job_id": job_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "files": files_map,
            "folders": folders_state,
            "debate_history": debate_history,
        }
    finally:
        conn.close()


def export_portable_plan(
    db_path: Path,
    job_id: str,
    dest_dir: Path,
) -> dict[str, Any]:
    """Export the full tree plan to dest_dir/_cronofotos_plan.json atomically."""
    plan_data = build_portable_plan(db_path, job_id)
    plan_path = dest_dir / PORTABLE_PLAN_FILENAME
    save_portable_plan_atomic(plan_path, plan_data)
    return {
        "job_id": job_id,
        "plan_path": str(plan_path),
        "file_count": len(plan_data["files"]),
        "folder_count": len(plan_data["folders"]),
        "plan": plan_data,
    }


def _extract_sha_folder_pairs(plan_data: dict[str, Any]) -> dict[str, str]:
    """Extract sha256 -> target_folder mapping from a portable plan dict."""
    mapping: dict[str, str] = {}
    files_obj = plan_data.get("files") or plan_data.get("sha256_map") or plan_data.get("items")
    if isinstance(files_obj, dict):
        for sha, val in files_obj.items():
            if isinstance(val, str) and val.strip():
                mapping[str(sha)] = val.strip()
            elif isinstance(val, dict):
                for key in ("target_folder", "folder", "carpeta_destino", "dest_folder"):
                    if isinstance(val.get(key), str) and val[key].strip():
                        mapping[str(sha)] = val[key].strip()
                        break
    elif isinstance(files_obj, list):
        for item in files_obj:
            if not isinstance(item, dict):
                continue
            sha = item.get("sha256")
            if not sha:
                continue
            for key in ("target_folder", "folder", "carpeta_destino", "dest_folder"):
                if isinstance(item.get(key), str) and item[key].strip():
                    mapping[str(sha)] = item[key].strip()
                    break
    return mapping


def import_portable_plan(
    db_path: Path,
    job_id: str,
    plan_data: dict[str, Any],
) -> dict[str, Any]:
    """Restore sha256->folder mappings, folder states, and debate history into SQLite."""
    sha_map = _extract_sha_folder_pairs(plan_data)
    folders_obj = plan_data.get("folders")
    debate_history = plan_data.get("debate_history")
    now_iso = datetime.now(timezone.utc).isoformat()

    history_by_folder: dict[str, list[dict[str, Any]]] = {}
    if isinstance(debate_history, list):
        for entry in debate_history:
            if isinstance(entry, dict) and entry.get("folder"):
                history_by_folder.setdefault(str(entry["folder"]), []).append(entry)

    updated_items = 0
    touched_folders: set[str] = set()
    folders_synced = 0

    conn = open_connection(db_path)
    try:
        with conn:
            for sha, target_folder in sha_map.items():
                cur = conn.execute(
                    """
                    UPDATE media_items
                    SET target_folder = ?
                    WHERE job_id = ? AND sha256 = ? AND target_folder != ?
                    """,
                    (target_folder, job_id, sha, target_folder),
                )
                if cur.rowcount > 0:
                    updated_items += cur.rowcount
                    touched_folders.add(target_folder)

            for folder in sorted(touched_folders):
                recompute_filenames(conn, job_id, folder)

            if isinstance(folders_obj, dict):
                for folder_name, meta in folders_obj.items():
                    if not isinstance(meta, dict):
                        continue
                    st = str(meta.get("status") or meta.get("state") or "pending").strip().lower()
                    if st not in ("pending", "debated"):
                        st = "pending"
                    hist = meta.get("history")
                    if not isinstance(hist, list):
                        hist = history_by_folder.get(str(folder_name), [])
                    conn.execute(
                        """
                        INSERT INTO folder_debate_state (job_id, folder, status, history_json, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(job_id, folder) DO UPDATE SET
                            status = excluded.status,
                            history_json = excluded.history_json,
                            updated_at = excluded.updated_at
                        """,
                        (job_id, str(folder_name), st, json.dumps(hist, ensure_ascii=False), now_iso),
                    )
                    folders_synced += 1
    finally:
        conn.close()

    return {
        "job_id": job_id,
        "updated_items": updated_items,
        "folders_synced": folders_synced,
    }


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
    batch_size: int = 100,
    solo_primer_lote: bool = False,
    incremental_mode: bool = False,
    master_index_path: str | Path | None = None,
    pause_event: threading.Event | None = None,
    on_batch_progress: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    """Export staged media_items sequentially in batches without mutating source files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    cleaned = cleanup_orphaned_tmp_files(dest_dir)
    lib_index = load_library_index(dest_dir)

    eff_batch_size = batch_size if batch_size and batch_size > 0 else 100
    conn = open_connection(db_path)
    exported = 0
    exported_bytes = 0
    duplicates = 0
    skipped = 0
    failed = 0
    sidecars_merged = 0
    batches_info: list[dict[str, Any]] = []

    try:
        rows = conn.execute(
            """
            SELECT id, source_path, ext, sha256, is_duplicate, is_placeholder,
                   sidecar_path, target_folder, target_filename, metadata_json, export_state, route_reason
            FROM media_items
            WHERE job_id = ?
            ORDER BY target_folder ASC, target_filename ASC
            """,
            (job_id,),
        ).fetchall()

        chunks = [rows[i : i + eff_batch_size] for i in range(0, len(rows), eff_batch_size)]
        total_batches = len(chunks)
        batches_info = [
            {
                "batch_index": idx + 1,
                "total_batches": total_batches,
                "label": f"Lote {idx + 1} de {total_batches}",
                "status": "pendiente",
                "total_items": len(chunk),
                "processed_items": 0,
                "duplicates_skipped": 0,
                "errors": 0,
                "elapsed_seconds": 0.0,
                "error_message": None,
            }
            for idx, chunk in enumerate(chunks)
        ]

        global_idx = -1
        for b_idx, chunk in enumerate(chunks):
            batch_info = batches_info[b_idx]

            # Skip already completed batches when resuming/continuing
            if all(
                r["export_state"] == "VERIFIED"
                or r["route_reason"] in OMITTED_ROUTE_REASONS
                or (incremental_mode and bool(r["is_duplicate"]))
                or r["is_placeholder"]
                or str(r["ext"]).lower() in IGNORED_SIDECAR_EXTENSIONS
                for r in chunk
            ):
                global_idx += len(chunk)
                batch_info["status"] = "completado"
                batch_info["processed_items"] = len(chunk)
                skipped += len(chunk)
                continue

            batch_info["status"] = "en_proceso"
            if on_batch_progress:
                on_batch_progress(batches_info)
            t0 = time.monotonic()
            batch_exported = 0
            batch_errors = 0

            for row in chunk:
                global_idx += 1
                if pause_event is not None:
                    pause_event.wait()

                if (
                    row["export_state"] == "VERIFIED"
                    or row["route_reason"] in OMITTED_ROUTE_REASONS
                    or (incremental_mode and bool(row["is_duplicate"]))
                ):
                    skipped += 1
                    batch_info["processed_items"] += 1
                    continue
                if row["is_placeholder"]:
                    skipped += 1
                    batch_info["processed_items"] += 1
                    continue
                if str(row["ext"]).lower() in IGNORED_SIDECAR_EXTENSIONS:
                    skipped += 1
                    batch_info["processed_items"] += 1
                    continue

                src = Path(row["source_path"])
                if not src.is_file():
                    conn.execute(
                        "UPDATE media_items SET export_state = 'FAILED' WHERE id = ?",
                        (row["id"],),
                    )
                    conn.commit()
                    failed += 1
                    batch_errors += 1
                    batch_info["errors"] = batch_errors
                    batch_info["error_message"] = f"Archivo origen no encontrado: {src}"
                    batch_info["processed_items"] += 1
                    continue

                effective_target_folder = str(row["target_folder"])
                if incremental_mode and effective_target_folder.startswith("01_Recuerdos/"):
                    effective_target_folder = effective_target_folder[len("01_Recuerdos/") :]
                target_dir = dest_dir / effective_target_folder
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
                    batch_errors += 1
                    batch_info["errors"] = batch_errors
                    batch_info["error_message"] = f"SHA-256 no coincide para {src.name}"
                    batch_info["processed_items"] += 1
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

                if crash_after is not None and global_idx == crash_after:
                    raise RuntimeError("Simulated mid-export crash")

                os.replace(to_win_long_path(tmp_path), to_win_long_path(final_path))
                conn.execute(
                    "UPDATE media_items SET export_state = 'VERIFIED' WHERE id = ?",
                    (row["id"],),
                )
                conn.commit()

                file_size = final_path.stat().st_size if final_path.exists() else 0
                exported += 1
                exported_bytes += file_size
                batch_exported += 1
                batch_info["processed_items"] += 1
                if row["sidecar_path"]:
                    sidecars_merged += 1
                if row["is_duplicate"]:
                    duplicates += 1
                    batch_info["duplicates_skipped"] += 1
                else:
                    lib_index[expected_sha] = {
                        "rel_path": f"{effective_target_folder}/{row['target_filename']}",
                        "size_bytes": file_size,
                    }

            batch_info["elapsed_seconds"] = round(time.monotonic() - t0, 3)
            batch_info["status"] = "error" if (batch_errors > 0 and batch_exported == 0) else "completado"
            if on_batch_progress:
                on_batch_progress(batches_info)

            if solo_primer_lote and b_idx == 0 and total_batches > 1:
                break
    finally:
        conn.close()
        try:
            save_library_index(dest_dir, lib_index)
        except OSError:
            pass
        try:
            master_path = db_path.parent / MASTER_INDEX_FILENAME
            master_index = load_index_file(master_path)
            if master_index_path:
                custom_p = Path(str(master_index_path).strip())
                if custom_p.is_file():
                    master_index.update(load_index_file(custom_p))
            master_index.update(lib_index)
            save_index_file(master_path, master_index)
        except OSError:
            pass

    return {
        "exported": exported,
        "exported_bytes": exported_bytes,
        "exported_gb": round(exported_bytes / (1024 ** 3), 4),
        "duplicates": duplicates,
        "skipped": skipped,
        "failed": failed,
        "sidecars_merged": sidecars_merged,
        "cleaned_tmp": cleaned,
        "batch_size": eff_batch_size,
        "solo_primer_lote": solo_primer_lote,
        "batches": batches_info,
    }
