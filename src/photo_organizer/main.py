"""Local web application bound to 127.0.0.1 with SSE, tree staging CRUD, and SPA.

Uses FastAPI + Uvicorn when installed, and falls back cleanly to a stdlib
ThreadingHTTPServer when optional ASGI dependencies are absent.
"""

from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from email.parser import BytesParser
from email.policy import default as email_default_policy
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from datetime import datetime
from photo_organizer.ai_classifier import (
    DEFAULT_AGY_MODEL,
    GeminiClassifier,
    _load_whitelisted_face_clusters,
    classify_with_agy,
    debate_folder_batch_with_agy,
    generate_tree_ai_suggestions,
    resolve_folder_whitelisted_people,
    suggest_tree_optimizations,
)
from photo_organizer.clustering import offline_reverse_geocode
from photo_organizer.config import (
    AGY_ALLOWED_NAMES,
    IMAGE_EXTENSIONS,
    PEOPLE_IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    OrganizerConfig,
    is_standard_agy_binary,
    normalize_win_path,
    resolve_cli_binary,
    validate_source_dest_pair,
)
from photo_organizer.db import init_db, open_connection, recompute_filenames
from photo_organizer.exporter import (
    PORTABLE_PLAN_FILENAME,
    export_portable_plan,
    import_portable_plan,
    refine_exported_tree,
    run_two_phase_export,
)
from photo_organizer.media import BinaryHealthCheck, MediaMetadataService
from photo_organizer.thumbnails import (
    ThumbnailCache,
    discover_frequent_faces,
    extract_top_face_crops,
    load_face_clusters,
    update_face_cluster,
)
from photo_organizer.worker import PipelineOrchestrator

try:
    from fastapi import FastAPI, HTTPException, Query, Request, Response
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when fastapi is not installed
    FASTAPI_AVAILABLE = False

STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "testserver"}


def _parse_multipart_form(
    content_type: str,
    body: bytes,
) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """Parse multipart/form-data payload using Python stdlib email parser."""
    if "multipart/form-data" not in (content_type or "").lower():
        raise _ApiError(400, "Se requiere Content-Type multipart/form-data")
    header_blob = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("latin-1")
    msg = BytesParser(policy=email_default_policy).parsebytes(header_blob + body)
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    if not msg.is_multipart():
        raise _ApiError(400, "Cuerpo multipart inválido")
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename is not None:
            files[str(name)] = (str(filename), payload)
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[str(name)] = payload.decode(charset, errors="replace")
    return fields, files


class _ApiError(Exception):
    """HTTP error raised by shared route handlers."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _is_allowed_origin(method: str, host_header: str, origin: str | None) -> tuple[bool, str]:
    """Validate Host and Origin headers for mutating HTTP methods."""
    if method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return True, ""
    host = host_header.split(":")[0].lower()
    if host and host not in ALLOWED_HOSTS:
        return False, "Forbidden Host"
    if origin:
        origin_host = (urlparse(origin).hostname or "").lower()
        if origin_host not in ALLOWED_HOSTS:
            return False, "Forbidden Origin"
    return True, ""


class _LocalAppCore:
    """Shared controller for FastAPI and stdlib HTTP server backends."""

    def __init__(self, cfg: OrganizerConfig) -> None:
        self.cfg = cfg
        init_db(cfg.db_path)
        self.media_service = MediaMetadataService(cfg.exiftool_path, cfg.ffmpeg_path)
        self.thumbs = ThumbnailCache(cfg.thumb_cache_dir, ffmpeg_bin=self.media_service.ffmpeg_bin)
        self.orchestrator = PipelineOrchestrator(
            cfg.db_path,
            self.media_service,
            max_workers=cfg.max_workers,
            antigravity_cli=cfg.antigravity_cli or "agy",
            ai_model=cfg.gemini_model or DEFAULT_AGY_MODEL,
        )
        self.state: dict[str, Any] = {
            "job_task": None,
            "active_job_id": None,
            "antigravity_cli": cfg.antigravity_cli or "agy",
            "model": cfg.gemini_model or DEFAULT_AGY_MODEL,
        }
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()
        self._ai_cancel_event = threading.Event()
        self._face_discover_lock = threading.Lock()
        self._face_discover_state: dict[str, Any] = {
            "status": "idle",
            "job_id": None,
            "processed": 0,
            "total": 0,
            "clusters_found": 0,
            "error": None,
        }

    def _ensure_bg_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop is None or not self._loop.is_running():
                loop = asyncio.new_event_loop()
                ready = threading.Event()

                def _run_loop() -> None:
                    asyncio.set_event_loop(loop)
                    ready.set()
                    loop.run_forever()

                t = threading.Thread(target=_run_loop, daemon=True)
                t.start()
                ready.wait(timeout=5.0)
                self._loop = loop
                self._loop_thread = t
            return self._loop

    def health(self) -> dict[str, Any]:
        status = BinaryHealthCheck.check(self.cfg.exiftool_path, self.cfg.ffmpeg_path)
        agy_cmd = self.state.get("antigravity_cli") or "agy"
        agy_path = resolve_cli_binary(agy_cmd, allowed_names=AGY_ALLOWED_NAMES)
        agy_ok = bool(agy_path)
        model = self.state.get("model") or DEFAULT_AGY_MODEL
        return {
            "exiftool_available": status.exiftool_available,
            "exiftool_path": status.exiftool_path,
            "ffmpeg_available": status.ffmpeg_available,
            "ffmpeg_path": status.ffmpeg_path,
            "agy_ok": agy_ok,
            "ai_ok": agy_ok,
            "agy_path": agy_path,
            "antigravity_cli": agy_cmd,
            "ai_model": model,
            "model": model,
            "warnings": status.warnings,
        }

    def get_settings(self) -> dict[str, Any]:
        return {
            "exiftool_path": self.cfg.exiftool_path or "",
            "ffmpeg_path": self.cfg.ffmpeg_path or "",
            "antigravity_cli": self.state["antigravity_cli"],
            "model": self.state["model"],
            "gemini_model": self.state["model"],
            "ai_configured": True,
        }

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("exiftool_path") is not None:
            self.cfg.exiftool_path = str(payload["exiftool_path"]).strip() or None
        if payload.get("ffmpeg_path") is not None:
            self.cfg.ffmpeg_path = str(payload["ffmpeg_path"]).strip() or None
        cli_cmd = payload.get("antigravity_cli")
        cli_warning: str | None = None
        if cli_cmd is not None:
            cleaned_cli = str(cli_cmd).strip() or "agy"
            if not is_standard_agy_binary(cleaned_cli):
                cli_warning = (
                    "El comando Antigravity CLI difiere del ejecutable estándar 'agy' "
                    "(agy.exe/.cmd/.bat) y no se ejecutará."
                )
            self.state["antigravity_cli"] = cleaned_cli
        model = str(payload.get("model") or payload.get("gemini_model") or "").strip()
        if model:
            self.state["model"] = model

        self.media_service.close()
        self.media_service = MediaMetadataService(self.cfg.exiftool_path, self.cfg.ffmpeg_path)
        self.thumbs.ffmpeg_bin = self.media_service.ffmpeg_bin
        self.orchestrator.media_service = self.media_service
        self.orchestrator.antigravity_cli = self.state["antigravity_cli"]
        self.orchestrator.ai_model = self.state["model"]
        settings = self.get_settings()
        if cli_warning:
            settings["warning"] = cli_warning
        return settings

    def browse_folder(self, initial_dir: str | None = None) -> dict[str, Any]:
        """Open a native folder picker dialog on the local Windows/desktop host."""
        selected = ""
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(
                initialdir=initial_dir or None,
                title="Seleccionar carpeta",
            ) or ""
            root.destroy()
            return {"path": selected, "cancelled": not bool(selected)}
        except Exception:
            pass

        if os.name == "nt":
            try:
                ps_bin = resolve_cli_binary("powershell")
                if not ps_bin:
                    return {"path": "", "cancelled": True, "error": "powershell binary missing"}
                ps_script = (
                    "Add-Type -AssemblyName System.Windows.Forms; "
                    "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
                    "$d.Description = 'Seleccionar carpeta'; "
                    "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
                    "{ Write-Output $d.SelectedPath }"
                )
                proc = subprocess.run(
                    [ps_bin, "-NoProfile", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                selected = (proc.stdout or "").strip()
                return {"path": selected, "cancelled": not bool(selected)}
            except Exception as exc:
                return {"path": "", "cancelled": True, "error": str(exc)}

        return {"path": "", "cancelled": True}

    def scan_start(self, payload: dict[str, Any], loop: asyncio.AbstractEventLoop | None = None) -> dict[str, Any]:
        source_dir = str(payload.get("source_dir") or "")
        dest_dir = str(payload.get("dest_dir") or "")
        time_gap_hours = float(payload.get("time_gap_hours", 4.0))
        gps_distance_km = float(payload.get("gps_distance_km", 25.0))
        batch_size = int(payload.get("batch_size") or 100)
        solo_primer_lote = bool(payload.get("solo_primer_lote", False))
        incremental_mode = bool(payload.get("incremental_mode") or payload.get("incremental") or False)
        master_index_path = payload.get("master_index_path") or payload.get("previous_index_path")
        if isinstance(master_index_path, str):
            master_index_path = master_index_path.strip() or None
        if time_gap_hours <= 0 or gps_distance_km <= 0:
            raise _ApiError(400, "time_gap_hours and gps_distance_km must be positive.")
        try:
            src, dst = validate_source_dest_pair(source_dir, dest_dir)
        except ValueError as exc:
            raise _ApiError(400, str(exc)) from exc

        async def _runner() -> None:
            jid = await self.orchestrator.run_job(
                src,
                dst,
                time_gap_hours=time_gap_hours,
                gps_distance_km=gps_distance_km,
                batch_size=batch_size,
                solo_primer_lote=solo_primer_lote,
                incremental_mode=incremental_mode,
                master_index_path=master_index_path,
            )
            self.state["active_job_id"] = jid
            self.state.setdefault("job_options", {})[jid] = {
                "incremental_mode": incremental_mode,
                "master_index_path": master_index_path,
            }

        if loop is not None:
            self.state["job_task"] = loop.create_task(_runner())
        else:
            bg_loop = self._ensure_bg_loop()
            self.state["job_task"] = asyncio.run_coroutine_threadsafe(_runner(), bg_loop)
        return self.orchestrator.current_snapshot.to_dict()

    def scan_pause(self) -> dict[str, Any]:
        self.orchestrator.pause()
        return self.orchestrator.current_snapshot.to_dict()

    def scan_resume(self) -> dict[str, Any]:
        self.orchestrator.resume()
        return self.orchestrator.current_snapshot.to_dict()

    def scan_status(self) -> dict[str, Any]:
        return self.orchestrator.current_snapshot.to_dict()

    def get_tree(
        self,
        job_id: str | None = None,
        folder: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        jid = job_id or self.state["active_job_id"] or self.orchestrator.current_snapshot.job_id
        if not jid:
            return {"job_id": None, "folders": [], "items": [], "page": page, "total": 0}

        page = max(1, page)
        page_size = max(1, min(200, page_size))
        offset = (page - 1) * page_size
        conn = open_connection(self.cfg.db_path)
        try:
            folder_rows = conn.execute(
                """
                SELECT target_folder AS folder, COUNT(*) AS count
                FROM media_items
                WHERE job_id = ?
                GROUP BY target_folder
                ORDER BY target_folder ASC
                """,
                (jid,),
            ).fetchall()
            folder_counts: dict[str, int] = {str(r["folder"]): int(r["count"]) for r in folder_rows}
            custom_rows = conn.execute(
                "SELECT folder FROM custom_folders WHERE job_id = ?",
                (jid,),
            ).fetchall()
            for cr in custom_rows:
                folder_counts.setdefault(str(cr["folder"]), 0)
            folders = [
                {"folder": name, "count": folder_counts[name]}
                for name in sorted(folder_counts.keys())
            ]

            if folder:
                count_row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM media_items WHERE job_id = ? AND target_folder = ?",
                    (jid, folder),
                ).fetchone()
                item_rows = conn.execute(
                    """
                    SELECT id, target_folder, target_filename, captured_at, location_label,
                           is_duplicate, is_placeholder, export_state, route_reason
                    FROM media_items
                    WHERE job_id = ? AND target_folder = ?
                    ORDER BY target_filename ASC
                    LIMIT ? OFFSET ?
                    """,
                    (jid, folder, page_size, offset),
                ).fetchall()
            else:
                count_row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM media_items WHERE job_id = ?",
                    (jid,),
                ).fetchone()
                item_rows = conn.execute(
                    """
                    SELECT id, target_folder, target_filename, captured_at, location_label,
                           is_duplicate, is_placeholder, export_state, route_reason
                    FROM media_items
                    WHERE job_id = ?
                    ORDER BY target_folder ASC, target_filename ASC
                    LIMIT ? OFFSET ?
                    """,
                    (jid, page_size, offset),
                ).fetchall()
        finally:
            conn.close()

        total = int(count_row["cnt"]) if count_row else 0
        return {
            "job_id": jid,
            "folders": folders,
            "items": [dict(r) for r in item_rows],
            "page": page,
            "page_size": page_size,
            "total": total,
        }

    def rename_folder(self, job_id: str, old_folder: str, new_folder: str) -> dict[str, Any]:
        conn = open_connection(self.cfg.db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE media_items SET target_folder = ? WHERE job_id = ? AND target_folder = ?",
                    (new_folder, job_id, old_folder),
                )
                conn.execute(
                    "UPDATE OR IGNORE custom_folders SET folder = ? WHERE job_id = ? AND folder = ?",
                    (new_folder, job_id, old_folder),
                )
                conn.execute(
                    "DELETE FROM custom_folders WHERE job_id = ? AND folder = ?",
                    (job_id, old_folder),
                )
                count = recompute_filenames(conn, job_id, new_folder)
            return {"folder": new_folder, "updated": count}
        finally:
            conn.close()

    def merge_folders(self, job_id: str, from_folder: str, to_folder: str) -> dict[str, Any]:
        conn = open_connection(self.cfg.db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE media_items SET target_folder = ? WHERE job_id = ? AND target_folder = ?",
                    (to_folder, job_id, from_folder),
                )
                conn.execute(
                    "DELETE FROM custom_folders WHERE job_id = ? AND folder = ?",
                    (job_id, from_folder),
                )
                count = recompute_filenames(conn, job_id, to_folder)
            return {"merged": from_folder, "into": to_folder, "updated": count}
        finally:
            conn.close()

    def create_folder(self, job_id: str, folder: str) -> dict[str, Any]:
        clean = folder.strip().strip("/\\")
        if not clean or ".." in Path(clean).parts:
            raise _ApiError(400, "Invalid folder name")
        conn = open_connection(self.cfg.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO custom_folders (job_id, folder, created_at)
                    VALUES (?, ?, datetime('now'))
                    """,
                    (job_id, clean),
                )
            return {"folder": clean, "created": True}
        finally:
            conn.close()

    def move_item(self, job_id: str, item_id: str, to_folder: str) -> dict[str, Any]:
        conn = open_connection(self.cfg.db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE media_items SET target_folder = ? WHERE job_id = ? AND id = ?",
                    (to_folder, job_id, item_id),
                )
                count = recompute_filenames(conn, job_id, to_folder)
            return {"item_id": item_id, "to_folder": to_folder, "updated": count}
        finally:
            conn.close()

    def get_thumbnail(self, item_id: str) -> bytes:
        conn = open_connection(self.cfg.db_path)
        try:
            row = conn.execute(
                "SELECT id, source_path, sha256 FROM media_items WHERE id = ?",
                (item_id,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            raise _ApiError(404, "Item not found")
        return self.thumbs.get_or_create(row["id"], Path(row["source_path"]), row["sha256"])

    def cancel_ai(self) -> dict[str, Any]:
        """Signal any in-flight AI classification or folder regeneration to stop."""
        self._ai_cancel_event.set()
        return {"cancelled": True}

    def ai_classify(
        self,
        item_ids: list[str],
        confirmed: bool,
        batch_size: int = 100,
        solo_primer_lote: bool = False,
    ) -> dict[str, Any]:
        if confirmed:
            self._ai_cancel_event.clear()
        clf = GeminiClassifier(
            self.cfg.db_path,
            cli_command=self.state["antigravity_cli"],
            model=self.state["model"],
        )
        return clf.classify_batch(
            item_ids,
            confirmed=confirmed,
            batch_size=int(batch_size or 100),
            solo_primer_lote=bool(solo_primer_lote),
            cancel_event=self._ai_cancel_event,
        )

    def _people_dir(self) -> Path:
        return self.cfg.people_dir

    def _read_people_index(self) -> list[dict[str, Any]]:
        pdir = self._people_dir()
        idx_path = pdir / "people.json"
        if not idx_path.is_file():
            return []
        try:
            raw = json.loads(idx_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                return []
        except (OSError, json.JSONDecodeError):
            return []
        valid: list[dict[str, Any]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            fname = str(entry.get("filename") or "")
            if fname and (pdir / fname).is_file():
                valid.append(entry)
        return valid

    def _write_people_index(self, entries: list[dict[str, Any]]) -> None:
        pdir = self._people_dir()
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "people.json").write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_people(self) -> dict[str, Any]:
        return {"people": self._read_people_index()}

    def add_person(self, label: str, filename: str, data: bytes) -> dict[str, Any]:
        clean_label = (label or "").strip()
        if not clean_label:
            raise _ApiError(400, "La etiqueta de la persona es obligatoria")
        if not data:
            raise _ApiError(400, "El archivo de imagen está vacío")
        ext = Path(filename or "foto.jpg").suffix.lower() or ".jpg"
        if ext not in PEOPLE_IMAGE_EXTENSIONS:
            raise _ApiError(400, f"Formato de imagen no soportado: {ext}")

        pdir = self._people_dir()
        pdir.mkdir(parents=True, exist_ok=True)
        person_id = uuid.uuid4().hex[:12]
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", clean_label).strip("_") or "persona"
        stored_name = f"{slug}__{person_id}{ext}"
        (pdir / stored_name).write_bytes(data)

        entry = {
            "id": person_id,
            "label": clean_label,
            "filename": stored_name,
            "url": f"/api/people/{person_id}/image",
        }
        entries = self._read_people_index()
        entries.append(entry)
        self._write_people_index(entries)
        return {"person": entry, **entry}

    def delete_person(self, person_id: str) -> dict[str, Any]:
        entries = self._read_people_index()
        target = next((e for e in entries if str(e.get("id")) == person_id), None)
        if target is None:
            raise _ApiError(404, "Persona no encontrada")
        pdir = self._people_dir()
        fpath = pdir / str(target.get("filename") or "")
        if fpath.is_file():
            try:
                fpath.unlink()
            except OSError:
                pass
        remaining = [e for e in entries if str(e.get("id")) != person_id]
        self._write_people_index(remaining)
        return {"deleted": True, "id": person_id}

    def get_person_image(self, person_id: str) -> tuple[bytes, str]:
        entries = self._read_people_index()
        target = next((e for e in entries if str(e.get("id")) == person_id), None)
        if target is None:
            raise _ApiError(404, "Persona no encontrada")
        fpath = self._people_dir() / str(target.get("filename") or "")
        if not fpath.is_file():
            raise _ApiError(404, "Imagen de referencia no encontrada")
        ctype, _ = mimetypes.guess_type(str(fpath))
        return fpath.read_bytes(), (ctype or "image/jpeg")

    def _face_clusters_dir(self) -> Path:
        return self.cfg.data_dir / "face_clusters"

    def get_face_discovery_status(self) -> dict[str, Any]:
        with self._face_discover_lock:
            return dict(self._face_discover_state)

    def start_face_discovery(self, job_id: str | None = None) -> dict[str, Any]:
        conn = open_connection(self.cfg.db_path)
        try:
            jid = job_id or self.state.get("active_job_id")
            if not jid:
                row = conn.execute("SELECT id FROM jobs ORDER BY created_at DESC LIMIT 1").fetchone()
                jid = str(row["id"]) if row else None
            if not jid:
                raise _ApiError(404, "No hay un trabajo escaneado para analizar rostros")
            rows = conn.execute(
                """
                SELECT id, source_path, ext
                FROM media_items
                WHERE job_id = ?
                ORDER BY COALESCE(captured_at, '') ASC, source_path ASC
                """,
                (jid,),
            ).fetchall()
        finally:
            conn.close()

        media_records = [
            {
                "id": str(r["id"]),
                "source_path": str(r["source_path"]),
                "ext": str(r["ext"] or ""),
            }
            for r in rows
            if str(r["ext"] or "").lower() in IMAGE_EXTENSIONS
        ]

        with self._face_discover_lock:
            self._face_discover_state = {
                "status": "running",
                "job_id": jid,
                "processed": 0,
                "total": len(media_records),
                "clusters_found": 0,
                "error": None,
            }
            snapshot = dict(self._face_discover_state)

        def _run() -> None:
            def _on_progress(processed: int, total: int, clusters_found: int) -> None:
                with self._face_discover_lock:
                    self._face_discover_state["processed"] = processed
                    self._face_discover_state["total"] = total
                    self._face_discover_state["clusters_found"] = clusters_found

            try:
                clusters = discover_frequent_faces(
                    media_records,
                    self.cfg.data_dir,
                    progress_cb=_on_progress,
                )
                with self._face_discover_lock:
                    self._face_discover_state["status"] = "completed"
                    self._face_discover_state["processed"] = len(media_records)
                    self._face_discover_state["clusters_found"] = len(clusters)
            except Exception as exc:
                with self._face_discover_lock:
                    self._face_discover_state["status"] = "failed"
                    self._face_discover_state["error"] = str(exc)

        threading.Thread(target=_run, daemon=True).start()
        return snapshot

    def list_face_clusters(self, *, include_ignored: bool = False) -> dict[str, Any]:
        clusters = load_face_clusters(self._face_clusters_dir(), include_ignored=include_ignored)
        return {
            "clusters": clusters,
            "discovery": self.get_face_discovery_status(),
        }

    def update_face_cluster_entry(
        self,
        cluster_id: str,
        *,
        label: str | None = None,
        ignored: bool | None = None,
    ) -> dict[str, Any]:
        updated = update_face_cluster(
            self._face_clusters_dir(),
            cluster_id,
            label=label,
            ignored=ignored,
        )
        if updated is None:
            raise _ApiError(404, "Clúster facial no encontrado")
        return {"cluster": updated, **updated}

    def get_face_cluster_image(self, cluster_id: str) -> tuple[bytes, str]:
        fdir = self._face_clusters_dir()
        img_path = fdir / f"{cluster_id}.jpg"
        if not img_path.is_file():
            jf = fdir / f"{cluster_id}.json"
            if jf.is_file():
                try:
                    data = json.loads(jf.read_text(encoding="utf-8"))
                    cp = Path(str(data.get("crop_path") or ""))
                    if cp.is_file():
                        img_path = cp
                except (OSError, json.JSONDecodeError):
                    pass
        if not img_path.is_file():
            raise _ApiError(404, "Miniatura de rostro no encontrada")
        ctype, _ = mimetypes.guess_type(str(img_path))
        return img_path.read_bytes(), (ctype or "image/jpeg")

    def regenerate_ai_names(self, job_id: str) -> dict[str, Any]:
        """Invalidate AI name cache and re-run AI folder naming on 01_Recuerdos/ folders."""
        self._ai_cancel_event.clear()
        conn = open_connection(self.cfg.db_path)
        updated = 0
        total_folders = 0
        cancelled = False
        cli_cmd = self.state["antigravity_cli"]
        model = self.state["model"]
        agy_available = bool(resolve_cli_binary(cli_cmd, allowed_names=AGY_ALLOWED_NAMES))
        last_error: str | None = None
        try:
            with conn:
                conn.execute("DELETE FROM ai_cache")
            rows = conn.execute(
                """
                SELECT target_folder, COUNT(*) AS cnt, MIN(source_path) AS first_source_path,
                       AVG(latitude) AS avg_lat, AVG(longitude) AS avg_lon
                FROM media_items
                WHERE job_id = ? AND target_folder LIKE '01_Recuerdos/%'
                GROUP BY target_folder
                """,
                (job_id,),
            ).fetchall()
            total_folders = len(rows)
            with conn:
                for row in rows:
                    if self._ai_cancel_event.is_set():
                        cancelled = True
                        break
                    old_folder = str(row["target_folder"])
                    parts = old_folder.split("/")
                    if len(parts) < 3:
                        continue
                    year_part = parts[1]
                    ym_prefix = parts[2].split(" - ")[0]
                    orig_parent = Path(str(row["first_source_path"])).parent.name
                    loc_label = offline_reverse_geocode(row["avg_lat"], row["avg_lon"])
                    gps_txt = (
                        f" | GPS: {loc_label}"
                        if loc_label and not loc_label.startswith("GPS ")
                        else ""
                    )
                    summary = f"{ym_prefix} | {row['cnt']} fotos | carpeta: {orig_parent}{gps_txt}"

                    folder_items = conn.execute(
                        """
                        SELECT id, source_path, ext, sha256
                        FROM media_items
                        WHERE job_id = ? AND target_folder = ?
                        ORDER BY COALESCE(captured_at, '') ASC, COALESCE(sha256, '') ASC, source_path ASC
                        """,
                        (job_id, old_folder),
                    ).fetchall()
                    img_items = [
                        it for it in folder_items if str(it["ext"]).lower() not in VIDEO_EXTENSIONS
                    ]
                    scene_indices = (
                        [0, len(img_items) // 2, min(len(img_items) - 1, (len(img_items) * 3) // 4)]
                        if img_items
                        else []
                    )
                    tmp_scene_dir: str | None = None
                    add_dirs: list[Path] = []
                    if scene_indices:
                        tmp_scene_dir = tempfile.mkdtemp(prefix="agy_regen_")
                        tmp_scene_path = Path(tmp_scene_dir)
                        seen_idx: set[int] = set()
                        for idx in scene_indices:
                            if idx not in seen_idx:
                                seen_idx.add(idx)
                                it = img_items[idx]
                                tpath = self.thumbs.get_or_create_resized(
                                    str(it["id"]),
                                    Path(str(it["source_path"])),
                                    sha256=it["sha256"],
                                    max_dim=320,
                                )
                                if tpath.is_file():
                                    (tmp_scene_path / f"scene_{idx}.jpg").write_bytes(tpath.read_bytes())
                        extract_top_face_crops(
                            [Path(str(it["source_path"])) for it in img_items[:12]],
                            tmp_scene_path,
                            max_faces=4,
                            file_prefix="face",
                        )
                        add_dirs.append(tmp_scene_path)

                    _has_fc, _wl_fc = _load_whitelisted_face_clusters(self._face_clusters_dir())
                    _confirmed_people = (
                        resolve_folder_whitelisted_people(
                            [str(it["id"]) for it in folder_items],
                            _wl_fc,
                        )
                        if _has_fc
                        else []
                    )
                    if _confirmed_people:
                        summary = f"{summary} | Personas confirmadas: {', '.join(_confirmed_people)}"

                    err_info: dict[str, str] = {}
                    try:
                        ai_name = classify_with_agy(
                            summary,
                            cli_command=cli_cmd,
                            model=model,
                            cancel_event=self._ai_cancel_event,
                            people_dir=self.cfg.people_dir,
                            add_dirs=add_dirs,
                            error_out=err_info,
                        )
                    finally:
                        if tmp_scene_dir:
                            import shutil as _shutil

                            _shutil.rmtree(tmp_scene_dir, ignore_errors=True)
                    if err_info.get("reason"):
                        last_error = err_info["reason"]
                    if self._ai_cancel_event.is_set():
                        cancelled = True
                        break
                    safe_ai = "".join("_" if c in '<>:"/\\|?*' else c for c in ai_name).strip()
                    if not safe_ai:
                        continue
                    new_folder = f"01_Recuerdos/{year_part}/{ym_prefix} - {safe_ai}"
                    if new_folder != old_folder:
                        conn.execute(
                            "UPDATE media_items SET target_folder = ? WHERE job_id = ? AND target_folder = ?",
                            (new_folder, job_id, old_folder),
                        )
                        recompute_filenames(conn, job_id, new_folder)
                        updated += 1
        finally:
            conn.close()
        engine_name = f"agy · {model}" if agy_available else "Fallback local"
        result: dict[str, Any] = {
            "job_id": job_id,
            "updated_folders": updated,
            "total_folders": total_folders,
            "cache_cleared": True,
            "cancelled": cancelled,
            "engine": engine_name,
            "ai_engine": "agy" if agy_available else "heuristica_local",
            "model": model,
        }
        if last_error:
            result["error_reason"] = last_error
        return result

    def get_ai_tree_suggestions(self, job_id: str | None = None) -> dict[str, Any]:
        """Build folder summaries + <=256px representative thumbnails and request AI suggestions."""
        self._ai_cancel_event.clear()
        conn = open_connection(self.cfg.db_path)
        try:
            jid = job_id
            if not jid:
                row = conn.execute("SELECT id FROM jobs ORDER BY created_at DESC LIMIT 1").fetchone()
                if not row:
                    raise _ApiError(404, "No hay trabajos activos para analizar")
                jid = str(row["id"])

            rows = conn.execute(
                """
                SELECT id, source_path, ext, sha256, captured_at, location_label, target_folder
                FROM media_items
                WHERE job_id = ? AND target_folder LIKE '01_Recuerdos/%'
                ORDER BY target_folder ASC, COALESCE(captured_at, '') ASC, COALESCE(sha256, '') ASC, source_path ASC
                """,
                (jid,),
            ).fetchall()
        finally:
            conn.close()

        by_folder: dict[str, list[Any]] = {}
        for r in rows:
            by_folder.setdefault(str(r["target_folder"]), []).append(r)

        folder_summaries: list[dict[str, Any]] = []
        folder_to_rep_id: dict[str, str] = {}
        for folder, items in by_folder.items():
            mid = len(items) // 2
            rep = items[mid]
            for offset in range(len(items)):
                cand_idx = mid + (offset if offset % 2 == 0 else -offset) // 2
                if 0 <= cand_idx < len(items) and str(items[cand_idx]["ext"]).lower() not in VIDEO_EXTENSIONS:
                    rep = items[cand_idx]
                    break
            rep_id = str(rep["id"])
            folder_to_rep_id[folder] = rep_id
            thumb_path = self.thumbs.get_or_create_resized(
                rep_id,
                Path(str(rep["source_path"])),
                sha256=rep["sha256"],
                max_dim=256,
            )
            img_items = [it for it in items if str(it["ext"]).lower() not in VIDEO_EXTENSIONS]
            sample_paths: list[str] = []
            if img_items:
                sample_indices = sorted(
                    {
                        0,
                        len(img_items) // 4,
                        len(img_items) // 2,
                        min(len(img_items) - 1, (len(img_items) * 3) // 4),
                        len(img_items) - 1,
                    }
                )
                for s_idx in sample_indices:
                    it = img_items[s_idx]
                    tp = self.thumbs.get_or_create_resized(
                        str(it["id"]),
                        Path(str(it["source_path"])),
                        sha256=it["sha256"],
                        max_dim=320,
                    )
                    sample_paths.append(str(tp))
            if not sample_paths:
                sample_paths = [str(thumb_path)]

            parsed_dts: list[datetime] = []
            loc_label: str | None = None
            for it in items:
                if not loc_label and it["location_label"]:
                    loc_label = str(it["location_label"])
                ts = it["captured_at"]
                if ts:
                    try:
                        parsed_dts.append(datetime.fromisoformat(str(ts).replace("Z", "+00:00")))
                    except ValueError:
                        pass
            parsed_dts.sort()
            max_gap = 0.0
            for idx in range(1, len(parsed_dts)):
                gap = (parsed_dts[idx] - parsed_dts[idx - 1]).total_seconds() / 3600.0
                if gap > max_gap:
                    max_gap = gap

            is_midnight_merged = (
                len(parsed_dts) >= 2
                and parsed_dts[0].date() != parsed_dts[-1].date()
                and (parsed_dts[-1] - parsed_dts[0]).total_seconds() <= 14 * 3600
            )

            folder_summaries.append(
                {
                    "folder": folder,
                    "count": len(items),
                    "media_ids": [str(it["id"]) for it in items],
                    "start_date": parsed_dts[0].isoformat() if parsed_dts else None,
                    "end_date": parsed_dts[-1].isoformat() if parsed_dts else None,
                    "location_label": loc_label,
                    "max_gap_hours": round(max_gap, 2),
                    "is_midnight_merged": is_midnight_merged,
                    "thumb_path": str(thumb_path),
                    "thumb_paths": sample_paths,
                }
            )

        res = suggest_tree_optimizations(
            folder_summaries,
            cli_command=self.state["antigravity_cli"],
            model=self.state["model"],
            timeout=60.0,
            cancel_event=self._ai_cancel_event,
            people_dir=self.cfg.people_dir,
            face_clusters_dir=self._face_clusters_dir(),
            data_dir=self.cfg.data_dir,
        )
        enriched: list[dict[str, Any]] = []
        for s in res["suggestions"]:
            item = dict(s)
            before = item.get("before")
            rep_key = before[0] if isinstance(before, list) and before else before
            rid = folder_to_rep_id.get(str(rep_key or ""))
            if rid:
                item["thumb_item_id"] = rid
                item["thumb_url"] = f"/api/thumbnails/{rid}"
            enriched.append(item)
        out: dict[str, Any] = {
            "job_id": jid,
            "engine": res["engine"],
            "suggestions": enriched,
        }
        if res.get("error_reason"):
            out["error_reason"] = res["error_reason"]
        return out

    def apply_ai_tree_suggestions(
        self,
        job_id: str,
        suggestions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply selected rename, merge, and split suggestions cleanly to SQLite staging."""
        conn = open_connection(self.cfg.db_path)
        applied = 0
        touched_folders: set[str] = set()
        try:
            with conn:
                for s in suggestions:
                    if not isinstance(s, dict):
                        continue
                    stype = str(s.get("type", "")).strip().lower()
                    if stype == "rename":
                        before = str(s.get("before", "")).strip()
                        after = str(s.get("after", "")).strip()
                        if before and after and before != after:
                            cur = conn.execute(
                                "UPDATE media_items SET target_folder = ? WHERE job_id = ? AND target_folder = ?",
                                (after, job_id, before),
                            )
                            if cur.rowcount > 0:
                                touched_folders.add(after)
                                applied += 1
                    elif stype == "merge":
                        before_list = s.get("before")
                        after = str(s.get("after", "")).strip()
                        if isinstance(before_list, list) and after:
                            merged_any = False
                            for bf in before_list:
                                b_str = str(bf).strip()
                                if b_str and b_str != after:
                                    cur = conn.execute(
                                        "UPDATE media_items SET target_folder = ? WHERE job_id = ? AND target_folder = ?",
                                        (after, job_id, b_str),
                                    )
                                    if cur.rowcount > 0:
                                        merged_any = True
                            if merged_any:
                                touched_folders.add(after)
                                applied += 1
                    elif stype == "split":
                        before = str(s.get("before", "")).strip()
                        after_list = s.get("after")
                        if not before or not isinstance(after_list, list) or len(after_list) < 2:
                            continue
                        targets = [str(a).strip() for a in after_list if str(a).strip()]
                        if len(targets) < 2:
                            continue
                        rows = conn.execute(
                            """
                            SELECT id, captured_at
                            FROM media_items
                            WHERE job_id = ? AND target_folder = ?
                            ORDER BY COALESCE(captured_at, '') ASC, COALESCE(sha256, '') ASC, source_path ASC
                            """,
                            (job_id, before),
                        ).fetchall()
                        if not rows:
                            continue
                        # Find largest internal time gaps to split into len(targets) groups
                        gaps: list[tuple[float, int]] = []
                        dts: list[datetime | None] = []
                        for r in rows:
                            ts = r["captured_at"]
                            dt_val = None
                            if ts:
                                try:
                                    dt_val = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                                except ValueError:
                                    dt_val = None
                            dts.append(dt_val)
                        for idx in range(1, len(rows)):
                            d_prev, d_curr = dts[idx - 1], dts[idx]
                            g = (d_curr - d_prev).total_seconds() if (d_prev and d_curr) else 0.0
                            gaps.append((g, idx))
                        num_splits = len(targets) - 1
                        if any(g > 0 for g, _ in gaps):
                            split_indices = sorted(
                                idx for _, idx in sorted(gaps, key=lambda x: (-x[0], x[1]))[:num_splits]
                            )
                        else:
                            step = max(1, len(rows) // len(targets))
                            split_indices = [step * k for k in range(1, len(targets))]
                        bounds = [0, *split_indices, len(rows)]
                        for grp_idx in range(len(targets)):
                            target_f = targets[grp_idx]
                            chunk_rows = rows[bounds[grp_idx] : bounds[grp_idx + 1]]
                            for r in chunk_rows:
                                conn.execute(
                                    "UPDATE media_items SET target_folder = ? WHERE id = ?",
                                    (target_f, r["id"]),
                                )
                            if chunk_rows:
                                touched_folders.add(target_f)
                        applied += 1

                for tf in touched_folders:
                    recompute_filenames(conn, job_id, tf)
        finally:
            conn.close()
        return {"status": "ok", "job_id": job_id, "applied": applied}

    def open_destination(self, job_id: str | None = None, path: str | None = None) -> dict[str, Any]:
        """Open the destination library directory in the local OS file explorer."""
        dest_str = (path or "").strip()
        if not dest_str and job_id:
            conn = open_connection(self.cfg.db_path)
            try:
                row = conn.execute("SELECT dest_dir FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row:
                    dest_str = str(row["dest_dir"])
            finally:
                conn.close()
        if not dest_str:
            raise _ApiError(400, "No destination directory specified")
        dest_path = Path(dest_str)
        if not dest_path.exists():
            raise _ApiError(404, "Destination directory does not exist")
        if os.name == "nt":
            os.startfile(str(dest_path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(dest_path)])
        else:
            subprocess.Popen(["xdg-open", str(dest_path)])
        return {"opened": True, "path": str(dest_path)}

    def open_system_folder(self, job_id: str | None = None, path: str | None = None) -> dict[str, Any]:
        """Open the configured destination folder (or a subpath inside it) in the OS file explorer."""
        conn = open_connection(self.cfg.db_path)
        try:
            if job_id:
                row = conn.execute("SELECT dest_dir FROM jobs WHERE id = ?", (job_id,)).fetchone()
            else:
                row = conn.execute("SELECT dest_dir FROM jobs ORDER BY created_at DESC LIMIT 1").fetchone()
        finally:
            conn.close()
        if not row or not row["dest_dir"]:
            raise _ApiError(404, "No hay Carpeta Destino configurada para este trabajo")

        allowed_dest = normalize_win_path(str(row["dest_dir"]))
        requested = (path or "").strip()
        target_path = normalize_win_path(requested) if requested else allowed_dest

        allowed_cmp = Path(os.path.normcase(str(allowed_dest)))
        target_cmp = Path(os.path.normcase(str(target_path)))
        if target_cmp != allowed_cmp and allowed_cmp not in target_cmp.parents:
            raise _ApiError(403, "Solo se permite abrir la Carpeta Destino configurada")

        if not target_path.exists():
            raise _ApiError(404, "Destination directory does not exist")
        if os.name == "nt":
            os.startfile(str(target_path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target_path)])
        else:
            subprocess.Popen(["xdg-open", str(target_path)])
        return {"opened": True, "path": str(target_path)}

    def export_tree(
        self,
        job_id: str,
        batch_size: int = 100,
        solo_primer_lote: bool = False,
        incremental_mode: bool | None = None,
        master_index_path: str | None = None,
    ) -> dict[str, Any]:
        conn = open_connection(self.cfg.db_path)
        try:
            job = conn.execute(
                "SELECT id, dest_dir FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
        if not job:
            raise _ApiError(404, "Job not found")
        job_opts = self.state.get("job_options", {}).get(job_id, {})
        eff_incremental = (
            bool(incremental_mode)
            if incremental_mode is not None
            else bool(job_opts.get("incremental_mode", False))
        )
        eff_master_index = (
            master_index_path.strip() or None
            if isinstance(master_index_path, str)
            else job_opts.get("master_index_path")
        )
        return run_two_phase_export(
            self.cfg.db_path,
            job_id,
            Path(job["dest_dir"]),
            self.media_service,
            batch_size=int(batch_size or 100),
            solo_primer_lote=bool(solo_primer_lote),
            incremental_mode=eff_incremental,
            master_index_path=eff_master_index,
        )

    def _resolve_job_and_dest(self, job_id: str | None) -> tuple[str, Path]:
        conn = open_connection(self.cfg.db_path)
        try:
            if job_id:
                row = conn.execute("SELECT id, dest_dir FROM jobs WHERE id = ?", (job_id,)).fetchone()
            else:
                row = conn.execute("SELECT id, dest_dir FROM jobs ORDER BY created_at DESC LIMIT 1").fetchone()
            if not row:
                raise _ApiError(404, "No hay trabajo activo")
            return str(row["id"]), Path(str(row["dest_dir"]))
        finally:
            conn.close()

    def export_tree_plan(self, job_id: str | None = None) -> dict[str, Any]:
        jid, dest_dir = self._resolve_job_and_dest(job_id)
        return export_portable_plan(self.cfg.db_path, jid, dest_dir)

    def import_tree_plan(
        self,
        job_id: str | None = None,
        plan: dict[str, Any] | None = None,
        plan_path: str | None = None,
    ) -> dict[str, Any]:
        jid, dest_dir = self._resolve_job_and_dest(job_id)
        plan_data = plan
        resolved_path = Path(plan_path) if plan_path else (dest_dir / PORTABLE_PLAN_FILENAME)
        if plan_data is None:
            if not resolved_path.is_file():
                raise _ApiError(404, f"No se encontró el archivo de plan: {resolved_path}")
            try:
                plan_data = json.loads(resolved_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise _ApiError(400, f"Plan JSON inválido: {exc}") from exc
        if not isinstance(plan_data, dict):
            raise _ApiError(400, "El plan debe ser un objeto JSON")
        res = import_portable_plan(self.cfg.db_path, jid, plan_data)
        res["plan_path"] = str(resolved_path)
        return res

    def refine_tree(
        self,
        job_id: str | None = None,
        dest_dir: str | None = None,
        plan: dict[str, Any] | None = None,
        plan_path: str | None = None,
    ) -> dict[str, Any]:
        resolved_job_id: str | None = job_id
        resolved_dest: Path | None = Path(dest_dir) if dest_dir and str(dest_dir).strip() else None
        if resolved_dest is None or resolved_job_id is None:
            try:
                jid, db_dest = self._resolve_job_and_dest(resolved_job_id)
                if resolved_job_id is None:
                    resolved_job_id = jid
                if resolved_dest is None:
                    resolved_dest = db_dest
            except _ApiError:
                if resolved_dest is None:
                    raise
        try:
            return refine_exported_tree(
                self.cfg.db_path,
                resolved_dest,
                job_id=resolved_job_id,
                plan_data=plan,
                plan_path=plan_path,
            )
        except FileNotFoundError as exc:
            raise _ApiError(404, str(exc)) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise _ApiError(400, str(exc)) from exc

    def run_ai_debate(
        self,
        job_id: str | None = None,
        *,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Run resumable 2-agent consensus debate (Curador + Auditor) in batches of 5 folders."""
        from datetime import timezone as _tz

        jid, dest_dir = self._resolve_job_and_dest(job_id)
        cli_cmd = self.state.get("antigravity_cli") or "agy"
        model = self.state.get("model") or DEFAULT_AGY_MODEL

        conn = open_connection(self.cfg.db_path)
        try:
            rows = conn.execute(
                """
                SELECT id, source_path, ext, sha256, captured_at, location_label, target_folder
                FROM media_items
                WHERE job_id = ? AND target_folder LIKE '01_Recuerdos/%'
                ORDER BY target_folder ASC, COALESCE(captured_at, '') ASC, COALESCE(sha256, '') ASC, source_path ASC
                """,
                (jid,),
            ).fetchall()

            by_folder: dict[str, list[Any]] = {}
            for r in rows:
                by_folder.setdefault(str(r["target_folder"]), []).append(r)

            all_folders = list(by_folder.keys())
            now_iso = datetime.now(_tz.utc).isoformat()
            with conn:
                if not resume:
                    for f in all_folders:
                        conn.execute(
                            """
                            INSERT INTO folder_debate_state (job_id, folder, status, history_json, updated_at)
                            VALUES (?, ?, 'pending', '[]', ?)
                            ON CONFLICT(job_id, folder) DO UPDATE SET
                                status = 'pending',
                                history_json = '[]',
                                updated_at = excluded.updated_at
                            """,
                            (jid, f, now_iso),
                        )
                else:
                    for f in all_folders:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO folder_debate_state (job_id, folder, status, history_json, updated_at)
                            VALUES (?, ?, 'pending', '[]', ?)
                            """,
                            (jid, f, now_iso),
                        )

            state_rows = conn.execute(
                "SELECT folder, status FROM folder_debate_state WHERE job_id = ?",
                (jid,),
            ).fetchall()
            status_by_folder = {str(sr["folder"]): str(sr["status"]) for sr in state_rows}
        finally:
            conn.close()

        pending_folders = [f for f in all_folders if status_by_folder.get(f, "pending") != "debated"]
        total_count = len(all_folders)
        debated_count = total_count - len(pending_folders)

        batch_size = 5
        paused_reason: str | None = None

        for i in range(0, len(pending_folders), batch_size):
            batch_folders = pending_folders[i : i + batch_size]
            batch_summaries: list[dict[str, Any]] = []
            for folder in batch_folders:
                items = by_folder[folder]
                img_items = [it for it in items if str(it["ext"]).lower() not in VIDEO_EXTENSIONS]
                sample_paths: list[str] = []
                thumb_path_str = ""
                if img_items:
                    idxs = sorted({0, len(img_items) // 2, min(len(img_items) - 1, (len(img_items) * 3) // 4)})
                    for s_idx in idxs:
                        it = img_items[s_idx]
                        tp = self.thumbs.get_or_create_resized(
                            str(it["id"]),
                            Path(str(it["source_path"])),
                            sha256=it["sha256"],
                            max_dim=320,
                        )
                        sample_paths.append(str(tp))
                    thumb_path_str = sample_paths[0]
                loc_label = next((str(it["location_label"]) for it in items if it["location_label"]), None)
                dates = [str(it["captured_at"]) for it in items if it["captured_at"]]
                batch_summaries.append(
                    {
                        "folder": folder,
                        "count": len(items),
                        "media_ids": [str(it["id"]) for it in items],
                        "start_date": min(dates) if dates else None,
                        "end_date": max(dates) if dates else None,
                        "location_label": loc_label,
                        "thumb_path": thumb_path_str,
                        "thumb_paths": sample_paths,
                    }
                )

            res = debate_folder_batch_with_agy(
                batch_summaries,
                cli_command=cli_cmd,
                model=model,
                timeout=60.0,
                people_dir=self.cfg.people_dir,
                face_clusters_dir=self._face_clusters_dir(),
                data_dir=self.cfg.data_dir,
            )
            if not res.get("ok"):
                paused_reason = str(res.get("error_reason") or "Error de cuota o autenticación en agy")
                export_portable_plan(self.cfg.db_path, jid, dest_dir)
                return {
                    "job_id": jid,
                    "status": "paused",
                    "paused": True,
                    "pause_reason": paused_reason,
                    "completed_folders": debated_count,
                    "total_folders": total_count,
                    "pending_folders": total_count - debated_count,
                    "progress_label": f"{debated_count}/{total_count} carpetas listas",
                    "can_resume": True,
                    "engine": f"agy · {model}",
                }

            decisions = res.get("decisions", [])
            conn = open_connection(self.cfg.db_path)
            try:
                touched: set[str] = set()
                batch_ts = datetime.now(_tz.utc).isoformat()
                with conn:
                    for dec in decisions:
                        old_folder = str(dec.get("folder") or "")
                        if not old_folder:
                            continue
                        final_name = str(dec.get("final_name") or "").strip()
                        merge_with = dec.get("merge_with")
                        new_folder = old_folder
                        if merge_with and isinstance(merge_with, str) and merge_with.strip():
                            new_folder = merge_with.strip()
                        elif final_name:
                            parts = old_folder.split("/")
                            if len(parts) >= 3:
                                prefix = parts[2].split(" - ")[0]
                                new_folder = f"{parts[0]}/{parts[1]}/{prefix} - {final_name}"
                        if new_folder != old_folder:
                            conn.execute(
                                "UPDATE media_items SET target_folder = ? WHERE job_id = ? AND target_folder = ?",
                                (new_folder, jid, old_folder),
                            )
                            conn.execute(
                                "DELETE FROM folder_debate_state WHERE job_id = ? AND folder = ?",
                                (jid, old_folder),
                            )
                            touched.add(new_folder)

                        hist_entry = [
                            {
                                "curador": dec.get("curador"),
                                "auditor": new_folder,
                                "reason": dec.get("reason"),
                            }
                        ]
                        conn.execute(
                            """
                            INSERT INTO folder_debate_state (job_id, folder, status, history_json, updated_at)
                            VALUES (?, ?, 'debated', ?, ?)
                            ON CONFLICT(job_id, folder) DO UPDATE SET
                                status = 'debated',
                                history_json = excluded.history_json,
                                updated_at = excluded.updated_at
                            """,
                            (jid, new_folder, json.dumps(hist_entry, ensure_ascii=False), batch_ts),
                        )
                        debated_count += 1

                    for tf in sorted(touched):
                        recompute_filenames(conn, jid, tf)
            finally:
                conn.close()

            # Atomic checkpoint after each batch of 5
            export_portable_plan(self.cfg.db_path, jid, dest_dir)

        plan_res = export_portable_plan(self.cfg.db_path, jid, dest_dir)
        return {
            "job_id": jid,
            "status": "completed",
            "paused": False,
            "completed_folders": debated_count,
            "total_folders": total_count,
            "pending_folders": max(0, total_count - debated_count),
            "progress_label": f"{debated_count}/{total_count} carpetas listas",
            "can_resume": False,
            "plan_path": plan_res["plan_path"],
            "engine": f"agy · {model}",
        }


if FASTAPI_AVAILABLE:

    class ScanStartRequest(BaseModel):
        source_dir: str
        dest_dir: str
        time_gap_hours: float = Field(default=4.0, gt=0.0)
        gps_distance_km: float = Field(default=25.0, gt=0.0)
        batch_size: int = Field(default=100, ge=1)
        solo_primer_lote: bool = False
        incremental_mode: bool = False
        incremental: bool = False
        master_index_path: str | None = None
        previous_index_path: str | None = None

    class FolderRenameRequest(BaseModel):
        job_id: str
        old_folder: str
        new_folder: str

    class FolderMergeRequest(BaseModel):
        job_id: str
        from_folder: str
        to_folder: str

    class FolderCreateRequest(BaseModel):
        job_id: str
        folder: str

    class ItemMoveRequest(BaseModel):
        job_id: str
        item_id: str
        to_folder: str

    class SettingsUpdateRequest(BaseModel):
        exiftool_path: str | None = None
        ffmpeg_path: str | None = None
        antigravity_cli: str | None = None
        model: str | None = None
        gemini_model: str | None = None

    class BrowseFolderRequest(BaseModel):
        initial_dir: str | None = None

    class AIClassifyRequest(BaseModel):
        job_id: str
        item_ids: list[str]
        confirmed: bool = False
        batch_size: int = Field(default=100, ge=1)
        solo_primer_lote: bool = False

    class AIRegenerateRequest(BaseModel):
        job_id: str

    class AITreeSuggestionsRequest(BaseModel):
        job_id: str | None = None

    class AITreeApplyRequest(BaseModel):
        job_id: str
        suggestions: list[dict[str, Any]] = Field(default_factory=list)

    class ExportRequest(BaseModel):
        job_id: str
        batch_size: int = Field(default=100, ge=1)
        solo_primer_lote: bool = False
        incremental_mode: bool | None = None
        incremental: bool | None = None
        master_index_path: str | None = None
        previous_index_path: str | None = None

    class OpenDestinationRequest(BaseModel):
        job_id: str | None = None
        path: str | None = None

    class PlanExportRequest(BaseModel):
        job_id: str | None = None

    class PlanImportRequest(BaseModel):
        job_id: str | None = None
        plan: dict[str, Any] | None = None
        plan_path: str | None = None

    class TreeRefineRequest(BaseModel):
        job_id: str | None = None
        dest_dir: str | None = None
        plan: dict[str, Any] | None = None
        plan_path: str | None = None

    class AIDebateRequest(BaseModel):
        job_id: str | None = None
        resume: bool = False


def create_app(config: OrganizerConfig | None = None) -> Any:
    """Create and configure the local Photo Organizer FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is not installed in this Python environment.")

    cfg = config or OrganizerConfig()
    core = _LocalAppCore(cfg)
    app = FastAPI(title="Local Photo & Video Organizer", version="0.1.0")

    @app.middleware("http")
    async def validate_local_origin(request: Request, call_next: Any) -> Response:
        ok, reason = _is_allowed_origin(
            request.method,
            request.headers.get("host") or "",
            request.headers.get("origin"),
        )
        if not ok:
            return JSONResponse(status_code=403, content={"detail": reason})
        return await call_next(request)

    @app.get("/")
    async def serve_index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return core.health()

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        return core.get_settings()

    @app.post("/api/settings")
    async def update_settings(payload: SettingsUpdateRequest) -> dict[str, Any]:
        return core.update_settings(payload.model_dump())

    @app.get("/api/people")
    async def list_people() -> dict[str, Any]:
        return await asyncio.to_thread(core.list_people)

    @app.post("/api/people")
    async def add_person(request: Request) -> JSONResponse:
        try:
            body = await request.body()
            fields, files = _parse_multipart_form(request.headers.get("content-type", ""), body)
            label = fields.get("label", "")
            file_entry = files.get("file") or (next(iter(files.values())) if files else None)
            if not file_entry:
                raise _ApiError(400, "Se requiere un archivo de imagen")
            filename, data = file_entry
            res = await asyncio.to_thread(core.add_person, label, filename, data)
            return JSONResponse(status_code=201, content=res)
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.delete("/api/people/{person_id}")
    async def delete_person(person_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(core.delete_person, person_id)
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.get("/api/people/{person_id}/image")
    async def get_person_image(person_id: str) -> Response:
        try:
            data, ctype = await asyncio.to_thread(core.get_person_image, person_id)
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        return Response(content=data, media_type=ctype)

    @app.post("/api/faces/discover")
    async def discover_faces(request: Request) -> JSONResponse:
        try:
            try:
                body = await request.json()
            except Exception:
                body = {}
            job_id = body.get("job_id") if isinstance(body, dict) else None
            res = await asyncio.to_thread(core.start_face_discovery, job_id)
            return JSONResponse(status_code=202, content=res)
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.get("/api/faces/discover/status")
    async def discover_faces_status() -> dict[str, Any]:
        return core.get_face_discovery_status()

    @app.get("/api/faces/clusters")
    async def list_face_clusters(include_ignored: bool = False) -> dict[str, Any]:
        return await asyncio.to_thread(core.list_face_clusters, include_ignored=include_ignored)

    @app.post("/api/faces/clusters/{cluster_id}")
    async def update_face_cluster_route(cluster_id: str, request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                body = {}
            label = body.get("label") if "label" in body else None
            ignored = body.get("ignored") if "ignored" in body else None
            return await asyncio.to_thread(
                core.update_face_cluster_entry,
                cluster_id,
                label=label,
                ignored=ignored,
            )
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.get("/api/faces/clusters/{cluster_id}/image")
    async def get_face_cluster_image(cluster_id: str) -> Response:
        try:
            data, ctype = await asyncio.to_thread(core.get_face_cluster_image, cluster_id)
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        return Response(content=data, media_type=ctype)

    @app.post("/api/browse-folder")
    async def browse_folder(payload: BrowseFolderRequest | None = None) -> dict[str, Any]:
        initial = payload.initial_dir if payload else None
        return await asyncio.to_thread(core.browse_folder, initial)

    @app.post("/api/scan/start")
    async def scan_start(payload: ScanStartRequest) -> dict[str, Any]:
        try:
            snap = core.scan_start(payload.model_dump(), loop=asyncio.get_running_loop())
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        await asyncio.sleep(0)
        return snap

    @app.post("/api/scan/pause")
    async def scan_pause() -> dict[str, Any]:
        return core.scan_pause()

    @app.post("/api/scan/resume")
    async def scan_resume() -> dict[str, Any]:
        return core.scan_resume()

    @app.get("/api/scan/status")
    async def scan_status() -> dict[str, Any]:
        return core.scan_status()

    @app.get("/api/events")
    async def sse_events(request: Request) -> StreamingResponse:
        async def _stream() -> Any:
            while True:
                if await request.is_disconnected():
                    break
                payload = json.dumps(core.orchestrator.current_snapshot.to_dict())
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    @app.get("/api/tree")
    async def get_tree(
        job_id: str | None = None,
        folder: str | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        return await asyncio.to_thread(core.get_tree, job_id, folder, page, page_size)

    @app.post("/api/tree/folders/rename")
    async def rename_folder(payload: FolderRenameRequest) -> dict[str, Any]:
        return await asyncio.to_thread(
            core.rename_folder,
            payload.job_id,
            payload.old_folder,
            payload.new_folder,
        )

    @app.post("/api/tree/folders/merge")
    async def merge_folders(payload: FolderMergeRequest) -> dict[str, Any]:
        return await asyncio.to_thread(
            core.merge_folders,
            payload.job_id,
            payload.from_folder,
            payload.to_folder,
        )

    @app.post("/api/tree/folders/create")
    async def create_folder(payload: FolderCreateRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(core.create_folder, payload.job_id, payload.folder)
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/api/tree/items/move")
    async def move_item(payload: ItemMoveRequest) -> dict[str, Any]:
        return await asyncio.to_thread(
            core.move_item,
            payload.job_id,
            payload.item_id,
            payload.to_folder,
        )

    @app.get("/api/thumbnails/{item_id}")
    async def get_thumbnail(item_id: str) -> Response:
        try:
            data = await asyncio.to_thread(core.get_thumbnail, item_id)
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        return Response(content=data, media_type="image/jpeg")

    @app.post("/api/ai/classify")
    async def ai_classify(payload: AIClassifyRequest) -> dict[str, Any]:
        return core.ai_classify(
            payload.item_ids,
            payload.confirmed,
            batch_size=payload.batch_size,
            solo_primer_lote=payload.solo_primer_lote,
        )

    @app.post("/api/ai/regenerate")
    async def ai_regenerate(payload: AIRegenerateRequest) -> dict[str, Any]:
        return await asyncio.to_thread(core.regenerate_ai_names, payload.job_id)

    @app.post("/api/tree/ai-suggestions")
    async def ai_tree_suggestions(payload: AITreeSuggestionsRequest) -> dict[str, Any]:
        return await asyncio.to_thread(core.get_ai_tree_suggestions, payload.job_id)

    @app.post("/api/tree/apply-suggestions")
    async def ai_tree_apply(payload: AITreeApplyRequest) -> dict[str, Any]:
        return await asyncio.to_thread(core.apply_ai_tree_suggestions, payload.job_id, payload.suggestions)

    @app.get("/api/tree/plan/export")
    async def export_plan_get(job_id: str | None = None) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(core.export_tree_plan, job_id)
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/api/tree/plan/export")
    async def export_plan_post(payload: PlanExportRequest | None = None) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(core.export_tree_plan, payload.job_id if payload else None)
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/api/tree/plan/import")
    async def import_plan_post(payload: PlanImportRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                core.import_tree_plan,
                payload.job_id,
                payload.plan,
                payload.plan_path,
            )
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/api/tree/refine")
    async def refine_tree_post(payload: TreeRefineRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                core.refine_tree,
                payload.job_id,
                payload.dest_dir,
                payload.plan,
                payload.plan_path,
            )
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/api/tree/ai-debate")
    async def ai_debate(payload: AIDebateRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                core.run_ai_debate,
                payload.job_id,
                resume=payload.resume,
            )
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/api/ai/cancel")
    async def ai_cancel() -> dict[str, Any]:
        return core.cancel_ai()

    @app.post("/api/export")
    async def export_tree(payload: ExportRequest) -> dict[str, Any]:
        inc_val: bool | None = None
        if payload.incremental_mode is not None or payload.incremental is not None:
            inc_val = bool(payload.incremental_mode or payload.incremental)
        idx_path = payload.master_index_path or payload.previous_index_path
        try:
            return await asyncio.to_thread(
                core.export_tree,
                payload.job_id,
                payload.batch_size,
                payload.solo_primer_lote,
                inc_val,
                idx_path,
            )
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/api/open-destination")
    async def open_destination(payload: OpenDestinationRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(core.open_destination, payload.job_id, payload.path)
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/api/system/open-folder")
    async def open_system_folder(payload: OpenDestinationRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(core.open_system_folder, payload.job_id, payload.path)
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def _make_stdlib_handler(core: _LocalAppCore) -> type[http.server.BaseHTTPRequestHandler]:
    """Build a ThreadingHTTPServer handler mirroring all FastAPI routes."""

    class LocalHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_bytes(self, code: int, content_type: str, data: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data: blob:;",
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, code: int, body: dict[str, Any]) -> None:
            self._send_bytes(code, "application/json; charset=utf-8", json.dumps(body).encode("utf-8"))

        def _read_raw_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return b""
            return self.rfile.read(length)

        def _read_json(self) -> dict[str, Any]:
            raw_bytes = self._read_raw_body()
            if not raw_bytes:
                return {}
            raw = raw_bytes.decode("utf-8")
            return json.loads(raw) if raw else {}

        def _serve_file(self, path: Path) -> None:
            if not path.is_file():
                self._send_json(404, {"detail": "Not found"})
                return
            ctype, _ = mimetypes.guess_type(str(path))
            self._send_bytes(200, ctype or "application/octet-stream", path.read_bytes())

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/":
                self._serve_file(STATIC_DIR / "index.html")
                return
            if path.startswith("/static/"):
                rel = path[len("/static/"):].lstrip("/")
                target = (STATIC_DIR / rel).resolve()
                if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
                    self._send_json(403, {"detail": "Forbidden"})
                    return
                self._serve_file(target)
                return

            try:
                if path == "/api/health":
                    self._send_json(200, core.health())
                elif path == "/api/settings":
                    self._send_json(200, core.get_settings())
                elif path == "/api/people":
                    self._send_json(200, core.list_people())
                elif path.startswith("/api/people/") and path.endswith("/image"):
                    person_id = path[len("/api/people/") : -len("/image")].strip("/")
                    data, ctype = core.get_person_image(person_id)
                    self._send_bytes(200, ctype, data)
                elif path == "/api/faces/discover/status":
                    self._send_json(200, core.get_face_discovery_status())
                elif path == "/api/faces/clusters":
                    qs = parse_qs(parsed.query)
                    inc = qs.get("include_ignored", ["0"])[0].lower() in {"1", "true", "yes"}
                    self._send_json(200, core.list_face_clusters(include_ignored=inc))
                elif path.startswith("/api/faces/clusters/") and path.endswith("/image"):
                    cluster_id = path[len("/api/faces/clusters/") : -len("/image")].strip("/")
                    data, ctype = core.get_face_cluster_image(cluster_id)
                    self._send_bytes(200, ctype, data)
                elif path == "/api/scan/status":
                    self._send_json(200, core.scan_status())
                elif path == "/api/tree":
                    qs = parse_qs(parsed.query)
                    job_id = qs.get("job_id", [None])[0]
                    folder = qs.get("folder", [None])[0]
                    page = int(qs.get("page", ["1"])[0])
                    page_size = int(qs.get("page_size", ["50"])[0])
                    self._send_json(200, core.get_tree(job_id, folder, page, page_size))
                elif path == "/api/tree/plan/export":
                    qs = parse_qs(parsed.query)
                    job_id = qs.get("job_id", [None])[0]
                    self._send_json(200, core.export_tree_plan(job_id))
                elif path.startswith("/api/thumbnails/"):
                    item_id = path.rsplit("/", 1)[-1]
                    data = core.get_thumbnail(item_id)
                    self._send_bytes(200, "image/jpeg", data)
                elif path == "/api/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header(
                        "Content-Security-Policy",
                        "default-src 'self'; img-src 'self' data: blob:;",
                    )
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    while True:
                        try:
                            payload = json.dumps(core.orchestrator.current_snapshot.to_dict())
                            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                            self.wfile.flush()
                            time.sleep(0.25)
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            break
                else:
                    self._send_json(404, {"detail": "Not found"})
            except _ApiError as exc:
                self._send_json(exc.status_code, {"detail": exc.detail})
            except Exception as exc:
                self._send_json(500, {"detail": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            ok, reason = _is_allowed_origin(
                "POST",
                self.headers.get("Host") or "",
                self.headers.get("Origin"),
            )
            if not ok:
                self._send_json(403, {"detail": reason})
                return

            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path == "/api/people":
                    raw_body = self._read_raw_body()
                    fields, files = _parse_multipart_form(self.headers.get("Content-Type", ""), raw_body)
                    label = fields.get("label", "")
                    file_entry = files.get("file") or (next(iter(files.values())) if files else None)
                    if not file_entry:
                        raise _ApiError(400, "Se requiere un archivo de imagen")
                    filename, data = file_entry
                    self._send_json(201, core.add_person(label, filename, data))
                    return
                body = self._read_json()
                if path == "/api/settings":
                    self._send_json(200, core.update_settings(body))
                elif path == "/api/browse-folder":
                    self._send_json(200, core.browse_folder(body.get("initial_dir")))
                elif path == "/api/scan/start":
                    self._send_json(200, core.scan_start(body))
                elif path == "/api/scan/pause":
                    self._send_json(200, core.scan_pause())
                elif path == "/api/scan/resume":
                    self._send_json(200, core.scan_resume())
                elif path == "/api/tree/folders/rename":
                    self._send_json(
                        200,
                        core.rename_folder(body["job_id"], body["old_folder"], body["new_folder"]),
                    )
                elif path == "/api/tree/folders/merge":
                    self._send_json(
                        200,
                        core.merge_folders(body["job_id"], body["from_folder"], body["to_folder"]),
                    )
                elif path == "/api/tree/folders/create":
                    self._send_json(200, core.create_folder(body["job_id"], body["folder"]))
                elif path == "/api/tree/items/move":
                    self._send_json(
                        200,
                        core.move_item(body["job_id"], body["item_id"], body["to_folder"]),
                    )
                elif path == "/api/ai/classify":
                    self._send_json(
                        200,
                        core.ai_classify(
                            body.get("item_ids", []),
                            bool(body.get("confirmed", False)),
                            batch_size=int(body.get("batch_size") or 100),
                            solo_primer_lote=bool(body.get("solo_primer_lote", False)),
                        ),
                    )
                elif path == "/api/ai/regenerate":
                    self._send_json(200, core.regenerate_ai_names(body["job_id"]))
                elif path == "/api/tree/ai-suggestions":
                    self._send_json(200, core.get_ai_tree_suggestions(body.get("job_id")))
                elif path == "/api/tree/apply-suggestions":
                    self._send_json(
                        200,
                        core.apply_ai_tree_suggestions(
                            body["job_id"],
                            body.get("suggestions", []),
                        ),
                    )
                elif path == "/api/tree/plan/export":
                    self._send_json(200, core.export_tree_plan(body.get("job_id")))
                elif path == "/api/tree/plan/import":
                    self._send_json(
                        200,
                        core.import_tree_plan(
                            body.get("job_id"),
                            body.get("plan"),
                            body.get("plan_path"),
                        ),
                    )
                elif path == "/api/tree/refine":
                    self._send_json(
                        200,
                        core.refine_tree(
                            body.get("job_id"),
                            body.get("dest_dir"),
                            body.get("plan"),
                            body.get("plan_path"),
                        ),
                    )
                elif path == "/api/tree/ai-debate":
                    self._send_json(
                        200,
                        core.run_ai_debate(
                            body.get("job_id"),
                            resume=bool(body.get("resume", False)),
                        ),
                    )
                elif path == "/api/ai/cancel":
                    self._send_json(200, core.cancel_ai())
                elif path == "/api/export":
                    inc_val: bool | None = None
                    if "incremental_mode" in body or "incremental" in body:
                        inc_val = bool(body.get("incremental_mode") or body.get("incremental"))
                    idx_path = body.get("master_index_path") or body.get("previous_index_path")
                    self._send_json(
                        200,
                        core.export_tree(
                            body["job_id"],
                            batch_size=int(body.get("batch_size") or 100),
                            solo_primer_lote=bool(body.get("solo_primer_lote", False)),
                            incremental_mode=inc_val,
                            master_index_path=idx_path,
                        ),
                    )
                elif path == "/api/open-destination":
                    self._send_json(
                        200,
                        core.open_destination(body.get("job_id"), body.get("path")),
                    )
                elif path == "/api/faces/discover":
                    self._send_json(202, core.start_face_discovery(body.get("job_id")))
                elif path.startswith("/api/faces/clusters/"):
                    cluster_id = path[len("/api/faces/clusters/") :].strip("/")
                    label = body.get("label") if "label" in body else None
                    ignored = body.get("ignored") if "ignored" in body else None
                    self._send_json(
                        200,
                        core.update_face_cluster_entry(cluster_id, label=label, ignored=ignored),
                    )
                elif path == "/api/system/open-folder":
                    self._send_json(
                        200,
                        core.open_system_folder(body.get("job_id"), body.get("path")),
                    )
                else:
                    self._send_json(404, {"detail": "Not found"})
            except _ApiError as exc:
                self._send_json(exc.status_code, {"detail": exc.detail})
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"detail": str(exc)})
            except Exception as exc:
                self._send_json(500, {"detail": str(exc)})

        def do_DELETE(self) -> None:  # noqa: N802
            ok, reason = _is_allowed_origin(
                "DELETE",
                self.headers.get("Host") or "",
                self.headers.get("Origin"),
            )
            if not ok:
                self._send_json(403, {"detail": reason})
                return

            path = urlparse(self.path).path
            try:
                if path.startswith("/api/people/"):
                    person_id = path[len("/api/people/") :].strip("/")
                    if not person_id or "/" in person_id:
                        raise _ApiError(404, "Not found")
                    self._send_json(200, core.delete_person(person_id))
                else:
                    self._send_json(404, {"detail": "Not found"})
            except _ApiError as exc:
                self._send_json(exc.status_code, {"detail": exc.detail})
            except Exception as exc:
                self._send_json(500, {"detail": str(exc)})

    return LocalHandler


def _resolve_port(default_port: int) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=None)
    args, _ = parser.parse_known_args()
    if args.port is not None:
        return args.port
    env_port = os.environ.get("PORT")
    if env_port and env_port.isdigit():
        return int(env_port)
    return default_port


def run_server() -> None:
    """Start the local server bound strictly to 127.0.0.1."""
    cfg = OrganizerConfig()
    port = _resolve_port(cfg.port)

    if FASTAPI_AVAILABLE:
        try:
            import uvicorn  # type: ignore[import-not-found]

            uvicorn.run(create_app(cfg), host="127.0.0.1", port=port)
            return
        except ImportError:
            pass

    core = _LocalAppCore(cfg)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _make_stdlib_handler(core))
    print(f"Photo Organizer listening on http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
