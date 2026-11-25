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
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from photo_organizer.ai_classifier import DEFAULT_AGY_MODEL, GeminiClassifier, classify_with_agy
from photo_organizer.clustering import offline_reverse_geocode
from photo_organizer.config import OrganizerConfig, resolve_cli_binary, validate_source_dest_pair
from photo_organizer.db import init_db, open_connection, recompute_filenames
from photo_organizer.exporter import run_two_phase_export
from photo_organizer.media import BinaryHealthCheck, MediaMetadataService
from photo_organizer.thumbnails import ThumbnailCache
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
        agy_path = resolve_cli_binary(agy_cmd)
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
        if cli_cmd is not None:
            self.state["antigravity_cli"] = str(cli_cmd).strip() or "agy"
        model = str(payload.get("model") or payload.get("gemini_model") or "").strip()
        if model:
            self.state["model"] = model

        self.media_service.close()
        self.media_service = MediaMetadataService(self.cfg.exiftool_path, self.cfg.ffmpeg_path)
        self.thumbs.ffmpeg_bin = self.media_service.ffmpeg_bin
        self.orchestrator.media_service = self.media_service
        self.orchestrator.antigravity_cli = self.state["antigravity_cli"]
        self.orchestrator.ai_model = self.state["model"]
        return self.get_settings()

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
            )
            self.state["active_job_id"] = jid

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

    def ai_classify(
        self,
        item_ids: list[str],
        confirmed: bool,
        batch_size: int = 100,
        solo_primer_lote: bool = False,
    ) -> dict[str, Any]:
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
        )

    def regenerate_ai_names(self, job_id: str) -> dict[str, Any]:
        """Invalidate AI name cache and re-run AI folder naming on 01_Recuerdos/ folders."""
        conn = open_connection(self.cfg.db_path)
        updated = 0
        total_folders = 0
        cli_cmd = self.state["antigravity_cli"]
        model = self.state["model"]
        agy_available = bool(resolve_cli_binary(cli_cmd))
        try:
            with conn:
                conn.execute("DELETE FROM ai_cache")
            rows = conn.execute(
                """
                SELECT target_folder, COUNT(*) AS cnt, MIN(source_path) AS sample_path,
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
                    old_folder = str(row["target_folder"])
                    parts = old_folder.split("/")
                    if len(parts) < 3:
                        continue
                    year_part = parts[1]
                    ym_prefix = parts[2].split(" - ")[0]
                    orig_parent = Path(str(row["sample_path"])).parent.name
                    loc_label = offline_reverse_geocode(row["avg_lat"], row["avg_lon"])
                    gps_txt = (
                        f" | GPS: {loc_label}"
                        if loc_label and not loc_label.startswith("GPS ")
                        else ""
                    )
                    summary = f"{ym_prefix} | {row['cnt']} fotos | carpeta: {orig_parent}{gps_txt}"
                    ai_name = classify_with_agy(
                        summary,
                        cli_command=cli_cmd,
                        model=model,
                    )
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
        return {
            "job_id": job_id,
            "updated_folders": updated,
            "total_folders": total_folders,
            "cache_cleared": True,
            "engine": engine_name,
            "ai_engine": "agy" if agy_available else "fallback_local",
            "model": model,
        }

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

    def export_tree(
        self,
        job_id: str,
        batch_size: int = 100,
        solo_primer_lote: bool = False,
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
        return run_two_phase_export(
            self.cfg.db_path,
            job_id,
            Path(job["dest_dir"]),
            self.media_service,
            batch_size=int(batch_size or 100),
            solo_primer_lote=bool(solo_primer_lote),
        )


if FASTAPI_AVAILABLE:

    class ScanStartRequest(BaseModel):
        source_dir: str
        dest_dir: str
        time_gap_hours: float = Field(default=4.0, gt=0.0)
        gps_distance_km: float = Field(default=25.0, gt=0.0)
        batch_size: int = Field(default=100, ge=1)
        solo_primer_lote: bool = False

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

    class ExportRequest(BaseModel):
        job_id: str
        batch_size: int = Field(default=100, ge=1)
        solo_primer_lote: bool = False

    class OpenDestinationRequest(BaseModel):
        job_id: str | None = None
        path: str | None = None


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

    @app.post("/api/export")
    async def export_tree(payload: ExportRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                core.export_tree,
                payload.job_id,
                payload.batch_size,
                payload.solo_primer_lote,
            )
        except _ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/api/open-destination")
    async def open_destination(payload: OpenDestinationRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(core.open_destination, payload.job_id, payload.path)
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
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, code: int, body: dict[str, Any]) -> None:
            self._send_bytes(code, "application/json; charset=utf-8", json.dumps(body).encode("utf-8"))

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
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
                elif path == "/api/scan/status":
                    self._send_json(200, core.scan_status())
                elif path == "/api/tree":
                    qs = parse_qs(parsed.query)
                    job_id = qs.get("job_id", [None])[0]
                    folder = qs.get("folder", [None])[0]
                    page = int(qs.get("page", ["1"])[0])
                    page_size = int(qs.get("page_size", ["50"])[0])
                    self._send_json(200, core.get_tree(job_id, folder, page, page_size))
                elif path.startswith("/api/thumbnails/"):
                    item_id = path.rsplit("/", 1)[-1]
                    data = core.get_thumbnail(item_id)
                    self._send_bytes(200, "image/jpeg", data)
                elif path == "/api/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
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
                elif path == "/api/export":
                    self._send_json(
                        200,
                        core.export_tree(
                            body["job_id"],
                            batch_size=int(body.get("batch_size") or 100),
                            solo_primer_lote=bool(body.get("solo_primer_lote", False)),
                        ),
                    )
                elif path == "/api/open-destination":
                    self._send_json(
                        200,
                        core.open_destination(body.get("job_id"), body.get("path")),
                    )
                else:
                    self._send_json(404, {"detail": "Not found"})
            except _ApiError as exc:
                self._send_json(exc.status_code, {"detail": exc.detail})
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"detail": str(exc)})
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
