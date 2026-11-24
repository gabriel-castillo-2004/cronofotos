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
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from photo_organizer.ai_classifier import GeminiClassifier, mask_api_key
from photo_organizer.config import OrganizerConfig, validate_source_dest_pair
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
        )
        self.state: dict[str, Any] = {
            "job_task": None,
            "active_job_id": None,
            "gemini_api_key": "",
            "gemini_model": cfg.gemini_model,
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
        return {
            "exiftool_available": status.exiftool_available,
            "exiftool_path": status.exiftool_path,
            "ffmpeg_available": status.ffmpeg_available,
            "ffmpeg_path": status.ffmpeg_path,
            "warnings": status.warnings,
        }

    def get_settings(self) -> dict[str, Any]:
        return {
            "exiftool_path": self.cfg.exiftool_path or "",
            "ffmpeg_path": self.cfg.ffmpeg_path or "",
            "gemini_api_key": mask_api_key(self.state["gemini_api_key"]),
            "gemini_model": self.state["gemini_model"],
            "gemini_configured": bool(self.state["gemini_api_key"]),
        }

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("exiftool_path") is not None:
            self.cfg.exiftool_path = str(payload["exiftool_path"]).strip() or None
        if payload.get("ffmpeg_path") is not None:
            self.cfg.ffmpeg_path = str(payload["ffmpeg_path"]).strip() or None
        api_key = payload.get("gemini_api_key")
        if api_key is not None and not str(api_key).startswith("*"):
            self.state["gemini_api_key"] = str(api_key).strip()
        model = str(payload.get("gemini_model") or "").strip()
        if model:
            self.state["gemini_model"] = model

        self.media_service.close()
        self.media_service = MediaMetadataService(self.cfg.exiftool_path, self.cfg.ffmpeg_path)
        self.thumbs.ffmpeg_bin = self.media_service.ffmpeg_bin
        self.orchestrator.media_service = self.media_service
        return self.get_settings()

    def scan_start(self, payload: dict[str, Any], loop: asyncio.AbstractEventLoop | None = None) -> dict[str, Any]:
        source_dir = str(payload.get("source_dir") or "")
        dest_dir = str(payload.get("dest_dir") or "")
        time_gap_hours = float(payload.get("time_gap_hours", 4.0))
        gps_distance_km = float(payload.get("gps_distance_km", 25.0))
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
            folders = [dict(r) for r in folder_rows]

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
                count = recompute_filenames(conn, job_id, to_folder)
            return {"merged": from_folder, "into": to_folder, "updated": count}
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

    def ai_classify(self, item_ids: list[str], confirmed: bool) -> dict[str, Any]:
        clf = GeminiClassifier(
            self.cfg.db_path,
            api_key=self.state["gemini_api_key"],
            model=self.state["gemini_model"],
        )
        return clf.classify_batch(item_ids, confirmed=confirmed)

    def export_tree(self, job_id: str) -> dict[str, Any]:
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
        )


if FASTAPI_AVAILABLE:

    class ScanStartRequest(BaseModel):
        source_dir: str
        dest_dir: str
        time_gap_hours: float = Field(default=4.0, gt=0.0)
        gps_distance_km: float = Field(default=25.0, gt=0.0)

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
        gemini_api_key: str | None = None
        gemini_model: str = "gemini-2.5-flash"

    class AIClassifyRequest(BaseModel):
        job_id: str
        item_ids: list[str]
        confirmed: bool = False

    class ExportRequest(BaseModel):
        job_id: str


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
        return {"folder": payload.folder, "created": True}

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
        return core.ai_classify(payload.item_ids, payload.confirmed)

    @app.post("/api/export")
    async def export_tree(payload: ExportRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(core.export_tree, payload.job_id)
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
                    self._send_json(200, {"folder": body["folder"], "created": True})
                elif path == "/api/tree/items/move":
                    self._send_json(
                        200,
                        core.move_item(body["job_id"], body["item_id"], body["to_folder"]),
                    )
                elif path == "/api/ai/classify":
                    self._send_json(
                        200,
                        core.ai_classify(body.get("item_ids", []), bool(body.get("confirmed", False))),
                    )
                elif path == "/api/export":
                    self._send_json(200, core.export_tree(body["job_id"]))
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
