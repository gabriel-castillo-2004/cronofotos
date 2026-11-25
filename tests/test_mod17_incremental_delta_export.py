"""Tests for post-delivery modification MOD-17: Incremental (Delta) export with reusable master index.

Verifies:
1. UI (`index.html`, `app.js`):
   - Optional input "Índice maestro previo (_cronofotos_index.json)" and checkbox
     "Modo incremental (exportar solo archivos nuevos)".
   - `app.js` wires incremental mode and master index path to the backend API.
2. Scan & deduplication (`worker.py`, `main.py`):
   - Files whose `sha256` already exists in the previous master index (`_cronofotos_index.json`
     or `<data_dir>/master_index.json`) are marked as omitted (`ya respaldado en índice maestro`).
3. Incremental export (`exporter.py`, `main.py`):
   - Exporting in incremental mode into a clean destination folder copies only new files,
     organized directly by year and event (`YYYY/YYYY-MM-DD - Evento/...`), omitting files
     present in the previous master index.
   - Updates both `<dest_dir>/_cronofotos_index.json` and `<data_dir>/master_index.json`.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

STATIC_DIR = SRC_DIR / "photo_organizer" / "static"


def _read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def _make_jpeg_bytes(tag: bytes) -> bytes:
    """Build minimal valid JPEG bytes without requiring Pillow."""
    payload = tag + b"\x00\x01\x02\x03"
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xfe"
        + len(payload + b"\x00\x00").to_bytes(2, "big")
        + payload
        + b"\xff\xd9"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _http_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"detail": raw}
        return exc.code, parsed


def _wait_scan_completed(base_url: str, timeout_sec: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        code, data = _http_json("GET", base_url, "/api/scan/status")
        assert code == 200
        last_status = data
        if data.get("status") in {"completed", "error"}:
            return data
        time.sleep(0.05)
    raise AssertionError(f"Scan did not finish within {timeout_sec}s: {last_status}")


def _start_test_server(tmp_path: Path) -> tuple[Any, http.server.ThreadingHTTPServer, str, Path]:
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.main import _LocalAppCore, _make_stdlib_handler

    data_dir = tmp_path / "app_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = OrganizerConfig(
        db_path=data_dir / "organizer.db",
        thumb_cache_dir=data_dir / "thumbs",
    )
    core = _LocalAppCore(cfg)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_stdlib_handler(core))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return core, server, f"http://127.0.0.1:{server.server_port}", data_dir


def test_mod17_01_ui_incremental_delta_controls() -> None:
    """UI exposes optional previous master index field and incremental mode checkbox wired in app.js."""
    html = _read_static("index.html")
    js = _read_static("app.js")

    assert "Índice maestro previo (_cronofotos_index.json)" in html
    assert "Modo incremental (exportar solo archivos nuevos)" in html

    # Verify there is a text/path input for the previous master index and a checkbox for incremental mode
    assert re.search(
        r"Índice maestro previo \(_cronofotos_index\.json\)[\s\S]{0,400}<input[^>]+>",
        html,
    )
    assert re.search(
        r"<input[^>]+type=[\"']checkbox[\"'][^>]*>[\s\S]{0,120}Modo incremental \(exportar solo archivos nuevos\)"
        r"|Modo incremental \(exportar solo archivos nuevos\)[\s\S]{0,120}<input[^>]+type=[\"']checkbox[\"']",
        html,
    )

    # Verify app.js wires incremental/master index controls into backend requests
    assert re.search(r"incremental", js, re.IGNORECASE)
    assert re.search(r"master.?index|index.?path|_cronofotos_index", js, re.IGNORECASE)


def test_mod17_02_scan_marks_master_index_matches_as_omitted(tmp_path: Path) -> None:
    """Files whose sha256 exists in <data_dir>/master_index.json are marked omitted during scan."""
    core, server, base_url, data_dir = _start_test_server(tmp_path)
    try:
        old_bytes = _make_jpeg_bytes(b"backed_up_2023_01")
        new_bytes = _make_jpeg_bytes(b"brand_new_2024_08")
        old_sha = _sha256_bytes(old_bytes)

        master_index_path = data_dir / "master_index.json"
        master_index_path.write_text(
            json.dumps(
                {
                    old_sha: {
                        "rel_path": "2023/2023-05-10 - Viaje/2023-05-10_10-00-00_001.jpg",
                        "size_bytes": len(old_bytes),
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        src_dir = tmp_path / "source_photos"
        src_dir.mkdir()
        (src_dir / "PXL_20230510_100000.jpg").write_bytes(old_bytes)
        (src_dir / "PXL_20240815_140000.jpg").write_bytes(new_bytes)

        clean_dest = tmp_path / "clean_delta_dest"
        clean_dest.mkdir()

        code, _ = _http_json(
            "POST",
            base_url,
            "/api/scan/start",
            {
                "source_dir": str(src_dir),
                "dest_dir": str(clean_dest),
                "incremental_mode": True,
                "incremental": True,
            },
        )
        assert code == 200
        status = _wait_scan_completed(base_url)
        assert status["status"] == "completed"
        job_id = status["job_id"]

        conn = sqlite3.connect(core.cfg.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT source_path, sha256, is_duplicate, target_folder, route_reason "
                "FROM media_items WHERE job_id = ?",
                (job_id,),
            ).fetchall()
        finally:
            conn.close()

        by_name = {Path(r["source_path"]).name: dict(r) for r in rows}
        old_row = by_name["PXL_20230510_100000.jpg"]
        new_row = by_name["PXL_20240815_140000.jpg"]

        old_reason_text = f"{old_row.get('route_reason') or ''} {old_row.get('target_folder') or ''}".lower()
        assert "ya respaldado en índice maestro" in old_reason_text
        assert new_row["is_duplicate"] == 0
    finally:
        server.shutdown()
        server.server_close()


def test_mod17_03_incremental_export_to_clean_folder_and_updates_both_indexes(tmp_path: Path) -> None:
    """Incremental export into a clean folder omits previously indexed files,
    exports only new files under YYYY/YYYY-MM-DD - Evento/..., and updates both
    _cronofotos_index.json and <data_dir>/master_index.json.
    """
    _core, server, base_url, data_dir = _start_test_server(tmp_path)
    try:
        old_bytes = _make_jpeg_bytes(b"already_archived_photo")
        new_bytes = _make_jpeg_bytes(b"new_delta_photo_2024")
        old_sha = _sha256_bytes(old_bytes)
        new_sha = _sha256_bytes(new_bytes)

        master_index_path = data_dir / "master_index.json"
        master_index_path.write_text(
            json.dumps(
                {
                    old_sha: {
                        "rel_path": "2022/2022-11-01 - Recuerdo/old.jpg",
                        "size_bytes": len(old_bytes),
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        src_dir = tmp_path / "src_mixed"
        src_dir.mkdir()
        old_src = src_dir / "PXL_20221101_090000.jpg"
        new_src = src_dir / "PXL_20240815_143000.jpg"
        old_src.write_bytes(old_bytes)
        new_src.write_bytes(new_bytes)

        clean_dest = tmp_path / "Clean_Delta_Drive"
        clean_dest.mkdir()

        code, _ = _http_json(
            "POST",
            base_url,
            "/api/scan/start",
            {
                "source_dir": str(src_dir),
                "dest_dir": str(clean_dest),
                "incremental_mode": True,
                "incremental": True,
            },
        )
        assert code == 200
        status = _wait_scan_completed(base_url)
        assert status["status"] == "completed"
        job_id = status["job_id"]

        exp_code, exp_res = _http_json(
            "POST",
            base_url,
            "/api/export",
            {
                "job_id": job_id,
                "incremental_mode": True,
                "incremental": True,
            },
        )
        assert exp_code == 200
        assert exp_res["exported"] == 1

        exported_media = [
            p
            for p in clean_dest.rglob("*.jpg")
            if "_Duplicados" not in p.parts
        ]
        assert len(exported_media) == 1
        rel = exported_media[0].relative_to(clean_dest).as_posix()
        # Organized directly by YYYY/YYYY-MM-DD - Evento/... for dragging into Google Drive
        assert re.match(r"^2024/2024-08-15 - [^/]+/", rel), f"Unexpected incremental path: {rel}"
        assert exported_media[0].read_bytes() == new_bytes

        # Both _cronofotos_index.json and <data_dir>/master_index.json must be updated
        dest_index_file = clean_dest / "_cronofotos_index.json"
        assert dest_index_file.is_file()
        dest_index = json.loads(dest_index_file.read_text(encoding="utf-8"))
        assert new_sha in dest_index

        assert master_index_path.is_file()
        master_index = json.loads(master_index_path.read_text(encoding="utf-8"))
        assert old_sha in master_index
        assert new_sha in master_index

        # Source files untouched
        assert old_src.read_bytes() == old_bytes
        assert new_src.read_bytes() == new_bytes
    finally:
        server.shutdown()
        server.server_close()


def test_mod17_04_custom_previous_master_index_omits_backed_up_files(tmp_path: Path) -> None:
    """Specifying a custom previous _cronofotos_index.json omits matching files when exporting to a clean folder."""
    _core, server, base_url, data_dir = _start_test_server(tmp_path)
    try:
        prev_bytes = _make_jpeg_bytes(b"custom_prev_index_photo")
        fresh_bytes = _make_jpeg_bytes(b"fresh_photo_2025")
        prev_sha = _sha256_bytes(prev_bytes)
        fresh_sha = _sha256_bytes(fresh_bytes)

        prev_index_dir = tmp_path / "external_backup"
        prev_index_dir.mkdir()
        custom_index_path = prev_index_dir / "_cronofotos_index.json"
        custom_index_path.write_text(
            json.dumps(
                {
                    prev_sha: {
                        "rel_path": "2023/2023-07-20 - Verano/2023-07-20_12-00-00_001.jpg",
                        "size_bytes": len(prev_bytes),
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        src_dir = tmp_path / "src_custom"
        src_dir.mkdir()
        (src_dir / "PXL_20230720_120000.jpg").write_bytes(prev_bytes)
        (src_dir / "PXL_20250110_180000.jpg").write_bytes(fresh_bytes)

        clean_dest = tmp_path / "Clean_Dest_Custom"
        clean_dest.mkdir()

        code, _ = _http_json(
            "POST",
            base_url,
            "/api/scan/start",
            {
                "source_dir": str(src_dir),
                "dest_dir": str(clean_dest),
                "incremental_mode": True,
                "incremental": True,
                "master_index_path": str(custom_index_path),
            },
        )
        assert code == 200
        status = _wait_scan_completed(base_url)
        assert status["status"] == "completed"
        job_id = status["job_id"]

        exp_code, exp_res = _http_json(
            "POST",
            base_url,
            "/api/export",
            {
                "job_id": job_id,
                "incremental_mode": True,
                "incremental": True,
                "master_index_path": str(custom_index_path),
            },
        )
        assert exp_code == 200
        assert exp_res["exported"] == 1

        exported_files = [p for p in clean_dest.rglob("*.jpg") if "_Duplicados" not in p.parts]
        assert len(exported_files) == 1
        assert exported_files[0].read_bytes() == fresh_bytes

        rel = exported_files[0].relative_to(clean_dest).as_posix()
        assert re.match(r"^2025/2025-01-10 - [^/]+/", rel), f"Unexpected incremental path: {rel}"

        master_index_file = data_dir / "master_index.json"
        assert master_index_file.is_file()
        master_index = json.loads(master_index_file.read_text(encoding="utf-8"))
        assert fresh_sha in master_index
    finally:
        server.shutdown()
        server.server_close()
