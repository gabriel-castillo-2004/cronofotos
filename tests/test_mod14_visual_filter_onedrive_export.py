"""Tests for post-delivery modification MOD-14:
1. Real visual analysis with `agy` (`--add-dir`, 60s timeout), no `[SPLIT]` on midnight-merged folders,
   no fixed `Sesión Mañana`/`Sesión Tarde` labels, and `<img>` thumbnail preview in AI suggestions.
2. Folder filter in `Árbol Propuesto` with `× Ver todos` reset button (`app.js`, `main.py`).
3. Safe OneDrive placeholder handling during scan, reported in `Actividad en vivo` and count > 0.
4. Export summary with final count + bytes/GB and `Abrir carpeta en Explorador` (`POST /api/system/open-folder`)
   restricted to the configured destination directory.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
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


def _make_jpeg_bytes(tag: bytes, min_bytes: int = 20000) -> bytes:
    payload = tag + b"\x00\x01\x02\x03"
    if len(payload) < min_bytes:
        payload = payload + (b"\xaa" * (min_bytes - len(payload)))
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xfe"
        + len(payload + b"\x00\x00").to_bytes(2, "big")
        + payload
        + b"\xff\xd9"
    )


def _http_json(method: str, base_url: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"detail": body}
        return exc.code, parsed


def test_mod14_01_generate_tree_ai_suggestions_passes_add_dir_and_60s_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_tree_ai_suggestions passes thumbnail dir via --add-dir with timeout=60s and returns Spanish titles."""
    from photo_organizer import ai_classifier

    fn = getattr(ai_classifier, "generate_tree_ai_suggestions", None) or ai_classifier.suggest_tree_optimizations
    thumb_dir = tmp_path / "ai_thumbs"
    thumb_dir.mkdir()
    thumb_file = thumb_dir / "rep_1.jpg"
    thumb_file.write_bytes(_make_jpeg_bytes(b"visual_test"))

    summaries = [
        {
            "folder": "01_Recuerdos/2024/2024-08-10 - Eventos del día",
            "count": 8,
            "start_date": "2024-08-10T10:00:00+00:00",
            "end_date": "2024-08-10T13:00:00+00:00",
            "location_label": "CDMX, MX",
            "max_gap_hours": 1.0,
            "thumb_path": str(thumb_file),
        }
    ]

    captured_calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured_calls.append((list(cmd), dict(kwargs)))
        payload = {
            "suggestions": [
                {
                    "type": "rename",
                    "before": summaries[0]["folder"],
                    "after": "01_Recuerdos/2024/2024-08-10 - Paseo en el Parque",
                    "reason": "Se observan árboles y caminata al aire libre",
                }
            ]
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload, ensure_ascii=False), stderr="")

    shim = tmp_path / "agy.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")

    with monkeypatch.context() as mp:
        mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if "agy" in str(b) else None)
        mp.setattr(subprocess, "run", fake_run)
        res = fn(summaries, cli_command="agy")

    assert len(captured_calls) == 1
    cmd, kwargs = captured_calls[0]
    assert "--add-dir" in cmd
    add_dir_idx = cmd.index("--add-dir")
    assert add_dir_idx + 1 < len(cmd)
    passed_dir = Path(cmd[add_dir_idx + 1])
    assert passed_dir.exists()
    assert float(kwargs.get("timeout", 0)) >= 60.0
    assert "agy" in str(res.get("engine", "")).lower()
    assert len(res["suggestions"]) == 1
    assert res["suggestions"][0]["after"] == "01_Recuerdos/2024/2024-08-10 - Paseo en el Parque"


def test_mod14_02_no_split_on_midnight_merged_and_forbids_fixed_session_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Midnight-merged folders must not be split, and fixed 'Sesión Mañana'/'Sesión Tarde' labels are forbidden."""
    from photo_organizer import ai_classifier

    fn = getattr(ai_classifier, "generate_tree_ai_suggestions", None) or ai_classifier.suggest_tree_optimizations
    thumb_file = tmp_path / "night.jpg"
    thumb_file.write_bytes(_make_jpeg_bytes(b"night"))

    summaries = [
        {
            "folder": "01_Recuerdos/2024/2024-08-10 - Eventos del día",
            "count": 32,
            "start_date": "2024-08-10T22:30:00+00:00",
            "end_date": "2024-08-11T02:15:00+00:00",
            "location_label": "CDMX, MX",
            "max_gap_hours": 3.5,
            "is_midnight_merged": True,
            "thumb_path": str(thumb_file),
        },
        {
            "folder": "01_Recuerdos/2024/2024-08-12 - Eventos del día",
            "count": 30,
            "start_date": "2024-08-12T08:00:00+00:00",
            "end_date": "2024-08-12T18:00:00+00:00",
            "location_label": "CDMX, MX",
            "max_gap_hours": 5.0,
            "is_midnight_merged": False,
            "thumb_path": str(thumb_file),
        },
    ]

    # 1) Deterministic fallback check (agy unavailable)
    with monkeypatch.context() as mp:
        mp.setattr(shutil, "which", lambda _b, **_kw: None)
        fb_res = fn(summaries, cli_command="agy")

    fb_serialized = json.dumps(fb_res["suggestions"], ensure_ascii=False)
    assert "Sesión Mañana" not in fb_serialized
    assert "Sesión Tarde" not in fb_serialized
    fb_splits = [s["before"] for s in fb_res["suggestions"] if s.get("type") == "split"]
    assert summaries[0]["folder"] not in fb_splits

    # 2) agy response sanitization check
    shim = tmp_path / "agy.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        payload = {
            "suggestions": [
                {
                    "type": "split",
                    "before": summaries[0]["folder"],
                    "after": ["01_Recuerdos/2024/2024-08-10 - Cena", "01_Recuerdos/2024/2024-08-11 - Regreso"],
                    "reason": "Cruce de medianoche",
                },
                {
                    "type": "split",
                    "before": summaries[1]["folder"],
                    "after": ["01_Recuerdos/2024/2024-08-12 - Sesión Mañana", "01_Recuerdos/2024/2024-08-12 - Sesión Tarde"],
                    "reason": "Etiquetas fijas prohibidas",
                },
            ]
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload, ensure_ascii=False), stderr="")

    with monkeypatch.context() as mp:
        mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if "agy" in str(b) else None)
        mp.setattr(subprocess, "run", fake_run)
        agy_res = fn(summaries, cli_command="agy")

    agy_serialized = json.dumps(agy_res["suggestions"], ensure_ascii=False)
    assert "Sesión Mañana" not in agy_serialized
    assert "Sesión Tarde" not in agy_serialized
    agy_splits = [s["before"] for s in agy_res["suggestions"] if s.get("type") == "split"]
    assert summaries[0]["folder"] not in agy_splits


def test_mod14_03_ai_suggestions_ui_renders_thumbnail_img() -> None:
    """AI optimizer panel in app.js renders representative <img> thumbnail alongside each suggestion."""
    html = _read_static("index.html")
    js = _read_static("app.js")
    assert "ai-suggestions-list" in html
    assert "renderAiSuggestions" in js
    fn_match = re.search(r"function\s+renderAiSuggestions\b[\s\S]{0,2500}", js)
    assert fn_match is not None
    fn_body = fn_match.group(0)
    assert ('createElement("img")' in fn_body) or ("createElement('img')" in fn_body) or ("<img" in fn_body)


def test_mod14_04_folder_filter_and_clear_filter_button(tmp_path: Path) -> None:
    """Clicking a folder filters /api/tree items and UI exposes '× Ver todos' to clear the filter."""
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.db import init_db
    from photo_organizer.main import _LocalAppCore, _make_stdlib_handler

    src_dir = tmp_path / "src_filter"
    dest_dir = tmp_path / "dest_filter"
    src_dir.mkdir()
    (src_dir / "PXL_20240601_100000.jpg").write_bytes(_make_jpeg_bytes(b"day1"))
    (src_dir / "PXL_20240605_150000.jpg").write_bytes(_make_jpeg_bytes(b"day2"))

    cfg = OrganizerConfig(db_path=tmp_path / "organizer.db", thumb_cache_dir=tmp_path / "thumbs")
    init_db(cfg.db_path)
    core = _LocalAppCore(cfg)
    job_id = asyncio.run(core.orchestrator.run_job(src_dir, dest_dir))

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_stdlib_handler(core))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        st_all, data_all = _http_json("GET", base_url, f"/api/tree?job_id={job_id}")
        assert st_all == 200
        assert len(data_all["folders"]) >= 2
        target_folder = data_all["folders"][0]["folder"]
        encoded_folder = urllib.request.quote(target_folder, safe="")
        st_filt, data_filt = _http_json("GET", base_url, f"/api/tree?job_id={job_id}&folder={encoded_folder}")
        assert st_filt == 200
        assert len(data_filt["items"]) >= 1
        assert all(it["target_folder"] == target_folder for it in data_filt["items"])
        assert len(data_filt["items"]) < len(data_all["items"])
    finally:
        server.shutdown()
        server.server_close()

    html = _read_static("index.html")
    js = _read_static("app.js")
    assert "Ver todos" in (html + "\n" + js)


def test_mod14_05_onedrive_placeholder_detected_without_read_and_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OneDrive cloud-only placeholders are skipped without reading bytes, counted in snapshot, and shown in UI."""
    from photo_organizer import worker
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.db import init_db, open_connection
    from photo_organizer.media import MediaMetadataService
    from photo_organizer.worker import PipelineOrchestrator

    src_dir = tmp_path / "src_od"
    dest_dir = tmp_path / "dest_od"
    src_dir.mkdir()
    real_img = src_dir / "PXL_20240701_120000.jpg"
    real_img.write_bytes(_make_jpeg_bytes(b"real_photo"))
    cloud_img = src_dir / "PXL_20240701_120500_cloud.jpg"
    cloud_img.write_bytes(b"")

    hashed_paths: list[Path] = []
    orig_hash = worker.compute_sha256_streaming

    def spy_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
        hashed_paths.append(path)
        return orig_hash(path, chunk_size)

    cfg = OrganizerConfig(db_path=tmp_path / "organizer.db", thumb_cache_dir=tmp_path / "thumbs")
    init_db(cfg.db_path)
    snapshots: list[dict[str, Any]] = []
    orch = PipelineOrchestrator(
        cfg.db_path,
        MediaMetadataService(),
        on_progress=lambda snap: snapshots.append(snap.to_dict()),
    )

    with monkeypatch.context() as mp:
        mp.setattr(worker, "is_onedrive_placeholder", lambda p, **_kw: Path(p).name == cloud_img.name)
        mp.setattr(worker, "compute_sha256_streaming", spy_hash)
        job_id = asyncio.run(orch.run_job(src_dir, dest_dir))

    assert all(p.name != cloud_img.name for p in hashed_paths)

    conn = open_connection(cfg.db_path)
    try:
        row = conn.execute(
            "SELECT is_placeholder, route_reason FROM media_items WHERE job_id = ? AND source_path = ?",
            (job_id, str(cloud_img)),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert int(row["is_placeholder"]) == 1
    assert row["route_reason"] == "onedrive_placeholder"

    last_snap = snapshots[-1]
    placeholder_count = (
        last_snap.get("placeholders_count")
        or last_snap.get("onedrive_placeholders")
        or last_snap.get("placeholder_count")
        or 0
    )
    assert int(placeholder_count) == 1

    html = _read_static("index.html")
    js = _read_static("app.js")
    combined = html + "\n" + js
    assert "placeholder" in combined.lower() or "nube" in combined.lower() or "onedrive" in combined.lower()


def test_mod14_06_export_summary_bytes_and_open_folder_restricted_to_dest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export returns count + exported bytes, and POST /api/system/open-folder is restricted to dest_dir."""
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.db import init_db
    from photo_organizer.main import _LocalAppCore, _make_stdlib_handler

    src_dir = tmp_path / "src_exp"
    dest_dir = tmp_path / "dest_exp"
    outside_dir = tmp_path / "outside_dir"
    src_dir.mkdir()
    outside_dir.mkdir()
    p1 = src_dir / "PXL_20240710_100000.jpg"
    p1.write_bytes(_make_jpeg_bytes(b"exp1", min_bytes=22000))

    cfg = OrganizerConfig(db_path=tmp_path / "organizer.db", thumb_cache_dir=tmp_path / "thumbs")
    init_db(cfg.db_path)
    core = _LocalAppCore(cfg)
    job_id = asyncio.run(core.orchestrator.run_job(src_dir, dest_dir))

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_stdlib_handler(core))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    opened_paths: list[str] = []
    try:
        st_exp, res_exp = _http_json("POST", base_url, "/api/export", {"job_id": job_id})
        assert st_exp == 200
        assert res_exp["exported"] == 1
        exported_bytes = (
            res_exp.get("exported_bytes")
            or res_exp.get("total_bytes")
            or res_exp.get("bytes_exported")
            or 0
        )
        assert int(exported_bytes) >= 22000

        with monkeypatch.context() as mp:
            if hasattr(os, "startfile"):
                mp.setattr(os, "startfile", lambda p: opened_paths.append(str(p)), raising=False)
            mp.setattr(subprocess, "Popen", lambda args, **_kw: opened_paths.append(str(args[-1])))

            st_ok, res_ok = _http_json(
                "POST",
                base_url,
                "/api/system/open-folder",
                {"job_id": job_id, "path": str(dest_dir)},
            )
            assert st_ok == 200
            assert res_ok.get("opened") is True

            st_bad, _ = _http_json(
                "POST",
                base_url,
                "/api/system/open-folder",
                {"job_id": job_id, "path": str(outside_dir)},
            )
            assert st_bad in (400, 403)
    finally:
        server.shutdown()
        server.server_close()

    js = _read_static("app.js")
    html = _read_static("index.html")
    combined = html + "\n" + js
    assert "/api/system/open-folder" in js
    assert "Abrir carpeta en Explorador" in combined
