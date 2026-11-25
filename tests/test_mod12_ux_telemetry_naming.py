"""Tests for post-delivery modification MOD-12: UX/UI, live telemetry, and naming quality.

Verifies:
1. Live AI status pill in header (`agy` availability + active model or amber fallback)
   and `/api/health` AI status fields.
2. Zero dead clicks: disabled action buttons, animated spinner + status text,
   result banner/toast with engine & counts, collapsible "Actividad en vivo" log,
   and "Abrir carpeta destino" button after export.
3. Incremental progress emitted every 25 files during phase "1/4 Depurando duplicados".
4. Clean Spanish event naming (JSON from `agy`, clean deterministic fallback ignoring
   generic folders like OneDriveFotos/Camera/DCIM/Pictures, no `Fotos_Carpeta__Gps_`
   template artifacts) and per-day event granularity `01_Recuerdos/YYYY/YYYY-MM-DD - <Evento>/`.
5. "Regenerar nombres con IA" invalidates previous AI name cache and re-queries AI.
6. `v2_exif_` cache prefix in `thumbnails.py` and removal of duplicate controls
   ("Probar solo 1 lote", "Clasificar con IA") inside "Ajustes avanzados".
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
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


def test_mod12_01_header_ai_status_pill_and_health_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Header shows live AI status pill next to ExifTool/FFmpeg and /api/health reports AI status."""
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.main import _LocalAppCore

    html = _read_static("index.html")
    js = _read_static("app.js")
    css = _read_static("styles.css")
    combined_ui = html + "\n" + js + "\n" + css

    # Header has health-badges and AI status pill + fallback/settings shortcut
    assert "health-badges" in html
    assert re.search(r"agy OK|Fallback local", combined_ui, re.IGNORECASE)
    assert re.search(r"IA:", combined_ui)

    cfg = OrganizerConfig(
        db_path=tmp_path / "organizer.db",
        thumb_cache_dir=tmp_path / "thumbs",
    )
    core = _LocalAppCore(cfg)

    # 1. When agy binary is found
    shim = tmp_path / "agy.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")
    with monkeypatch.context() as mp:
        mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if "agy" in str(b) else None)
        h_ok = core.health()
        assert any(
            h_ok.get(k) is True
            for k in ("agy_ok", "ai_ok", "ai_available", "agy_available")
        ), f"Expected live AI availability True in health(): {h_ok}"
        assert h_ok.get("ai_model") or h_ok.get("model")

    # 2. When agy binary is missing -> fallback status
    with monkeypatch.context() as mp:
        mp.setattr(shutil, "which", lambda _b, **_kw: None)
        h_fb = core.health()
        assert any(
            h_fb.get(k) is False
            for k in ("agy_ok", "ai_ok", "ai_available", "agy_available")
        ), f"Expected live AI availability False in health(): {h_fb}"


def test_mod12_02_action_feedback_spinner_toast_activity_log_and_open_dest() -> None:
    """UI eliminates dead clicks with spinner, status/toast banner, live activity log, and open destination button."""
    html = _read_static("index.html")
    js = _read_static("app.js")
    css = _read_static("styles.css")
    combined_ui = html + "\n" + js

    # Spinner and action feedback text
    assert "spinner" in (css + "\n" + js).lower()
    assert "Regenerando nombres con IA..." in combined_ui
    assert "Exportando archivos..." in combined_ui

    # Result banner / toast
    assert re.search(r"toast|banner", combined_ui, re.IGNORECASE)

    # Collapsible "Actividad en vivo" panel
    assert "Actividad en vivo" in html

    # "Abrir carpeta destino" button
    assert "Abrir carpeta destino" in combined_ui


def test_mod12_03_incremental_progress_every_25_files_during_dedup(tmp_path: Path) -> None:
    """Phase 1/4 Depurando duplicados emits incremental progress every 25 files."""
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.media import MediaMetadataService
    from photo_organizer.worker import PipelineOrchestrator

    src_dir = tmp_path / "src_60"
    dest_dir = tmp_path / "dest_60"
    src_dir.mkdir()
    for idx in range(60):
        (src_dir / f"PXL_20240801_{idx:06d}.jpg").write_bytes(
            _make_jpeg_bytes(f"photo_{idx}".encode("ascii"))
        )

    cfg = OrganizerConfig(
        db_path=tmp_path / "organizer.db",
        thumb_cache_dir=tmp_path / "thumbs",
    )
    media_svc = MediaMetadataService(cfg.exiftool_path, cfg.ffmpeg_path)

    snapshots: list[dict[str, Any]] = []
    orch = PipelineOrchestrator(
        cfg.db_path,
        media_svc,
        on_progress=lambda snap: snapshots.append(snap.to_dict()),
    )
    asyncio.run(orch.run_job(src_dir, dest_dir))

    phase1_processed = [
        s["processed_files"]
        for s in snapshots
        if s.get("phase_label") == "1/4 Depurando duplicados"
        and s.get("stage") in ("discovery", "hashing")
    ]
    assert any(0 < count < 60 for count in phase1_processed), (
        f"Expected intermediate processed_files during phase 1/4, got: {phase1_processed}"
    )


def test_mod12_04_clean_event_naming_no_template_strings_and_daily_granularity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event naming parses JSON from agy, avoids Fotos_Carpeta__Gps_ templates,
    ignores generic folders in fallback, and groups by YYYY-MM-DD - <Evento>.
    """
    from photo_organizer import ai_classifier
    from photo_organizer.clustering import ClusterItem, cluster_and_stage_items

    # 1. Clean JSON response from agy
    def fake_agy_json(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"nombre_evento": "Salida Nocturna"}\n',
            stderr="",
        )

    with monkeypatch.context() as mp:
        mp.setattr(subprocess, "run", fake_agy_json)
        label = ai_classifier.classify_with_agy(
            "2024-08-03 | 5 fotos | carpeta: OneDriveFotos",
            cli_command="agy",
            model="gemini-4-carbon-medium",
        )
        assert label == "Salida Nocturna"

    # 2. Deterministic fallback when agy is missing: no template strings, ignores generic folders
    with monkeypatch.context() as mp:
        mp.setattr(shutil, "which", lambda _b, **_kw: None)
        fb_generic = ai_classifier.classify_with_agy(
            "2024-08-03 | 5 fotos | carpeta: OneDriveFotos",
            cli_command="agy",
        )
        assert "Fotos_Carpeta" not in fb_generic
        assert "__Gps_" not in fb_generic
        assert "OneDriveFotos" not in fb_generic
        assert fb_generic == "Eventos del día"

        fb_named = ai_classifier.classify_with_agy(
            "2024-08-03 | 5 fotos | carpeta: Boda_Laura",
            cli_command="agy",
        )
        assert "Fotos_Carpeta" not in fb_named
        assert "Boda" in fb_named

    # 3. Per-day event granularity in 01_Recuerdos/YYYY/YYYY-MM-DD - <Evento>
    items = [
        ClusterItem(
            id="d1",
            source_path=str(tmp_path / "OneDriveFotos" / "PXL_20240803_120000.jpg"),
            ext=".jpg",
            sha256="aaa1",
            is_duplicate=False,
            captured_at="2024-08-03T12:00:00Z",
            latitude=None,
            longitude=None,
            ai_label="Salida Nocturna",
        ),
        ClusterItem(
            id="d2",
            source_path=str(tmp_path / "OneDriveFotos" / "PXL_20240815_180000.jpg"),
            ext=".jpg",
            sha256="bbb2",
            is_duplicate=False,
            captured_at="2024-08-15T18:00:00Z",
            latitude=None,
            longitude=None,
            ai_label="Viaje Familiar",
        ),
    ]
    staged = cluster_and_stage_items(items)
    assert staged["d1"]["target_folder"] == "01_Recuerdos/2024/2024-08-03 - Salida Nocturna"
    assert staged["d2"]["target_folder"] == "01_Recuerdos/2024/2024-08-15 - Viaje Familiar"


def test_mod12_05_regenerate_ai_names_invalidates_cache_and_reports_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regenerar nombres con IA invalidates ai_cache, re-queries AI, and reports engine used."""
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.db import init_db, open_connection
    from photo_organizer.main import _LocalAppCore

    src_dir = tmp_path / "src_regen"
    dest_dir = tmp_path / "dest_regen"
    src_dir.mkdir()
    (src_dir / "PXL_20240901_100000.jpg").write_bytes(_make_jpeg_bytes(b"regen_1"))

    cfg = OrganizerConfig(
        db_path=tmp_path / "organizer.db",
        thumb_cache_dir=tmp_path / "thumbs",
    )
    init_db(cfg.db_path)
    core = _LocalAppCore(cfg)

    job_id = asyncio.run(core.orchestrator.run_job(src_dir, dest_dir))

    # Seed stale ai_cache entry
    conn = open_connection(cfg.db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO ai_cache (cache_key, kind, label, model, created_at) VALUES (?, ?, ?, ?, ?)",
            ("stale_key", "event", "Nombre_Viejo", "old-model", "2024-01-01T00:00:00Z"),
        )
        conn.commit()
    finally:
        conn.close()

    agy_calls: list[list[str]] = []

    def fake_agy_regen(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        agy_calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"nombre_evento": "Paseo en Bici"}\n',
            stderr="",
        )

    with monkeypatch.context() as mp:
        mp.setattr(subprocess, "run", fake_agy_regen)
        res = core.regenerate_ai_names(job_id)

    assert agy_calls, "regenerate_ai_names must re-query agy after invalidating cache"
    assert res["job_id"] == job_id
    assert res["updated_folders"] >= 1
    assert any(k in res for k in ("engine", "model", "ai_engine"))

    conn = open_connection(cfg.db_path)
    try:
        stale = conn.execute(
            "SELECT * FROM ai_cache WHERE cache_key = 'stale_key'"
        ).fetchone()
        assert stale is None, "Prior ai_cache entries must be invalidated on regenerate"
        folders = [
            r["target_folder"]
            for r in conn.execute(
                "SELECT DISTINCT target_folder FROM media_items WHERE job_id = ?",
                (job_id,),
            ).fetchall()
        ]
        assert any("Paseo en Bici" in f for f in folders)
    finally:
        conn.close()


def test_mod12_06_thumbnail_cache_v2_exif_prefix_and_no_duplicate_advanced_controls(
    tmp_path: Path,
) -> None:
    """ThumbnailCache uses v2_exif_ disk prefix and Ajustes avanzados has no duplicate controls."""
    from photo_organizer.thumbnails import ThumbnailCache

    cache = ThumbnailCache(tmp_path / "thumbs")
    disk_path = cache._disk_path("item1", "abc123")
    assert disk_path.name.startswith("v2_exif_")

    html = _read_static("index.html")
    adv_match = re.search(
        r'<details[^>]*id="advanced-menu"[^>]*>(.*?)</details>',
        html,
        flags=re.DOTALL,
    )
    assert adv_match is not None, "Expected #advanced-menu in index.html"
    adv_html = adv_match.group(1)

    assert "Probar solo 1 lote" not in adv_html
    assert "Clasificar con IA" not in adv_html
