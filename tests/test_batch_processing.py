"""Tests for post-delivery modification MOD-10 / MOD-1: batch processing,
detailed batch progress, and quick single-batch test mode (`solo_primer_lote`).
"""

from __future__ import annotations

import asyncio
import inspect
import re
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


def _make_jpeg(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal valid JPEG SOI + JFIF APP0 + unique payload + EOI (no Pillow dependency)
    payload = bytes([color[0] % 256, color[1] % 256, color[2] % 256]) * (size[0] % 13 + 4)
    data = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xfe"
        + len(payload + b"\x00\x00").to_bytes(2, "big")
        + payload
        + b"\xff\xd9"
    )
    path.write_bytes(data)


def _create_source_library(
    src_dir: Path,
    unique_count: int = 55,
    duplicate_count: int = 3,
) -> dict[Path, bytes]:
    """Create a deterministic source library and return initial file bytes."""
    snapshot: dict[Path, bytes] = {}
    for i in range(unique_count):
        p = src_dir / f"2024-05-{(i % 28) + 1:02d}" / f"photo_{i:03d}.jpg"
        _make_jpeg(p, ((i * 17) % 255, (i * 53) % 255, (i * 97) % 255), size=(32 + (i % 7), 32))
        snapshot[p] = p.read_bytes()

    # Add exact byte-for-byte duplicates of the first `duplicate_count` files
    first_files = sorted(snapshot.keys())[:duplicate_count]
    for idx, orig in enumerate(first_files):
        dup_path = src_dir / "dups" / f"dup_{idx:02d}.jpg"
        dup_path.parent.mkdir(parents=True, exist_ok=True)
        dup_path.write_bytes(orig.read_bytes())
        snapshot[dup_path] = dup_path.read_bytes()

    return snapshot


def test_mod10_01_frontend_batch_controls_and_progress_panel() -> None:
    """UI exposes Tamaño de lote (50,100,250,500), Probar solo 1 lote, and batch panel."""
    html = _read_static("index.html")
    js = _read_static("app.js")
    css = _read_static("styles.css")

    # Top controls: Tamaño de lote selector with 50, 100, 250, 500 (100 default)
    assert "Tamaño de lote" in html
    for opt in ("50", "100", "250", "500"):
        assert re.search(rf"<option[^>]*value=[\"']{opt}[\"'][^>]*>", html), (
            f"Missing batch size option {opt} in index.html"
        )
    assert re.search(r"<option[^>]*value=[\"']100[\"'][^>]*selected", html) or (
        "100" in js
    )

    # Top controls: test-mode (single batch) checkbox (single source of truth after MOD-12)
    assert "Modo prueba (100 archivos)" in html or "Probar solo 1 lote" in html
    assert re.search(r"<input[^>]*type=[\"']checkbox[\"']", html)

    # Visual batch progress panel in HTML + rendering in app.js
    combined_ui = html + "\n" + js + "\n" + css
    assert re.search(r"lote", combined_ui, re.IGNORECASE)
    assert re.search(r"solo_primer_lote", js)
    assert re.search(r"Lote\s+.*\s+de\s+", js) or re.search(r"Lote\s+\$\{", js)
    for state_name in ("pendiente", "en_proceso", "completado", "error"):
        assert state_name in js


def test_mod10_02_global_scan_batches_and_solo_primer_lote(tmp_path: Path) -> None:
    """Global read-only scan deduplicates globally, splits into batches, and respects solo_primer_lote."""
    import time
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.main import _LocalAppCore

    src_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    initial_files = _create_source_library(src_dir, unique_count=55, duplicate_count=3)

    cfg = OrganizerConfig(
        db_path=tmp_path / "organizer.db",
        thumb_cache_dir=tmp_path / "thumbs",
    )
    core = _LocalAppCore(cfg)

    core.scan_start(
        {
            "source_dir": str(src_dir),
            "dest_dir": str(dest_dir),
            "batch_size": 50,
            "solo_primer_lote": True,
        }
    )

    # Wait briefly for async scan to finish Lote 1
    status_data: dict[str, Any] = {}
    for _ in range(100):
        status_data = core.scan_status()
        if status_data.get("status") in {"completed", "paused", "partial", "stopped", "failed"}:
            break
        time.sleep(0.05)

    assert status_data.get("status") != "failed"
    assert status_data.get("total_files") == len(initial_files)
    assert status_data.get("duplicates_count") == 3

    # Batch metadata must show 2 batches (50 + 8) with only Lote 1 completed
    batches = status_data.get("batches") or status_data.get("lotes") or []
    assert len(batches) == 2
    assert batches[0]["status"] == "completado"
    assert batches[1]["status"] == "pendiente"

    # Source directory must remain 100% untouched
    for p, expected_bytes in initial_files.items():
        assert p.exists()
        assert p.read_bytes() == expected_bytes


def test_mod10_03_export_batches_solo_primer_lote_and_resume_without_repeat(tmp_path: Path) -> None:
    """Export processes sequentially by batch, stops after Lote 1 when solo_primer_lote=True,
    continues remaining batches without repeating completed ones, and never alters source.
    """
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.db import open_connection
    from photo_organizer.main import _LocalAppCore

    src_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    initial_files = _create_source_library(src_dir, unique_count=55, duplicate_count=2)

    cfg = OrganizerConfig(
        db_path=tmp_path / "organizer.db",
        thumb_cache_dir=tmp_path / "thumbs",
    )
    core = _LocalAppCore(cfg)

    # Run full scan first (default batch_size=100, or explicit 50)
    job_id = asyncio.run(
        core.orchestrator.run_job(src_dir, dest_dir)
    )

    # Export only Lote 1 (batch_size=50, solo_primer_lote=True)
    sig = inspect.signature(core.export_tree)
    export_kwargs: dict[str, Any] = {}
    if "batch_size" in sig.parameters:
        export_kwargs["batch_size"] = 50
    elif "tamano_lote" in sig.parameters:
        export_kwargs["tamano_lote"] = 50
    if "solo_primer_lote" in sig.parameters:
        export_kwargs["solo_primer_lote"] = True

    res_first = core.export_tree(job_id, **export_kwargs)
    assert res_first["exported"] == 50

    batches_first = res_first.get("batches") or res_first.get("lotes") or []
    assert len(batches_first) == 2
    assert batches_first[0]["status"] == "completado"
    assert batches_first[1]["status"] == "pendiente"
    assert "elapsed_seconds" in batches_first[0] or "tiempo_transcurrido" in batches_first[0]

    # Continue remaining batches (solo_primer_lote=False) — Lote 1 must NOT be repeated
    if "solo_primer_lote" in sig.parameters:
        export_kwargs["solo_primer_lote"] = False
    res_rest = core.export_tree(job_id, **export_kwargs)
    assert res_rest["exported"] == len(initial_files) - 50

    conn = open_connection(cfg.db_path)
    try:
        verified = conn.execute(
            "SELECT COUNT(*) AS cnt FROM media_items WHERE job_id = ? AND export_state = 'VERIFIED'",
            (job_id,),
        ).fetchone()["cnt"]
        assert verified == len(initial_files)
    finally:
        conn.close()

    # Source files must remain completely untouched
    current_src_files = [p for p in src_dir.rglob("*") if p.is_file()]
    assert len(current_src_files) == len(initial_files)
    for p, expected_bytes in initial_files.items():
        assert p.read_bytes() == expected_bytes


def test_mod10_04_ai_classify_batches_and_solo_primer_lote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clasificar con IA processes sequentially in batches, tracks batch states/counts,
    and supports solo_primer_lote and continuing remaining batches.
    """
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.db import open_connection
    from photo_organizer.main import _LocalAppCore

    src_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    _create_source_library(src_dir, unique_count=55, duplicate_count=0)

    def fake_agy_ok(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="Eventos_2024\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_agy_ok)

    cfg = OrganizerConfig(
        db_path=tmp_path / "organizer.db",
        thumb_cache_dir=tmp_path / "thumbs",
    )
    core = _LocalAppCore(cfg)
    job_id = asyncio.run(core.orchestrator.run_job(src_dir, dest_dir))

    conn = open_connection(cfg.db_path)
    try:
        item_ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM media_items WHERE job_id = ? ORDER BY id ASC",
                (job_id,),
            ).fetchall()
        ]
    finally:
        conn.close()

    assert len(item_ids) == 55

    sig = inspect.signature(core.ai_classify)
    ai_kwargs: dict[str, Any] = {}
    if "batch_size" in sig.parameters:
        ai_kwargs["batch_size"] = 50
    elif "tamano_lote" in sig.parameters:
        ai_kwargs["tamano_lote"] = 50
    if "solo_primer_lote" in sig.parameters:
        ai_kwargs["solo_primer_lote"] = True

    res_first = core.ai_classify(item_ids, confirmed=True, **ai_kwargs)
    batches_first = res_first.get("batches") or res_first.get("lotes") or []
    assert len(batches_first) == 2
    assert batches_first[0]["status"] == "completado"
    assert batches_first[1]["status"] == "pendiente"
    assert len(res_first.get("results", {})) == 50

    if "solo_primer_lote" in sig.parameters:
        ai_kwargs["solo_primer_lote"] = False
    res_rest = core.ai_classify(item_ids, confirmed=True, **ai_kwargs)
    batches_rest = res_rest.get("batches") or res_rest.get("lotes") or []
    assert all(b["status"] == "completado" for b in batches_rest)
