"""Tests for post-delivery modification MOD-11: Piloto Automático redesign.

Verifies:
1. Minimalist frictionless UI:
   - Simplified top bar with 2 folder selectors ("Carpeta con fotos nuevas" and
     "Mi Biblioteca Organizada"), "Organizar Automáticamente" primary button,
     "Modo prueba (100 archivos)" checkbox, and compact "Configuración" access.
   - Single clean progress bar with 4 clear phases ("1/4 Depurando duplicados",
     "2/4 Clasificando fotos y capturas", "3/4 Nombrando carpetas con IA",
     "4/4 Listo") and summary counters (nuevas, duplicadas omitidas, capturas separadas),
     replacing the open batch-cards grid.
   - Manual tree editing buttons hidden under "Ajustes avanzados" secondary menu,
     plus direct "Regenerar nombres con IA" button.
   - EXIF orientation fix (`ImageOps.exif_transpose`) in thumbnail generation.
2. Incremental deduplication against Destination Library via `_cronofotos_index.json`.
3. Smart date extraction priority (EXIF -> FFmpeg video -> filename patterns)
   and automatic routing (`02_Capturas_y_WhatsApp/YYYY-MM/` vs `01_Recuerdos/YYYY/`),
   plus automatic Spanish event folder naming via `agy` with deterministic fallback.
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


def test_mod11_01_autopilot_ui_minimalist_controls_and_phases() -> None:
    """UI exposes minimalist Piloto Automático controls, 4 phases, summary counters,
    Ajustes avanzados menu, and Regenerar nombres con IA button.
    """
    html = _read_static("index.html")
    js = _read_static("app.js")
    combined = html + "\n" + js

    # Simplified top controls
    assert "Carpeta con fotos nuevas" in html
    assert "Mi Biblioteca Organizada" in html
    assert "Organizar Automáticamente" in html
    assert "Modo prueba (100 archivos)" in html
    assert "Configuración" in html

    # Open batch cards grid replaced by single clean progress bar with 4 phases
    assert '<details id="batch-panel" class="batch-panel" open>' not in html
    for phase in (
        "1/4 Depurando duplicados",
        "2/4 Clasificando fotos y capturas",
        "3/4 Nombrando carpetas con IA",
        "4/4 Listo",
    ):
        assert phase in combined, f"Missing progress phase: {phase}"

    # Summary counters
    assert re.search(r"nuevas", combined, re.IGNORECASE)
    assert re.search(r"duplicadas\s+omitidas", combined, re.IGNORECASE)
    assert re.search(r"capturas\s+separadas", combined, re.IGNORECASE)

    # Manual tree editing hidden under "Ajustes avanzados" + direct AI rename button
    assert "Ajustes avanzados" in html
    assert "Regenerar nombres con IA" in html
    assert "Renombrar" in html
    assert "Fusionar" in html
    assert "Nueva carpeta" in html

    # app.js wires autopilot and AI rename actions to backend API
    assert "/api/" in js


def test_mod11_02_thumbnail_exif_transpose_orientation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ThumbnailCache applies ImageOps.exif_transpose so portrait EXIF photos render upright."""
    from photo_organizer import thumbnails

    thumb_src = (SRC_DIR / "photo_organizer" / "thumbnails.py").read_text(encoding="utf-8")
    assert "ImageOps" in thumb_src
    assert "exif_transpose" in thumb_src

    sample_jpg = tmp_path / "portrait_exif6.jpg"
    sample_jpg.write_bytes(_make_jpeg_bytes(b"exif6"))

    transpose_calls: list[Any] = []
    saved_sizes: list[tuple[int, int]] = []

    class _FakeImg:
        def __init__(self, size: tuple[int, int]) -> None:
            self.size = size

        def __enter__(self) -> "_FakeImg":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def convert(self, _mode: str) -> "_FakeImg":
            return self

        def thumbnail(self, max_size: tuple[int, int]) -> None:
            w, h = self.size
            ratio = min(max_size[0] / w, max_size[1] / h, 1.0)
            self.size = (max(1, int(w * ratio)), max(1, int(h * ratio)))

        def save(self, buf: Any, format: str = "JPEG", quality: int = 80) -> None:
            _ = (format, quality)
            saved_sizes.append(self.size)
            buf.write(b"FAKEJPEG:" + f"{self.size[0]}x{self.size[1]}".encode("ascii"))

    class _FakeImageModule:
        @staticmethod
        def open(_path: Path) -> _FakeImg:
            # Raw sensor pixels are landscape (400x200), EXIF orientation=6 means portrait
            return _FakeImg((400, 200))

    class _FakeImageOpsModule:
        @staticmethod
        def exif_transpose(img: _FakeImg) -> _FakeImg:
            transpose_calls.append(img.size)
            # Rotate 90 deg -> portrait (200x400)
            return _FakeImg((img.size[1], img.size[0]))

    monkeypatch.setattr(thumbnails, "Image", _FakeImageModule)
    monkeypatch.setattr(thumbnails, "ImageOps", _FakeImageOpsModule, raising=False)

    cache = thumbnails.ThumbnailCache(tmp_path / "thumbs", thumb_size=(320, 320))
    out = cache.get_or_create("portrait1", sample_jpg, "sha_portrait_1")

    assert len(transpose_calls) == 1
    assert saved_sizes and saved_sizes[0][1] > saved_sizes[0][0]
    assert out.startswith(b"FAKEJPEG:")


def test_mod11_03_incremental_dedup_cronofotos_index(tmp_path: Path) -> None:
    """Maintains _cronofotos_index.json in destination library, indexes pre-existing files,
    and skips already-archived files on subsequent runs without touching source.
    """
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.exporter import run_two_phase_export
    from photo_organizer.media import MediaMetadataService
    from photo_organizer.worker import PipelineOrchestrator

    dest_dir = tmp_path / "Mi_Biblioteca_Organizada"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Pre-existing photo already inside destination library before CronoFotos runs
    existing_bytes = _make_jpeg_bytes(b"already_in_library_2023")
    existing_dest = dest_dir / "01_Recuerdos" / "2023" / "existing.jpg"
    existing_dest.parent.mkdir(parents=True, exist_ok=True)
    existing_dest.write_bytes(existing_bytes)

    # First source batch: 1 duplicate of pre-existing dest file + 1 brand new photo
    src_1 = tmp_path / "fotos_nuevas_1"
    src_1.mkdir()
    dup_of_existing = src_1 / "PXL_20240501_100000.jpg"
    dup_of_existing.write_bytes(existing_bytes)
    new_photo_1 = src_1 / "PXL_20240502_110000.jpg"
    new_bytes_1 = _make_jpeg_bytes(b"new_photo_may_2024")
    new_photo_1.write_bytes(new_bytes_1)

    cfg = OrganizerConfig(
        db_path=tmp_path / "organizer.db",
        thumb_cache_dir=tmp_path / "thumbs",
    )
    media_svc = MediaMetadataService(cfg.exiftool_path, cfg.ffmpeg_path)
    orch = PipelineOrchestrator(cfg.db_path, media_svc)

    job_1 = asyncio.run(orch.run_job(src_1, dest_dir))
    res_1 = run_two_phase_export(cfg.db_path, job_1, dest_dir, media_svc)

    index_file = dest_dir / "_cronofotos_index.json"
    assert index_file.is_file(), "_cronofotos_index.json must be created inside dest_dir"
    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    assert isinstance(index_data, (dict, list))

    # Only new_photo_1 should be exported as a new memory; dup_of_existing must be skipped/deduped
    assert res_1["exported"] <= 2
    memories_after_1 = list((dest_dir / "01_Recuerdos").rglob("*.jpg")) if (dest_dir / "01_Recuerdos").exists() else list(dest_dir.rglob("*.jpg"))
    # existing.jpg + new_photo_1 (not duplicated)
    assert len([p for p in memories_after_1 if "_Duplicados" not in p.parts]) == 2

    # Second run months later: mixed old (already exported) + genuinely new photo
    src_2 = tmp_path / "fotos_nuevas_2"
    src_2.mkdir()
    old_mixed = src_2 / "PXL_20240502_110000.jpg"
    old_mixed.write_bytes(new_bytes_1)
    new_photo_2 = src_2 / "PXL_20240915_153000.jpg"
    new_photo_2.write_bytes(_make_jpeg_bytes(b"brand_new_sept_2024"))

    job_2 = asyncio.run(orch.run_job(src_2, dest_dir))
    res_2 = run_two_phase_export(cfg.db_path, job_2, dest_dir, media_svc)

    # old_mixed must be skipped/omitted automatically; only new_photo_2 is newly archived
    memories_after_2 = [
        p for p in dest_dir.rglob("*.jpg") if "_Duplicados" not in p.parts
    ]
    assert len(memories_after_2) == 3
    assert res_2["exported"] <= 2

    # Source files remain completely untouched
    assert dup_of_existing.read_bytes() == existing_bytes
    assert new_photo_1.read_bytes() == new_bytes_1
    assert old_mixed.read_bytes() == new_bytes_1


def test_mod11_04_smart_date_priority_and_folder_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Date extraction follows EXIF -> FFmpeg video -> filename patterns,
    and routes screenshots/WhatsApp without camera EXIF to 02_Capturas_y_WhatsApp/YYYY-MM/
    while keeping photographic memories in 01_Recuerdos/YYYY/.
    """
    from photo_organizer.clustering import ClusterItem, cluster_and_stage_items
    from photo_organizer.media import MediaMetadataService

    # Test filename date fallback patterns when EXIF & FFmpeg are absent
    svc = MediaMetadataService(exiftool_bin=None, ffmpeg_bin=None)
    monkeypatch.setattr(svc, "exiftool_bin", None)
    monkeypatch.setattr(svc, "ffmpeg_bin", None)

    wa_file = tmp_path / "IMG-20240512-WA0003.jpg"
    ss_file = tmp_path / "Screenshot_20240615-102030.png"
    vid_file = tmp_path / "VID_20240720_183000.mp4"
    pxl_file = tmp_path / "PXL_20240803_120000.jpg"
    for f in (wa_file, ss_file, vid_file, pxl_file):
        f.write_bytes(_make_jpeg_bytes(f.name.encode("utf-8")))

    meta_wa = svc.read_metadata(wa_file)
    meta_ss = svc.read_metadata(ss_file)
    meta_vid = svc.read_metadata(vid_file)
    meta_pxl = svc.read_metadata(pxl_file)

    assert (meta_wa.get("captured_at") or "").startswith("2024-05-12")
    assert (meta_ss.get("captured_at") or "").startswith("2024-06-15")
    assert (meta_vid.get("captured_at") or "").startswith("2024-07-20")
    assert (meta_pxl.get("captured_at") or "").startswith("2024-08-03")

    # Test routing: screenshots & WhatsApp without camera EXIF -> 02_Capturas_y_WhatsApp/YYYY-MM
    # Photographic memories -> 01_Recuerdos/YYYY/...
    items = [
        ClusterItem(
            id="ss1",
            source_path=str(ss_file),
            ext=".png",
            sha256="aaa1",
            is_duplicate=False,
            captured_at="2024-06-15T10:20:30Z",
            latitude=None,
            longitude=None,
        ),
        ClusterItem(
            id="wa1",
            source_path=str(wa_file),
            ext=".jpg",
            sha256="bbb2",
            is_duplicate=False,
            captured_at="2024-05-12T00:00:00Z",
            latitude=None,
            longitude=None,
        ),
        ClusterItem(
            id="pxl1",
            source_path=str(pxl_file),
            ext=".jpg",
            sha256="ccc3",
            is_duplicate=False,
            captured_at="2024-08-03T12:00:00Z",
            latitude=20.65,
            longitude=-103.35,
        ),
    ]
    staged = cluster_and_stage_items(items)

    assert staged["ss1"]["target_folder"].startswith("02_Capturas_y_WhatsApp/2024-06")
    assert staged["wa1"]["target_folder"].startswith("02_Capturas_y_WhatsApp/2024-05")
    assert staged["pxl1"]["target_folder"].startswith("01_Recuerdos/2024/")


def test_mod11_05_ai_event_folder_naming_with_agy_and_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uses `agy` (with deterministic fallback) to name grouped event folders in Spanish."""
    from photo_organizer import ai_classifier
    from photo_organizer.clustering import ClusterItem, cluster_and_stage_items

    # 1. When `agy` succeeds, it receives event summary context and returns Spanish folder name
    agy_calls: list[list[str]] = []

    def fake_agy_event(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        agy_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Viaje a Oaxaca\n", stderr="")

    with monkeypatch.context() as mp:
        mp.setattr(subprocess, "run", fake_agy_event)
        label = ai_classifier.classify_with_agy(
            "2024-08-03 | 12 fotos | carpeta: Oaxaca_2024 | GPS: 17.07,-96.72",
            cli_command="agy",
            model="gemini-4-hydrogen-medium",
        )
        assert agy_calls and "agy" in agy_calls[0][0]
        assert "Oaxaca" in label

    # 2. When `agy` is missing, deterministic fallback produces a clean Spanish folder name
    with monkeypatch.context() as mp:
        mp.setattr(shutil, "which", lambda _b, **_kw: None)
        fb_label = ai_classifier.classify_with_agy(
            "2024-08-03 | 12 fotos | carpeta: Playa_Cancun",
            cli_command="agy",
        )
        assert isinstance(fb_label, str) and len(fb_label.strip()) > 0

    staged = cluster_and_stage_items(
        [
            ClusterItem(
                id="m1",
                source_path=str(tmp_path / "Playa_Cancun" / "PXL_20240803_120000.jpg"),
                ext=".jpg",
                sha256="ddd4",
                is_duplicate=False,
                captured_at="2024-08-03T12:00:00Z",
                latitude=21.16,
                longitude=-86.85,
                ai_label=label,
            )
        ]
    )
    assert staged["m1"]["target_folder"].startswith("01_Recuerdos/2024/")
    assert "Oaxaca" in staged["m1"]["target_folder"]
