"""Tests for post-delivery modifications (MOD-09 / MOD-1).

Verifies:
1. Antigravity CLI (`agy`) integration replacing Gemini API Key, with heuristic fallback.
2. Left column ("Árbol Propuesto") `.folder-actions` wrapping without horizontal overflow.
3. Startup empty state in "Elementos" panel and "Examinar..." native folder picker buttons.
"""

from __future__ import annotations

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


def test_mod_01_antigravity_cli_settings_modal_and_models() -> None:
    """Settings modal replaces Gemini API Key with Antigravity CLI path and Antigravity models."""
    html = _read_static("index.html")

    assert "Gemini API Key" not in html
    assert "cfg-gemini-key" not in html
    assert "Antigravity CLI" in html
    assert "agy" in html

    for model_name in (
        "gemini-4-carbon-medium",
        "gemini-4-hydrogen-medium",
        "gemini-4-neon-medium",
    ):
        assert model_name in html


def test_mod_02_antigravity_cli_backend_settings_and_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend uses `agy` subprocess for AI classification with heuristic fallback on error/timeout."""
    from photo_organizer import ai_classifier
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.main import _LocalAppCore

    cfg = OrganizerConfig(
        db_path=tmp_path / "organizer.db",
        thumb_cache_dir=tmp_path / "thumbs",
    )
    core = _LocalAppCore(cfg)
    settings = core.get_settings()

    # Settings must expose Antigravity CLI command/path (default 'agy') and not require API key
    cli_value = (
        settings.get("antigravity_cli")
        or settings.get("antigravity_cli_path")
        or settings.get("agy_path")
        or settings.get("cli_command")
    )
    assert cli_value == "agy"
    assert settings.get("model") in {
        "gemini-4-carbon-medium",
        "gemini-4-hydrogen-medium",
        "gemini-4-neon-medium",
    }

    # Verify subprocess invocation when `agy` succeeds
    calls: list[list[str]] = []

    def fake_run_ok(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Viaje_Playa\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run_ok)
    result_ok = ai_classifier.classify_with_agy(
        "2024/07/playa_01.jpg",
        cli_command="agy",
        model="gemini-4-hydrogen-medium",
    )
    assert calls and "agy" in calls[0][0]
    assert "Viaje" in result_ok

    # Verify heuristic fallback when `agy` binary is missing
    def fake_run_missing(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("agy not found")

    monkeypatch.setattr(subprocess, "run", fake_run_missing)
    fb_missing = ai_classifier.classify_with_agy(
        "Screenshots/captura_2024.png",
        cli_command="agy",
        model="gemini-4-hydrogen-medium",
    )
    assert isinstance(fb_missing, str) and len(fb_missing.strip()) > 0
    assert fb_missing != "unclassified_local_fallback"

    # Verify heuristic fallback when `agy` times out
    def fake_run_timeout(cmd: list[str], **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)

    monkeypatch.setattr(subprocess, "run", fake_run_timeout)
    fb_timeout = ai_classifier.classify_with_agy(
        "WhatsApp/IMG-20240501-WA0001.jpg",
        cli_command="agy",
        model="gemini-4-hydrogen-medium",
    )
    assert isinstance(fb_timeout, str) and len(fb_timeout.strip()) > 0


def test_mod_03_left_pane_folder_actions_wrap_without_overflow() -> None:
    """Left pane folder-actions wraps compact buttons so they never clip or overflow."""
    html = _read_static("index.html")
    css = _read_static("styles.css")

    assert "Renombrar" in html
    assert "Fusionar" in html
    assert "Nueva carpeta" in html

    # Base CSS (outside @media) must apply flex-wrap: wrap to .folder-actions
    base_css = css.split("@media", 1)[0]
    folder_actions_blocks = re.findall(
        r"\.folder-actions[^{]*\{[^}]*\}",
        base_css,
        flags=re.DOTALL,
    )
    assert any("flex-wrap: wrap" in block for block in folder_actions_blocks)


def test_mod_04_startup_empty_state_and_browse_buttons() -> None:
    """Startup UI shows centered empty state in Elementos and Examinar... buttons for paths."""
    html = _read_static("index.html")
    css = _read_static("styles.css")
    js = _read_static("app.js")
    main_py = (SRC_DIR / "photo_organizer" / "main.py").read_text(encoding="utf-8")

    # Examinar... buttons next to Carpeta Origen and Carpeta Destino
    assert html.count("Examinar...") >= 2

    # Empty state markup/rendering and centered styling
    assert "empty" in (html + js).lower()
    assert "empty-state" in css or "empty" in css

    # Backend exposes a folder browse endpoint wired from app.js
    assert re.search(r"/api/.*(browse|folder|dialog)", main_py)
    assert re.search(r"/api/.*(browse|folder|dialog)", js)


def test_mod_05_cli_resolution_and_missing_binary_fallbacks_without_file_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External CLIs (agy, ffmpeg, exiftool) resolve via shutil.which and trigger fallbacks when absent without raising FileNotFoundError."""
    import os
    import shutil
    import subprocess

    from photo_organizer import ai_classifier
    from photo_organizer.media import MediaMetadataService
    from photo_organizer.thumbnails import ThumbnailCache

    # 1. agy resolved via shutil.which passes full shim path to subprocess.run
    shim = tmp_path / ("agy.cmd" if os.name == "nt" else "agy")
    shim.write_text("@echo off\n" if os.name == "nt" else "#!/bin/sh\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run_capture(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Playa_2024\n", stderr="")

    with monkeypatch.context() as mp:
        mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if b == "agy" else None)
        mp.setattr(subprocess, "run", fake_run_capture)
        out = ai_classifier.classify_with_agy("beach", cli_command="agy")
        assert out == "Playa_2024"
        assert calls and calls[0][0] == str(shim)

    # 1b. Missing agy without monkeypatching subprocess.run falls back without FileNotFoundError
    monkeypatch.setattr(shutil, "which", lambda _b, **_kw: None)
    fb_direct = ai_classifier.classify_with_agy("beach/foto_01.jpg", cli_command="agy")
    assert fb_direct == "Viaje"

    # 2. Missing exiftool triggers fallback cleanly even if constructed with bare "exiftool"
    monkeypatch.setattr(shutil, "which", lambda _b, **_kw: None)
    sample_jpg = tmp_path / "photo.jpg"
    sample_jpg.write_bytes(b"not-a-real-jpg")
    svc = MediaMetadataService.__new__(MediaMetadataService)
    svc.exiftool_bin = "exiftool"
    svc.ffmpeg_bin = "ffmpeg"
    svc.timeout_sec = 2.0
    svc._local = __import__("threading").local()
    svc._lock = __import__("threading").Lock()
    svc._processes = []
    meta = svc.read_metadata(sample_jpg)
    assert isinstance(meta, dict)
    assert meta.get("Extractor") == "pillow_piexif_fallback"

    # 3. ThumbnailCache with bare missing ffmpeg_bin returns video fallback without FileNotFoundError
    fake_mp4 = tmp_path / "clip.mp4"
    fake_mp4.write_bytes(b"0000")
    tcache = ThumbnailCache(tmp_path / "thumbs", ffmpeg_bin="ffmpeg")
    thumb_bytes = tcache.get_or_create("vid1", fake_mp4, "deadbeef")
    assert thumb_bytes.startswith(b"THUMB:VIDEO:")
