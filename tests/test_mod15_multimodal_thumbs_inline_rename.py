"""Tests for post-delivery modification MOD-15:
1. Real multimodal reading of up to 3 representative thumbnails per folder (start, mid, 75%, <= 320px)
   in a temp directory passed via --add-dir to `agy`, explicit disk-reading prompt for activity/event/
   venue/artist rename suggestions (even with GPS), and exposing summarized technical failure reasons
   instead of hiding them behind `Fallback determinista local`.
2. Inline quick-rename button (`✏️`) next to each folder in `Árbol Propuesto` (`index.html`, `app.js`)
   saving through `/api/tree/folders/rename`.
"""

from __future__ import annotations

import asyncio
import http.server
import io
import json
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


def _make_jpeg_file(path: Path, tag: bytes = b"img") -> Path:
    path.write_bytes(_make_jpeg_bytes(tag))
    return path


def _jpeg_max_dim(path: Path) -> int | None:
    """Return max(width, height) from JPEG SOF marker if present, else None."""
    data = path.read_bytes()
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            h = int.from_bytes(data[i + 5 : i + 7], "big")
            w = int.from_bytes(data[i + 7 : i + 9], "big")
            return max(w, h)
        if marker == 0xD9:
            break
        if marker in (0xD8, 0x01) or (0xD0 <= marker <= 0xD7):
            i += 2
            continue
        if i + 4 > len(data):
            break
        seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
        i += 2 + max(seg_len, 2)
    return None


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


def test_mod15_01_generate_tree_ai_suggestions_multimodal_3_thumbs_and_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_tree_ai_suggestions saves up to 3 <=320px thumbnails in a temp --add-dir and instructs agy to read them."""
    from photo_organizer import ai_classifier

    src_imgs = tmp_path / "src_imgs"
    src_imgs.mkdir()
    img_paths = [
        _make_jpeg_file(src_imgs / f"img_{i}.jpg", tag=f"img_{i}".encode("ascii"))
        for i in range(5)
    ]

    folder_name = "01_Recuerdos/2024/2024-08-10 - CDMX, MX"
    summaries = [
        {
            "folder": folder_name,
            "count": 5,
            "start_date": "2024-08-10T20:00:00+00:00",
            "end_date": "2024-08-10T23:00:00+00:00",
            "location_label": "CDMX, MX",
            "max_gap_hours": 0.5,
            "is_midnight_merged": False,
            "thumb_path": str(img_paths[2]),
            "thumb_paths": [str(p) for p in img_paths],
        }
    ]

    observed_files: list[tuple[str, int | None]] = []
    captured_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(list(cmd))
        assert "--add-dir" in cmd
        add_dir = Path(cmd[cmd.index("--add-dir") + 1])
        assert add_dir.exists() and add_dir.is_dir()
        for f in sorted(add_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                observed_files.append((f.name, _jpeg_max_dim(f)))

        payload = {
            "suggestions": [
                {
                    "type": "rename",
                    "before": folder_name,
                    "after": "01_Recuerdos/2024/2024-08-10 - Concierto de Luis Miguel",
                    "reason": "Se observa escenario, luces y público en concierto",
                }
            ]
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload, ensure_ascii=False), stderr="")

    shim = tmp_path / "agy.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")

    with monkeypatch.context() as mp:
        mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if "agy" in str(b) else None)
        mp.setattr(subprocess, "run", fake_run)
        res = ai_classifier.generate_tree_ai_suggestions(summaries, cli_command="agy")

    assert len(captured_cmds) == 1
    cmd = captured_cmds[0]
    assert "-p" in cmd
    prompt = cmd[cmd.index("-p") + 1].lower()
    assert any(tok in prompt for tok in ("disco", "leer", "lee", "imagen", "archivo"))
    assert any(tok in prompt for tok in ("actividad", "evento", "recinto", "artista"))
    assert "rename" in prompt

    assert 1 <= len(observed_files) <= 3
    for _fname, max_dim in observed_files:
        if max_dim is not None:
            assert max_dim <= 320

    assert "agy" in str(res.get("engine", "")).lower()
    assert len(res["suggestions"]) == 1
    assert res["suggestions"][0]["type"] == "rename"
    assert res["suggestions"][0]["after"] == "01_Recuerdos/2024/2024-08-10 - Concierto de Luis Miguel"


def test_mod15_02_agy_failure_and_invalid_output_expose_technical_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When agy fails or returns invalid JSON, a summarized technical reason is surfaced instead of hiding behind Fallback."""
    from photo_organizer import ai_classifier

    thumb_file = _make_jpeg_file(tmp_path / "sample.jpg", tag=b"sample")
    summaries = [
        {
            "folder": "01_Recuerdos/2024/2024-09-01 - Eventos del día",
            "count": 4,
            "start_date": "2024-09-01T12:00:00+00:00",
            "end_date": "2024-09-01T14:00:00+00:00",
            "location_label": None,
            "max_gap_hours": 0.5,
            "is_midnight_merged": False,
            "thumb_path": str(thumb_file),
            "thumb_paths": [str(thumb_file)],
        }
    ]
    shim = tmp_path / "agy.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")

    # 1) Non-zero exit code with stderr message
    def fail_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="auth token expired")

    with monkeypatch.context() as mp:
        mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if "agy" in str(b) else None)
        mp.setattr(subprocess, "run", fail_run)
        res_fail = ai_classifier.generate_tree_ai_suggestions(summaries, cli_command="agy")

    serialized_fail = json.dumps(res_fail, ensure_ascii=False).lower()
    assert "fallback" in serialized_fail
    assert any(tok in serialized_fail for tok in ("auth token expired", "código 1", "exit 1", "returncode", "error"))

    # 2) Zero exit code with invalid JSON stdout
    def bad_json_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="not-valid-json-output", stderr="")

    with monkeypatch.context() as mp:
        mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if "agy" in str(b) else None)
        mp.setattr(subprocess, "run", bad_json_run)
        res_bad = ai_classifier.generate_tree_ai_suggestions(summaries, cli_command="agy")

    serialized_bad = json.dumps(res_bad, ensure_ascii=False).lower()
    assert "fallback" in serialized_bad
    assert any(tok in serialized_bad for tok in ("json", "inválid", "invalid", "salida", "parse"))


def test_mod15_03_inline_rename_button_in_proposed_tree_ui_and_endpoint(tmp_path: Path) -> None:
    """Proposed Tree renders a discrete ✏️ inline rename button per folder and renames via /api/tree/folders/rename."""
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.db import init_db
    from photo_organizer.main import _LocalAppCore, _make_stdlib_handler

    html = _read_static("index.html")
    js = _read_static("app.js")
    assert "Árbol Propuesto" in html
    assert "✏️" in js or "\\u270f" in js.lower()
    assert "/api/tree/folders/rename" in js

    load_tree_match = re.search(r"async function loadTree\b[\s\S]{0,4500}", js)
    assert load_tree_match is not None
    load_tree_body = load_tree_match.group(0)
    assert ("✏️" in load_tree_body) or ("\\u270f" in load_tree_body.lower())
    assert "input" in load_tree_body.lower()

    src_dir = tmp_path / "src_rename"
    dest_dir = tmp_path / "dest_rename"
    src_dir.mkdir()
    _make_jpeg_file(src_dir / "PXL_20240810_200000.jpg", tag=b"concert")

    cfg = OrganizerConfig(db_path=tmp_path / "organizer.db", thumb_cache_dir=tmp_path / "thumbs")
    init_db(cfg.db_path)
    core = _LocalAppCore(cfg)
    job_id = asyncio.run(core.orchestrator.run_job(src_dir, dest_dir))

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_stdlib_handler(core))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        st_tree, data_tree = _http_json("GET", base_url, f"/api/tree?job_id={job_id}")
        assert st_tree == 200
        assert len(data_tree["folders"]) >= 1
        old_folder = data_tree["folders"][0]["folder"]
        old_count = data_tree["folders"][0]["count"]
        new_folder = f"{old_folder} con Mamá"

        st_ren, _ = _http_json(
            "POST",
            base_url,
            "/api/tree/folders/rename",
            {"job_id": job_id, "old_folder": old_folder, "new_folder": new_folder},
        )
        assert st_ren == 200

        st_after, data_after = _http_json("GET", base_url, f"/api/tree?job_id={job_id}")
        assert st_after == 200
        folder_map = {f["folder"]: f["count"] for f in data_after["folders"]}
        assert new_folder in folder_map
        assert old_folder not in folder_map
        assert folder_map[new_folder] == old_count
    finally:
        server.shutdown()
        server.server_close()
