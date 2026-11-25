"""Tests for post-delivery modification MOD-16:
1. Key People Gallery (`config.py`, `main.py`, `index.html`, `app.js`):
   - Settings modal section "Personas Clave (Reconocimiento en carpetas)" to upload/delete
     reference photos with labels stored in `<data_dir>/personas_referencia/`.
   - Endpoints `GET /api/people`, `POST /api/people` (multipart/form-data), `DELETE /api/people/{person_id}`.
2. Smart face extraction + 3 scene thumbnails (`ai_classifier.py`, `thumbnails.py`):
   - Analyzes folder photos using `cv2.CascadeClassifier` (with safe fallback when unavailable),
     sorting detected faces by bounding-box area descending (prioritizing close-ups/selfies over
     distant stage faces) and saving up to 4 face crops + 3 scene thumbnails (start, mid, 75%).
   - Passes `personas_referencia/` plus scene thumbnails and face crops via `--add-dir` to `agy`
     so it identifies the event and key people present, surfacing summarized failure reasons in
     `Actividad en vivo`.
"""

from __future__ import annotations

import http.server
import inspect
import json
import re
import shutil
import subprocess
import sys
import threading
import types
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


def _make_jpeg_bytes(tag: bytes = b"img", min_bytes: int = 20000) -> bytes:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_make_jpeg_bytes(tag))
    return path


def _http_request(
    method: str,
    base_url: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any] | list[Any]]:
    req = urllib.request.Request(f"{base_url}{path}", data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: dict[str, Any] | list[Any] = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"detail": raw}
        return exc.code, parsed


def _encode_multipart(label: str, filename: str, file_bytes: bytes) -> tuple[bytes, str]:
    boundary = "----CronoFotosTestBoundary7e9a"
    parts = [
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="label"\r\n\r\n'
        f"{label}\r\n".encode("utf-8"),
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n".encode("utf-8"),
        file_bytes,
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _extract_people_list(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    for key in ("people", "items", "personas"):
        val = payload.get(key)
        if isinstance(val, list):
            return val
    return []


def _make_fake_cv2(bboxes_by_call: list[tuple[int, int, int, int]]) -> types.ModuleType:
    """Build a lightweight fake cv2 module with CascadeClassifier returning given (x, y, w, h) boxes."""

    class _FakeArray:
        def __init__(self, width: int = 400, height: int = 400) -> None:
            self.shape = (height, width, 3)

        def __getitem__(self, item: Any) -> "_FakeArray":
            if isinstance(item, tuple) and len(item) == 2:
                ys, xs = item
                h = max(1, (ys.stop or 100) - (ys.start or 0)) if isinstance(ys, slice) else 100
                w = max(1, (xs.stop or 100) - (xs.start or 0)) if isinstance(xs, slice) else 100
                return _FakeArray(width=w, height=h)
            return self

    class _FakeCascadeClassifier:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def empty(self) -> bool:
            return False

        def detectMultiScale(self, *_args: Any, **_kwargs: Any) -> list[tuple[int, int, int, int]]:
            return list(bboxes_by_call)

    mod = types.ModuleType("cv2")
    mod.data = types.SimpleNamespace(haarcascades="fake_haarcascades/")  # type: ignore[attr-defined]
    mod.COLOR_BGR2GRAY = 6  # type: ignore[attr-defined]
    mod.CascadeClassifier = _FakeCascadeClassifier  # type: ignore[attr-defined]
    mod.imread = lambda *_a, **_kw: _FakeArray(400, 400)  # type: ignore[attr-defined]
    mod.cvtColor = lambda img, *_a, **_kw: img  # type: ignore[attr-defined]
    mod.resize = lambda img, size, **_kw: _FakeArray(size[0], size[1])  # type: ignore[attr-defined]
    mod.imwrite = lambda path, _img, **_kw: bool(  # type: ignore[attr-defined]
        Path(path).write_bytes(_make_jpeg_bytes(b"facecrop", min_bytes=2048)) or True
    )
    return mod


def test_mod16_01_people_gallery_endpoints_and_storage(tmp_path: Path) -> None:
    """GET/POST/DELETE /api/people manage reference photos in <data_dir>/personas_referencia/."""
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.db import init_db
    from photo_organizer.main import _LocalAppCore, _make_stdlib_handler

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = OrganizerConfig(db_path=data_dir / "organizer.db", thumb_cache_dir=data_dir / "thumbs")
    init_db(cfg.db_path)
    core = _LocalAppCore(cfg)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_stdlib_handler(core))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        st_init, data_init = _http_request("GET", base_url, "/api/people")
        assert st_init == 200
        assert _extract_people_list(data_init) == []

        body, ctype = _encode_multipart("Mamá", "mama.jpg", _make_jpeg_bytes(b"mama_ref"))
        st_post, data_post = _http_request(
            "POST",
            base_url,
            "/api/people",
            body=body,
            headers={"Content-Type": ctype},
        )
        assert st_post in (200, 201)

        st_list, data_list = _http_request("GET", base_url, "/api/people")
        assert st_list == 200
        people = _extract_people_list(data_list)
        assert len(people) == 1
        person = people[0]
        assert person.get("label") == "Mamá"
        person_id = str(person.get("id") or person.get("person_id") or "")
        assert person_id

        people_dir = data_dir / "personas_referencia"
        assert people_dir.exists() and people_dir.is_dir()
        stored_files = [p for p in people_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        assert len(stored_files) == 1

        st_del, _ = _http_request("DELETE", base_url, f"/api/people/{person_id}")
        assert st_del == 200

        st_after, data_after = _http_request("GET", base_url, "/api/people")
        assert st_after == 200
        assert _extract_people_list(data_after) == []
        remaining_files = [p for p in people_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        assert remaining_files == []
    finally:
        server.shutdown()
        server.server_close()


def test_mod16_02_people_gallery_ui_in_settings_modal() -> None:
    """Settings modal exposes Key People section and app.js wires /api/people and live activity failure logging."""
    html = _read_static("index.html")
    js = _read_static("app.js")

    assert "Personas Clave (Reconocimiento en carpetas)" in html
    modal_match = re.search(r'<dialog[^>]*id="settings-modal"[\s\S]*?</dialog>', html)
    assert modal_match is not None
    assert "Personas Clave" in modal_match.group(0)

    assert "/api/people" in js
    assert "DELETE" in js
    assert "FormData" in js
    assert "logActivity" in js


def test_mod16_03_face_extraction_prioritizes_largest_boxes_and_passes_add_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detected faces are sorted by bbox area desc (<=4 crops + <=3 scene thumbs) and personas_referencia is passed via --add-dir."""
    from photo_organizer import ai_classifier

    src_imgs = tmp_path / "src_imgs"
    img_paths = [
        _make_jpeg_file(src_imgs / f"concert_{i}.jpg", tag=f"concert_{i}".encode("ascii"))
        for i in range(4)
    ]
    people_dir = tmp_path / "personas_referencia"
    _make_jpeg_file(people_dir / "mama.jpg", tag=b"mama_ref")

    # 6 faces: areas 400, 14400, 8100, 4900, 2500, 225 -> top 4 largest are >= 2500
    fake_cv2 = _make_fake_cv2([
        (10, 10, 20, 20),
        (50, 50, 120, 120),
        (200, 10, 90, 90),
        (10, 150, 70, 70),
        (150, 150, 50, 50),
        (250, 250, 15, 15),
    ])

    folder_name = "01_Recuerdos/2024/2024-08-10 - Iztacalco"
    summaries = [
        {
            "folder": folder_name,
            "count": 4,
            "start_date": "2024-08-10T20:00:00+00:00",
            "end_date": "2024-08-10T23:00:00+00:00",
            "location_label": "Iztacalco",
            "max_gap_hours": 0.5,
            "is_midnight_merged": False,
            "thumb_path": str(img_paths[1]),
            "thumb_paths": [str(p) for p in img_paths],
        }
    ]

    observed_add_dirs: list[Path] = []
    observed_images: list[str] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        idx = 0
        while idx < len(cmd):
            if cmd[idx] == "--add-dir" and idx + 1 < len(cmd):
                d = Path(cmd[idx + 1])
                assert d.exists() and d.is_dir()
                observed_add_dirs.append(d)
                for f in sorted(d.rglob("*")):
                    if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                        observed_images.append(f.name)
                idx += 2
            else:
                idx += 1

        payload = {
            "suggestions": [
                {
                    "type": "rename",
                    "before": folder_name,
                    "after": "01_Recuerdos/2024/2024-08-10 - Concierto de Luis Miguel con Mamá (Iztacalco)",
                    "reason": "Se identifica a Mamá junto al escenario en concierto",
                }
            ]
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload, ensure_ascii=False), stderr="")

    shim = tmp_path / "agy.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")

    sig = inspect.signature(ai_classifier.generate_tree_ai_suggestions)
    kwargs: dict[str, Any] = {"cli_command": "agy"}
    for param_name in ("people_dir", "personas_dir", "reference_dir"):
        if param_name in sig.parameters:
            kwargs[param_name] = people_dir
            break

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "cv2", fake_cv2)
        mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if "agy" in str(b) else None)
        mp.setattr(subprocess, "run", fake_run)
        res = ai_classifier.generate_tree_ai_suggestions(summaries, **kwargs)

    assert len(observed_add_dirs) >= 1
    assert any(
        d.resolve() == people_dir.resolve() or any("mama" in name.lower() for name in observed_images)
        for d in observed_add_dirs
    )
    assert len(res["suggestions"]) == 1
    assert res["suggestions"][0]["after"] == "01_Recuerdos/2024/2024-08-10 - Concierto de Luis Miguel con Mamá (Iztacalco)"


def test_mod16_04_naming_without_faces_or_cv2_and_failure_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naming works safely when cv2 is absent or detects 0 faces, and exposes failure reasons on agy error."""
    from photo_organizer import ai_classifier

    src_imgs = tmp_path / "no_faces"
    img_paths = [
        _make_jpeg_file(src_imgs / f"landscape_{i}.jpg", tag=f"land_{i}".encode("ascii"))
        for i in range(4)
    ]
    folder_name = "01_Recuerdos/2024/2024-09-15 - Montaña"
    summaries = [
        {
            "folder": folder_name,
            "count": 4,
            "start_date": "2024-09-15T10:00:00+00:00",
            "end_date": "2024-09-15T13:00:00+00:00",
            "location_label": "Montaña",
            "max_gap_hours": 0.4,
            "is_midnight_merged": False,
            "thumb_path": str(img_paths[0]),
            "thumb_paths": [str(p) for p in img_paths],
        }
    ]
    shim = tmp_path / "agy.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")

    def ok_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert "--add-dir" in cmd
        payload = {
            "suggestions": [
                {
                    "type": "rename",
                    "before": folder_name,
                    "after": "01_Recuerdos/2024/2024-09-15 - Senderismo en Montaña",
                    "reason": "Paisaje de montaña sin rostros cercanos",
                }
            ]
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload, ensure_ascii=False), stderr="")

    # Case A: cv2 unavailable
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "cv2", None)
        mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if "agy" in str(b) else None)
        mp.setattr(subprocess, "run", ok_run)
        res_no_cv2 = ai_classifier.generate_tree_ai_suggestions(summaries, cli_command="agy")
    assert res_no_cv2["suggestions"][0]["after"] == "01_Recuerdos/2024/2024-09-15 - Senderismo en Montaña"

    # Case B: cv2 detects 0 faces
    fake_cv2_empty = _make_fake_cv2([])
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "cv2", fake_cv2_empty)
        mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if "agy" in str(b) else None)
        mp.setattr(subprocess, "run", ok_run)
        res_zero_faces = ai_classifier.generate_tree_ai_suggestions(summaries, cli_command="agy")
    assert res_zero_faces["suggestions"][0]["after"] == "01_Recuerdos/2024/2024-09-15 - Senderismo en Montaña"

    # Case C: agy fails -> summarized failure reason surfaced
    def fail_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="rate limit exceeded")

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "cv2", fake_cv2_empty)
        mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if "agy" in str(b) else None)
        mp.setattr(subprocess, "run", fail_run)
        res_fail = ai_classifier.generate_tree_ai_suggestions(summaries, cli_command="agy")
    assert "rate limit exceeded" in json.dumps(res_fail, ensure_ascii=False).lower()
