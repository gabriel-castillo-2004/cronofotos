"""Tests for post-delivery modification:
Automatic frequent face discovery with named-people whitelist in CronoFotos:
1. Face detection & frequency clustering (`thumbnails.py`, `ai_classifier.py`, `main.py`):
   - `POST /api/faces/discover` with queryable progress scans current job photos,
     detects close faces, and groups them by embedding similarity (OpenCV
     `FaceDetectorYN` + `FaceRecognizerSF` with ONNX cache in `<data_dir>/models/`,
     and safe histogram/crop fallback when download or OpenCV is unavailable).
   - Saves representative crop + JSON records in `<data_dir>/face_clusters/` with
     `cluster_id`, `count`, `media_ids`, `label` (""), `ignored` (False),
     ordered by `count` descending.
2. "Personas Frecuentes" whitelist panel (`main.py`, `index.html`, `app.js`, `styles.css`):
   - `GET /api/faces/clusters` and `POST /api/faces/clusters/{cluster_id}` to assign
     `label` or mark `ignored=true`.
   - Settings modal / `👥 Personas` tab has `Detectar rostros frecuentes` button and
     a grid with Top 25 faces (thumbnail, count, text input e.g. `Yo (ancla)` / `Mamá`,
     `Guardar` and `Ignorar` buttons).
3. Strict AI naming whitelist rule (`ai_classifier.py`):
   - AI naming, `Optimizar Árbol`, and `Debate IA (2 Agentes)` cross-reference each
     folder's `media_ids` ONLY against clusters with non-empty `label` and `ignored=false`.
   - Passes only confirmed names present in that folder (excluding `Yo (ancla)` as an
     external companion) along with the 3 scene thumbnails to `agy`, completely ignoring
     unnamed or ignored faces.
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
import time
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
    json_body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | list[Any]]:
    data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
    headers = {"Content-Type": "application/json"} if json_body is not None else {}
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers=headers, method=method)
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


def _extract_clusters_list(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    for key in ("clusters", "items", "faces", "people"):
        val = payload.get(key)
        if isinstance(val, list):
            return val
    return []


def _load_clusters_from_dir(face_clusters_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for jf in sorted(face_clusters_dir.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        if isinstance(data, list):
            records.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict):
            if isinstance(data.get("clusters"), list):
                records.extend(item for item in data["clusters"] if isinstance(item, dict))
            elif "cluster_id" in data:
                records.append(data)
    return records


def _make_fake_cv2_sf(embeddings_by_filename: dict[str, list[list[float]]]) -> types.ModuleType:
    """Build fake cv2 with FaceDetectorYN + FaceRecognizerSF for deterministic embedding clustering."""

    class _FakeArray:
        def __init__(self, width: int = 400, height: int = 400, tag: str = "") -> None:
            self.shape = (height, width, 3)
            self.tag = tag

        def __getitem__(self, item: Any) -> "_FakeArray":
            if isinstance(item, tuple) and len(item) == 2:
                ys, xs = item
                h = max(1, (ys.stop or 100) - (ys.start or 0)) if isinstance(ys, slice) else 100
                w = max(1, (xs.stop or 100) - (xs.start or 0)) if isinstance(xs, slice) else 100
                return _FakeArray(width=w, height=h, tag=self.tag)
            return self

    class _FakeDetectorYN:
        @staticmethod
        def create(*_args: Any, **_kwargs: Any) -> "_FakeDetectorYN":
            return _FakeDetectorYN()

        def setInputSize(self, _size: tuple[int, int]) -> None:
            pass

        def detect(self, img: _FakeArray) -> tuple[int, list[list[float]]]:
            embs = embeddings_by_filename.get(img.tag, [])
            # YuNet format: [x, y, w, h, ...landmarks..., score]
            faces = [[50.0, 50.0, 120.0, 120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.98] for _ in embs]
            return len(faces), faces

    class _FakeRecognizerSF:
        @staticmethod
        def create(*_args: Any, **_kwargs: Any) -> "_FakeRecognizerSF":
            return _FakeRecognizerSF()

        def alignCrop(self, img: _FakeArray, face_box: list[float]) -> _FakeArray:
            idx = 0
            return _FakeArray(112, 112, tag=f"{img.tag}#{idx}")

        def feature(self, aligned: _FakeArray) -> list[list[float]]:
            base_tag = aligned.tag.split("#")[0]
            embs = embeddings_by_filename.get(base_tag, [[0.0, 0.0, 1.0]])
            return [embs[0]]

        def match(self, feat1: Any, feat2: Any, *_args: Any) -> float:
            v1 = feat1[0] if isinstance(feat1[0], list) else feat1
            v2 = feat2[0] if isinstance(feat2[0], list) else feat2
            dot = sum(float(a) * float(b) for a, b in zip(v1, v2))
            n1 = sum(float(a) * float(a) for a in v1) ** 0.5 or 1.0
            n2 = sum(float(b) * float(b) for b in v2) ** 0.5 or 1.0
            return dot / (n1 * n2)

    mod = types.ModuleType("cv2")
    mod.FaceDetectorYN = _FakeDetectorYN  # type: ignore[attr-defined]
    mod.FaceRecognizerSF = _FakeRecognizerSF  # type: ignore[attr-defined]
    mod.FR_COSINE = 0  # type: ignore[attr-defined]
    mod.imread = lambda p, *_a, **_kw: _FakeArray(400, 400, tag=Path(str(p)).name)  # type: ignore[attr-defined]
    mod.resize = lambda img, size, **_kw: _FakeArray(size[0], size[1], tag=getattr(img, "tag", ""))  # type: ignore[attr-defined]
    mod.imwrite = lambda path, _img, **_kw: bool(  # type: ignore[attr-defined]
        Path(path).write_bytes(_make_jpeg_bytes(b"facecrop", min_bytes=2048)) or True
    )
    return mod


def test_mod23_01_face_discovery_endpoint_frequency_clustering_and_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /api/faces/discover groups faces by frequency, stores crops + JSON in <data_dir>/face_clusters/, and falls back safely without ONNX download."""
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.db import init_db, open_connection
    from photo_organizer.main import _LocalAppCore, _make_stdlib_handler

    data_dir = tmp_path / "data"
    src_dir = tmp_path / "src"
    data_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    # Create 6 scanned photos: 3 with Person A, 2 with Person B, 1 with Person C
    photos = {
        "a1.jpg": [[1.0, 0.0, 0.0]],
        "a2.jpg": [[0.99, 0.05, 0.0]],
        "a3.jpg": [[0.98, 0.02, 0.0]],
        "b1.jpg": [[0.0, 1.0, 0.0]],
        "b2.jpg": [[0.03, 0.99, 0.0]],
        "c1.jpg": [[0.0, 0.0, 1.0]],
    }
    for fname in photos:
        _make_jpeg_file(src_dir / fname, tag=fname.encode("ascii"))

    cfg = OrganizerConfig(db_path=data_dir / "organizer.db", thumb_cache_dir=data_dir / "thumbs")
    init_db(cfg.db_path)

    # Seed a completed scan job + media items in SQLite
    job_id = "job_faces_1"
    now_iso = "2024-08-10T12:00:00+00:00"
    conn = open_connection(cfg.db_path)
    try:
        conn.execute(
            "INSERT INTO jobs (id, source_dir, dest_dir, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, str(src_dir), str(tmp_path / "dest"), "completed", now_iso, now_iso),
        )
        for idx, fname in enumerate(photos, start=1):
            fpath = src_dir / fname
            conn.execute(
                """
                INSERT INTO media_items (
                    id, job_id, source_path, ext, size_bytes, sha256,
                    captured_at, target_folder, target_filename, export_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"m{idx}",
                    job_id,
                    str(fpath),
                    ".jpg",
                    fpath.stat().st_size,
                    f"sha_{idx}",
                    f"2024-08-10T10:0{idx}:00+00:00",
                    "01_Recuerdos/2024/2024-08-10 - Evento",
                    fname,
                    "PENDING_EXPORT",
                ),
            )
        conn.commit()
    finally:
        conn.close()

    # Pre-seed dummy ONNX model files in <data_dir>/models/ so no network download is needed
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "face_detection_yunet_2023mar.onnx").write_bytes(b"onnx-yunet")
    (models_dir / "face_recognition_sface_2021dec.onnx").write_bytes(b"onnx-sface")

    fake_cv2 = _make_fake_cv2_sf(photos)
    core = _LocalAppCore(cfg)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_stdlib_handler(core))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        orig_urlopen = urllib.request.urlopen
        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "cv2", fake_cv2)
            # Block any external model download attempt
            mp.setattr(
                urllib.request,
                "urlopen",
                lambda req, *a, **kw: orig_urlopen(req, *a, **kw)
                if "127.0.0.1" in str(getattr(req, "full_url", req))
                else (_ for _ in ()).throw(urllib.error.URLError("offline")),
            )
            st_disc, disc_data = _http_request("POST", base_url, "/api/faces/discover", json_body={"job_id": job_id})
            assert st_disc in (200, 202)

            # Poll GET /api/faces/clusters (and/or discover status) until discovery finishes
            clusters: list[dict[str, Any]] = []
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                st_c, c_data = _http_request("GET", base_url, "/api/faces/clusters")
                assert st_c == 200
                clusters = _extract_clusters_list(c_data)
                if len(clusters) >= 3:
                    break
                time.sleep(0.1)

        assert len(clusters) >= 3
        counts = [int(c["count"]) for c in clusters]
        assert counts == sorted(counts, reverse=True), "Clusters must be sorted by count descending"
        assert counts[0] == 3
        assert counts[1] == 2
        assert counts[2] == 1

        for c in clusters:
            assert c.get("cluster_id")
            assert isinstance(c.get("media_ids"), list) and len(c["media_ids"]) == c["count"]
            assert c.get("label", "") == ""
            assert c.get("ignored") is False

        face_clusters_dir = data_dir / "face_clusters"
        assert face_clusters_dir.exists() and face_clusters_dir.is_dir()
        stored_records = _load_clusters_from_dir(face_clusters_dir)
        assert len(stored_records) >= 3
        stored_crops = [
            p for p in face_clusters_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        assert len(stored_crops) >= 3
    finally:
        server.shutdown()
        server.server_close()


def test_mod23_02_face_discovery_safe_fallback_without_cv2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When cv2 is unavailable and network is offline, discovery falls back safely without crashing."""
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.db import init_db, open_connection
    from photo_organizer.main import _LocalAppCore, _make_stdlib_handler

    data_dir = tmp_path / "data"
    src_dir = tmp_path / "src"
    data_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(3):
        _make_jpeg_file(src_dir / f"img_{idx}.jpg", tag=b"same_face_hist")

    cfg = OrganizerConfig(db_path=data_dir / "organizer.db", thumb_cache_dir=data_dir / "thumbs")
    init_db(cfg.db_path)
    job_id = "job_fallback_1"
    now_iso = "2024-08-10T12:00:00+00:00"
    conn = open_connection(cfg.db_path)
    try:
        conn.execute(
            "INSERT INTO jobs (id, source_dir, dest_dir, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, str(src_dir), str(tmp_path / "dest"), "completed", now_iso, now_iso),
        )
        for idx in range(3):
            fpath = src_dir / f"img_{idx}.jpg"
            conn.execute(
                """
                INSERT INTO media_items (
                    id, job_id, source_path, ext, size_bytes, sha256,
                    captured_at, target_folder, target_filename, export_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"mf{idx}",
                    job_id,
                    str(fpath),
                    ".jpg",
                    fpath.stat().st_size,
                    f"shaf_{idx}",
                    f"2024-08-10T10:0{idx}:00+00:00",
                    "01_Recuerdos/2024/2024-08-10 - Evento",
                    fpath.name,
                    "PENDING_EXPORT",
                ),
            )
        conn.commit()
    finally:
        conn.close()

    core = _LocalAppCore(cfg)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_stdlib_handler(core))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "cv2", None)
            st_disc, _ = _http_request("POST", base_url, "/api/faces/discover", json_body={"job_id": job_id})
            assert st_disc in (200, 202)
            time.sleep(0.3)
            st_c, _ = _http_request("GET", base_url, "/api/faces/clusters")
            assert st_c == 200
    finally:
        server.shutdown()
        server.server_close()


def test_mod23_03_whitelist_cluster_endpoints_label_and_ignore(tmp_path: Path) -> None:
    """GET /api/faces/clusters and POST /api/faces/clusters/{cluster_id} assign labels and mark ignored=true."""
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.db import init_db
    from photo_organizer.main import _LocalAppCore, _make_stdlib_handler

    data_dir = tmp_path / "data"
    face_dir = data_dir / "face_clusters"
    face_dir.mkdir(parents=True, exist_ok=True)

    # Seed 3 clusters in <data_dir>/face_clusters/
    seeded = [
        {"cluster_id": "fc_001", "count": 12, "media_ids": [f"m{i}" for i in range(12)], "label": "", "ignored": False},
        {"cluster_id": "fc_002", "count": 7, "media_ids": [f"m{i}" for i in range(7)], "label": "", "ignored": False},
        {"cluster_id": "fc_003", "count": 3, "media_ids": ["m1", "m2", "m3"], "label": "", "ignored": False},
    ]
    for item in seeded:
        crop = _make_jpeg_file(face_dir / f"{item['cluster_id']}.jpg", tag=item["cluster_id"].encode("ascii"))
        record = {**item, "crop_path": str(crop)}
        (face_dir / f"{item['cluster_id']}.json").write_text(json.dumps(record), encoding="utf-8")

    cfg = OrganizerConfig(db_path=data_dir / "organizer.db", thumb_cache_dir=data_dir / "thumbs")
    init_db(cfg.db_path)
    core = _LocalAppCore(cfg)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_stdlib_handler(core))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        st_get, data_get = _http_request("GET", base_url, "/api/faces/clusters")
        assert st_get == 200
        clusters = _extract_clusters_list(data_get)
        assert [c["cluster_id"] for c in clusters[:3]] == ["fc_001", "fc_002", "fc_003"]

        # Assign label "Yo (ancla)" to fc_001 and "Mamá" to fc_002
        st_p1, _ = _http_request("POST", base_url, "/api/faces/clusters/fc_001", json_body={"label": "Yo (ancla)"})
        assert st_p1 == 200
        st_p2, _ = _http_request("POST", base_url, "/api/faces/clusters/fc_002", json_body={"label": "Mamá"})
        assert st_p2 == 200

        # Mark fc_003 as ignored=true
        st_p3, _ = _http_request("POST", base_url, "/api/faces/clusters/fc_003", json_body={"ignored": True})
        assert st_p3 == 200

        st_after, data_after = _http_request("GET", base_url, "/api/faces/clusters")
        assert st_after == 200
        after_list = _extract_clusters_list(data_after)
        by_id = {c["cluster_id"]: c for c in after_list}
        assert by_id["fc_001"]["label"] == "Yo (ancla)"
        assert by_id["fc_001"]["ignored"] is False
        assert by_id["fc_002"]["label"] == "Mamá"
        assert by_id["fc_002"]["ignored"] is False
        # fc_003 is either omitted from visible list or marked ignored=True
        if "fc_003" in by_id:
            assert by_id["fc_003"]["ignored"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_mod23_04_frequent_people_ui_controls_and_top25_grid() -> None:
    """Settings modal / Personas tab exposes 'Detectar rostros frecuentes' and Top 25 grid with Guardar and Ignorar."""
    html = _read_static("index.html")
    js = _read_static("app.js")
    css = _read_static("styles.css")

    assert "Detectar rostros frecuentes" in html
    assert "Yo (ancla)" in html or "Yo (ancla)" in js
    assert "Mamá" in html or "Mamá" in js

    assert "/api/faces/discover" in js
    assert "/api/faces/clusters" in js
    assert "25" in js
    assert "Guardar" in js
    assert "Ignorar" in js
    assert re.search(r"display\s*:\s*grid", css)


def test_mod23_05_ai_naming_strict_whitelist_excludes_unnamed_ignored_and_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AI naming, Optimizar Árbol, and Debate IA pass only confirmed non-ignored folder people (excluding Yo (ancla)) + 3 scene thumbs."""
    from photo_organizer import ai_classifier

    data_dir = tmp_path / "data"
    people_dir = data_dir / "personas_referencia"
    people_dir.mkdir(parents=True, exist_ok=True)
    face_dir = data_dir / "face_clusters"
    face_dir.mkdir(parents=True, exist_ok=True)

    clusters = [
        {"cluster_id": "c_mama", "count": 5, "media_ids": ["m1", "m2"], "label": "Mamá", "ignored": False},
        {"cluster_id": "c_yo", "count": 5, "media_ids": ["m1", "m3"], "label": "Yo (ancla)", "ignored": False},
        {"cluster_id": "c_unnamed", "count": 4, "media_ids": ["m1", "m2", "m3"], "label": "", "ignored": False},
        {"cluster_id": "c_ignored", "count": 4, "media_ids": ["m1", "m2"], "label": "ExtraIgnorado", "ignored": True},
        {"cluster_id": "c_other", "count": 2, "media_ids": ["m99"], "label": "TíoCarlos", "ignored": False},
    ]
    for cl in clusters:
        crop = _make_jpeg_file(face_dir / f"{cl['cluster_id']}.jpg", tag=cl["cluster_id"].encode("ascii"))
        (face_dir / f"{cl['cluster_id']}.json").write_text(
            json.dumps({**cl, "crop_path": str(crop)}),
            encoding="utf-8",
        )

    src_imgs = tmp_path / "folder_imgs"
    img_paths = [_make_jpeg_file(src_imgs / f"scene_{i}.jpg", tag=f"scene_{i}".encode("ascii")) for i in range(4)]

    folder_name = "01_Recuerdos/2024/2024-08-10 - Iztacalco"
    summaries = [
        {
            "folder": folder_name,
            "count": 3,
            "media_ids": ["m1", "m2", "m3"],
            "start_date": "2024-08-10T20:00:00+00:00",
            "end_date": "2024-08-10T23:00:00+00:00",
            "location_label": "Iztacalco",
            "max_gap_hours": 0.5,
            "is_midnight_merged": False,
            "thumb_path": str(img_paths[0]),
            "thumb_paths": [str(p) for p in img_paths],
        }
    ]

    captured_prompts: list[str] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        prompt = cmd[-1]
        captured_prompts.append(prompt)
        if "Agente Curador" in prompt:
            out = {"proposals": [{"folder": folder_name, "proposed_name": "01_Recuerdos/2024/2024-08-10 - Concierto con Mamá", "reason": "Mamá presente"}]}
        elif "Agente Auditor" in prompt:
            out = {"decisions": [{"folder": folder_name, "final_name": "01_Recuerdos/2024/2024-08-10 - Concierto con Mamá", "merge_with": None, "reason": "Aprobado"}]}
        else:
            out = {"suggestions": [{"type": "rename", "before": folder_name, "after": "01_Recuerdos/2024/2024-08-10 - Concierto con Mamá", "reason": "Mamá presente"}]}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(out, ensure_ascii=False), stderr="")

    shim = tmp_path / "agy.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")

    def _build_kwargs(fn: Any) -> dict[str, Any]:
        sig = inspect.signature(fn)
        kw: dict[str, Any] = {"cli_command": "agy"}
        if "people_dir" in sig.parameters:
            kw["people_dir"] = people_dir
        if "face_clusters_dir" in sig.parameters:
            kw["face_clusters_dir"] = face_dir
        if "data_dir" in sig.parameters:
            kw["data_dir"] = data_dir
        return kw

    with monkeypatch.context() as mp:
        mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if "agy" in str(b) else None)
        mp.setattr(subprocess, "run", fake_run)

        ai_classifier.generate_tree_ai_suggestions(summaries, **_build_kwargs(ai_classifier.generate_tree_ai_suggestions))
        ai_classifier.suggest_tree_optimizations(summaries, **_build_kwargs(ai_classifier.suggest_tree_optimizations))
        ai_classifier.debate_folder_batch_with_agy(summaries, **_build_kwargs(ai_classifier.debate_folder_batch_with_agy))

    assert len(captured_prompts) >= 3
    for prompt in captured_prompts:
        if "Agente Auditor" in prompt:
            continue
        assert "Mamá" in prompt
        assert "ExtraIgnorado" not in prompt
        assert "TíoCarlos" not in prompt
        assert "Yo (ancla)" not in prompt
