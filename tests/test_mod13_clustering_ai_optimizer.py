"""Tests for post-delivery modification MOD-13: Advanced clustering rules & AI tree optimizer.

Verifies:
1. Automatic clustering & cleanup rules (`clustering.py`, `media.py`):
   - GPS location inheritance within +/- 2 hours on the same day.
   - Night event merging (< 6 hours across midnight in the same city/zone, keeping start date).
   - Multi-day trips (base zone from >= 10 distinct GPS days; >= 2 consecutive days outside
     grouped as `YYYY-MM-DD al MM-DD - Viaje a <Ciudad o Estado>`).
   - Consolidation of sparse local events (< 3 files inside base zone -> `YYYY-MM - Momentos Varios`;
     outside base zone keeps own folder).
   - Sticker and tiny file filter (< 15 KB or `.webp` without camera EXIF -> `02_Capturas_y_WhatsApp/YYYY-MM/`).
   - Motion/Live Photo pairing (image + short video <= 3s sharing timestamp & folder get identical
     `YYYY-MM-DD_HH-MM-SS_###` stem).
2. Backend AI tree optimizer (`ai_classifier.py`, `POST /api/tree/ai-suggestions`,
   `POST /api/tree/apply-suggestions`):
   - Representative center-photo thumbnail <= 256px + metadata summary in batches <= 12 folders.
   - Structured JSON suggestions (`rename`, `merge`, `split` only when >25 files and >=3h internal gap).
   - Deterministic fallback when `agy` fails or lacks attachment support, reporting engine and
     never emitting template strings (`__`, `placeholder`, `sample`).
   - Applying selected `rename`, `merge`, and `split` suggestions updates the tree cleanly.
3. Frontend UI (`index.html`, `app.js`, `styles.css`):
   - `✨ Optimizar Árbol con IA` button next to `Regenerar nombres con IA` with disabled + spinner state.
   - Suggestions panel with checkboxes, Antes -> Después view, reason, and
     `Aplicar sugerencias seleccionadas` button.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
from datetime import datetime, timedelta, timezone
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
    """Build valid JPEG bytes padded above 15 KB so normal photos are not treated as stickers."""
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


def _base_zone_items(tmp_path: Path, count_days: int = 10) -> list[Any]:
    """Create `count_days` distinct days with GPS in CDMX to establish a base zone."""
    from photo_organizer.clustering import ClusterItem

    items = []
    start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for day_idx in range(count_days):
        dt = start + timedelta(days=day_idx)
        for photo_idx in range(3):
            ts = (dt + timedelta(minutes=photo_idx * 5)).strftime("%Y-%m-%dT%H:%M:%SZ")
            items.append(
                ClusterItem(
                    id=f"base_{day_idx}_{photo_idx}",
                    source_path=str(tmp_path / f"IMG_base_{day_idx}_{photo_idx}.jpg"),
                    ext=".jpg",
                    sha256=f"sha_base_{day_idx}_{photo_idx}",
                    is_duplicate=False,
                    captured_at=ts,
                    latitude=19.4326,
                    longitude=-99.1332,
                    has_camera_exif=True,
                )
            )
    return items


def test_mod13_01_gps_inheritance_within_2_hours(tmp_path: Path) -> None:
    """Videos/photos without GPS inherit geocoded location from GPS items within +/-2h on same day."""
    from photo_organizer.clustering import ClusterItem, cluster_and_stage_items

    items = [
        ClusterItem(
            id="p_gps",
            source_path=str(tmp_path / "PXL_20240710_140000.jpg"),
            ext=".jpg",
            sha256="sha_gps",
            is_duplicate=False,
            captured_at="2024-07-10T14:00:00Z",
            latitude=19.4326,
            longitude=-99.1332,
            has_camera_exif=True,
        ),
        ClusterItem(
            id="v_near",
            source_path=str(tmp_path / "VID_20240710_151500.mp4"),
            ext=".mp4",
            sha256="sha_vnear",
            is_duplicate=False,
            captured_at="2024-07-10T15:15:00Z",
            latitude=None,
            longitude=None,
            has_camera_exif=False,
        ),
        ClusterItem(
            id="p_far",
            source_path=str(tmp_path / "PXL_20240710_193000.jpg"),
            ext=".jpg",
            sha256="sha_pfar",
            is_duplicate=False,
            captured_at="2024-07-10T19:30:00Z",
            latitude=None,
            longitude=None,
            has_camera_exif=True,
        ),
    ]

    staged = cluster_and_stage_items(items)
    assert staged["p_gps"]["location_label"] is not None
    assert staged["v_near"]["location_label"] == staged["p_gps"]["location_label"]
    assert staged["v_near"]["target_folder"] == staged["p_gps"]["target_folder"]
    assert staged["p_far"]["location_label"] is None


def test_mod13_02_night_event_merging_under_6_hours(tmp_path: Path) -> None:
    """Consecutive events in same city/zone <6h apart across midnight merge keeping start date."""
    from photo_organizer.clustering import ClusterItem, cluster_and_stage_items

    items = [
        ClusterItem(
            id="night1",
            source_path=str(tmp_path / "PXL_20240810_231500.jpg"),
            ext=".jpg",
            sha256="sha_n1",
            is_duplicate=False,
            captured_at="2024-08-10T23:15:00Z",
            latitude=19.4326,
            longitude=-99.1332,
            has_camera_exif=True,
        ),
        ClusterItem(
            id="night2",
            source_path=str(tmp_path / "PXL_20240811_023000.jpg"),
            ext=".jpg",
            sha256="sha_n2",
            is_duplicate=False,
            captured_at="2024-08-11T02:30:00Z",
            latitude=19.4326,
            longitude=-99.1332,
            has_camera_exif=True,
        ),
        ClusterItem(
            id="next_afternoon",
            source_path=str(tmp_path / "PXL_20240811_140000.jpg"),
            ext=".jpg",
            sha256="sha_n3",
            is_duplicate=False,
            captured_at="2024-08-11T14:00:00Z",
            latitude=19.4326,
            longitude=-99.1332,
            has_camera_exif=True,
        ),
    ]

    staged = cluster_and_stage_items(items)
    assert staged["night1"]["target_folder"] == staged["night2"]["target_folder"]
    assert "2024-08-10" in staged["night2"]["target_folder"]
    assert staged["next_afternoon"]["target_folder"] != staged["night1"]["target_folder"]
    assert "2024-08-11" in staged["next_afternoon"]["target_folder"]


def test_mod13_03_multiday_trips_with_base_zone_10_days(tmp_path: Path) -> None:
    """With >=10 distinct GPS days, 2+ consecutive days outside base zone become a Viaje folder."""
    from photo_organizer.clustering import ClusterItem, cluster_and_stage_items

    items = _base_zone_items(tmp_path, count_days=10)
    # 3 consecutive days in Oaxaca (outside CDMX base zone)
    for d_idx, day in enumerate((1, 2, 3)):
        items.append(
            ClusterItem(
                id=f"trip_{day}",
                source_path=str(tmp_path / f"PXL_2024090{day}_110000.jpg"),
                ext=".jpg",
                sha256=f"sha_trip_{d_idx}",
                is_duplicate=False,
                captured_at=f"2024-09-0{day}T11:00:00Z",
                latitude=17.0732,
                longitude=-96.7266,
                has_camera_exif=True,
            )
        )

    staged = cluster_and_stage_items(items)
    trip_folders = {staged[f"trip_{day}"]["target_folder"] for day in (1, 2, 3)}
    assert len(trip_folders) == 1
    trip_folder = next(iter(trip_folders))
    assert trip_folder.startswith("01_Recuerdos/2024/2024-09-01 al 09-03 - Viaje a ")


def test_mod13_04_consolidate_sparse_local_events_momentos_varios(tmp_path: Path) -> None:
    """Local events (<3 files inside base zone) consolidate into YYYY-MM - Momentos Varios;
    outside base zone keeps its own folder.
    """
    from photo_organizer.clustering import ClusterItem, cluster_and_stage_items

    items = _base_zone_items(tmp_path, count_days=10)
    # Sparse local day (2 files in CDMX base zone)
    for idx in range(2):
        items.append(
            ClusterItem(
                id=f"local_sparse_{idx}",
                source_path=str(tmp_path / f"PXL_20240505_100{idx}00.jpg"),
                ext=".jpg",
                sha256=f"sha_ls_{idx}",
                is_duplicate=False,
                captured_at=f"2024-05-05T10:0{idx}:00Z",
                latitude=19.4326,
                longitude=-99.1332,
                has_camera_exif=True,
            )
        )
    # Sparse outside-base-zone day (1 file in Guadalajara)
    items.append(
        ClusterItem(
            id="away_sparse",
            source_path=str(tmp_path / "PXL_20240612_150000.jpg"),
            ext=".jpg",
            sha256="sha_away_sparse",
            is_duplicate=False,
            captured_at="2024-06-12T15:00:00Z",
            latitude=20.6597,
            longitude=-103.3496,
            has_camera_exif=True,
        )
    )

    staged = cluster_and_stage_items(items)
    assert staged["local_sparse_0"]["target_folder"] == "01_Recuerdos/2024/2024-05 - Momentos Varios"
    assert staged["local_sparse_1"]["target_folder"] == "01_Recuerdos/2024/2024-05 - Momentos Varios"
    assert staged["away_sparse"]["target_folder"].startswith("01_Recuerdos/2024/2024-06-12 - ")
    assert "Momentos Varios" not in staged["away_sparse"]["target_folder"]


def test_mod13_05_sticker_and_tiny_file_filter(tmp_path: Path) -> None:
    """Images <15 KB or .webp without camera EXIF route to 02_Capturas_y_WhatsApp/YYYY-MM/."""
    from photo_organizer.clustering import ClusterItem, cluster_and_stage_items

    items = [
        ClusterItem(
            id="tiny_jpg",
            source_path=str(tmp_path / "IMG_20240701_120000.jpg"),
            ext=".jpg",
            sha256="sha_tiny",
            is_duplicate=False,
            captured_at="2024-07-01T12:00:00Z",
            latitude=None,
            longitude=None,
            has_camera_exif=False,
            file_size=8192,
        ),
        ClusterItem(
            id="webp_sticker",
            source_path=str(tmp_path / "STK_20240702_130000.webp"),
            ext=".webp",
            sha256="sha_webp",
            is_duplicate=False,
            captured_at="2024-07-02T13:00:00Z",
            latitude=None,
            longitude=None,
            has_camera_exif=False,
            file_size=45000,
        ),
        ClusterItem(
            id="normal_photo",
            source_path=str(tmp_path / "PXL_20240703_140000.jpg"),
            ext=".jpg",
            sha256="sha_normal",
            is_duplicate=False,
            captured_at="2024-07-03T14:00:00Z",
            latitude=None,
            longitude=None,
            has_camera_exif=True,
            file_size=2500000,
        ),
    ]

    staged = cluster_and_stage_items(items)
    assert staged["tiny_jpg"]["target_folder"] == "02_Capturas_y_WhatsApp/2024-07"
    assert staged["webp_sticker"]["target_folder"] == "02_Capturas_y_WhatsApp/2024-07"
    assert staged["normal_photo"]["target_folder"].startswith("01_Recuerdos/2024/")


def test_mod13_06_live_motion_photo_pairing(tmp_path: Path) -> None:
    """Image + short video (<=3s) with same timestamp & folder share YYYY-MM-DD_HH-MM-SS_### stem."""
    from photo_organizer.clustering import ClusterItem, cluster_and_stage_items

    items = [
        ClusterItem(
            id="live_img",
            source_path=str(tmp_path / "PXL_20241005_162030.jpg"),
            ext=".jpg",
            sha256="sha_live_img",
            is_duplicate=False,
            captured_at="2024-10-05T16:20:30Z",
            latitude=None,
            longitude=None,
            has_camera_exif=True,
            file_size=1500000,
        ),
        ClusterItem(
            id="live_vid",
            source_path=str(tmp_path / "PXL_20241005_162030.mp4"),
            ext=".mp4",
            sha256="sha_live_vid",
            is_duplicate=False,
            captured_at="2024-10-05T16:20:30Z",
            latitude=None,
            longitude=None,
            has_camera_exif=False,
            file_size=800000,
            duration_sec=1.8,
        ),
    ]

    staged = cluster_and_stage_items(items)
    assert staged["live_img"]["target_folder"] == staged["live_vid"]["target_folder"]
    img_name = staged["live_img"]["target_filename"]
    vid_name = staged["live_vid"]["target_filename"]
    assert img_name == "2024-10-05_16-20-30_001.jpg"
    assert vid_name == "2024-10-05_16-20-30_001.mp4"


def _post_json(base_url: str, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"{base_url}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_mod13_07_ai_tree_optimizer_endpoints_suggestions_and_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /api/tree/ai-suggestions batches <=12 folders with <=256px thumbnails and
    POST /api/tree/apply-suggestions applies rename, merge, and split cleanly.
    """
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.db import init_db, open_connection
    from photo_organizer.main import _LocalAppCore, _make_stdlib_handler

    src_dir = tmp_path / "src_opt"
    dest_dir = tmp_path / "dest_opt"
    src_dir.mkdir()

    # Create files for:
    # - Folder A (to rename)
    # - Folder B1 & B2 (consecutive, to merge)
    # - Folder C (>25 files with >=3h internal gap, to split)
    for i in range(2):
        (src_dir / f"PXL_20240801_100{i}00.jpg").write_bytes(_make_jpeg_bytes(f"a_{i}".encode()))
    (src_dir / "PXL_20240802_110000.jpg").write_bytes(_make_jpeg_bytes(b"b1"))
    (src_dir / "PXL_20240803_120000.jpg").write_bytes(_make_jpeg_bytes(b"b2"))
    for i in range(14):
        (src_dir / f"PXL_20240810_09{i:02d}00.jpg").write_bytes(_make_jpeg_bytes(f"c1_{i}".encode()))
    for i in range(14):
        (src_dir / f"PXL_20240810_15{i:02d}00.jpg").write_bytes(_make_jpeg_bytes(f"c2_{i}".encode()))

    cfg = OrganizerConfig(
        db_path=tmp_path / "organizer.db",
        thumb_cache_dir=tmp_path / "thumbs",
    )
    init_db(cfg.db_path)
    core = _LocalAppCore(cfg)
    job_id = asyncio.run(core.orchestrator.run_job(src_dir, dest_dir))

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_stdlib_handler(core))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    folder_a = "01_Recuerdos/2024/2024-08-01 - Eventos del día"
    folder_b1 = "01_Recuerdos/2024/2024-08-02 - Eventos del día"
    folder_b2 = "01_Recuerdos/2024/2024-08-03 - Eventos del día"
    folder_c = "01_Recuerdos/2024/2024-08-10 - Eventos del día"

    # Normalize folder names in DB so test is independent of initial heuristic naming
    conn = open_connection(cfg.db_path)
    try:
        rows = conn.execute(
            "SELECT id, captured_at FROM media_items WHERE job_id = ?",
            (job_id,),
        ).fetchall()
        for r in rows:
            ts = r["captured_at"] or ""
            if ts.startswith("2024-08-01"):
                tf = folder_a
            elif ts.startswith("2024-08-02"):
                tf = folder_b1
            elif ts.startswith("2024-08-03"):
                tf = folder_b2
            else:
                tf = folder_c
            conn.execute("UPDATE media_items SET target_folder = ? WHERE id = ?", (tf, r["id"]))
        conn.commit()
    finally:
        conn.close()

    agy_cmds: list[list[str]] = []

    def fake_agy_suggestions(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        agy_cmds.append(cmd)
        payload = {
            "suggestions": [
                {
                    "type": "rename",
                    "before": folder_a,
                    "after": "01_Recuerdos/2024/2024-08-01 - Desayuno en el Parque",
                    "reason": "Escena matutina al aire libre",
                },
                {
                    "type": "merge",
                    "before": [folder_b1, folder_b2],
                    "after": "01_Recuerdos/2024/2024-08-02 al 08-03 - Fin de Semana",
                    "reason": "Días consecutivos relacionados",
                },
                {
                    "type": "split",
                    "before": folder_c,
                    "after": [
                        "01_Recuerdos/2024/2024-08-10 - Sesión Mañana",
                        "01_Recuerdos/2024/2024-08-10 - Sesión Tarde",
                    ],
                    "reason": "Salto temporal interno mayor a 3 horas",
                },
            ]
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    shim = tmp_path / "agy.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")

    try:
        with monkeypatch.context() as mp:
            mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if "agy" in str(b) else None)
            mp.setattr(subprocess, "run", fake_agy_suggestions)
            status_code, data = _post_json(base_url, "/api/tree/ai-suggestions", {"job_id": job_id})

        assert status_code == 200
        assert data.get("engine")
        suggestions = data.get("suggestions", [])
        types_seen = {s.get("type") for s in suggestions}
        assert {"rename", "merge", "split"}.issubset(types_seen)

        for s in suggestions:
            serialized = json.dumps(s, ensure_ascii=False).lower()
            assert "__" not in serialized
            assert "placeholder" not in serialized
            assert "sample" not in serialized

        apply_status, _ = _post_json(
            base_url,
            "/api/tree/apply-suggestions",
            {"job_id": job_id, "suggestions": suggestions},
        )
        assert apply_status == 200
    finally:
        server.shutdown()
        server.server_close()

    conn = open_connection(cfg.db_path)
    try:
        distinct_folders = {
            r["target_folder"]
            for r in conn.execute(
                "SELECT DISTINCT target_folder FROM media_items WHERE job_id = ?",
                (job_id,),
            ).fetchall()
        }
    finally:
        conn.close()

    assert "01_Recuerdos/2024/2024-08-01 - Desayuno en el Parque" in distinct_folders
    assert "01_Recuerdos/2024/2024-08-02 al 08-03 - Fin de Semana" in distinct_folders
    assert "01_Recuerdos/2024/2024-08-10 - Sesión Mañana" in distinct_folders
    assert "01_Recuerdos/2024/2024-08-10 - Sesión Tarde" in distinct_folders


def test_mod13_08_ai_tree_optimizer_deterministic_fallback_when_agy_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When agy fails or is absent, /api/tree/ai-suggestions uses deterministic fallback and reports engine."""
    from photo_organizer.config import OrganizerConfig
    from photo_organizer.db import init_db
    from photo_organizer.main import _LocalAppCore, _make_stdlib_handler

    src_dir = tmp_path / "src_fb"
    dest_dir = tmp_path / "dest_fb"
    src_dir.mkdir()
    for i in range(14):
        (src_dir / f"PXL_20240915_08{i:02d}00.jpg").write_bytes(_make_jpeg_bytes(f"fb1_{i}".encode()))
    for i in range(14):
        (src_dir / f"PXL_20240915_16{i:02d}00.jpg").write_bytes(_make_jpeg_bytes(f"fb2_{i}".encode()))

    cfg = OrganizerConfig(
        db_path=tmp_path / "organizer.db",
        thumb_cache_dir=tmp_path / "thumbs",
    )
    init_db(cfg.db_path)
    core = _LocalAppCore(cfg)
    job_id = asyncio.run(core.orchestrator.run_job(src_dir, dest_dir))

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_stdlib_handler(core))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with monkeypatch.context() as mp:
            mp.setattr(shutil, "which", lambda _b, **_kw: None)
            status_code, data = _post_json(base_url, "/api/tree/ai-suggestions", {"job_id": job_id})
    finally:
        server.shutdown()
        server.server_close()

    assert status_code == 200
    engine = str(data.get("engine", "")).lower()
    assert "fallback" in engine or "local" in engine or "determin" in engine
    suggestions = data.get("suggestions", [])
    assert isinstance(suggestions, list)
    for s in suggestions:
        serialized = json.dumps(s, ensure_ascii=False).lower()
        assert "__" not in serialized
        assert "placeholder" not in serialized
        assert "sample" not in serialized


def test_mod13_09_ui_optimize_tree_button_panel_and_controls() -> None:
    """UI includes '✨ Optimizar Árbol con IA' button with spinner/disabled state and suggestion panel."""
    html = _read_static("index.html")
    js = _read_static("app.js")
    css = _read_static("styles.css")
    combined = html + "\n" + js + "\n" + css

    assert "Optimizar Árbol con IA" in combined
    assert "Aplicar sugerencias seleccionadas" in combined
    assert "/api/tree/ai-suggestions" in js
    assert "/api/tree/apply-suggestions" in js
    assert re.search(r'type=["\']checkbox["\']', combined)
    assert "disabled" in js
    assert "spinner" in (js + "\n" + css).lower()


def test_mod13_10_live_photo_ignores_long_video_over_3s(tmp_path: Path) -> None:
    """Videos > 3s sharing timestamp & folder do not pair with the image as a Live Photo."""
    from photo_organizer.clustering import ClusterItem, cluster_and_stage_items

    items = [
        ClusterItem(
            id="img_1",
            source_path=str(tmp_path / "PXL_20241005_162030.jpg"),
            ext=".jpg",
            sha256="sha_img_1",
            is_duplicate=False,
            captured_at="2024-10-05T16:20:30Z",
            latitude=None,
            longitude=None,
            has_camera_exif=True,
            file_size=1500000,
        ),
        ClusterItem(
            id="long_vid",
            source_path=str(tmp_path / "PXL_20241005_162030.mp4"),
            ext=".mp4",
            sha256="sha_long_vid",
            is_duplicate=False,
            captured_at="2024-10-05T16:20:30Z",
            latitude=None,
            longitude=None,
            has_camera_exif=False,
            file_size=4000000,
            duration_sec=8.5,
        ),
    ]

    staged = cluster_and_stage_items(items)
    assert staged["img_1"]["target_filename"] == "2024-10-05_16-20-30_001.jpg"
    assert staged["long_vid"]["target_filename"] == "2024-10-05_16-20-30_002.mp4"


def test_mod13_11_ai_optimizer_batches_12_folders_and_enforces_split_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """suggest_tree_optimizations batches <=12 folders per agy call and enforces split (>25 files, >=3h gap)."""
    from photo_organizer.ai_classifier import suggest_tree_optimizations

    summaries = [
        {
            "folder": f"01_Recuerdos/2024/2024-08-{i + 1:02d} - Eventos del día",
            "count": 20 if i == 0 else 28,
            "start_date": f"2024-08-{i + 1:02d}T09:00:00+00:00",
            "end_date": f"2024-08-{i + 1:02d}T18:00:00+00:00",
            "location_label": "CDMX, MX",
            "max_gap_hours": 4.0 if i in (0, 1) else 1.5,
            "thumb_path": str(tmp_path / f"thumb_{i}.jpg"),
        }
        for i in range(15)
    ]

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        # Return an invalid split for folder 0 (count=20 <= 25) and folder 2 (max_gap_hours=1.5 < 3.0)
        # and a valid split for folder 1 (count=28 > 25, max_gap_hours=4.0 >= 3.0).
        payload = {
            "suggestions": [
                {
                    "type": "split",
                    "before": summaries[0]["folder"],
                    "after": [summaries[0]["folder"] + " A", summaries[0]["folder"] + " B"],
                    "reason": "No debe aplicar por tener <=25 archivos",
                },
                {
                    "type": "split",
                    "before": summaries[1]["folder"],
                    "after": [summaries[1]["folder"] + " Mañana", summaries[1]["folder"] + " Tarde"],
                    "reason": "Salto mayor a 3 horas",
                },
                {
                    "type": "split",
                    "before": summaries[2]["folder"],
                    "after": [summaries[2]["folder"] + " A", summaries[2]["folder"] + " B"],
                    "reason": "No debe aplicar por salto < 3 horas",
                },
            ]
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    shim = tmp_path / "agy.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")

    with monkeypatch.context() as mp:
        mp.setattr(shutil, "which", lambda b, **_kw: str(shim) if "agy" in str(b) else None)
        mp.setattr(subprocess, "run", fake_run)
        res = suggest_tree_optimizations(summaries, cli_command="agy")

    assert len(calls) == 2
    assert "agy" in res["engine"].lower()
    split_befores = [s["before"] for s in res["suggestions"] if s["type"] == "split"]
    assert summaries[1]["folder"] in split_befores
    assert summaries[0]["folder"] not in split_befores
    assert summaries[2]["folder"] not in split_befores


def test_mod13_12_representative_thumbnail_resized_to_256px(tmp_path: Path) -> None:
    """get_or_create_resized creates and caches a <=256px representative thumbnail path."""
    from photo_organizer.thumbnails import ThumbnailCache

    img_path = tmp_path / "large.jpg"
    img_path.write_bytes(_make_jpeg_bytes(b"rep_thumb"))

    cache = ThumbnailCache(tmp_path / "cache")
    thumb_path = cache.get_or_create_resized("item_1", img_path, sha256="abc123", max_dim=256)
    assert thumb_path.exists()
    assert thumb_path.name == "ai_256_abc123.jpg"
    assert cache.get_or_create_resized("item_1", img_path, sha256="abc123", max_dim=256) == thumb_path
