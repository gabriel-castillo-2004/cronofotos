"""Tests for post-delivery modification MOD-19:
1. Safe in-place refinement on Windows (`Refinar árbol ya exportado`):
   - Applies a modified `_cronofotos_plan.json` over an already-exported library in `dest_dir`.
   - Moves files within `dest_dir` file-by-file without re-copying from `source_dir`.
   - Recalculates `###` collision suffixes when merging/moving files with colliding names.
   - Removes residual empty folders.
   - Automatically updates `_cronofotos_index.json`, `_cronofotos_plan.json`, and `<data_dir>/master_plan.json`.
2. Incremental Delta mode (`Solo archivos nuevos (Delta contra plan/índice previo)`):
   - Checkbox in `index.html` wired in `app.js`.
   - Skips during scan any file whose `sha256` already exists in `_cronofotos_plan.json` or `_cronofotos_index.json`.
   - Exports only new events organized by year (`YYYY/YYYY-MM-DD - Evento/...`) ready for Google Drive.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import re
import shutil
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _discover_refine_endpoint(js: str) -> str | None:
    match = re.search(r"""['"](/api/[^'"]*refin[^'"]*)['"]""", js, re.IGNORECASE)
    return match.group(1) if match else None


def test_mod19_01_ui_refine_and_delta_controls() -> None:
    """UI exposes 'Refinar árbol ya exportado' button and 'Solo archivos nuevos (Delta contra plan/índice previo)' checkbox."""
    html = _read_static("index.html")
    js = _read_static("app.js")

    assert "Refinar árbol ya exportado" in html
    assert "Solo archivos nuevos (Delta contra plan/índice previo)" in html
    assert re.search(
        r"<input[^>]+type=[\"']checkbox[\"'][^>]*>[\s\S]{0,160}Solo archivos nuevos \(Delta contra plan/índice previo\)"
        r"|Solo archivos nuevos \(Delta contra plan/índice previo\)[\s\S]{0,160}<input[^>]+type=[\"']checkbox[\"']",
        html,
    )

    refine_ep = _discover_refine_endpoint(js)
    assert refine_ep is not None, "app.js must wire an endpoint for 'Refinar árbol ya exportado'"
    assert re.search(r"delta|incremental", js, re.IGNORECASE)


def test_mod19_02_refine_inplace_merge_rename_collision_and_cleanup(tmp_path: Path) -> None:
    """Refining an already-exported library moves files in-place without source_dir,
    recalculates ### suffixes on collisions, removes empty folders, and updates
    _cronofotos_index.json, _cronofotos_plan.json, and <data_dir>/master_plan.json.
    """
    _core, server, base_url, data_dir = _start_test_server(tmp_path)
    try:
        src_dir = tmp_path / "source_library"
        dest_dir = tmp_path / "exported_library"
        src_dir.mkdir()
        dest_dir.mkdir()

        # Two files with identical timestamp in different source folders -> will collide when merged
        sub_a = src_dir / "AlbumA"
        sub_b = src_dir / "AlbumB"
        sub_c = src_dir / "AlbumC"
        sub_a.mkdir()
        sub_b.mkdir()
        sub_c.mkdir()

        bytes_a = _make_jpeg_bytes(b"photo_a_same_ts")
        bytes_b = _make_jpeg_bytes(b"photo_b_same_ts")
        bytes_c = _make_jpeg_bytes(b"photo_c_same_ts")

        sha_a = _sha256_bytes(bytes_a)
        sha_b = _sha256_bytes(bytes_b)
        sha_c = _sha256_bytes(bytes_c)

        (sub_a / "PXL_20240501_120000.jpg").write_bytes(bytes_a)
        (sub_b / "PXL_20240501_120000.jpg").write_bytes(bytes_b)
        (sub_c / "PXL_20240502_150000.jpg").write_bytes(bytes_c)

        code, _ = _http_json(
            "POST",
            base_url,
            "/api/scan/start",
            {"source_dir": str(src_dir), "dest_dir": str(dest_dir)},
        )
        assert code == 200
        status = _wait_scan_completed(base_url)
        assert status["status"] == "completed"
        job_id = status["job_id"]

        # Place the three files into three separate folders before initial export
        conn = sqlite3.connect(_core.cfg.db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE media_items SET target_folder = ?, target_filename = ? WHERE job_id = ? AND sha256 = ?",
                    ("01_Recuerdos/2024/2024-05-01 - Manana", "2024-05-01_12-00-00_001.jpg", job_id, sha_a),
                )
                conn.execute(
                    "UPDATE media_items SET target_folder = ?, target_filename = ? WHERE job_id = ? AND sha256 = ?",
                    ("01_Recuerdos/2024/2024-05-01 - Tarde", "2024-05-01_12-00-00_001.jpg", job_id, sha_b),
                )
                conn.execute(
                    "UPDATE media_items SET target_folder = ?, target_filename = ? WHERE job_id = ? AND sha256 = ?",
                    ("01_Recuerdos/2024/2024-05-02 - Paseo", "2024-05-02_15-00-00_001.jpg", job_id, sha_c),
                )
        finally:
            conn.close()

        exp_code, exp_res = _http_json("POST", base_url, "/api/export", {"job_id": job_id})
        assert exp_code == 200, f"Initial export failed: {exp_res}"
        assert exp_res.get("exported", 0) == 3

        # Delete source_dir completely to prove in-place refinement does not re-copy from source
        shutil.rmtree(src_dir)
        assert not src_dir.exists()

        # Export current plan, then modify it to merge A+B and rename C
        plan_code, plan_res = _http_json("POST", base_url, "/api/tree/plan/export", {"job_id": job_id})
        assert plan_code == 200, f"Plan export failed: {plan_res}"
        plan_path = dest_dir / "_cronofotos_plan.json"
        assert plan_path.exists()
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))

        merged_folder = "01_Recuerdos/2024/2024-05-01 - Dia Completo"
        renamed_folder = "01_Recuerdos/2024/2024-05-02 - Paseo por el Centro"

        files_map = plan_data.get("files", {})
        files_map[sha_a] = merged_folder
        files_map[sha_b] = merged_folder
        files_map[sha_c] = renamed_folder
        plan_data["files"] = files_map
        plan_path.write_text(json.dumps(plan_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # Trigger in-place refine with the modified _cronofotos_plan.json
        refine_ep = _discover_refine_endpoint(_read_static("app.js")) or "/api/tree/refine"
        ref_code, ref_res = _http_json(
            "POST",
            base_url,
            refine_ep,
            {
                "job_id": job_id,
                "dest_dir": str(dest_dir),
                "plan": plan_data,
                "plan_path": str(plan_path),
            },
        )
        assert ref_code == 200, f"Refine failed ({ref_code}): {ref_res}"

        # Verify zero file loss and exact SHA-256 preservation
        exported_media = [
            p for p in dest_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
        ]
        assert len(exported_media) == 3
        actual_shas = {_sha256_file(p): p for p in exported_media}
        assert set(actual_shas.keys()) == {sha_a, sha_b, sha_c}

        # Verify collision suffix recalculation in merged folder
        merged_dir = dest_dir / merged_folder
        merged_files = sorted(p.name for p in merged_dir.iterdir() if p.is_file())
        assert len(merged_files) == 2
        assert merged_files == ["2024-05-01_12-00-00_001.jpg", "2024-05-01_12-00-00_002.jpg"]

        # Verify renamed folder has its file
        renamed_dir = dest_dir / renamed_folder
        assert any(p.is_file() for p in renamed_dir.iterdir())

        # Verify residual empty folders were removed
        assert not (dest_dir / "01_Recuerdos/2024/2024-05-01 - Manana").exists()
        assert not (dest_dir / "01_Recuerdos/2024/2024-05-01 - Tarde").exists()
        assert not (dest_dir / "01_Recuerdos/2024/2024-05-02 - Paseo").exists()

        # Verify _cronofotos_index.json, _cronofotos_plan.json, and <data_dir>/master_plan.json were updated
        idx_data = json.loads((dest_dir / "_cronofotos_index.json").read_text(encoding="utf-8"))
        assert merged_folder in str(idx_data.get(sha_a, {}))
        assert merged_folder in str(idx_data.get(sha_b, {}))
        assert renamed_folder in str(idx_data.get(sha_c, {}))

        updated_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        assert updated_plan["files"][sha_a] == merged_folder
        assert updated_plan["files"][sha_b] == merged_folder
        assert updated_plan["files"][sha_c] == renamed_folder

        master_plan_path = data_dir / "master_plan.json"
        assert master_plan_path.exists(), "<data_dir>/master_plan.json must be updated automatically"
        master_plan_text = master_plan_path.read_text(encoding="utf-8")
        assert merged_folder in master_plan_text
        assert renamed_folder in master_plan_text
    finally:
        server.shutdown()
        server.server_close()


def test_mod19_03_delta_scan_omits_files_in_previous_plan_and_index(tmp_path: Path) -> None:
    """Delta scan omits files whose sha256 already exists in _cronofotos_plan.json or _cronofotos_index.json."""
    core, server, base_url, data_dir = _start_test_server(tmp_path)
    try:
        plan_old_bytes = _make_jpeg_bytes(b"already_in_previous_plan")
        idx_old_bytes = _make_jpeg_bytes(b"already_in_previous_index")
        brand_new_bytes = _make_jpeg_bytes(b"brand_new_delta_2025")

        plan_old_sha = _sha256_bytes(plan_old_bytes)
        idx_old_sha = _sha256_bytes(idx_old_bytes)
        brand_new_sha = _sha256_bytes(brand_new_bytes)

        prev_dir = tmp_path / "previous_backup_meta"
        prev_dir.mkdir()
        (prev_dir / "_cronofotos_plan.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "files": {plan_old_sha: "01_Recuerdos/2023/2023-06-10 - Playa"},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (prev_dir / "_cronofotos_index.json").write_text(
            json.dumps(
                {
                    idx_old_sha: {
                        "rel_path": "2023/2023-07-01 - Cena/2023-07-01_20-00-00_001.jpg",
                        "size_bytes": len(idx_old_bytes),
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Also place _cronofotos_plan.json in data_dir/master_plan.json and dest_dir so any lookup path finds it
        clean_dest = tmp_path / "Delta_Export_2025"
        clean_dest.mkdir()
        (clean_dest / "_cronofotos_plan.json").write_text(
            (prev_dir / "_cronofotos_plan.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (clean_dest / "_cronofotos_index.json").write_text(
            (prev_dir / "_cronofotos_index.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        src_dir = tmp_path / "mixed_camera_roll"
        src_dir.mkdir()
        (src_dir / "PXL_20230610_110000.jpg").write_bytes(plan_old_bytes)
        (src_dir / "PXL_20230701_200000.jpg").write_bytes(idx_old_bytes)
        (src_dir / "PXL_20250214_180000.jpg").write_bytes(brand_new_bytes)

        code, _ = _http_json(
            "POST",
            base_url,
            "/api/scan/start",
            {
                "source_dir": str(src_dir),
                "dest_dir": str(clean_dest),
                "incremental_mode": True,
                "incremental": True,
                "master_index_path": str(prev_dir),
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
        assert len(by_name) == 3

        for fname in ("PXL_20230610_110000.jpg", "PXL_20230701_200000.jpg"):
            row = by_name[fname]
            reason_blob = f"{row.get('route_reason') or ''} {row.get('target_folder') or ''}".lower()
            assert (
                row["is_duplicate"] == 1
                or "índice" in reason_blob
                or "already_in_library" in reason_blob
                or "plan" in reason_blob
            ), f"Expected previously indexed/planned file {fname} to be omitted, got {row}"

        assert by_name["PXL_20250214_180000.jpg"]["is_duplicate"] == 0
    finally:
        server.shutdown()
        server.server_close()


def test_mod19_04_delta_export_only_new_events_by_year(tmp_path: Path) -> None:
    """Delta export writes only new events organized by year (YYYY/YYYY-MM-DD - Evento/) for Google Drive."""
    _core, server, base_url, _data_dir = _start_test_server(tmp_path)
    try:
        old_bytes = _make_jpeg_bytes(b"old_plan_photo")
        new_bytes = _make_jpeg_bytes(b"new_event_2025_03")
        old_sha = _sha256_bytes(old_bytes)
        new_sha = _sha256_bytes(new_bytes)

        prev_plan_file = tmp_path / "_cronofotos_plan.json"
        prev_plan_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "files": {old_sha: "01_Recuerdos/2022/2022-09-15 - Fiesta"},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        src_dir = tmp_path / "src_delta"
        src_dir.mkdir()
        (src_dir / "PXL_20220915_100000.jpg").write_bytes(old_bytes)
        (src_dir / "PXL_20250320_160000.jpg").write_bytes(new_bytes)

        clean_dest = tmp_path / "Drive_Delta_Ready"
        clean_dest.mkdir()
        shutil.copy2(prev_plan_file, clean_dest / "_cronofotos_plan.json")

        code, _ = _http_json(
            "POST",
            base_url,
            "/api/scan/start",
            {
                "source_dir": str(src_dir),
                "dest_dir": str(clean_dest),
                "incremental_mode": True,
                "incremental": True,
                "master_index_path": str(prev_plan_file),
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
                "master_index_path": str(prev_plan_file),
            },
        )
        assert exp_code == 200, f"Delta export failed: {exp_res}"
        assert exp_res.get("exported", 0) == 1

        exported_media = [
            p for p in clean_dest.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
        ]
        assert len(exported_media) == 1
        assert _sha256_file(exported_media[0]) == new_sha

        rel_parts = exported_media[0].relative_to(clean_dest).parts
        assert rel_parts[0] == "2025", f"Expected year-first layout YYYY/YYYY-MM-DD - Evento/, got {rel_parts}"
        assert rel_parts[1].startswith("2025-03-20 - "), f"Expected event folder, got {rel_parts}"
    finally:
        server.shutdown()
        server.server_close()
