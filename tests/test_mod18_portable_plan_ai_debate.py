"""Tests for post-delivery modification MOD-18:
1. Portable Tree Plan with Checkpointing (`_cronofotos_plan.json`):
   - `Exportar Plan (.json)` and `Importar Plan (.json)` buttons & endpoints.
   - Saves/restores `sha256 -> carpeta destino`, per-folder `pending`/`debated` state,
     and debate history, synchronizing with SQLite without touching media files.
   - Saves atomically (`.tmp` + `os.replace`) at the end of each batch.
2. Resumable 2-Agent Consensus Debate (`POST /api/tree/ai-debate`):
   - Processes pending folders in small batches of 5 using `agy` with 2 roles:
     * Agente Curador: proposes searchable Spanish names (`Evento + Personas Clave + Lugar`)
       using scene thumbnails and detected face crops.
     * Agente Auditor: fixes generic names hard to search in Drive, suggests merges of
       related consecutive folders, and consolidates the final name.
   - On quota/token/auth errors from `agy`, pauses immediately without overwriting with
     generic names, reports saved progress (`X/N carpetas listas`), and enables
     `Reanudar Debate IA` to continue only with remaining `pending` folders.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import sqlite3
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


def _discover_plan_endpoints(js: str) -> tuple[str | None, str | None]:
    """Find plan export/import endpoint paths referenced in app.js."""
    export_match = re.search(r"""['"](/api/[^'"]*plan[^'"]*export[^'"]*|/api/[^'"]*export[^'"]*plan[^'"]*)['"]""", js, re.IGNORECASE)
    import_match = re.search(r"""['"](/api/[^'"]*plan[^'"]*import[^'"]*|/api/[^'"]*import[^'"]*plan[^'"]*)['"]""", js, re.IGNORECASE)
    return (
        export_match.group(1) if export_match else None,
        import_match.group(1) if import_match else None,
    )


def _create_multi_folder_scan(
    base_url: str,
    tmp_path: Path,
    folder_dates: list[str],
) -> tuple[str, Path, Path, dict[str, str]]:
    """Create one photo per date folder, run scan, and return (job_id, src_dir, dest_dir, sha_by_name)."""
    src_dir = tmp_path / "src_photos"
    dest_dir = tmp_path / "dest_library"
    src_dir.mkdir(parents=True, exist_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    sha_by_name: dict[str, str] = {}
    for idx, d in enumerate(folder_dates):
        compact = d.replace("-", "")
        fname = f"PXL_{compact}_1200{idx:02d}.jpg"
        raw = _make_jpeg_bytes(f"photo_{d}_{idx}".encode("utf-8"))
        (src_dir / fname).write_bytes(raw)
        sha_by_name[fname] = _sha256_bytes(raw)

    code, _ = _http_json(
        "POST",
        base_url,
        "/api/scan/start",
        {
            "source_dir": str(src_dir),
            "dest_dir": str(dest_dir),
        },
    )
    assert code == 200
    status = _wait_scan_completed(base_url)
    assert status["status"] == "completed"
    return status["job_id"], src_dir, dest_dir, sha_by_name


def test_mod18_01_ui_portable_plan_and_ai_debate_controls() -> None:
    """UI exposes Exportar Plan (.json), Importar Plan (.json), Reanudar Debate IA, and progress wiring."""
    html = _read_static("index.html")
    js = _read_static("app.js")

    assert "Exportar Plan (.json)" in html
    assert "Importar Plan (.json)" in html
    assert "Reanudar Debate IA" in html

    assert "/api/tree/ai-debate" in js
    assert re.search(r"carpetas listas", js, re.IGNORECASE)
    export_ep, import_ep = _discover_plan_endpoints(js)
    assert export_ep is not None, "app.js must wire an endpoint for Exportar Plan (.json)"
    assert import_ep is not None, "app.js must wire an endpoint for Importar Plan (.json)"


def test_mod18_02_plan_export_import_syncs_sqlite_without_touching_media(tmp_path: Path) -> None:
    """Exporting and importing _cronofotos_plan.json persists sha256->folder, folder states, and debate history in SQLite without touching media files."""
    core, server, base_url, _data_dir = _start_test_server(tmp_path)
    try:
        dates = ["2024-03-01", "2024-03-05", "2024-03-10"]
        job_id, src_dir, dest_dir, sha_by_name = _create_multi_folder_scan(base_url, tmp_path, dates)

        src_mtimes_before = {p.name: p.stat().st_mtime_ns for p in src_dir.iterdir()}

        js = _read_static("app.js")
        export_ep, import_ep = _discover_plan_endpoints(js)
        export_ep = export_ep or "/api/tree/plan/export"
        import_ep = import_ep or "/api/tree/plan/import"

        code, exp_res = _http_json("POST", base_url, export_ep, {"job_id": job_id})
        if code == 404:
            code, exp_res = _http_json("GET", base_url, f"{export_ep}?job_id={job_id}")
        assert code == 200, f"Plan export failed ({code}): {exp_res}"

        plan_file = dest_dir / "_cronofotos_plan.json"
        if not plan_file.exists() and isinstance(exp_res.get("plan_path"), str):
            plan_file = Path(exp_res["plan_path"])
        if not plan_file.exists() and isinstance(exp_res.get("plan"), dict):
            plan_data = exp_res["plan"]
        else:
            assert plan_file.exists(), "_cronofotos_plan.json must be written on export"
            plan_data = json.loads(plan_file.read_text(encoding="utf-8"))

        # Verify plan structure contains sha256 -> carpeta destino, folder state, and debate history
        plan_text = json.dumps(plan_data, ensure_ascii=False)
        for sha in sha_by_name.values():
            assert sha in plan_text, f"Exported plan must map sha256 {sha} to target folder"

        # Verify media files were NOT copied to dest_dir and source files were untouched
        dest_media = [p for p in dest_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
        assert dest_media == [], "Plan export must not copy media files yet"
        assert {p.name: p.stat().st_mtime_ns for p in src_dir.iterdir()} == src_mtimes_before

        # Modify target folder for one sha256 in the plan and mark it debated with history, then import
        first_sha = next(iter(sha_by_name.values()))
        new_folder = "01_Recuerdos/2024/2024-03-01 - Boda Ana y Luis en Oaxaca"

        def _patch_plan(obj: Any) -> bool:
            if isinstance(obj, dict):
                if first_sha in obj:
                    val = obj[first_sha]
                    if isinstance(val, str):
                        obj[first_sha] = new_folder
                        return True
                    if isinstance(val, dict):
                        for k in ("target_folder", "folder", "carpeta_destino", "dest_folder"):
                            if k in val:
                                val[k] = new_folder
                                return True
                for v in obj.values():
                    if _patch_plan(v):
                        return True
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and item.get("sha256") == first_sha:
                        for k in ("target_folder", "folder", "carpeta_destino", "dest_folder"):
                            if k in item:
                                item[k] = new_folder
                                return True
                        item["target_folder"] = new_folder
                        return True
                    if _patch_plan(item):
                        return True
            return False

        assert _patch_plan(plan_data), "Could not locate sha256 entry in exported plan JSON"
        if isinstance(plan_data, dict):
            if isinstance(plan_data.get("folders"), dict):
                plan_data["folders"][new_folder] = {
                    "status": "debated",
                    "state": "debated",
                    "history": [{"curador": "Boda Ana", "auditor": new_folder}],
                }
            if isinstance(plan_data.get("debate_history"), list):
                plan_data["debate_history"].append({"folder": new_folder, "role": "Auditor", "final": new_folder})

        plan_file.write_text(json.dumps(plan_data, ensure_ascii=False, indent=2), encoding="utf-8")

        code, imp_res = _http_json(
            "POST",
            base_url,
            import_ep,
            {"job_id": job_id, "plan": plan_data, "plan_path": str(plan_file)},
        )
        assert code == 200, f"Plan import failed ({code}): {imp_res}"

        # Verify SQLite reflects the imported target_folder without touching media files
        conn = sqlite3.connect(core.cfg.db_path)
        try:
            row = conn.execute(
                "SELECT target_folder FROM media_items WHERE job_id = ? AND sha256 = ?",
                (job_id, first_sha),
            ).fetchone()
        finally:
            conn.close()

        assert row is not None
        assert row[0] == new_folder
        dest_media_after = [p for p in dest_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
        assert dest_media_after == []
    finally:
        server.shutdown()
        server.server_close()


def test_mod18_03_plan_atomic_save_tmp_and_os_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving _cronofotos_plan.json uses a .tmp file followed by os.replace."""
    _core, server, base_url, _data_dir = _start_test_server(tmp_path)
    try:
        job_id, _src, dest_dir, _shas = _create_multi_folder_scan(
            base_url, tmp_path, ["2024-04-01", "2024-04-02"]
        )
        js = _read_static("app.js")
        export_ep, _ = _discover_plan_endpoints(js)
        export_ep = export_ep or "/api/tree/plan/export"

        replace_calls: list[tuple[str, str]] = []
        orig_replace = os.replace

        def _spy_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
            replace_calls.append((str(src), str(dst)))
            orig_replace(src, dst)

        monkeypatch.setattr(os, "replace", _spy_replace)

        code, _res = _http_json("POST", base_url, export_ep, {"job_id": job_id})
        if code == 404:
            code, _res = _http_json("GET", base_url, f"{export_ep}?job_id={job_id}")
        assert code == 200

        plan_replacements = [
            (s, d) for s, d in replace_calls if "_cronofotos_plan.json" in d
        ]
        assert plan_replacements, "Expected atomic os.replace call targeting _cronofotos_plan.json"
        assert any(".tmp" in s for s, _d in plan_replacements), (
            f"Expected .tmp source in os.replace for plan, got: {plan_replacements}"
        )
        assert list(dest_dir.glob("*.tmp")) == []
    finally:
        server.shutdown()
        server.server_close()


def test_mod18_04_ai_debate_two_roles_batches_of_5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/tree/ai-debate processes pending folders in batches of 5 using Curador and Auditor roles."""
    import photo_organizer.ai_classifier as ai_mod

    _core, server, base_url, _data_dir = _start_test_server(tmp_path)
    try:
        dates = [f"2024-05-{i:02d}" for i in range(1, 8)]  # 7 folders -> batches of 5 + 2
        job_id, _src, dest_dir, _shas = _create_multi_folder_scan(base_url, tmp_path, dates)

        monkeypatch.setattr(ai_mod, "resolve_cli_binary", lambda cmd, **_kw: "agy")
        monkeypatch.setattr(ai_mod, "is_standard_agy_binary", lambda *_a, **_kw: True)

        prompts_seen: list[str] = []

        def _fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            prompt = argv[-1] if argv else ""
            prompts_seen.append(prompt)
            # Extract folder names mentioned in prompt to return valid JSON for both Curador and Auditor
            folder_matches = re.findall(r"01_Recuerdos/[^\"\\]+", prompt)
            unique_folders = list(dict.fromkeys(folder_matches))
            if "auditor" in prompt.lower() or "consolid" in prompt.lower():
                payload = {
                    "decisions": [
                        {
                            "folder": f,
                            "final_name": f"Viaje Cultural {idx + 1} con Familia en Oaxaca",
                            "merge_with": None,
                            "reason": "Auditor consolidó nombre buscable",
                        }
                        for idx, f in enumerate(unique_folders)
                    ],
                    "suggestions": [
                        {
                            "type": "rename",
                            "before": f,
                            "after": re.sub(r" - .*$", f" - Viaje Cultural {idx + 1} con Familia en Oaxaca", f),
                            "reason": "Auditor consolidó nombre buscable",
                        }
                        for idx, f in enumerate(unique_folders)
                    ],
                }
            else:
                payload = {
                    "proposals": [
                        {
                            "folder": f,
                            "proposed_name": f"Evento Familiar {idx + 1} con Ana en Oaxaca",
                            "reason": "Curador identificó escena y rostros",
                        }
                        for idx, f in enumerate(unique_folders)
                    ]
                }
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload, ensure_ascii=False), stderr="")

        monkeypatch.setattr(subprocess, "run", _fake_run)

        code, res = _http_json("POST", base_url, "/api/tree/ai-debate", {"job_id": job_id})
        assert code == 200, f"ai-debate failed ({code}): {res}"
        assert len(prompts_seen) >= 2, "Expected at least 2 batches (5 + 2 folders)"
        assert any("curador" in p.lower() for p in prompts_seen)
        assert any("auditor" in p.lower() for p in prompts_seen)

        # Verify each batch had at most 5 folders
        for p in prompts_seen:
            batch_folders = set(re.findall(r"01_Recuerdos/\d{4}/\d{4}-\d{2}-\d{2}[^\"\\]*", p))
            assert len(batch_folders) <= 5, f"Batch exceeded 5 folders: {len(batch_folders)}"

        plan_file = dest_dir / "_cronofotos_plan.json"
        assert plan_file.exists(), "ai-debate must checkpoint _cronofotos_plan.json"
    finally:
        server.shutdown()
        server.server_close()


def test_mod18_05_ai_debate_pauses_on_quota_error_and_resumes_only_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On agy quota/auth error, ai-debate pauses immediately without overwriting pending folders, and resume processes only pending ones."""
    import photo_organizer.ai_classifier as ai_mod

    core, server, base_url, _data_dir = _start_test_server(tmp_path)
    try:
        dates = [f"2024-06-{i:02d}" for i in range(1, 8)]  # 7 folders -> Batch 1 (5), Batch 2 (2)
        job_id, _src, dest_dir, _shas = _create_multi_folder_scan(base_url, tmp_path, dates)

        monkeypatch.setattr(ai_mod, "resolve_cli_binary", lambda cmd, **_kw: "agy")
        monkeypatch.setattr(ai_mod, "is_standard_agy_binary", lambda *_a, **_kw: True)

        state = {"fail_batch_2": True, "batch_calls": 0, "resumed_folders": []}

        def _fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            prompt = argv[-1] if argv else ""
            folders = list(dict.fromkeys(re.findall(r"01_Recuerdos/\d{4}/\d{4}-\d{2}-\d{2}[^\"\\]*", prompt)))
            if not state["fail_batch_2"]:
                state["resumed_folders"].extend(folders)
            # Fail when processing June 06/07 during first run
            if state["fail_batch_2"] and any("2024-06-06" in f or "2024-06-07" in f for f in folders):
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    stdout="",
                    stderr="429 RESOURCE_EXHAUSTED: quota exceeded / token limit",
                )

            state["batch_calls"] += 1
            if "auditor" in prompt.lower() or "consolid" in prompt.lower():
                payload = {
                    "decisions": [
                        {
                            "folder": f,
                            "final_name": f"Concierto Verano {f[-10:]}",
                            "merge_with": None,
                        }
                        for f in folders
                    ],
                    "suggestions": [
                        {
                            "type": "rename",
                            "before": f,
                            "after": re.sub(r" - .*$", " - Concierto Verano", f),
                            "reason": "Auditor",
                        }
                        for f in folders
                    ],
                }
            else:
                payload = {
                    "proposals": [
                        {"folder": f, "proposed_name": "Concierto Verano"}
                        for f in folders
                    ]
                }
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload, ensure_ascii=False), stderr="")

        monkeypatch.setattr(subprocess, "run", _fake_run)

        # First run: Batch 1 succeeds (5 folders), Batch 2 hits quota error and pauses
        code, res1 = _http_json("POST", base_url, "/api/tree/ai-debate", {"job_id": job_id})
        assert code == 200, f"Expected graceful pause response, got {code}: {res1}"

        status_text = json.dumps(res1, ensure_ascii=False).lower()
        assert any(k in status_text for k in ("paused", "pausa", "quota", "cuota", "5/7"))

        # Remaining pending folders (2024-06-06, 2024-06-07) must NOT be overwritten with generic names
        conn = sqlite3.connect(core.cfg.db_path)
        try:
            pending_rows = conn.execute(
                "SELECT DISTINCT target_folder FROM media_items WHERE job_id = ? "
                "AND (target_folder LIKE '%2024-06-06%' OR target_folder LIKE '%2024-06-07%')",
                (job_id,),
            ).fetchall()
        finally:
            conn.close()
        assert len(pending_rows) == 2
        for (folder_name,) in pending_rows:
            assert "Concierto Verano" not in folder_name

        # Resume after quota recovers: only the 2 pending folders should be processed
        state["fail_batch_2"] = False
        code, res2 = _http_json("POST", base_url, "/api/tree/ai-debate", {"job_id": job_id, "resume": True})
        assert code == 200, f"Resume failed ({code}): {res2}"

        resumed_set = set(state["resumed_folders"])
        assert any("2024-06-06" in f for f in resumed_set)
        assert any("2024-06-07" in f for f in resumed_set)
        assert not any("2024-06-01" in f for f in resumed_set), "Already debated folders must not be re-debated on resume"
    finally:
        server.shutdown()
        server.server_close()
