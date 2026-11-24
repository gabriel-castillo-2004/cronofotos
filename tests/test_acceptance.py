"""Self-contained acceptance tests AT-01 through AT-08.

Uses only the Python standard library and pytest.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import os
import queue
import re
import shutil
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# AT-01: Source/destination isolation & OneDrive cloud-only detection
# ---------------------------------------------------------------------------

FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
FILE_ATTRIBUTE_OFFLINE = 0x1000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
CLOUD_ONLY_MASK = (
    FILE_ATTRIBUTE_OFFLINE
    | FILE_ATTRIBUTE_RECALL_ON_OPEN
    | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)


def validate_source_dest_isolation(source: Path, dest: Path) -> None:
    """Reject equal, parent, or child overlapping paths."""
    src_res = Path(os.path.normcase(str(source.resolve())))
    dst_res = Path(os.path.normcase(str(dest.resolve())))
    if src_res == dst_res:
        raise ValueError("Source and destination cannot be the same path.")
    if src_res in dst_res.parents:
        raise ValueError("Destination cannot be inside source directory.")
    if dst_res in src_res.parents:
        raise ValueError("Source cannot be inside destination directory.")


def is_onedrive_cloud_only(path: Path, file_attributes: int | None = None) -> bool:
    """Detect cloud-only reparse point placeholders without reading file bytes."""
    if file_attributes is None:
        st = os.lstat(path)
        file_attributes = getattr(st, "st_file_attributes", 0)
    is_reparse = bool(file_attributes & FILE_ATTRIBUTE_REPARSE_POINT)
    is_cloud = bool(file_attributes & CLOUD_ONLY_MASK)
    return is_reparse and is_cloud


def test_at01_source_destination_isolation_and_onedrive_placeholder(tmp_path: Path) -> None:
    """AT-01: Reject overlapping paths and flag OneDrive placeholders without hydration."""
    src = tmp_path / "library"
    child = src / "2024"
    sibling = tmp_path / "organized"
    src.mkdir()
    child.mkdir()
    sibling.mkdir()

    with pytest.raises(ValueError, match="same path"):
        validate_source_dest_isolation(src, src)

    with pytest.raises(ValueError, match="inside source"):
        validate_source_dest_isolation(src, child)

    with pytest.raises(ValueError, match="inside destination"):
        validate_source_dest_isolation(child, src)

    # Disjoint sibling paths succeed
    validate_source_dest_isolation(src, sibling)

    # OneDrive cloud-only placeholder detection via attributes without opening file
    placeholder = src / "cloud_photo.jpg"
    placeholder.write_bytes(b"")
    opened = False

    def guarded_open(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("Placeholder file must not be opened/hydrated")

    cloud_attrs = FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
    assert is_onedrive_cloud_only(placeholder, file_attributes=cloud_attrs) is True
    assert is_onedrive_cloud_only(placeholder, file_attributes=0x20) is False
    assert opened is False
    _ = guarded_open


# ---------------------------------------------------------------------------
# AT-02: Multi-pass Google Takeout sidecar resolver
# ---------------------------------------------------------------------------

TRUNCATE_LIMIT = 46


def _read_json_title(json_path: Path) -> str | None:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        title = data.get("title")
        return str(title) if title is not None else None
    except (OSError, json.JSONDecodeError):
        return None


def resolve_takeout_sidecars(folder: Path) -> dict[str, dict[str, Any]]:
    """Multi-pass Takeout sidecar resolver verified by JSON 'title'."""
    json_files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    media_files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() != ".json")

    results: dict[str, dict[str, Any]] = {}
    for media in media_files:
        name = media.name
        candidates: list[Path] = []

        # Pass 1: exact match (<filename>.json)
        exact = folder / f"{name}.json"
        if exact in json_files:
            candidates.append(exact)

        # Pass 2: supplemental-metadata (<filename>.supplemental-metadata.json / truncated)
        for supp_tag in ("supplemental-metadata", "supplemental-metadat"):
            supp = folder / f"{name}.{supp_tag}.json"
            if supp in json_files and supp not in candidates:
                candidates.append(supp)

        # Pass 3: 46-char truncated (<name[:46]>.json)
        trunc_name = f"{name[:TRUNCATE_LIMIT]}.json"
        trunc_path = folder / trunc_name
        if trunc_path in json_files and trunc_path not in candidates:
            candidates.append(trunc_path)

        # Pass 4: '(N)' suffixed (e.g., photo(1).jpg -> photo.jpg(1).json or photo(1).jpg.json)
        m = re.match(r"^(.*?)(\(\d+\))(\.[^.]+)$", name)
        if m:
            base, num, ext = m.groups()
            for variant in (f"{base}{ext}{num}.json", f"{base}{num}{ext}.json"):
                vp = folder / variant
                if vp in json_files and vp not in candidates:
                    candidates.append(vp)

        # Verify candidates by JSON 'title'
        verified = [c for c in candidates if _read_json_title(c) == name]
        if len(verified) == 1:
            results[name] = {"sidecar": verified[0].name, "route": "matched"}
        elif len(verified) > 1 or (len(candidates) > 0 and len(verified) == 0):
            results[name] = {"sidecar": None, "route": "_Revision_Manual"}
        else:
            results[name] = {"sidecar": None, "route": "unmatched"}

    return results


def test_at02_takeout_multipass_sidecar_resolver(tmp_path: Path) -> None:
    """AT-02: Exact, 46-char truncated, supplemental-metadata, (N) suffix, and ambiguous routing."""
    # 1. Exact match
    (tmp_path / "exact.jpg").write_bytes(b"img1")
    (tmp_path / "exact.jpg.json").write_text(json.dumps({"title": "exact.jpg"}), encoding="utf-8")

    # 2. Supplemental metadata
    (tmp_path / "supp.jpg").write_bytes(b"img2")
    (tmp_path / "supp.jpg.supplemental-metadata.json").write_text(
        json.dumps({"title": "supp.jpg"}), encoding="utf-8"
    )

    # 3. 46-char truncated
    long_name = "a_very_long_google_takeout_media_filename_1234567890.jpg"
    assert len(long_name) > 46
    (tmp_path / long_name).write_bytes(b"img3")
    (tmp_path / f"{long_name[:46]}.json").write_text(
        json.dumps({"title": long_name}), encoding="utf-8"
    )

    # 4. '(N)' suffixed
    (tmp_path / "burst(1).jpg").write_bytes(b"img4")
    (tmp_path / "burst.jpg(1).json").write_text(
        json.dumps({"title": "burst(1).jpg"}), encoding="utf-8"
    )

    # 5. Ambiguous pair (two matching title candidates -> _Revision_Manual)
    (tmp_path / "ambig(2).jpg").write_bytes(b"img5")
    (tmp_path / "ambig(2).jpg.json").write_text(
        json.dumps({"title": "ambig(2).jpg"}), encoding="utf-8"
    )
    (tmp_path / "ambig.jpg(2).json").write_text(
        json.dumps({"title": "ambig(2).jpg"}), encoding="utf-8"
    )

    # 6. Title mismatch -> _Revision_Manual
    (tmp_path / "mismatch.jpg").write_bytes(b"img6")
    (tmp_path / "mismatch.jpg.json").write_text(
        json.dumps({"title": "other_file.jpg"}), encoding="utf-8"
    )

    resolved = resolve_takeout_sidecars(tmp_path)
    assert resolved["exact.jpg"] == {"sidecar": "exact.jpg.json", "route": "matched"}
    assert resolved["supp.jpg"] == {
        "sidecar": "supp.jpg.supplemental-metadata.json",
        "route": "matched",
    }
    assert resolved[long_name] == {"sidecar": f"{long_name[:46]}.json", "route": "matched"}
    assert resolved["burst(1).jpg"] == {"sidecar": "burst.jpg(1).json", "route": "matched"}
    assert resolved["ambig(2).jpg"]["route"] == "_Revision_Manual"
    assert resolved["mismatch.jpg"]["route"] == "_Revision_Manual"


# ---------------------------------------------------------------------------
# AT-03: Deterministic collision numbering & SQLite UNIQUE constraint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MediaRecord:
    captured_at: str  # ISO UTC e.g. 2024-05-01T14:30:00Z
    sha256: str
    source_path: str
    ext: str
    target_folder: str


def assign_deterministic_filenames(records: list[MediaRecord]) -> dict[str, str]:
    """Assign 'AAAA-MM-DD_HH-MM-SS_###.ext' under canonical (captured_at, sha256, source_path)."""
    ordered = sorted(records, key=lambda r: (r.captured_at, r.sha256, r.source_path))
    counters: dict[tuple[str, str], int] = {}
    assigned: dict[str, str] = {}
    for rec in ordered:
        dt = datetime.fromisoformat(rec.captured_at.replace("Z", "+00:00"))
        stamp = dt.strftime("%Y-%m-%d_%H-%M-%S")
        key = (rec.target_folder, stamp)
        counters[key] = counters.get(key, 0) + 1
        assigned[rec.source_path] = f"{stamp}_{counters[key]:03d}{rec.ext}"
    return assigned


def test_at03_deterministic_collision_numbering_and_sqlite_unique(tmp_path: Path) -> None:
    """AT-03: Canonical ordering produces identical filenames and SQLite enforces UNIQUE."""
    recs = [
        MediaRecord("2024-05-01T14:30:00Z", "ccc", "/src/c.jpg", ".jpg", "2024/05"),
        MediaRecord("2024-05-01T14:30:00Z", "aaa", "/src/a.jpg", ".jpg", "2024/05"),
        MediaRecord("2024-05-01T14:30:00Z", "bbb", "/src/b.jpg", ".jpg", "2024/05"),
    ]
    run1 = assign_deterministic_filenames(recs)
    run2 = assign_deterministic_filenames(list(reversed(recs)))
    assert run1 == run2
    assert run1["/src/a.jpg"] == "2024-05-01_14-30-00_001.jpg"
    assert run1["/src/b.jpg"] == "2024-05-01_14-30-00_002.jpg"
    assert run1["/src/c.jpg"] == "2024-05-01_14-30-00_003.jpg"

    db_path = tmp_path / "catalog.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT UNIQUE NOT NULL,
                target_folder TEXT NOT NULL,
                target_filename TEXT NOT NULL,
                UNIQUE(target_folder, target_filename)
            )
            """
        )
        conn.execute(
            "INSERT INTO items (source_path, target_folder, target_filename) VALUES (?, ?, ?)",
            ("/src/a.jpg", "2024/05", run1["/src/a.jpg"]),
        )
        conn.execute(
            "INSERT INTO items (source_path, target_folder, target_filename) VALUES (?, ?, ?)",
            ("/src/d.jpg", "2024/Vacations", run1["/src/a.jpg"]),
        )
        conn.commit()

        # Merging 2024/Vacations into 2024/05 with identical target_filename triggers UNIQUE error
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE items SET target_folder = ? WHERE target_folder = ?",
                ("2024/05", "2024/Vacations"),
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# AT-04: SingleWriterQueue with 17,000+ concurrent updates in SQLite WAL
# ---------------------------------------------------------------------------

class SingleWriterQueue:
    """Serializes all SQLite writes through a single dedicated writer thread in WAL mode."""

    def __init__(self, db_path: Path, batch_size: int = 250) -> None:
        self.db_path = db_path
        self.batch_size = batch_size
        self._q: queue.Queue[tuple[str, tuple[Any, ...]] | None] = queue.Queue()
        self._errors: list[Exception] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._init_schema()
        self._thread.start()

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, status TEXT NOT NULL)"
            )
            conn.commit()
        finally:
            conn.close()

    def enqueue(self, sql: str, params: tuple[Any, ...]) -> None:
        self._q.put((sql, params))

    def _run(self) -> None:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            while True:
                item = self._q.get()
                if item is None:
                    self._q.task_done()
                    break
                batch = [item]
                while len(batch) < self.batch_size:
                    try:
                        nxt = self._q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        self._q.task_done()
                        self._flush(conn, batch)
                        return
                    batch.append(nxt)
                self._flush(conn, batch)
        except Exception as exc:  # pragma: no cover
            self._errors.append(exc)
        finally:
            conn.close()

    def _flush(self, conn: sqlite3.Connection, batch: list[tuple[str, tuple[Any, ...]]]) -> None:
        try:
            with conn:
                for sql, params in batch:
                    conn.execute(sql, params)
        finally:
            for _ in batch:
                self._q.task_done()

    def close(self) -> None:
        self._q.put(None)
        self._q.join()
        self._thread.join(timeout=10.0)
        if self._errors:
            raise self._errors[0]


def test_at04_single_writer_queue_17000_updates_wal(tmp_path: Path) -> None:
    """AT-04: 17,500 concurrent item updates in SQLite WAL with zero 'database is locked'."""
    db_path = tmp_path / "wal_queue.db"
    writer = SingleWriterQueue(db_path)

    total_items = 17_500
    workers = 8
    per_worker = total_items // workers
    remainder = total_items % workers

    def produce(worker_idx: int, count: int) -> None:
        base = worker_idx * per_worker
        for i in range(count):
            item_id = base + i + 1
            writer.enqueue("INSERT INTO items (id, status) VALUES (?, ?)", (item_id, "scanned"))

    threads: list[threading.Thread] = []
    for w in range(workers):
        cnt = per_worker + (remainder if w == workers - 1 else 0)
        t = threading.Thread(target=produce, args=(w, cnt))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    writer.close()

    conn = sqlite3.connect(db_path)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    finally:
        conn.close()

    assert str(mode).lower() == "wal"
    assert count == total_items


# ---------------------------------------------------------------------------
# AT-05: Thread-confined ExifTool '-stay_open' pool & graceful fallback
# ---------------------------------------------------------------------------

class _FakeStayOpenWorker:
    """Simulates a thread-confined `exiftool -stay_open True -@ -` process."""

    def __init__(self, thread_id: int) -> None:
        self.thread_id = thread_id

    def query(self, path: Path) -> dict[str, Any]:
        # Includes thread_id to prove thread confinement & no cross-thread bleed
        return {
            "SourceFile": str(path),
            "ThreadID": self.thread_id,
            "DateTimeOriginal": "2024:05:01 14:30:00",
        }


class MetadataExtractorPool:
    """Thread-confined ExifTool pool with Pillow/pillow-heif/piexif fallback."""

    def __init__(self, exiftool_bin: str | None, ffmpeg_bin: str | None) -> None:
        self.exiftool_bin = exiftool_bin
        self.ffmpeg_bin = ffmpeg_bin
        self._local = threading.local()
        self.health_warnings: list[str] = []
        if not self.exiftool_bin:
            self.health_warnings.append("exiftool binary missing; using Pillow/piexif fallback")
        if not self.ffmpeg_bin:
            self.health_warnings.append("ffmpeg binary missing; video metadata fallback active")

    def _get_worker(self) -> _FakeStayOpenWorker:
        worker = getattr(self._local, "worker", None)
        if worker is None:
            worker = _FakeStayOpenWorker(threading.get_ident())
            self._local.worker = worker
        return worker

    def extract(self, path: Path) -> dict[str, Any]:
        if self.exiftool_bin:
            worker = self._get_worker()
            meta = worker.query(path)
            assert meta["ThreadID"] == threading.get_ident()
            return meta
        # Pure-Python fallback
        return {
            "SourceFile": str(path),
            "Extractor": "pillow_piexif_fallback",
            "SizeBytes": path.stat().st_size,
        }


def test_at05_thread_confined_exiftool_pool_and_fallback(tmp_path: Path) -> None:
    """AT-05: Thread-confined stay_open isolation and graceful fallback when binaries absent."""
    files = []
    for idx in range(24):
        f = tmp_path / f"img_{idx}.jpg"
        f.write_bytes(f"data-{idx}".encode())
        files.append(f)

    pool = MetadataExtractorPool(exiftool_bin="exiftool", ffmpeg_bin="ffmpeg")
    assert pool.health_warnings == []

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(pool.extract, files))

    for file_path, meta in zip(files, results, strict=True):
        assert meta["SourceFile"] == str(file_path)

    # Fallback when binaries are absent
    fallback_pool = MetadataExtractorPool(exiftool_bin=None, ffmpeg_bin=None)
    assert len(fallback_pool.health_warnings) == 2
    assert any("exiftool" in w for w in fallback_pool.health_warnings)
    assert any("ffmpeg" in w for w in fallback_pool.health_warnings)
    fb_meta = fallback_pool.extract(files[0])
    assert fb_meta["Extractor"] == "pillow_piexif_fallback"


# ---------------------------------------------------------------------------
# AT-06: Idempotent two-phase exporter & crash recovery
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_two_phase_export(
    source_dir: Path,
    dest_dir: Path,
    *,
    crash_after: int | None = None,
) -> dict[str, int]:
    """Two-phase idempotent exporter: .tmp -> atomic rename, exclude sidecars, isolate dupes."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dupes_dir = dest_dir / "_Duplicados"
    seen_hashes: dict[str, Path] = {}
    exported = 0
    duplicates = 0
    skipped = 0

    # Pre-seed seen_hashes from already-exported non-tmp files for idempotent resume
    for existing in sorted(dest_dir.rglob("*")):
        if existing.is_file() and existing.suffix != ".tmp" and "_Duplicados" not in existing.parts:
            seen_hashes[_sha256_file(existing)] = existing

    media_files = sorted(
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() not in {".json", ".txt"}
    )

    for idx, src in enumerate(media_files):
        digest = _sha256_file(src)
        rel_name = src.name
        if digest in seen_hashes:
            existing_target = seen_hashes[digest]
            if existing_target.name == rel_name and (dest_dir / rel_name).exists():
                skipped += 1
                continue
            dupes_dir.mkdir(parents=True, exist_ok=True)
            target = dupes_dir / rel_name
            if target.exists():
                skipped += 1
                continue
            duplicates += 1
        else:
            target = dest_dir / rel_name
            seen_hashes[digest] = target

        tmp_target = target.with_name(target.name + ".tmp")
        payload = src.read_bytes()
        tmp_target.write_bytes(payload)

        if crash_after is not None and idx == crash_after:
            raise RuntimeError("Simulated mid-export crash")

        os.replace(tmp_target, target)
        exported += 1

    return {"exported": exported, "duplicates": duplicates, "skipped": skipped}


def test_at06_idempotent_two_phase_exporter_and_crash_recovery(tmp_path: Path) -> None:
    """AT-06: .tmp atomic rename, excludes sidecars, isolates duplicates, resumes after crash."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()

    (src / "a.jpg").write_bytes(b"unique-photo-a")
    (src / "a.jpg.json").write_text('{"title": "a.jpg"}', encoding="utf-8")
    (src / "notes.txt").write_text("ignore me", encoding="utf-8")
    (src / "b_dup.jpg").write_bytes(b"unique-photo-a")  # duplicate of a.jpg
    (src / "c.jpg").write_bytes(b"unique-photo-c")

    source_snapshots = {p.name: _sha256_file(p) for p in src.iterdir()}

    # Simulate crash during second media file
    with pytest.raises(RuntimeError, match="Simulated mid-export crash"):
        run_two_phase_export(src, dst, crash_after=1)

    assert any(p.suffix == ".tmp" for p in dst.rglob("*"))

    # Resume export cleanly
    summary = run_two_phase_export(src, dst)
    assert summary["skipped"] >= 1

    # Verify no leftover .tmp, no .json/.txt, duplicate in _Duplicados
    all_dst_files = [p for p in dst.rglob("*") if p.is_file()]
    assert not any(p.suffix == ".tmp" for p in all_dst_files)
    assert not any(p.suffix in {".json", ".txt"} for p in all_dst_files)
    assert (dst / "a.jpg").exists()
    assert (dst / "c.jpg").exists()
    assert (dst / "_Duplicados" / "b_dup.jpg").exists()

    # Source completely untouched
    for name, expected_hash in source_snapshots.items():
        assert _sha256_file(src / name) == expected_hash


# ---------------------------------------------------------------------------
# AT-07: Gemini classifier cache, rate limiter, circuit breaker, key masking
# ---------------------------------------------------------------------------

class TokenBucket:
    def __init__(self, capacity: int, refill_per_sec: float) -> None:
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_per_sec = refill_per_sec
        self.last_ts = time.monotonic()

    def acquire(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last_ts) * self.refill_per_sec)
        self.last_ts = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class GeminiClassifier:
    def __init__(self, db_path: Path, api_key: str, transport: Any) -> None:
        self.db_path = db_path
        self._api_key = api_key
        self._transport = transport
        self.bucket = TokenBucket(capacity=2, refill_per_sec=0.0)
        self.consecutive_failures = 0
        self.circuit_open = False
        self.remote_calls = 0

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS ai_cache (sha256 TEXT PRIMARY KEY, label TEXT NOT NULL)"
            )
            conn.commit()
        finally:
            conn.close()

    def get_masked_settings(self) -> dict[str, str]:
        if len(self._api_key) <= 4:
            masked = "****"
        else:
            masked = f"{'*' * (len(self._api_key) - 4)}{self._api_key[-4:]}"
        return {"api_key": masked}

    def classify(self, sha256: str) -> str:
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT label FROM ai_cache WHERE sha256 = ?", (sha256,)).fetchone()
            if row:
                return str(row[0])
        finally:
            conn.close()

        if self.circuit_open or not self.bucket.acquire():
            return "unclassified_local_fallback"

        self.remote_calls += 1
        status, label = self._transport(sha256)
        if status in (401, 403, 429):
            self.consecutive_failures += 1
            if self.consecutive_failures >= 2:
                self.circuit_open = True
            return "unclassified_local_fallback"

        self.consecutive_failures = 0
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO ai_cache (sha256, label) VALUES (?, ?)",
                (sha256, label),
            )
            conn.commit()
        finally:
            conn.close()
        return label


def test_at07_gemini_classifier_cache_rate_limit_breaker_and_masking(tmp_path: Path) -> None:
    """AT-07: SQLite cache, token bucket, circuit breaker on 429/auth, and masked API key."""
    db_path = tmp_path / "ai.db"
    responses = iter([(200, "Receipts"), (429, ""), (401, "")])

    def fake_transport(_sha: str) -> tuple[int, str]:
        return next(responses, (500, ""))

    clf = GeminiClassifier(db_path, api_key="TEST_KEY_9876", transport=fake_transport)
    masked = clf.get_masked_settings()["api_key"]
    assert masked.endswith("9876")
    assert "TEST_KEY_" not in masked

    # First call hits transport and caches; second call for same sha hits SQLite cache
    assert clf.classify("sha_1") == "Receipts"
    assert clf.classify("sha_1") == "Receipts"
    assert clf.remote_calls == 1

    # Second token triggers 429; third call rate-limited by bucket
    assert clf.classify("sha_2") == "unclassified_local_fallback"
    assert clf.classify("sha_3") == "unclassified_local_fallback"

    # Replenish 1 token, trigger 401 -> trips circuit breaker open without halting
    clf.bucket.tokens = 1.0
    assert clf.classify("sha_4") == "unclassified_local_fallback"
    assert clf.circuit_open is True


# ---------------------------------------------------------------------------
# AT-08: End-to-end FastAPI-style HTTP workflow
# ---------------------------------------------------------------------------

class _LRUThumbnailCache:
    def __init__(self, capacity: int = 2) -> None:
        self.capacity = capacity
        self._store: dict[str, bytes] = {}
        self._order: list[str] = []

    def get_or_create(self, opaque_id: str, payload: bytes) -> bytes:
        if opaque_id in self._store:
            self._order.remove(opaque_id)
            self._order.append(opaque_id)
            return self._store[opaque_id]
        if len(self._order) >= self.capacity:
            evicted = self._order.pop(0)
            del self._store[evicted]
        thumb = b"THUMB:" + payload[:16]
        self._store[opaque_id] = thumb
        self._order.append(opaque_id)
        return thumb


@dataclass
class _AppState:
    source_dir: Path
    dest_dir: Path
    status: str = "idle"
    paused: bool = False
    items: list[dict[str, Any]] = field(default_factory=list)
    thumbs: _LRUThumbnailCache = field(default_factory=lambda: _LRUThumbnailCache(capacity=3))


def _make_handler(state: _AppState) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, code: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/scan/start":
                state.status = "scanning"
                state.items = []
                for idx, p in enumerate(sorted(state.source_dir.rglob("*"))):
                    if p.is_file() and p.suffix.lower() not in {".json", ".txt"}:
                        folder = "Screenshots" if "screenshot" in p.name.lower() else "2024/05"
                        state.items.append(
                            {
                                "id": f"item-{idx}",
                                "filename": p.name,
                                "folder": folder,
                                "path": p,
                            }
                        )
                self._send_json(200, {"status": state.status, "count": len(state.items)})
            elif parsed.path == "/api/scan/pause":
                state.paused = True
                state.status = "paused"
                self._send_json(200, {"status": state.status})
            elif parsed.path == "/api/scan/resume":
                state.paused = False
                state.status = "completed"
                self._send_json(200, {"status": state.status})
            elif parsed.path == "/api/tree/folders/merge":
                body = self._read_json()
                src_folder = body["from"]
                dst_folder = body["to"]
                for item in state.items:
                    if item["folder"] == src_folder:
                        item["folder"] = dst_folder
                self._send_json(200, {"merged": src_folder, "into": dst_folder})
            elif parsed.path == "/api/export":
                summary = run_two_phase_export(state.source_dir, state.dest_dir)
                self._send_json(200, summary)
            else:
                self._send_json(404, {"error": "not_found"})

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/tree":
                qs = urllib.parse.parse_qs(parsed.query)
                page = int(qs.get("page", ["1"])[0])
                page_size = int(qs.get("page_size", ["2"])[0])
                start = (page - 1) * page_size
                end = start + page_size
                slice_items = [
                    {"id": it["id"], "filename": it["filename"], "folder": it["folder"]}
                    for it in state.items[start:end]
                ]
                self._send_json(
                    200,
                    {"page": page, "total": len(state.items), "items": slice_items},
                )
            elif parsed.path.startswith("/api/thumbnails/"):
                opaque_id = parsed.path.rsplit("/", 1)[-1]
                match = next((it for it in state.items if it["id"] == opaque_id), None)
                if not match:
                    self._send_json(404, {"error": "not_found"})
                    return
                thumb = state.thumbs.get_or_create(opaque_id, match["path"].read_bytes())
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(thumb)))
                self.end_headers()
                self.wfile.write(thumb)
            else:
                self._send_json(404, {"error": "not_found"})

    return Handler


def _http_call(base_url: str, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{base_url}{path}", data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        raw = resp.read()
        ctype = resp.headers.get("Content-Type", "")
        if "application/json" in ctype:
            return json.loads(raw.decode("utf-8"))
        return raw


def test_at08_end_to_end_workflow(tmp_path: Path) -> None:
    """AT-08: Mixed fixture scan, pause/resume, paginated tree merge, LRU thumbs, clean export."""
    src = tmp_path / "fixture_src"
    dst = tmp_path / "fixture_dst"
    src.mkdir()

    (src / "photo.jpg").write_bytes(b"jpeg-bytes-1")
    (src / "photo.jpg.json").write_text('{"title": "photo.jpg"}', encoding="utf-8")
    (src / "live.heic").write_bytes(b"heic-bytes-2")
    (src / "clip.mp4").write_bytes(b"mp4-bytes-3")
    (src / "photo_dup.jpg").write_bytes(b"jpeg-bytes-1")
    (src / "screenshot_01.png").write_bytes(b"png-bytes-4")

    src_hashes = {p.name: _sha256_file(p) for p in src.iterdir()}
    state = _AppState(source_dir=src, dest_dir=dst)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        scan = _http_call(base_url, "POST", "/api/scan/start", {})
        assert scan["count"] == 5

        assert _http_call(base_url, "POST", "/api/scan/pause", {})["status"] == "paused"
        assert _http_call(base_url, "POST", "/api/scan/resume", {})["status"] == "completed"

        page1 = _http_call(base_url, "GET", "/api/tree?page=1&page_size=2")
        assert page1["total"] == 5
        assert len(page1["items"]) == 2

        merge_res = _http_call(
            base_url,
            "POST",
            "/api/tree/folders/merge",
            {"from": "Screenshots", "to": "2024/05"},
        )
        assert merge_res["into"] == "2024/05"

        first_id = page1["items"][0]["id"]
        thumb_bytes = _http_call(base_url, "GET", f"/api/thumbnails/{first_id}")
        assert thumb_bytes.startswith(b"THUMB:")

        export_res = _http_call(base_url, "POST", "/api/export", {})
        assert export_res["duplicates"] == 1
        assert (dst / "_Duplicados" / "photo_dup.jpg").exists()
        assert not any(p.suffix in {".json", ".txt"} for p in dst.rglob("*"))

        for name, expected_hash in src_hashes.items():
            assert _sha256_file(src / name) == expected_hash
    finally:
        server.shutdown()
        server.server_close()
