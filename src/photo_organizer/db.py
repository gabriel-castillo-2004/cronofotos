"""SQLite WAL persistence layer with a dedicated SingleWriterQueue."""

from __future__ import annotations

import asyncio
import queue
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

EXPORT_STATES = (
    "PENDING_EXPORT",
    "COPIED",
    "METADATA_EMBEDDED",
    "VERIFIED",
    "FAILED",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source_dir TEXT NOT NULL,
    dest_dir TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'idle',
    stage TEXT NOT NULL DEFAULT 'init',
    total_files INTEGER NOT NULL DEFAULT 0,
    processed_files INTEGER NOT NULL DEFAULT 0,
    duplicates_count INTEGER NOT NULL DEFAULT 0,
    manual_review_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS media_items (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    ext TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT,
    is_duplicate INTEGER NOT NULL DEFAULT 0,
    is_placeholder INTEGER NOT NULL DEFAULT 0,
    captured_at TEXT,
    latitude REAL,
    longitude REAL,
    location_label TEXT,
    sidecar_path TEXT,
    metadata_json TEXT,
    target_folder TEXT NOT NULL DEFAULT '_Revision_Manual',
    target_filename TEXT NOT NULL,
    export_state TEXT NOT NULL DEFAULT 'PENDING_EXPORT'
        CHECK (export_state IN ('PENDING_EXPORT', 'COPIED', 'METADATA_EMBEDDED', 'VERIFIED', 'FAILED')),
    route_reason TEXT,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    UNIQUE(job_id, target_folder, target_filename)
);

CREATE INDEX IF NOT EXISTS idx_media_job_folder
    ON media_items(job_id, target_folder, target_filename);

CREATE INDEX IF NOT EXISTS idx_media_job_sha
    ON media_items(job_id, sha256);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_cache (
    cache_key TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'item',
    label TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT 'gemini-2.5-flash',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS custom_folders (
    job_id TEXT NOT NULL,
    folder TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, folder),
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);
"""


def open_connection(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection configured with WAL mode and busy timeout."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path) -> None:
    """Initialize all database tables and WAL pragmas."""
    conn = open_connection(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


class SingleWriterQueue:
    """Serializes all SQLite writes through a single dedicated writer thread in WAL mode."""

    def __init__(self, db_path: Path, batch_size: int = 80) -> None:
        self.db_path = db_path
        self.batch_size = max(1, batch_size)
        self._q: queue.Queue[tuple[str, tuple[Any, ...]] | None] = queue.Queue()
        self._errors: list[Exception] = []
        self._closed = False
        init_db(self.db_path)
        self._thread = threading.Thread(target=self._run, name="sqlite-single-writer", daemon=True)
        self._thread.start()

    def enqueue(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        """Enqueue a write statement for batched execution."""
        if self._closed:
            raise RuntimeError("SingleWriterQueue is closed")
        self._q.put((sql, params))

    def flush(self) -> None:
        """Block until all currently queued writes are committed."""
        self._q.join()
        if self._errors:
            raise self._errors[0]

    def _run(self) -> None:
        conn = open_connection(self.db_path)
        try:
            while True:
                item = self._q.get()
                if item is None:
                    self._q.task_done()
                    break
                batch: list[tuple[str, tuple[Any, ...]]] = [item]
                stop_after = False
                while len(batch) < self.batch_size:
                    try:
                        nxt = self._q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        self._q.task_done()
                        stop_after = True
                        break
                    batch.append(nxt)
                self._flush_batch(conn, batch)
                if stop_after:
                    break
        except Exception as exc:  # pragma: no cover
            self._errors.append(exc)
        finally:
            conn.close()

    def _flush_batch(
        self,
        conn: sqlite3.Connection,
        batch: list[tuple[str, tuple[Any, ...]]],
    ) -> None:
        try:
            with conn:
                for sql, params in batch:
                    conn.execute(sql, params)
        except Exception as exc:
            self._errors.append(exc)
        finally:
            for _ in batch:
                self._q.task_done()

    def close(self) -> None:
        """Flush remaining items and shut down the writer thread."""
        if self._closed:
            return
        self._closed = True
        self._q.put(None)
        self._q.join()
        self._thread.join(timeout=15.0)
        if self._errors:
            raise self._errors[0]


def _format_stamp(captured_at: str | None) -> str:
    if not captured_at:
        return "0000-00-00_00-00-00"
    try:
        dt = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return "0000-00-00_00-00-00"


def recompute_filenames(conn: sqlite3.Connection, job_id: str, folder: str) -> int:
    """Deterministically recompute 'AAAA-MM-DD_HH-MM-SS_###.ext' within a folder."""
    import json as _json
    from photo_organizer.config import VIDEO_EXTENSIONS

    rows = conn.execute(
        """
        SELECT id, captured_at, sha256, source_path, ext, metadata_json
        FROM media_items
        WHERE job_id = ? AND target_folder = ?
        ORDER BY COALESCE(captured_at, '') ASC, COALESCE(sha256, '') ASC, source_path ASC
        """,
        (job_id, folder),
    ).fetchall()

    by_stamp: dict[str, list[Any]] = {}
    for row in rows:
        stamp = _format_stamp(row["captured_at"])
        by_stamp.setdefault(stamp, []).append(row)

    updates: list[tuple[str, str]] = []
    for stamp, group in by_stamp.items():
        images = [r for r in group if str(r["ext"]).lower() not in VIDEO_EXTENSIONS]
        short_videos = []
        other_videos = []
        for r in group:
            if str(r["ext"]).lower() not in VIDEO_EXTENSIONS:
                continue
            dur = None
            if r["metadata_json"]:
                try:
                    dur = _json.loads(r["metadata_json"]).get("duration_sec")
                except Exception:
                    dur = None
            if isinstance(dur, (int, float)) and dur <= 3.0:
                short_videos.append(r)
            else:
                other_videos.append(r)

        counter = 0
        paired: set[str] = set()
        sv_queue = list(short_videos)
        for img in images:
            counter += 1
            updates.append((f"{stamp}_{counter:03d}{img['ext']}", img["id"]))
            if sv_queue:
                vid = sv_queue.pop(0)
                paired.add(vid["id"])
                updates.append((f"{stamp}_{counter:03d}{vid['ext']}", vid["id"]))
        for vid in sv_queue + other_videos:
            if vid["id"] in paired:
                continue
            counter += 1
            updates.append((f"{stamp}_{counter:03d}{vid['ext']}", vid["id"]))

    # Phase 1: move to temporary unique names to avoid transient UNIQUE collisions
    for idx, (_new_name, item_id) in enumerate(updates):
        conn.execute(
            "UPDATE media_items SET target_filename = ? WHERE id = ?",
            (f"tmp-{idx}-{item_id}", item_id),
        )
    # Phase 2: assign final deterministic names
    for new_name, item_id in updates:
        conn.execute(
            "UPDATE media_items SET target_filename = ? WHERE id = ?",
            (new_name, item_id),
        )
    return len(updates)


async def async_fetch_all(
    db_path: Path,
    sql: str,
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    """Run a read-only SQLite query off the event loop."""

    def _query() -> list[dict[str, Any]]:
        conn = open_connection(db_path)
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    return await asyncio.to_thread(_query)


async def async_fetch_one(
    db_path: Path,
    sql: str,
    params: tuple[Any, ...] = (),
) -> dict[str, Any] | None:
    """Run a single-row read-only SQLite query off the event loop."""
    rows = await async_fetch_all(db_path, sql, params)
    return rows[0] if rows else None
