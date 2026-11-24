"""Optional Google Gemini classifier with SQLite cache, token-bucket rate limit, and breaker."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from photo_organizer.db import init_db, open_connection

try:  # Optional google-genai SDK
    from google import genai  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]


class TokenBucket:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, capacity: int = 10, refill_per_sec: float = 0.5) -> None:
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_per_sec = refill_per_sec
        self.last_ts = time.monotonic()

    def acquire(self) -> bool:
        now = time.monotonic()
        elapsed = max(0.0, now - self.last_ts)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
        self.last_ts = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


def mask_api_key(api_key: str | None) -> str:
    """Mask an API key for safe display in settings responses."""
    if not api_key:
        return ""
    if len(api_key) <= 4:
        return "****"
    return f"{'*' * (len(api_key) - 4)}{api_key[-4:]}"


class GeminiClassifier:
    """Optional Gemini classifier that never halts the local pipeline on remote errors."""

    def __init__(
        self,
        db_path: Path,
        api_key: str = "",
        model: str = "gemini-2.5-flash",
        transport: Callable[[str], tuple[int, str]] | None = None,
        bucket_capacity: int = 10,
        refill_per_sec: float = 0.5,
    ) -> None:
        self.db_path = db_path
        self._api_key = api_key
        self.model = model or "gemini-2.5-flash"
        self._transport = transport
        self.bucket = TokenBucket(capacity=bucket_capacity, refill_per_sec=refill_per_sec)
        self.consecutive_failures = 0
        self.circuit_open = False
        self.remote_calls = 0
        init_db(self.db_path)

    def get_masked_settings(self) -> dict[str, Any]:
        return {
            "api_key": mask_api_key(self._api_key),
            "model": self.model,
            "configured": bool(self._api_key),
            "circuit_open": self.circuit_open,
        }

    def _get_cached(self, cache_key: str) -> str | None:
        conn = open_connection(self.db_path)
        try:
            row = conn.execute(
                "SELECT label FROM ai_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            return str(row["label"]) if row else None
        finally:
            conn.close()

    def _put_cached(self, cache_key: str, kind: str, label: str) -> None:
        conn = open_connection(self.db_path)
        try:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT OR REPLACE INTO ai_cache (cache_key, kind, label, model, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_key, kind, label, self.model, now),
            )
            conn.commit()
        finally:
            conn.close()

    def _default_transport(self, prompt_key: str) -> tuple[int, str]:
        if not self._api_key or genai is None:
            return (503, "")
        try:
            client = genai.Client(api_key=self._api_key)
            resp = client.models.generate_content(
                model=self.model,
                contents=f"Give a short 2-4 word Spanish event/category label for: {prompt_key}",
            )
            text = (getattr(resp, "text", "") or "").strip()
            return (200, text or "Evento")
        except Exception as exc:
            msg = str(exc)
            if "429" in msg:
                return (429, "")
            if "401" in msg or "403" in msg or "API_KEY" in msg.upper():
                return (401, "")
            return (500, "")

    def classify(self, cache_key: str, kind: str = "item") -> str:
        """Classify an item or cluster signature with cache, rate-limit, and circuit breaker."""
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if self.circuit_open or not self._api_key or not self.bucket.acquire():
            return "unclassified_local_fallback"

        self.remote_calls += 1
        transport = self._transport or self._default_transport
        status, label = transport(cache_key)
        if status in (401, 403, 429) or status >= 500:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 2:
                self.circuit_open = True
            return "unclassified_local_fallback"

        self.consecutive_failures = 0
        clean_label = label.strip() or "Evento"
        self._put_cached(cache_key, kind, clean_label)
        return clean_label

    def classify_batch(
        self,
        keys: list[str],
        *,
        confirmed: bool = False,
        kind: str = "item",
    ) -> dict[str, Any]:
        """Require explicit user confirmation before dispatching a remote AI batch."""
        if not confirmed:
            return {
                "requires_confirmation": True,
                "item_count": len(keys),
                "model": self.model,
                "results": {},
            }
        results = {k: self.classify(k, kind=kind) for k in keys}
        return {
            "requires_confirmation": False,
            "item_count": len(keys),
            "model": self.model,
            "circuit_open": self.circuit_open,
            "results": results,
        }
