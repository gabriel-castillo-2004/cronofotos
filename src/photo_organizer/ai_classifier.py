"""Optional Antigravity CLI (`agy`) classifier with SQLite cache and heuristic fallback."""

from __future__ import annotations

import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from photo_organizer.db import init_db, open_connection

DEFAULT_AGY_MODEL = "gemini-4-hydrogen-medium"


def heuristic_classify(prompt_key: str) -> str:
    """Derive a clean Spanish category/event label from folder and filename metadata."""
    lower = (prompt_key or "").lower()
    rules = (
        (("screenshot", "captura", "pantallazo", "screen_shot"), "Capturas"),
        (("whatsapp", "-wa0", "wa_"), "WhatsApp"),
        (("telegram",), "Telegram"),
        (("instagram",), "Instagram"),
        (("download", "descarga"), "Descargas"),
        (("document", "scan", "recibo", "factura", "pdf"), "Documentos"),
        (("camera", "dcim"), "Camara"),
        (("viaje", "trip", "travel", "vacacion", "playa", "beach"), "Viaje"),
    )
    for keywords, label in rules:
        if any(kw in lower for kw in keywords):
            return label

    parts = [p for p in re.split(r"[/\\]+", prompt_key or "") if p]
    for part in parts:
        stem = Path(part).stem
        tokens = [
            t
            for t in re.split(r"[\s_\-]+", stem)
            if t and not t.isdigit() and len(t) > 2
        ]
        if tokens:
            return "_".join(t.capitalize() for t in tokens[:3])
    return "Evento_Local"


def classify_with_agy(
    prompt_key: str,
    *,
    cli_command: str = "agy",
    model: str = DEFAULT_AGY_MODEL,
    timeout: float = 15.0,
) -> str:
    """Invoke local Antigravity CLI (`agy`) via subprocess with heuristic fallback."""
    cmd_bin = (cli_command or "agy").strip() or "agy"
    selected_model = (model or DEFAULT_AGY_MODEL).strip() or DEFAULT_AGY_MODEL
    prompt = (
        "Responde únicamente con una etiqueta corta de 1 a 3 palabras en español "
        f"(sin puntuación extra) para clasificar este archivo o carpeta: {prompt_key}"
    )
    try:
        proc = subprocess.run(
            [cmd_bin, "-m", selected_model, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            first_line = next(
                (ln.strip() for ln in proc.stdout.splitlines() if ln.strip()),
                "",
            )
            if first_line:
                return first_line[:60]
    except (OSError, subprocess.SubprocessError):
        pass
    return heuristic_classify(prompt_key)


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
    """Optional Antigravity CLI classifier that falls back cleanly to local heuristics."""

    def __init__(
        self,
        db_path: Path,
        api_key: str = "",
        cli_command: str = "agy",
        model: str = DEFAULT_AGY_MODEL,
        transport: Callable[[str], tuple[int, str]] | None = None,
        bucket_capacity: int = 10,
        refill_per_sec: float = 0.5,
    ) -> None:
        self.db_path = db_path
        self._api_key = api_key
        self.cli_command = (cli_command or "agy").strip() or "agy"
        self.model = model or DEFAULT_AGY_MODEL
        self._transport = transport
        self.bucket = TokenBucket(capacity=bucket_capacity, refill_per_sec=refill_per_sec)
        self.consecutive_failures = 0
        self.circuit_open = False
        self.remote_calls = 0
        init_db(self.db_path)

    def get_masked_settings(self) -> dict[str, Any]:
        return {
            "antigravity_cli": self.cli_command,
            "model": self.model,
            "configured": bool(self.cli_command),
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
        label = classify_with_agy(
            prompt_key,
            cli_command=self.cli_command,
            model=self.model,
        )
        return (200, label)

    def classify(self, cache_key: str, kind: str = "item") -> str:
        """Classify an item or cluster signature with cache, rate-limit, and heuristic fallback."""
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if self.circuit_open or not self.bucket.acquire():
            return heuristic_classify(cache_key)

        self.remote_calls += 1
        transport = self._transport or self._default_transport
        status, label = transport(cache_key)
        if status in (401, 403, 429) or status >= 500:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 2:
                self.circuit_open = True
            return heuristic_classify(cache_key)

        self.consecutive_failures = 0
        clean_label = label.strip() or heuristic_classify(cache_key)
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
