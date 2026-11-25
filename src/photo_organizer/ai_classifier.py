"""Optional Antigravity CLI (`agy`) classifier with SQLite cache and heuristic fallback."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from photo_organizer.config import AGY_ALLOWED_NAMES, is_standard_agy_binary, resolve_cli_binary
from photo_organizer.db import init_db, open_connection

logger = logging.getLogger(__name__)
DEFAULT_AGY_MODEL = "gemini-4-hydrogen-medium"
_ORIG_SUBPROCESS_RUN = subprocess.run

_GENERIC_FOLDER_NAMES = frozenset(
    {
        "onedrivefotos",
        "onedrive",
        "camera",
        "dcim",
        "pictures",
        "imagenes",
        "imágenes",
        "fotos",
        "photos",
        "camera roll",
        "rollo de cámara",
        "mis imagenes",
        "mis imágenes",
        "sin_carpeta",
        "unknown",
        "media",
        "backup",
        "takeout",
        "google fotos",
        "google photos",
    }
)


def _clean_label(raw: str) -> str:
    """Sanitize an event label and strip template/punctuation artifacts."""
    cleaned = raw.strip().strip("\"'`")
    cleaned = re.sub(r"Fotos_Carpeta__Gps_[^\s]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:fallback_|placeholder|sample_)[^\s]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"__+", " ", cleaned)
    cleaned = re.sub(r"[<>:\"/\\|?*]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:60]


def _parse_agy_output(stdout: str) -> str:
    """Extract event name from JSON or plain-text `agy` output."""
    text = (stdout or "").strip()
    if not text:
        return ""

    # Strip optional markdown code fences (including unclosed leading fences)
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
    else:
        candidate = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()

    # Try direct JSON or embedded JSON object
    json_candidates = [candidate]
    obj_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if obj_match:
        json_candidates.append(obj_match.group(0))

    for jc in json_candidates:
        try:
            parsed = json.loads(jc)
            if isinstance(parsed, dict):
                for key in ("nombre_evento", "event_name", "nombre", "evento", "label"):
                    val = parsed.get(key)
                    if isinstance(val, str) and val.strip():
                        return _clean_label(val)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Salida JSON malformada o truncada de agy: %s", exc)
            continue

    # If the output looks like truncated/malformed JSON, let the caller fall back cleanly
    if candidate.startswith(("{", "[")) or re.search(
        r'"(?:nombre_evento|event_name|nombre|evento|label)"\s*:',
        candidate,
    ):
        return ""

    first_line = next((ln.strip() for ln in candidate.splitlines() if ln.strip()), "")
    if first_line.startswith(("{", "[")):
        return ""
    return _clean_label(first_line)


def heuristic_classify(prompt_key: str) -> str:
    """Derive a clean Spanish category/event label from folder, GPS, or filename metadata."""
    raw = (prompt_key or "").strip()
    if not raw:
        return "Eventos del día"

    # Handle structured event summaries: "YYYY-MM-DD | N fotos | carpeta: X | GPS: Y"
    if "|" in raw and ("carpeta:" in raw.lower() or "gps:" in raw.lower()):
        folder_match = re.search(r"carpeta:\s*([^|]+)", raw, flags=re.IGNORECASE)
        gps_match = re.search(r"gps:\s*([^|]+)", raw, flags=re.IGNORECASE)
        folder_name = folder_match.group(1).strip() if folder_match else ""
        gps_val = gps_match.group(1).strip() if gps_match else ""

        if folder_name and folder_name.lower() not in _GENERIC_FOLDER_NAMES:
            tokens = [
                t
                for t in re.split(r"[\s_\-]+", folder_name)
                if t and not t.isdigit() and len(t) > 1
            ]
            if tokens:
                return _clean_label(" ".join(t.capitalize() for t in tokens[:4]))

        if gps_val and not re.match(r"^(?:gps\s*)?-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?$", gps_val, re.IGNORECASE):
            return _clean_label(gps_val)

        return "Eventos del día"

    lower = raw.lower()
    rules = (
        (("screenshot", "captura", "pantallazo", "screen_shot"), "Capturas"),
        (("whatsapp", "-wa0", "wa_"), "WhatsApp"),
        (("telegram",), "Telegram"),
        (("instagram",), "Instagram"),
        (("download", "descarga"), "Descargas"),
        (("document", "scan", "recibo", "factura", "pdf"), "Documentos"),
        (("viaje", "trip", "travel", "vacacion", "playa", "beach"), "Viaje"),
    )
    for keywords, label in rules:
        if any(kw in lower for kw in keywords):
            return label

    parts = [p for p in re.split(r"[/\\]+", raw) if p]
    for part in parts:
        stem = Path(part).stem
        if stem.lower() in _GENERIC_FOLDER_NAMES:
            continue
        tokens = [
            t
            for t in re.split(r"[\s_\-]+", stem)
            if t and not t.isdigit() and len(t) > 2
        ]
        if tokens:
            return _clean_label(" ".join(t.capitalize() for t in tokens[:3]))
    return "Eventos del día"


def classify_with_agy(
    prompt_key: str,
    *,
    cli_command: str = "agy",
    model: str = DEFAULT_AGY_MODEL,
    timeout: float = 15.0,
    cancel_event: threading.Event | None = None,
) -> str:
    """Invoke local Antigravity CLI (`agy`) via subprocess with heuristic fallback."""
    if cancel_event is not None and cancel_event.is_set():
        return heuristic_classify(prompt_key)

    cmd_bin = (cli_command or "agy").strip() or "agy"
    resolved_bin = resolve_cli_binary(cmd_bin, allowed_names=AGY_ALLOWED_NAMES)
    if resolved_bin is None and (
        subprocess.run is _ORIG_SUBPROCESS_RUN or not is_standard_agy_binary(cmd_bin)
    ):
        return heuristic_classify(prompt_key)

    selected_model = (model or DEFAULT_AGY_MODEL).strip() or DEFAULT_AGY_MODEL
    prompt = (
        "Responde únicamente con un JSON válido de la forma "
        '{"nombre_evento": "Nombre Corto en Español"} '
        "(máximo 3 palabras, sin explicación adicional) para este grupo de fotos: "
        f"{prompt_key}"
    )
    argv = [resolved_bin or cmd_bin, "-m", selected_model, "-p", prompt]
    try:
        if subprocess.run is not _ORIG_SUBPROCESS_RUN:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout:
                parsed = _parse_agy_output(proc.stdout)
                if parsed:
                    return parsed
        else:
            proc_p = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + timeout
            while proc_p.poll() is None:
                if (cancel_event is not None and cancel_event.is_set()) or time.monotonic() >= deadline:
                    proc_p.terminate()
                    try:
                        proc_p.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc_p.kill()
                    return heuristic_classify(prompt_key)
                time.sleep(0.05)
            stdout_data, _ = proc_p.communicate()
            if proc_p.returncode == 0 and stdout_data:
                parsed = _parse_agy_output(stdout_data)
                if parsed:
                    return parsed
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning("Error ejecutando o parseando salida de agy: %s", exc)
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
        batch_size: int = 100,
        solo_primer_lote: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Require explicit user confirmation before dispatching a remote AI batch."""
        eff_batch_size = batch_size if batch_size and batch_size > 0 else 100
        chunks = [keys[i : i + eff_batch_size] for i in range(0, len(keys), eff_batch_size)]
        total_batches = len(chunks)

        if not confirmed:
            return {
                "requires_confirmation": True,
                "item_count": len(keys),
                "batch_size": eff_batch_size,
                "total_batches": total_batches,
                "solo_primer_lote": solo_primer_lote,
                "model": self.model,
                "results": {},
            }

        batches: list[dict[str, Any]] = [
            {
                "batch_index": idx + 1,
                "total_batches": total_batches,
                "label": f"Lote {idx + 1} de {total_batches}",
                "status": "pendiente",
                "total_items": len(chunk),
                "processed_items": 0,
                "duplicates_skipped": 0,
                "errors": 0,
                "elapsed_seconds": 0.0,
                "error_message": None,
            }
            for idx, chunk in enumerate(chunks)
        ]

        results: dict[str, str] = {}
        cancelled = False
        for b_idx, chunk in enumerate(chunks):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            batch_info = batches[b_idx]
            cached_chunk: dict[str, str] = {}
            for k in chunk:
                cached = self._get_cached(k)
                if cached is not None:
                    cached_chunk[k] = cached

            if len(cached_chunk) == len(chunk):
                results.update(cached_chunk)
                batch_info["status"] = "completado"
                batch_info["processed_items"] = len(chunk)
                batch_info["elapsed_seconds"] = 0.0
                continue

            batch_info["status"] = "en_proceso"
            t0 = time.monotonic()
            errors = 0
            for k in chunk:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                try:
                    results[k] = self.classify(k, kind=kind)
                    batch_info["processed_items"] += 1
                except Exception as exc:  # pragma: no cover
                    errors += 1
                    batch_info["errors"] = errors
                    batch_info["error_message"] = str(exc)

            batch_info["elapsed_seconds"] = round(time.monotonic() - t0, 3)
            batch_info["status"] = "error" if errors > 0 and batch_info["processed_items"] == 0 else "completado"

            if cancelled or (solo_primer_lote and b_idx == 0 and total_batches > 1):
                break

        return {
            "requires_confirmation": False,
            "item_count": len(keys),
            "batch_size": eff_batch_size,
            "solo_primer_lote": solo_primer_lote,
            "model": self.model,
            "circuit_open": self.circuit_open,
            "cancelled": cancelled,
            "batches": batches,
            "results": results,
        }
