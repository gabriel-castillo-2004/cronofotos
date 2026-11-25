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


_FORBIDDEN_TEMPLATE_TOKENS = ("__", "placeholder", "sample")


def _is_clean_suggestion_text(val: Any) -> bool:
    if not isinstance(val, str) or not val.strip():
        return False
    low = val.lower()
    return not any(tok in low for tok in _FORBIDDEN_TEMPLATE_TOKENS)


def _sanitize_suggestion(raw: dict[str, Any], folder_meta: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    stype = str(raw.get("type", "")).strip().lower()
    if stype not in {"rename", "merge", "split"}:
        return None
    reason = str(raw.get("reason", "")).strip() or "Optimización estructural sugerida"
    if not _is_clean_suggestion_text(reason):
        return None

    if stype == "rename":
        before = raw.get("before")
        after = raw.get("after")
        if not (_is_clean_suggestion_text(before) and _is_clean_suggestion_text(after)):
            return None
        if before == after:
            return None
        return {"type": "rename", "before": str(before).strip(), "after": str(after).strip(), "reason": reason}

    if stype == "merge":
        before = raw.get("before")
        after = raw.get("after")
        if not isinstance(before, list) or len(before) < 2:
            return None
        if not all(_is_clean_suggestion_text(b) for b in before) or not _is_clean_suggestion_text(after):
            return None
        return {
            "type": "merge",
            "before": [str(b).strip() for b in before],
            "after": str(after).strip(),
            "reason": reason,
        }

    # stype == "split"
    before = raw.get("before")
    after = raw.get("after")
    if not _is_clean_suggestion_text(before) or not isinstance(after, list) or len(after) < 2:
        return None
    if not all(_is_clean_suggestion_text(a) for a in after):
        return None
    meta = folder_meta.get(str(before).strip())
    if meta is not None:
        if int(meta.get("count", 0)) <= 25 or float(meta.get("max_gap_hours", 0.0)) < 3.0:
            return None
    return {
        "type": "split",
        "before": str(before).strip(),
        "after": [str(a).strip() for a in after],
        "reason": reason,
    }


def _parse_suggestions_json(stdout: str, folder_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fence_match.group(1).strip() if fence_match else text
    candidates = [candidate]
    obj_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if obj_match:
        candidates.append(obj_match.group(0))

    for jc in candidates:
        try:
            parsed = json.loads(jc)
            if isinstance(parsed, dict) and isinstance(parsed.get("suggestions"), list):
                cleaned: list[dict[str, Any]] = []
                for item in parsed["suggestions"]:
                    if isinstance(item, dict):
                        s = _sanitize_suggestion(item, folder_meta)
                        if s is not None:
                            cleaned.append(s)
                return cleaned
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _deterministic_tree_suggestions(folder_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    ordered = sorted(folder_summaries, key=lambda f: str(f.get("folder", "")))

    for f in ordered:
        folder = str(f.get("folder", ""))
        cnt = int(f.get("count", 0))
        gap = float(f.get("max_gap_hours", 0.0))
        loc = str(f.get("location_label") or "").strip()

        if cnt > 25 and gap >= 3.0:
            base = folder.split(" - ")[0] if " - " in folder else folder
            suggestions.append(
                {
                    "type": "split",
                    "before": folder,
                    "after": [f"{base} - Sesión Mañana", f"{base} - Sesión Tarde"],
                    "reason": "Carpeta con más de 25 archivos y salto temporal interno >= 3 horas",
                }
            )
        elif folder.endswith(" - Eventos del día") and loc and not loc.startswith("GPS "):
            prefix = folder.rsplit(" - ", 1)[0]
            city = loc.split(",")[0].strip()
            if city and _is_clean_suggestion_text(city):
                suggestions.append(
                    {
                        "type": "rename",
                        "before": folder,
                        "after": f"{prefix} - {city}",
                        "reason": f"Ubicación GPS detectada en {city}",
                    }
                )

    for i in range(len(ordered) - 1):
        f1 = ordered[i]
        f2 = ordered[i + 1]
        if int(f1.get("count", 0)) < 4 and int(f2.get("count", 0)) < 4:
            d1 = str(f1.get("end_date") or "")[:10]
            d2 = str(f2.get("start_date") or "")[:10]
            if d1 and d2 and f1.get("location_label") == f2.get("location_label"):
                try:
                    dt1 = datetime.fromisoformat(d1)
                    dt2 = datetime.fromisoformat(d2)
                    if 0 <= (dt2 - dt1).days <= 1:
                        p1 = str(f1["folder"]).split("/")
                        if len(p1) >= 3:
                            year = p1[1]
                            s_ymd = str(f1.get("start_date") or d1)[:10]
                            e_md = d2[5:10]
                            suggestions.append(
                                {
                                    "type": "merge",
                                    "before": [str(f1["folder"]), str(f2["folder"])],
                                    "after": f"01_Recuerdos/{year}/{s_ymd} al {e_md} - Eventos Cercanos",
                                    "reason": "Carpetas consecutivas pequeñas en la misma zona",
                                }
                            )
                except ValueError:
                    pass

    return suggestions


def suggest_tree_optimizations(
    folder_summaries: list[dict[str, Any]],
    *,
    cli_command: str = "agy",
    model: str = DEFAULT_AGY_MODEL,
    timeout: float = 20.0,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Analyze folder metadata + <=256px representative thumbnails in batches of <=12 via agy."""
    if not folder_summaries:
        return {"engine": "Fallback determinista local", "suggestions": []}

    folder_meta = {str(f["folder"]): f for f in folder_summaries if f.get("folder")}
    cmd_bin = (cli_command or "agy").strip() or "agy"
    resolved_bin = resolve_cli_binary(cmd_bin, allowed_names=AGY_ALLOWED_NAMES)
    selected_model = (model or DEFAULT_AGY_MODEL).strip() or DEFAULT_AGY_MODEL

    can_call_agy = resolved_bin is not None or (
        subprocess.run is not _ORIG_SUBPROCESS_RUN and is_standard_agy_binary(cmd_bin)
    )
    if not can_call_agy or (cancel_event is not None and cancel_event.is_set()):
        return {
            "engine": "Fallback determinista local",
            "suggestions": _deterministic_tree_suggestions(folder_summaries),
        }

    batch_size = 12
    all_suggestions: list[dict[str, Any]] = []
    agy_failed = False

    for i in range(0, len(folder_summaries), batch_size):
        if cancel_event is not None and cancel_event.is_set():
            agy_failed = True
            break
        chunk = folder_summaries[i : i + batch_size]
        compact = [
            {
                "folder": f.get("folder"),
                "count": f.get("count"),
                "start_date": f.get("start_date"),
                "end_date": f.get("end_date"),
                "location": f.get("location_label"),
                "max_gap_hours": f.get("max_gap_hours"),
                "thumbnail": f.get("thumb_path"),
            }
            for f in chunk
        ]
        prompt = (
            "Analiza estas carpetas (máximo 12 por lote) y responde ÚNICAMENTE con JSON válido "
            '{"suggestions": [{"type": "rename|merge|split", "before": ..., "after": ..., "reason": "..."}]}. '
            "Reglas: split solo si count > 25 y max_gap_hours >= 3.0. "
            "Carpetas: " + json.dumps(compact, ensure_ascii=False)
        )
        argv = [resolved_bin or cmd_bin, "-m", selected_model, "-p", prompt]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout:
                agy_failed = True
                break
            parsed = _parse_suggestions_json(proc.stdout, folder_meta)
            if parsed is None:
                agy_failed = True
                break
            all_suggestions.extend(parsed)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            logger.warning("Fallo al consultar agy para optimización de árbol: %s", exc)
            agy_failed = True
            break

    if agy_failed:
        return {
            "engine": "Fallback determinista local",
            "suggestions": _deterministic_tree_suggestions(folder_summaries),
        }
    return {
        "engine": f"agy · {selected_model}",
        "suggestions": all_suggestions,
    }
