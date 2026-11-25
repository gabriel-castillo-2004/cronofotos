"""Optional Antigravity CLI (`agy`) classifier with SQLite cache and heuristic fallback."""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from photo_organizer.config import (
    AGY_ALLOWED_NAMES,
    PEOPLE_IMAGE_EXTENSIONS,
    is_standard_agy_binary,
    resolve_cli_binary,
)
from photo_organizer.thumbnails import extract_top_face_crops
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
    people_dir: Path | str | None = None,
    add_dirs: list[Path | str] | None = None,
    error_out: dict[str, str] | None = None,
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
    people_labels = _load_people_labels(people_dir)
    people_hint = (
        f" Personas clave de referencia disponibles: {', '.join(people_labels)}."
        if people_labels
        else ""
    )
    prompt = (
        "Responde únicamente con un JSON válido de la forma "
        '{"nombre_evento": "Nombre Corto en Español"} '
        "(máximo 6 palabras, sin explicación adicional) identificando el evento y, "
        f"si aparecen personas clave de referencia, inclúyelas.{people_hint} "
        f"Grupo de fotos: {prompt_key}"
    )
    argv = [resolved_bin or cmd_bin, "-m", selected_model]
    for d in _collect_add_dirs(people_dir, add_dirs):
        argv.extend(["--add-dir", str(d)])
    argv.extend(["-p", prompt])
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
            elif proc.returncode != 0 and error_out is not None:
                detail = _summarize_failure_reason(proc.stderr or proc.stdout)
                error_out["reason"] = f"código {proc.returncode}: {detail}" if detail else f"código {proc.returncode}"
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
                    if error_out is not None and (cancel_event is None or not cancel_event.is_set()):
                        error_out["reason"] = "tiempo de espera agotado"
                    return heuristic_classify(prompt_key)
                time.sleep(0.05)
            stdout_data, stderr_data = proc_p.communicate()
            if proc_p.returncode == 0 and stdout_data:
                parsed = _parse_agy_output(stdout_data)
                if parsed:
                    return parsed
            elif proc_p.returncode != 0 and error_out is not None:
                detail = _summarize_failure_reason(stderr_data or stdout_data)
                error_out["reason"] = f"código {proc_p.returncode}: {detail}" if detail else f"código {proc_p.returncode}"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        if error_out is not None:
            error_out["reason"] = _summarize_failure_reason(str(exc)) or "error ejecutando agy"
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
_FORBIDDEN_FIXED_LABELS = ("sesión mañana", "sesión tarde", "sesion manana", "sesion tarde")


def _is_clean_suggestion_text(val: Any, *, forbid_fixed_labels: bool = False) -> bool:
    if not isinstance(val, str) or not val.strip():
        return False
    low = val.lower()
    if any(tok in low for tok in _FORBIDDEN_TEMPLATE_TOKENS):
        return False
    if forbid_fixed_labels and any(lbl in low for lbl in _FORBIDDEN_FIXED_LABELS):
        return False
    return True


def _sanitize_suggestion(
    raw: dict[str, Any],
    folder_meta: dict[str, dict[str, Any]],
    *,
    forbid_fixed_labels: bool = False,
) -> dict[str, Any] | None:
    stype = str(raw.get("type", "")).strip().lower()
    if stype not in {"rename", "merge", "split"}:
        return None
    reason = str(raw.get("reason", "")).strip() or "Optimización estructural sugerida"
    if not _is_clean_suggestion_text(reason, forbid_fixed_labels=forbid_fixed_labels):
        return None

    if stype == "rename":
        before = raw.get("before")
        after = raw.get("after")
        if not (
            _is_clean_suggestion_text(before, forbid_fixed_labels=forbid_fixed_labels)
            and _is_clean_suggestion_text(after, forbid_fixed_labels=forbid_fixed_labels)
        ):
            return None
        if before == after:
            return None
        return {"type": "rename", "before": str(before).strip(), "after": str(after).strip(), "reason": reason}

    if stype == "merge":
        before = raw.get("before")
        after = raw.get("after")
        if not isinstance(before, list) or len(before) < 2:
            return None
        if not all(_is_clean_suggestion_text(b, forbid_fixed_labels=forbid_fixed_labels) for b in before) or not _is_clean_suggestion_text(after, forbid_fixed_labels=forbid_fixed_labels):
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
    if not _is_clean_suggestion_text(before, forbid_fixed_labels=forbid_fixed_labels) or not isinstance(after, list) or len(after) < 2:
        return None
    if not all(_is_clean_suggestion_text(a, forbid_fixed_labels=forbid_fixed_labels) for a in after):
        return None
    meta = folder_meta.get(str(before).strip())
    if meta is not None:
        if meta.get("is_midnight_merged"):
            return None
        if int(meta.get("count", 0)) <= 25 or float(meta.get("max_gap_hours", 0.0)) < 3.0:
            return None
    return {
        "type": "split",
        "before": str(before).strip(),
        "after": [str(a).strip() for a in after],
        "reason": reason,
    }


def _parse_suggestions_json(
    stdout: str,
    folder_meta: dict[str, dict[str, Any]],
    *,
    forbid_fixed_labels: bool = False,
) -> list[dict[str, Any]] | None:
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
                        s = _sanitize_suggestion(item, folder_meta, forbid_fixed_labels=forbid_fixed_labels)
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

        if cnt > 25 and gap >= 3.0 and not f.get("is_midnight_merged"):
            base = folder.split(" - ")[0] if " - " in folder else folder
            suggestions.append(
                {
                    "type": "split",
                    "before": folder,
                    "after": [f"{base} - Primera Parte", f"{base} - Segunda Parte"],
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


def _select_representative_thumb_paths(folder_info: dict[str, Any], limit: int = 3) -> list[Path]:
    """Pick up to `limit` representative thumbnail paths (start, mid, 75%) for a folder."""
    raw_list = folder_info.get("thumb_paths")
    candidates: list[Path] = []
    if isinstance(raw_list, (list, tuple)):
        for item in raw_list:
            if item:
                p = Path(str(item))
                if p.exists() and p.is_file():
                    candidates.append(p)
    if not candidates and folder_info.get("thumb_path"):
        p = Path(str(folder_info["thumb_path"]))
        if p.exists() and p.is_file():
            candidates.append(p)
    if not candidates:
        return []
    if len(candidates) <= limit:
        seen: set[Path] = set()
        unique: list[Path] = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    indices = [0, len(candidates) // 2, min(len(candidates) - 1, (len(candidates) * 3) // 4)]
    selected: list[Path] = []
    seen_idx: set[int] = set()
    for idx in indices:
        if idx not in seen_idx:
            seen_idx.add(idx)
            selected.append(candidates[idx])
    return selected[:limit]


def _prepare_temp_thumb(src: Path, dst: Path, max_dim: int = 320) -> Path:
    """Save a <=320px JPEG copy into `dst` (falling back to raw copy if image decode fails)."""
    try:
        from PIL import Image, ImageOps

        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")
            img.thumbnail((max_dim, max_dim))
            img.save(dst, format="JPEG", quality=82, optimize=True)
            return dst
    except Exception:
        shutil.copy2(src, dst)
        return dst


def _load_people_labels(people_dir: Path | str | None) -> list[str]:
    """Read key-people labels from people.json or image filenames in `people_dir`."""
    if not people_dir:
        return []
    pdir = Path(people_dir)
    if not pdir.exists() or not pdir.is_dir():
        return []
    index_file = pdir / "people.json"
    if index_file.is_file():
        try:
            data = json.loads(index_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                labels = [str(item.get("label", "")).strip() for item in data if isinstance(item, dict)]
                labels = [lbl for lbl in labels if lbl]
                if labels:
                    return labels
        except (OSError, json.JSONDecodeError):
            pass
    labels: list[str] = []
    for f in sorted(pdir.iterdir()):
        if f.is_file() and f.suffix.lower() in PEOPLE_IMAGE_EXTENSIONS:
            stem = f.stem.split("__", 1)[0].replace("_", " ").strip()
            if stem:
                labels.append(stem)
    return labels


def _resolve_face_clusters_dir(
    face_clusters_dir: Path | str | None,
    data_dir: Path | str | None,
    people_dir: Path | str | None,
) -> Path | None:
    """Resolve `<data_dir>/face_clusters` from explicit path, data_dir, or people_dir."""
    if face_clusters_dir:
        return Path(face_clusters_dir)
    if data_dir:
        return Path(data_dir) / "face_clusters"
    if people_dir:
        return Path(people_dir).parent / "face_clusters"
    return None


def _is_anchor_person_label(label: str) -> bool:
    """Return True if label represents the anchor user ('Yo (ancla)')."""
    low = re.sub(r"\s+", " ", (label or "").strip().lower())
    return low in {"yo (ancla)", "yo(ancla)", "yo"} or "yo (ancla)" in low


def _load_face_cluster_records(face_clusters_dir: Path | None) -> list[dict[str, Any]]:
    """Load all face cluster JSON records from `face_clusters_dir`."""
    if not face_clusters_dir or not face_clusters_dir.exists() or not face_clusters_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for jf in sorted(face_clusters_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("cluster_id"):
            records.append(data)
        elif isinstance(data, list):
            records.extend(item for item in data if isinstance(item, dict) and item.get("cluster_id"))
    return records


def _load_whitelisted_face_clusters(face_clusters_dir: Path | None) -> tuple[bool, list[dict[str, Any]]]:
    """Return (has_any_clusters, whitelisted_clusters) filtering out unnamed and ignored entries."""
    all_records = _load_face_cluster_records(face_clusters_dir)
    if not all_records:
        return False, []
    whitelisted = [
        rec
        for rec in all_records
        if not bool(rec.get("ignored", False)) and str(rec.get("label") or "").strip() != ""
    ]
    return True, whitelisted


def resolve_folder_whitelisted_people(
    media_ids: list[str] | set[str] | None,
    whitelisted_clusters: list[dict[str, Any]],
) -> list[str]:
    """Cross-reference folder media_ids against whitelisted clusters, excluding 'Yo (ancla)'."""
    folder_mids = {str(mid) for mid in (media_ids or []) if mid}
    if not folder_mids or not whitelisted_clusters:
        return []
    confirmed: list[str] = []
    seen: set[str] = set()
    for cl in whitelisted_clusters:
        label = str(cl.get("label") or "").strip()
        if not label or _is_anchor_person_label(label):
            continue
        cl_mids = {str(mid) for mid in (cl.get("media_ids") or []) if mid}
        if folder_mids & cl_mids and label not in seen:
            seen.add(label)
            confirmed.append(label)
    return confirmed


def _collect_add_dirs(
    people_dir: Path | str | None,
    extra_dirs: list[Path | str] | None = None,
) -> list[Path]:
    """Return deduplicated existing directories to pass via `--add-dir`."""
    dirs: list[Path] = []
    seen: set[Path] = set()
    candidates: list[Path | str] = []
    if extra_dirs:
        candidates.extend(extra_dirs)
    if people_dir:
        pdir = Path(people_dir)
        if pdir.exists() and pdir.is_dir():
            has_imgs = any(
                f.is_file() and f.suffix.lower() in PEOPLE_IMAGE_EXTENSIONS
                for f in pdir.iterdir()
            )
            if has_imgs:
                candidates.append(pdir)
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw)
        if p.exists() and p.is_dir():
            res = p.resolve()
            if res not in seen:
                seen.add(res)
                dirs.append(p)
    return dirs


def _collect_candidate_image_paths(folder_info: dict[str, Any]) -> list[Path]:
    """Collect all existing candidate image paths for a folder."""
    raw_list = folder_info.get("thumb_paths")
    candidates: list[Path] = []
    seen: set[Path] = set()
    if isinstance(raw_list, (list, tuple)):
        for item in raw_list:
            if item:
                p = Path(str(item))
                if p.exists() and p.is_file() and p not in seen:
                    seen.add(p)
                    candidates.append(p)
    if folder_info.get("thumb_path"):
        p = Path(str(folder_info["thumb_path"]))
        if p.exists() and p.is_file() and p not in seen:
            seen.add(p)
            candidates.append(p)
    return candidates


def _summarize_failure_reason(raw: str, max_len: int = 140) -> str:
    """Return a concise single-line technical failure reason."""
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    msg = re.sub(r"\s+", " ", lines[0])
    return msg[:max_len]


def suggest_tree_optimizations(
    folder_summaries: list[dict[str, Any]],
    *,
    cli_command: str = "agy",
    model: str = DEFAULT_AGY_MODEL,
    timeout: float = 60.0,
    cancel_event: threading.Event | None = None,
    forbid_fixed_labels: bool = False,
    people_dir: Path | str | None = None,
    face_clusters_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Analyze folder metadata + 3 scene thumbs + up to 4 face crops + key people via agy."""
    if not folder_summaries:
        return {"engine": "Fallback determinista local", "suggestions": []}

    folder_meta = {str(f["folder"]): f for f in folder_summaries if f.get("folder")}
    cmd_bin = (cli_command or "agy").strip() or "agy"
    resolved_bin = resolve_cli_binary(cmd_bin, allowed_names=AGY_ALLOWED_NAMES)
    selected_model = (model or DEFAULT_AGY_MODEL).strip() or DEFAULT_AGY_MODEL

    can_call_agy = resolved_bin is not None or (
        subprocess.run is not _ORIG_SUBPROCESS_RUN and is_standard_agy_binary(cmd_bin)
    )
    if cancel_event is not None and cancel_event.is_set():
        reason = "operación cancelada"
        return {
            "engine": f"Fallback determinista local ({reason})",
            "error_reason": reason,
            "suggestions": _deterministic_tree_suggestions(folder_summaries),
        }
    if not can_call_agy:
        reason = f"binario '{cmd_bin}' no encontrado"
        return {
            "engine": f"Fallback determinista local ({reason})",
            "error_reason": reason,
            "suggestions": _deterministic_tree_suggestions(folder_summaries),
        }

    batch_size = 12
    all_suggestions: list[dict[str, Any]] = []
    failure_reason: str | None = None
    resolved_fc_dir = _resolve_face_clusters_dir(face_clusters_dir, data_dir, people_dir)
    has_clusters, whitelisted_clusters = _load_whitelisted_face_clusters(resolved_fc_dir)
    legacy_people_labels = [] if has_clusters else _load_people_labels(people_dir)

    for i in range(0, len(folder_summaries), batch_size):
        if cancel_event is not None and cancel_event.is_set():
            failure_reason = "operación cancelada"
            break
        chunk = folder_summaries[i : i + batch_size]

        tmp_dir = tempfile.mkdtemp(prefix="agy_thumbs_")
        atexit.register(shutil.rmtree, tmp_dir, True)
        tmp_root = Path(tmp_dir)
        compact: list[dict[str, Any]] = []
        has_prepared_thumbs = False
        batch_confirmed_people: list[str] = []
        seen_batch_people: set[str] = set()

        for f_idx, f in enumerate(chunk):
            rep_srcs = _select_representative_thumb_paths(f, limit=3)
            prepared_paths: list[str] = []
            for t_idx, src_p in enumerate(rep_srcs):
                dst_p = tmp_root / f"folder_{f_idx:02d}_scene_{t_idx + 1}.jpg"
                _prepare_temp_thumb(src_p, dst_p, max_dim=320)
                prepared_paths.append(str(dst_p))
                has_prepared_thumbs = True

            candidate_paths = _collect_candidate_image_paths(f)
            face_crops = extract_top_face_crops(
                candidate_paths,
                tmp_root,
                max_faces=4,
                file_prefix=f"folder_{f_idx:02d}_face",
            )
            if face_crops:
                has_prepared_thumbs = True

            confirmed_people = (
                resolve_folder_whitelisted_people(f.get("media_ids"), whitelisted_clusters)
                if has_clusters
                else []
            )
            for pname in confirmed_people:
                if pname not in seen_batch_people:
                    seen_batch_people.add(pname)
                    batch_confirmed_people.append(pname)

            compact_entry: dict[str, Any] = {
                "folder": f.get("folder"),
                "count": f.get("count"),
                "start_date": f.get("start_date"),
                "end_date": f.get("end_date"),
                "location": f.get("location_label"),
                "max_gap_hours": f.get("max_gap_hours"),
                "is_midnight_merged": bool(f.get("is_midnight_merged", False)),
                "thumbnails": prepared_paths if prepared_paths else ([f.get("thumb_path")] if f.get("thumb_path") else []),
                "face_crops": [str(p) for p in face_crops],
            }
            if confirmed_people:
                compact_entry["personas_confirmadas"] = confirmed_people
            compact.append(compact_entry)

        extra_dirs: list[Path | str] = []
        if has_prepared_thumbs:
            extra_dirs.append(tmp_root)
        else:
            thumb_dirs = [
                str(Path(str(f["thumb_path"])).parent)
                for f in chunk
                if f.get("thumb_path") and Path(str(f["thumb_path"])).exists()
            ]
            if thumb_dirs:
                try:
                    common = Path(os.path.commonpath(thumb_dirs))
                    extra_dirs.append(common if common.exists() else Path(thumb_dirs[0]))
                except ValueError:
                    extra_dirs.append(Path(thumb_dirs[0]))

        add_dirs = _collect_add_dirs(None if has_clusters else people_dir, extra_dirs)
        if has_clusters:
            people_hint = (
                f" Personas confirmadas presentes en estas carpetas: {', '.join(batch_confirmed_people)}."
                if batch_confirmed_people
                else ""
            )
        else:
            people_hint = (
                f" Personas clave de referencia en personas_referencia/: {', '.join(legacy_people_labels)}."
                if legacy_people_labels
                else ""
            )

        prompt = (
            "Lee directamente desde disco los archivos de imagen adjuntos en los directorios (--add-dir): "
            "hasta 3 miniaturas de escena por carpeta (inicio, mitad y 75%), hasta 4 recortes de rostros cercanos "
            "y las fotos de referencia de personas clave si existen. "
            "Identifica la actividad, evento, recinto o artista visible y las personas clave presentes "
            "(ej. 'Concierto de Luis Miguel con Mamá (Iztacalco)', 'Cena Familiar', 'Senderismo en Montaña') "
            "y propón sugerencias `rename` enriqueciendo el nombre actual de la carpeta aunque ya tenga ubicación GPS."
            f"{people_hint} "
            "Analiza estas carpetas (máximo 12 por lote) y responde ÚNICAMENTE con JSON válido "
            '{"suggestions": [{"type": "rename|merge|split", "before": ..., "after": ..., "reason": "..."}]}. '
            "Reglas: usa títulos descriptivos reales en español según lo que se ve en cada imagen; "
            "NUNCA uses etiquetas fijas como 'Sesión Mañana' o 'Sesión Tarde'; "
            "split solo si count > 25, max_gap_hours >= 3.0 y is_midnight_merged es false. "
            "Carpetas: " + json.dumps(compact, ensure_ascii=False)
        )
        argv = [resolved_bin or cmd_bin, "-m", selected_model]
        for add_d in add_dirs:
            argv.extend(["--add-dir", str(add_d)])
        argv.extend(["-p", prompt])
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode != 0:
                detail = _summarize_failure_reason(proc.stderr or proc.stdout)
                failure_reason = (
                    f"código {proc.returncode}: {detail}" if detail else f"código {proc.returncode}"
                )
                logger.warning("agy falló (%s)", failure_reason)
                break
            if not proc.stdout or not proc.stdout.strip():
                failure_reason = "salida vacía de agy"
                logger.warning("agy devolvió salida vacía")
                break
            parsed = _parse_suggestions_json(
                proc.stdout,
                folder_meta,
                forbid_fixed_labels=forbid_fixed_labels,
            )
            if parsed is None:
                snippet = _summarize_failure_reason(proc.stdout, max_len=80)
                failure_reason = f"salida JSON inválida de agy ({snippet})" if snippet else "salida JSON inválida de agy"
                logger.warning("agy devolvió salida JSON inválida: %s", snippet)
                break
            all_suggestions.extend(parsed)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            detail = _summarize_failure_reason(str(exc))
            failure_reason = f"error ejecutando agy: {detail}" if detail else "error ejecutando agy"
            logger.warning("Fallo al consultar agy para optimización de árbol: %s", exc)
            break

    if failure_reason is not None:
        return {
            "engine": f"Fallback determinista local ({failure_reason})",
            "error_reason": failure_reason,
            "suggestions": _deterministic_tree_suggestions(folder_summaries),
        }
    return {
        "engine": f"agy · {selected_model}",
        "suggestions": all_suggestions,
    }


def generate_tree_ai_suggestions(
    folder_summaries: list[dict[str, Any]],
    *,
    cli_command: str = "agy",
    model: str = DEFAULT_AGY_MODEL,
    timeout: float = 60.0,
    cancel_event: threading.Event | None = None,
    people_dir: Path | str | None = None,
    face_clusters_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Visual + structural tree optimization via agy (--add-dir) forbidding fixed session labels."""
    return suggest_tree_optimizations(
        folder_summaries,
        cli_command=cli_command,
        model=model,
        timeout=timeout,
        cancel_event=cancel_event,
        forbid_fixed_labels=True,
        people_dir=people_dir,
        face_clusters_dir=face_clusters_dir,
        data_dir=data_dir,
    )


_QUOTA_OR_AUTH_KEYWORDS = (
    "429",
    "resource_exhausted",
    "quota",
    "cuota",
    "token",
    "rate limit",
    "ratelimit",
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "auth",
    "exceeded",
    "limit",
)


def _is_quota_or_auth_error(raw: str) -> bool:
    low = (raw or "").lower()
    return any(kw in low for kw in _QUOTA_OR_AUTH_KEYWORDS)


def _extract_json_dict(stdout: str) -> dict[str, Any] | None:
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
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _invoke_agy_json(
    argv: list[str],
    *,
    timeout: float = 60.0,
) -> tuple[dict[str, Any] | None, str | None, bool]:
    """Run agy subprocess and return (parsed_dict, error_reason, is_quota_or_auth)."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        msg = _summarize_failure_reason(str(exc)) or "error ejecutando agy"
        return None, msg, _is_quota_or_auth_error(msg)

    if proc.returncode != 0:
        raw_err = proc.stderr or proc.stdout or ""
        detail = _summarize_failure_reason(raw_err)
        reason = f"código {proc.returncode}: {detail}" if detail else f"código {proc.returncode}"
        return None, reason, _is_quota_or_auth_error(raw_err)

    if not proc.stdout or not proc.stdout.strip():
        return None, "salida vacía de agy", False

    parsed = _extract_json_dict(proc.stdout)
    if parsed is None:
        return None, "salida JSON inválida de agy", False
    return parsed, None, False


def debate_folder_batch_with_agy(
    batch_summaries: list[dict[str, Any]],
    *,
    cli_command: str = "agy",
    model: str = DEFAULT_AGY_MODEL,
    timeout: float = 60.0,
    people_dir: Path | str | None = None,
    face_clusters_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run 2-role consensus debate (Curador -> Auditor) on a batch of up to 5 folders."""
    if not batch_summaries:
        return {"ok": True, "decisions": [], "history": []}

    cmd_bin = (cli_command or "agy").strip() or "agy"
    resolved_bin = resolve_cli_binary(cmd_bin, allowed_names=AGY_ALLOWED_NAMES)
    selected_model = (model or DEFAULT_AGY_MODEL).strip() or DEFAULT_AGY_MODEL
    can_call_agy = resolved_bin is not None or (
        subprocess.run is _ORIG_SUBPROCESS_RUN and is_standard_agy_binary(cmd_bin)
    )
    if not can_call_agy:
        return {
            "ok": False,
            "paused": True,
            "is_quota_or_auth": True,
            "error_reason": f"binario '{cmd_bin}' no encontrado",
        }

    resolved_fc_dir = _resolve_face_clusters_dir(face_clusters_dir, data_dir, people_dir)
    has_clusters, whitelisted_clusters = _load_whitelisted_face_clusters(resolved_fc_dir)
    legacy_people_labels = [] if has_clusters else _load_people_labels(people_dir)

    tmp_dir = tempfile.mkdtemp(prefix="agy_debate_")
    tmp_root = Path(tmp_dir)
    try:
        compact: list[dict[str, Any]] = []
        has_prepared = False
        batch_confirmed_people: list[str] = []
        seen_batch_people: set[str] = set()
        for f_idx, f in enumerate(batch_summaries[:5]):
            rep_srcs = _select_representative_thumb_paths(f, limit=3)
            prepared_paths: list[str] = []
            for t_idx, src_p in enumerate(rep_srcs):
                dst_p = tmp_root / f"folder_{f_idx:02d}_scene_{t_idx + 1}.jpg"
                _prepare_temp_thumb(src_p, dst_p, max_dim=320)
                prepared_paths.append(str(dst_p))
                has_prepared = True

            candidate_paths = _collect_candidate_image_paths(f)
            face_crops = extract_top_face_crops(
                candidate_paths,
                tmp_root,
                max_faces=4,
                file_prefix=f"folder_{f_idx:02d}_face",
            )
            if face_crops:
                has_prepared = True

            confirmed_people = (
                resolve_folder_whitelisted_people(f.get("media_ids"), whitelisted_clusters)
                if has_clusters
                else []
            )
            for pname in confirmed_people:
                if pname not in seen_batch_people:
                    seen_batch_people.add(pname)
                    batch_confirmed_people.append(pname)

            compact_entry: dict[str, Any] = {
                "folder": f.get("folder"),
                "count": f.get("count"),
                "start_date": f.get("start_date"),
                "end_date": f.get("end_date"),
                "location": f.get("location_label"),
                "thumbnails": prepared_paths if prepared_paths else ([f.get("thumb_path")] if f.get("thumb_path") else []),
                "face_crops": [str(p) for p in face_crops],
            }
            if confirmed_people:
                compact_entry["personas_confirmadas"] = confirmed_people
            compact.append(compact_entry)

        extra_dirs = [tmp_root] if has_prepared else []
        add_dirs = _collect_add_dirs(None if has_clusters else people_dir, extra_dirs)
        if has_clusters:
            people_hint = (
                f" Personas confirmadas presentes en estas carpetas: {', '.join(batch_confirmed_people)}."
                if batch_confirmed_people
                else ""
            )
        else:
            people_hint = (
                f" Personas clave disponibles: {', '.join(legacy_people_labels)}."
                if legacy_people_labels
                else ""
            )

        def _build_argv(prompt_text: str) -> list[str]:
            argv = [resolved_bin or cmd_bin, "-m", selected_model]
            for add_d in add_dirs:
                argv.extend(["--add-dir", str(add_d)])
            argv.extend(["-p", prompt_text])
            return argv

        # Role 1: Agente Curador
        curador_prompt = (
            "Actúa como Agente Curador de fotografías. Analiza las miniaturas de escena y rostros "
            "detectados para proponer nombres buscables en español con formato "
            "'Evento + Personas Clave + Lugar'. "
            f"{people_hint} "
            "Responde ÚNICAMENTE con JSON válido: "
            '{"proposals": [{"folder": "...", "proposed_name": "...", "reason": "..."}]}. '
            "Carpetas: " + json.dumps(compact, ensure_ascii=False)
        )
        curador_json, err1, is_q1 = _invoke_agy_json(_build_argv(curador_prompt), timeout=timeout)
        if curador_json is None:
            return {
                "ok": False,
                "paused": True,
                "is_quota_or_auth": is_q1,
                "error_reason": err1 or "Error en Agente Curador",
            }

        proposals = curador_json.get("proposals")
        if not isinstance(proposals, list):
            proposals = []

        # Role 2: Agente Auditor
        auditor_prompt = (
            "Actúa como Agente Auditor. Revisa las propuestas del Agente Curador, corrige nombres "
            "genéricos difíciles de buscar en Google Drive, sugiere fusiones de carpetas consecutivas "
            "relacionadas (merge_with) y consolida el nombre final en español. "
            "Responde ÚNICAMENTE con JSON válido: "
            '{"decisions": [{"folder": "...", "final_name": "...", "merge_with": null, "reason": "..."}]}. '
            "Propuestas del Curador: " + json.dumps(proposals, ensure_ascii=False) + " "
            "Carpetas: " + json.dumps(compact, ensure_ascii=False)
        )
        auditor_json, err2, is_q2 = _invoke_agy_json(_build_argv(auditor_prompt), timeout=timeout)
        if auditor_json is None:
            return {
                "ok": False,
                "paused": True,
                "is_quota_or_auth": is_q2,
                "error_reason": err2 or "Error en Agente Auditor",
            }

        proposals_by_folder = {
            str(p.get("folder")): p
            for p in proposals
            if isinstance(p, dict) and p.get("folder")
        }
        raw_decisions = auditor_json.get("decisions")
        if not isinstance(raw_decisions, list):
            raw_decisions = []

        decisions_by_folder: dict[str, dict[str, Any]] = {}
        for d in raw_decisions:
            if isinstance(d, dict) and d.get("folder"):
                decisions_by_folder[str(d["folder"])] = d

        final_decisions: list[dict[str, Any]] = []
        for f in batch_summaries[:5]:
            f_path = str(f.get("folder") or "")
            prop = proposals_by_folder.get(f_path, {})
            dec = decisions_by_folder.get(f_path, {})
            raw_name = str(
                dec.get("final_name")
                or prop.get("proposed_name")
                or ""
            ).strip()
            clean_name = _clean_label(raw_name)
            merge_with = dec.get("merge_with")
            if isinstance(merge_with, str):
                merge_with = merge_with.strip() or None
            else:
                merge_with = None
            final_decisions.append(
                {
                    "folder": f_path,
                    "final_name": clean_name,
                    "merge_with": merge_with,
                    "curador": prop.get("proposed_name"),
                    "auditor": clean_name,
                    "reason": str(dec.get("reason") or prop.get("reason") or "Consenso Curador + Auditor"),
                }
            )

        return {
            "ok": True,
            "decisions": final_decisions,
            "engine": f"agy · {selected_model}",
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
