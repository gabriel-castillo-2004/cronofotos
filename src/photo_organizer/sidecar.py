"""Multi-pass Google Takeout .json sidecar resolver and metadata merger."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRUNCATE_LIMIT = 46


def build_json_index(folder: Path) -> dict[str, Path]:
    """Index all .json sidecars in a directory by exact filename."""
    if not folder.is_dir():
        return {}
    return {
        p.name: p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() == ".json"
    }


def _read_sidecar_json(json_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _extract_sidecar_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Extract photoTakenTime and geoDataExif/geoData without mutating source files."""
    captured_at: str | None = None
    taken = data.get("photoTakenTime") or data.get("creationTime")
    if isinstance(taken, dict) and taken.get("timestamp") is not None:
        try:
            ts = int(taken["timestamp"])
            if ts > 0:
                captured_at = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
        except (TypeError, ValueError, OSError):
            pass

    lat: float | None = None
    lon: float | None = None
    for geo_key in ("geoDataExif", "geoData"):
        geo = data.get(geo_key)
        if isinstance(geo, dict):
            try:
                g_lat = float(geo.get("latitude", 0.0))
                g_lon = float(geo.get("longitude", 0.0))
                if g_lat != 0.0 or g_lon != 0.0:
                    lat, lon = g_lat, g_lon
                    break
            except (TypeError, ValueError):
                continue

    return {
        "title": data.get("title"),
        "captured_at": captured_at,
        "latitude": lat,
        "longitude": lon,
        "description": data.get("description") or None,
    }


def resolve_sidecar(
    media_path: Path,
    json_index: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Multi-pass Takeout sidecar resolver cross-checked against JSON 'title'."""
    if json_index is None:
        json_index = build_json_index(media_path.parent)

    name = media_path.name
    candidates: list[Path] = []

    # Pass 1: exact match (<filename>.json)
    exact_name = f"{name}.json"
    if exact_name in json_index:
        candidates.append(json_index[exact_name])

    # Pass 2: supplemental-metadata variants
    for supp_tag in ("supplemental-metadata", "supplemental-metadat"):
        supp_name = f"{name}.{supp_tag}.json"
        if supp_name in json_index and json_index[supp_name] not in candidates:
            candidates.append(json_index[supp_name])

    # Pass 3: 46-char Takeout truncation
    trunc_name = f"{name[:TRUNCATE_LIMIT]}.json"
    if trunc_name in json_index and json_index[trunc_name] not in candidates:
        candidates.append(json_index[trunc_name])

    # Pass 4: '(N)' suffix alignment
    match = re.match(r"^(.*?)(\(\d+\))(\.[^.]+)$", name)
    if match:
        base, num, ext = match.groups()
        for variant in (f"{base}{ext}{num}.json", f"{base}{num}{ext}.json"):
            if variant in json_index and json_index[variant] not in candidates:
                candidates.append(json_index[variant])

    verified: list[tuple[Path, dict[str, Any]]] = []
    for cand in candidates:
        data = _read_sidecar_json(cand)
        if data is not None and str(data.get("title", "")) == name:
            verified.append((cand, data))

    if len(verified) == 1:
        cand_path, cand_data = verified[0]
        return {
            "route": "matched",
            "sidecar_path": str(cand_path),
            "sidecar_name": cand_path.name,
            "metadata": _extract_sidecar_fields(cand_data),
        }

    if len(verified) > 1 or (len(candidates) > 0 and len(verified) == 0):
        return {
            "route": "_Revision_Manual",
            "sidecar_path": None,
            "sidecar_name": None,
            "metadata": {},
        }

    return {
        "route": "unmatched",
        "sidecar_path": None,
        "sidecar_name": None,
        "metadata": {},
    }


def merge_metadata(
    embedded: dict[str, Any],
    sidecar_meta: dict[str, Any],
) -> dict[str, Any]:
    """Merge embedded EXIF/QuickTime metadata with Takeout sidecar metadata."""
    merged = dict(embedded)
    if not merged.get("captured_at") and sidecar_meta.get("captured_at"):
        merged["captured_at"] = sidecar_meta["captured_at"]
    if merged.get("latitude") is None and sidecar_meta.get("latitude") is not None:
        merged["latitude"] = sidecar_meta["latitude"]
    if merged.get("longitude") is None and sidecar_meta.get("longitude") is not None:
        merged["longitude"] = sidecar_meta["longitude"]
    if not merged.get("description") and sidecar_meta.get("description"):
        merged["description"] = sidecar_meta["description"]
    return merged
