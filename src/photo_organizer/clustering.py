"""Temporal/spatial event clustering, offline reverse geocoding, and deterministic naming."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:  # Optional offline reverse_geocoder
    import reverse_geocoder as _rg  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _rg = None

SCREENSHOT_RE = re.compile(r"(screenshot|captura|pantallazo|screen[-_\s]?shot|img[-_]?e\d+)", re.IGNORECASE)
WHATSAPP_RE = re.compile(r"(whatsapp|img-\d{8}-wa\d+|vid-\d{8}-wa\d+|[-_]wa\d{3,})", re.IGNORECASE)
DOCUMENT_RE = re.compile(r"(scan|receipt|factura|ticket|documento|doc[-_]\d+)", re.IGNORECASE)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two WGS84 coordinates."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def offline_reverse_geocode(lat: float | None, lon: float | None) -> str | None:
    """Resolve city/region offline via reverse_geocoder when installed."""
    if lat is None or lon is None:
        return None
    if _rg is not None:
        try:
            matches = _rg.search((lat, lon), mode=1)
            if matches:
                city = matches[0].get("name", "")
                cc = matches[0].get("cc", "")
                label = f"{city}, {cc}".strip(", ")
                if label:
                    return label
        except Exception:
            pass
    return f"GPS {lat:.2f},{lon:.2f}"


def is_screenshot_or_document(filename: str) -> str | None:
    """Detect screenshots, WhatsApp forwards, or documents by filename heuristics."""
    if SCREENSHOT_RE.search(filename):
        return "Capturas"
    if WHATSAPP_RE.search(filename):
        return "WhatsApp"
    if DOCUMENT_RE.search(filename):
        return "Documentos"
    return None


@dataclass
class ClusterItem:
    id: str
    source_path: str
    ext: str
    sha256: str
    is_duplicate: bool
    captured_at: str | None
    latitude: float | None
    longitude: float | None
    route_reason: str | None = None
    ai_label: str | None = None
    has_camera_exif: bool = False


def _parse_dt(iso_str: str | None) -> datetime | None:
    if not iso_str:
        return None
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def cluster_and_stage_items(
    items: list[ClusterItem],
    time_gap_hours: float = 4.0,
    gps_distance_km: float = 25.0,
) -> dict[str, dict[str, Any]]:
    """Assign target_folder ('Año / AAAA-MM - Evento') and deterministic target_filename."""
    staged: dict[str, dict[str, Any]] = {}

    dated_items: list[tuple[datetime, ClusterItem]] = []
    for item in items:
        fname = Path(item.source_path).name
        if item.is_duplicate:
            staged[item.id] = {
                "target_folder": "_Duplicados",
                "location_label": None,
                "route_reason": "duplicate_sha256",
            }
            continue

        if item.route_reason == "_Revision_Manual":
            staged[item.id] = {
                "target_folder": "_Revision_Manual",
                "location_label": None,
                "route_reason": "ambiguous_sidecar",
            }
            continue

        special = is_screenshot_or_document(fname)
        dt = _parse_dt(item.captured_at)
        if special in ("Capturas", "WhatsApp") and not item.has_camera_exif:
            ym_folder = dt.strftime("%Y-%m") if dt else "Sin_Fecha"
            staged[item.id] = {
                "target_folder": f"02_Capturas_y_WhatsApp/{ym_folder}",
                "location_label": None,
                "route_reason": f"classified_{special.lower()}",
            }
            continue
        if special == "Documentos":
            staged[item.id] = {
                "target_folder": "_Revision_Manual/Documentos",
                "location_label": None,
                "route_reason": "classified_documentos",
            }
            continue

        if dt is None:
            staged[item.id] = {
                "target_folder": "_Revision_Manual",
                "location_label": None,
                "route_reason": "undated",
            }
            continue

        dated_items.append((dt, item))

    # Sort dated items chronologically for temporal/spatial clustering
    dated_items.sort(key=lambda pair: (pair[0], pair[1].sha256, pair[1].source_path))

    prev_dt: datetime | None = None
    prev_coords: tuple[float, float] | None = None
    cluster_idx = 0
    cluster_location: str | None = None

    for dt, item in dated_items:
        new_cluster = False
        if prev_dt is None:
            new_cluster = True
        elif dt.date() != prev_dt.date():
            new_cluster = True
        else:
            gap_hours = (dt - prev_dt).total_seconds() / 3600.0
            if gap_hours > time_gap_hours:
                new_cluster = True
            elif (
                prev_coords is not None
                and item.latitude is not None
                and item.longitude is not None
                and haversine_km(prev_coords[0], prev_coords[1], item.latitude, item.longitude)
                > gps_distance_km
            ):
                new_cluster = True

        if new_cluster:
            cluster_idx += 1
            loc = offline_reverse_geocode(item.latitude, item.longitude)
            cluster_location = None if (loc and loc.startswith("GPS ")) else loc

        year = dt.strftime("%Y")
        ymd = dt.strftime("%Y-%m-%d")
        event_name = item.ai_label or cluster_location or "Eventos del día"
        safe_event = re.sub(r'[<>:"/\\|?*]', "_", event_name).strip()
        target_folder = f"01_Recuerdos/{year}/{ymd} - {safe_event}"

        staged[item.id] = {
            "target_folder": target_folder,
            "location_label": cluster_location,
            "route_reason": "clustered",
        }
        prev_dt = dt
        if item.latitude is not None and item.longitude is not None:
            prev_coords = (item.latitude, item.longitude)

    # Deterministic collision numbering per folder ordered by (captured_at, sha256, source_path)
    ordered_all = sorted(
        items,
        key=lambda it: (it.captured_at or "", it.sha256 or "", it.source_path),
    )
    counters: dict[tuple[str, str], int] = {}
    for item in ordered_all:
        folder = staged[item.id]["target_folder"]
        dt = _parse_dt(item.captured_at)
        stamp = dt.strftime("%Y-%m-%d_%H-%M-%S") if dt else "0000-00-00_00-00-00"
        key = (folder, stamp)
        counters[key] = counters.get(key, 0) + 1
        staged[item.id]["target_filename"] = f"{stamp}_{counters[key]:03d}{item.ext}"

    return staged
