"""Temporal/spatial event clustering, offline reverse geocoding, and deterministic naming."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from photo_organizer.config import VIDEO_EXTENSIONS

try:  # Optional offline reverse_geocoder
    import reverse_geocoder as _rg  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _rg = None

SCREENSHOT_RE = re.compile(r"(screenshot|captura|pantallazo|screen[-_\s]?shot|img[-_]?e\d+)", re.IGNORECASE)
WHATSAPP_RE = re.compile(r"(whatsapp|img-\d{8}-wa\d+|vid-\d{8}-wa\d+|[-_]wa\d{3,})", re.IGNORECASE)
DOCUMENT_RE = re.compile(r"(scan|receipt|factura|ticket|documento|doc[-_]\d+)", re.IGNORECASE)

_TINY_IMAGE_BYTES = 15 * 1024
_GPS_INHERIT_WINDOW_SEC = 2 * 3600
_NIGHT_MERGE_GAP_HOURS = 6.0
_MIN_BASE_ZONE_DAYS = 10
_LIVE_PHOTO_MAX_VIDEO_SEC = 3.0

_KNOWN_CITIES: tuple[tuple[float, float, str], ...] = (
    (19.4326, -99.1332, "Ciudad de México, MX"),
    (17.0732, -96.7266, "Oaxaca, MX"),
    (20.6597, -103.3496, "Guadalajara, MX"),
    (25.6866, -100.3161, "Monterrey, MX"),
    (21.1619, -86.8515, "Cancún, MX"),
    (19.0414, -98.2063, "Puebla, MX"),
    (20.9674, -89.5926, "Mérida, MX"),
    (40.4168, -3.7038, "Madrid, ES"),
    (41.3874, 2.1686, "Barcelona, ES"),
    (40.7128, -74.0060, "New York, US"),
    (48.8566, 2.3522, "Paris, FR"),
    (51.5072, -0.1276, "London, GB"),
    (35.6762, 139.6503, "Tokyo, JP"),
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two WGS84 coordinates."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def offline_reverse_geocode(lat: float | None, lon: float | None) -> str | None:
    """Resolve city/region offline via reverse_geocoder or built-in city fallback."""
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
    best_city: str | None = None
    best_dist = 60.0
    for c_lat, c_lon, c_name in _KNOWN_CITIES:
        dist = haversine_km(lat, lon, c_lat, c_lon)
        if dist < best_dist:
            best_dist = dist
            best_city = c_name
    if best_city is not None:
        return best_city
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
    file_size: int | None = None
    duration_sec: float | None = None


def _parse_dt(iso_str: str | None) -> datetime | None:
    if not iso_str:
        return None
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_tiny_or_sticker(item: ClusterItem) -> bool:
    ext_low = item.ext.lower()
    if ext_low in VIDEO_EXTENSIONS:
        return False
    if ext_low == ".webp" and not item.has_camera_exif:
        return True
    if item.file_size is not None and 256 <= item.file_size < _TINY_IMAGE_BYTES:
        return True
    return False


def _same_zone(
    loc_a: str | None,
    coords_a: tuple[float, float] | None,
    loc_b: str | None,
    coords_b: tuple[float, float] | None,
    gps_distance_km: float,
) -> bool:
    if coords_a is not None and coords_b is not None:
        return haversine_km(coords_a[0], coords_a[1], coords_b[0], coords_b[1]) <= gps_distance_km
    if loc_a and loc_b:
        return loc_a == loc_b
    return True


def _clean_city_name(location_label: str) -> str:
    return location_label.split(",")[0].strip() or location_label


def _assign_deterministic_filenames(
    items: list[ClusterItem],
    staged: dict[str, dict[str, Any]],
) -> None:
    """Assign deterministic YYYY-MM-DD_HH-MM-SS_###.ext with Live/Motion photo pairing."""
    by_folder_stamp: dict[tuple[str, str], list[ClusterItem]] = {}
    ordered_all = sorted(
        items,
        key=lambda it: (it.captured_at or "", it.sha256 or "", it.source_path),
    )
    for item in ordered_all:
        folder = staged[item.id]["target_folder"]
        dt = _parse_dt(item.captured_at)
        stamp = dt.strftime("%Y-%m-%d_%H-%M-%S") if dt else "0000-00-00_00-00-00"
        by_folder_stamp.setdefault((folder, stamp), []).append(item)

    for (_folder, stamp), group in by_folder_stamp.items():
        images = [it for it in group if it.ext.lower() not in VIDEO_EXTENSIONS]
        short_videos = [
            it
            for it in group
            if it.ext.lower() in VIDEO_EXTENSIONS
            and it.duration_sec is not None
            and it.duration_sec <= _LIVE_PHOTO_MAX_VIDEO_SEC
        ]
        other_videos = [
            it
            for it in group
            if it.ext.lower() in VIDEO_EXTENSIONS
            and (it.duration_sec is None or it.duration_sec > _LIVE_PHOTO_MAX_VIDEO_SEC)
        ]

        counter = 0
        paired_video_ids: set[str] = set()
        short_vid_queue = list(short_videos)

        for img in images:
            counter += 1
            staged[img.id]["target_filename"] = f"{stamp}_{counter:03d}{img.ext}"
            if short_vid_queue:
                vid = short_vid_queue.pop(0)
                paired_video_ids.add(vid.id)
                staged[vid.id]["target_filename"] = f"{stamp}_{counter:03d}{vid.ext}"

        for vid in short_vid_queue + other_videos:
            if vid.id in paired_video_ids:
                continue
            counter += 1
            staged[vid.id]["target_filename"] = f"{stamp}_{counter:03d}{vid.ext}"


def cluster_and_stage_items(
    items: list[ClusterItem],
    time_gap_hours: float = 4.0,
    gps_distance_km: float = 25.0,
) -> dict[str, dict[str, Any]]:
    """Assign target_folder ('01_Recuerdos/Año/...') and deterministic target_filename."""
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

        dt = _parse_dt(item.captured_at)
        if _is_tiny_or_sticker(item):
            ym_folder = dt.strftime("%Y-%m") if dt else "Sin_Fecha"
            staged[item.id] = {
                "target_folder": f"02_Capturas_y_WhatsApp/{ym_folder}",
                "location_label": None,
                "route_reason": "classified_sticker_or_tiny",
            }
            continue

        special = is_screenshot_or_document(fname)
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

    # Sort dated items chronologically for deterministic processing
    dated_items.sort(key=lambda pair: (pair[0], pair[1].sha256, pair[1].source_path))

    # Rule 1: GPS location inheritance within +/- 2 hours on the same day
    eff_coords: dict[str, tuple[float, float] | None] = {}
    eff_loc: dict[str, str | None] = {}
    gps_anchors: list[tuple[datetime, ClusterItem, str | None]] = []

    for dt, item in dated_items:
        if item.latitude is not None and item.longitude is not None:
            loc = offline_reverse_geocode(item.latitude, item.longitude)
            eff_coords[item.id] = (item.latitude, item.longitude)
            eff_loc[item.id] = loc
            gps_anchors.append((dt, item, loc))
        else:
            eff_coords[item.id] = None
            eff_loc[item.id] = None

    for dt, item in dated_items:
        if eff_coords[item.id] is not None:
            continue
        best_anchor: tuple[float, datetime, ClusterItem, str | None] | None = None
        for a_dt, a_item, a_loc in gps_anchors:
            if a_dt.date() != dt.date():
                continue
            diff_sec = abs((a_dt - dt).total_seconds())
            if diff_sec <= _GPS_INHERIT_WINDOW_SEC:
                if best_anchor is None or diff_sec < best_anchor[0]:
                    best_anchor = (diff_sec, a_dt, a_item, a_loc)
        if best_anchor is not None:
            _, _, a_item, a_loc = best_anchor
            eff_coords[item.id] = (a_item.latitude, a_item.longitude)  # type: ignore[arg-type]
            eff_loc[item.id] = a_loc

    # Rule 2: Temporal/spatial clustering with night-event merging (< 6h across midnight)
    clusters: list[dict[str, Any]] = []
    for dt, item in dated_items:
        coords = eff_coords[item.id]
        loc = eff_loc[item.id]

        if not clusters:
            clusters.append(
                {
                    "start_dt": dt,
                    "end_dt": dt,
                    "start_date": dt.date(),
                    "end_date": dt.date(),
                    "location_label": loc,
                    "coords": coords,
                    "items": [item],
                }
            )
            continue

        cur = clusters[-1]
        prev_dt: datetime = cur["end_dt"]
        gap_hours = (dt - prev_dt).total_seconds() / 3600.0
        same_zone = _same_zone(
            cur["location_label"],
            cur["coords"],
            loc,
            coords,
            gps_distance_km,
        )

        merge_into_cur = False
        if dt.date() == prev_dt.date():
            if gap_hours <= time_gap_hours and same_zone:
                merge_into_cur = True
        else:
            if gap_hours < _NIGHT_MERGE_GAP_HOURS and same_zone:
                merge_into_cur = True

        if merge_into_cur:
            cur["items"].append(item)
            cur["end_dt"] = dt
            cur["end_date"] = dt.date()
            if cur["location_label"] is None and loc is not None:
                cur["location_label"] = loc
            if cur["coords"] is None and coords is not None:
                cur["coords"] = coords
        else:
            clusters.append(
                {
                    "start_dt": dt,
                    "end_dt": dt,
                    "start_date": dt.date(),
                    "end_date": dt.date(),
                    "location_label": loc,
                    "coords": coords,
                    "items": [item],
                }
            )

    # Rule 3: Calculate base zone when >= 10 distinct GPS days exist
    gps_days: set[date] = set()
    loc_counts: Counter[str] = Counter()
    for dt, item in dated_items:
        loc = eff_loc[item.id]
        if eff_coords[item.id] is not None:
            gps_days.add(dt.date())
        if loc and not loc.startswith("GPS "):
            loc_counts[loc] += 1

    base_zone: str | None = None
    if len(gps_days) >= _MIN_BASE_ZONE_DAYS and loc_counts:
        base_zone = loc_counts.most_common(1)[0][0]

    # Group 2+ consecutive days outside base_zone into multi-day trips
    final_groups: list[dict[str, Any]] = []
    idx = 0
    while idx < len(clusters):
        c = clusters[idx]
        c_loc = c["location_label"]
        is_away = bool(
            base_zone is not None
            and c_loc is not None
            and not c_loc.startswith("GPS ")
            and c_loc != base_zone
        )
        if is_away:
            trip_clusters = [c]
            j = idx + 1
            while j < len(clusters):
                nxt = clusters[j]
                nxt_loc = nxt["location_label"]
                day_gap = (nxt["start_date"] - trip_clusters[-1]["end_date"]).days
                if nxt_loc == c_loc and day_gap <= 1:
                    trip_clusters.append(nxt)
                    j += 1
                else:
                    break
            span_days = (trip_clusters[-1]["end_date"] - trip_clusters[0]["start_date"]).days + 1
            if span_days >= 2 and len(trip_clusters) >= 2:
                trip_items = [it for tc in trip_clusters for it in tc["items"]]
                final_groups.append(
                    {
                        "kind": "trip",
                        "start_dt": trip_clusters[0]["start_dt"],
                        "end_dt": trip_clusters[-1]["end_dt"],
                        "location_label": c_loc,
                        "items": trip_items,
                    }
                )
                idx = j
                continue

        final_groups.append(
            {
                "kind": "event",
                "start_dt": c["start_dt"],
                "end_dt": c["end_dt"],
                "location_label": c_loc,
                "items": c["items"],
            }
        )
        idx += 1

    # Rule 4: Assign target_folder, consolidating sparse (<3) local events into Momentos Varios
    for grp in final_groups:
        start_dt: datetime = grp["start_dt"]
        end_dt: datetime = grp["end_dt"]
        loc_label: str | None = grp["location_label"]
        grp_items: list[ClusterItem] = grp["items"]
        year = start_dt.strftime("%Y")

        if grp["kind"] == "trip":
            start_ymd = start_dt.strftime("%Y-%m-%d")
            end_md = end_dt.strftime("%m-%d")
            city_name = _clean_city_name(loc_label or "Destino")
            safe_city = re.sub(r'[<>:"/\\|?*]', "_", city_name).strip()
            target_folder = f"01_Recuerdos/{year}/{start_ymd} al {end_md} - Viaje a {safe_city}"
            route_reason = "multiday_trip"
        else:
            is_local = base_zone is not None and (loc_label is None or loc_label == base_zone)
            has_ai_override = any(it.ai_label for it in grp_items)
            if is_local and len(grp_items) < 3 and not has_ai_override:
                ym = start_dt.strftime("%Y-%m")
                target_folder = f"01_Recuerdos/{year}/{ym} - Momentos Varios"
                route_reason = "momentos_varios"
            else:
                ymd = start_dt.strftime("%Y-%m-%d")
                ai_lbl = next((it.ai_label for it in grp_items if it.ai_label), None)
                loc_event = loc_label if (loc_label and not loc_label.startswith("GPS ")) else None
                event_name = ai_lbl or loc_event or "Eventos del día"
                safe_event = re.sub(r'[<>:"/\\|?*]', "_", event_name).strip()
                target_folder = f"01_Recuerdos/{year}/{ymd} - {safe_event}"
                route_reason = "clustered"

        for it in grp_items:
            staged[it.id] = {
                "target_folder": target_folder,
                "location_label": eff_loc.get(it.id, loc_label),
                "route_reason": route_reason,
            }

    _assign_deterministic_filenames(items, staged)
    return staged
