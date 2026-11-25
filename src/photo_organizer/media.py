"""Thread-confined ExifTool '-stay_open' worker pool, ffmpeg wrapper, and Python fallback."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from photo_organizer.config import VIDEO_EXTENSIONS, resolve_cli_binary

try:  # Optional Pillow / pillow-heif / piexif support
    from PIL import ExifTags, Image
except ImportError:  # pragma: no cover
    ExifTags = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]

try:
    import pillow_heif  # type: ignore[import-not-found]

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pillow_heif = None  # type: ignore[assignment]

try:
    import piexif  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    piexif = None  # type: ignore[assignment]


@dataclass
class BinaryHealthStatus:
    """Health check result for external exiftool and ffmpeg binaries."""

    exiftool_path: str | None
    ffmpeg_path: str | None
    exiftool_available: bool
    ffmpeg_available: bool
    warnings: list[str] = field(default_factory=list)


class BinaryHealthCheck:
    """Detect exiftool and ffmpeg in PATH or custom Windows locations."""

    @staticmethod
    def resolve_binary(name: str, custom_path: str | None = None) -> str | None:
        return resolve_cli_binary(name, custom_path)

    @classmethod
    def check(
        cls,
        exiftool_custom: str | None = None,
        ffmpeg_custom: str | None = None,
    ) -> BinaryHealthStatus:
        exif = cls.resolve_binary("exiftool", exiftool_custom)
        ffmp = cls.resolve_binary("ffmpeg", ffmpeg_custom)
        warnings: list[str] = []
        if not exif:
            warnings.append("exiftool binary missing; using Pillow/piexif fallback")
        if not ffmp:
            warnings.append("ffmpeg binary missing; video metadata/thumbnail fallback active")
        return BinaryHealthStatus(
            exiftool_path=exif,
            ffmpeg_path=ffmp,
            exiftool_available=exif is not None,
            ffmpeg_available=ffmp is not None,
            warnings=warnings,
        )


class ExifToolProcess:
    """Single thread-confined ExifTool '-stay_open True -@ -' process."""

    MAX_FILES_PER_PROCESS = 1000

    def __init__(self, binary_path: str, timeout_sec: float = 15.0) -> None:
        self.binary_path = binary_path
        self.timeout_sec = timeout_sec
        self.thread_id = threading.get_ident()
        self._proc: subprocess.Popen[str] | None = None
        self._processed_count = 0

    def _ensure_started(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            if self._processed_count < self.MAX_FILES_PER_PROCESS:
                return self._proc
            self.close()

        resolved_bin = resolve_cli_binary(self.binary_path)
        if not resolved_bin:
            raise FileNotFoundError(f"ExifTool executable not found: {self.binary_path}")
        self._proc = subprocess.Popen(
            [resolved_bin, "-stay_open", "True", "-@", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._processed_count = 0
        return self._proc

    def execute(self, *args: str) -> str:
        """Run one command against the stay_open process using {ready} sentinel framing."""
        proc = self._ensure_started()
        assert proc.stdin is not None and proc.stdout is not None
        cmd_lines = [*args, "-execute\n"]
        proc.stdin.write("\n".join(cmd_lines))
        proc.stdin.flush()

        lines: list[str] = []
        result_holder: list[str] = []
        err_holder: list[BaseException] = []

        def _reader() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    if line.strip() == "{ready}":
                        result_holder.append("".join(lines))
                        return
                    lines.append(line)
            except BaseException as exc:  # pragma: no cover
                err_holder.append(exc)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout=self.timeout_sec)
        if t.is_alive():
            self.close()
            raise TimeoutError(f"ExifTool timed out after {self.timeout_sec}s")
        if err_holder:
            self.close()
            raise err_holder[0]

        self._processed_count += 1
        return result_holder[0] if result_holder else ""

    def read_json(self, path: Path) -> dict[str, Any]:
        raw = self.execute("-j", "-n", "-charset", "filename=UTF8", str(path))
        if not raw.strip():
            return {}
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        return {}

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.write("-stay_open\nFalse\n")
                    self._proc.stdin.flush()
            except OSError:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3.0)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()
            finally:
                self._proc = None
                self._processed_count = 0


def _parse_exif_datetime(raw: Any) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    cleaned = raw.strip().split("+")[0].split(".")[0]
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    return None


def _read_metadata_python_fallback(path: Path) -> dict[str, Any]:
    """Read basic image EXIF using Pillow/piexif when ExifTool is absent."""
    result: dict[str, Any] = {
        "SourceFile": str(path),
        "Extractor": "pillow_piexif_fallback",
        "captured_at": None,
        "latitude": None,
        "longitude": None,
        "width": None,
        "height": None,
    }
    if Image is None or path.suffix.lower() in VIDEO_EXTENSIONS:
        return result

    try:
        with Image.open(path) as img:
            result["width"], result["height"] = img.size
            exif = img.getexif()
            if exif and ExifTags is not None:
                tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
                for dt_key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                    if dt_key in tag_map:
                        parsed = _parse_exif_datetime(str(tag_map[dt_key]))
                        if parsed:
                            result["captured_at"] = parsed
                            break
    except Exception:
        pass
    return result


class MediaMetadataService:
    """Thread-confined ExifTool pool with Pillow/pillow-heif/piexif fallback."""

    def __init__(
        self,
        exiftool_bin: str | None = None,
        ffmpeg_bin: str | None = None,
        timeout_sec: float = 15.0,
    ) -> None:
        health = BinaryHealthCheck.check(exiftool_bin, ffmpeg_bin)
        self.exiftool_bin = health.exiftool_path
        self.ffmpeg_bin = health.ffmpeg_path
        self.health_warnings = health.warnings
        self.timeout_sec = timeout_sec
        self._local = threading.local()
        self._processes: list[ExifToolProcess] = []
        self._lock = threading.Lock()

    def _get_exiftool(self) -> ExifToolProcess:
        proc = getattr(self._local, "exiftool", None)
        if proc is None:
            assert self.exiftool_bin is not None
            proc = ExifToolProcess(self.exiftool_bin, timeout_sec=self.timeout_sec)
            self._local.exiftool = proc
            with self._lock:
                self._processes.append(proc)
        return proc

    def read_metadata(self, path: Path) -> dict[str, Any]:
        """Extract normalized metadata from an image or video without mutating source."""
        if self.exiftool_bin:
            try:
                proc = self._get_exiftool()
                raw = proc.read_json(path)
                captured_at = None
                for key in (
                    "DateTimeOriginal",
                    "CreateDate",
                    "MediaCreateDate",
                    "ModifyDate",
                ):
                    captured_at = _parse_exif_datetime(raw.get(key))
                    if captured_at:
                        break
                lat = raw.get("GPSLatitude")
                lon = raw.get("GPSLongitude")
                return {
                    "SourceFile": str(path),
                    "Extractor": "exiftool",
                    "captured_at": captured_at,
                    "latitude": float(lat) if isinstance(lat, (int, float)) else None,
                    "longitude": float(lon) if isinstance(lon, (int, float)) else None,
                    "width": raw.get("ImageWidth"),
                    "height": raw.get("ImageHeight"),
                    "make": raw.get("Make"),
                    "model": raw.get("Model"),
                }
            except FileNotFoundError:
                self.exiftool_bin = None
            except Exception:
                pass
        return _read_metadata_python_fallback(path)

    def embed_metadata(self, dest_path: Path, metadata: dict[str, Any]) -> bool:
        """Embed merged captured_at / GPS metadata into the destination copy only."""
        captured_at = metadata.get("captured_at")
        lat = metadata.get("latitude")
        lon = metadata.get("longitude")
        if not captured_at and lat is None and lon is None:
            return True

        if self.exiftool_bin:
            args = ["-overwrite_original"]
            if captured_at:
                exif_dt = captured_at.replace("T", " ").rstrip("Z").replace("-", ":", 2)
                args.extend(
                    [
                        f"-DateTimeOriginal={exif_dt}",
                        f"-CreateDate={exif_dt}",
                    ]
                )
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                args.extend(
                    [
                        f"-GPSLatitude={abs(lat)}",
                        f"-GPSLatitudeRef={'N' if lat >= 0 else 'S'}",
                        f"-GPSLongitude={abs(lon)}",
                        f"-GPSLongitudeRef={'E' if lon >= 0 else 'W'}",
                    ]
                )
            args.append(str(dest_path))
            try:
                self._get_exiftool().execute(*args)
                return True
            except FileNotFoundError:
                self.exiftool_bin = None
            except Exception:
                return False

        # Pure-Python JPEG EXIF fallback via piexif
        if piexif is not None and dest_path.suffix.lower() in {".jpg", ".jpeg"}:
            try:
                exif_dict = piexif.load(str(dest_path))
                if captured_at:
                    exif_dt = (
                        captured_at.replace("T", " ")
                        .rstrip("Z")
                        .replace("-", ":", 2)
                        .encode("ascii")
                    )
                    exif_dict.setdefault("Exif", {})[piexif.ExifIFD.DateTimeOriginal] = exif_dt
                piexif.insert(piexif.dump(exif_dict), str(dest_path))
                return True
            except Exception:
                return False
        return True

    def close(self) -> None:
        with self._lock:
            for proc in self._processes:
                proc.close()
            self._processes.clear()
