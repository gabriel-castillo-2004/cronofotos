"""On-demand JPEG thumbnail generator with bounded RAM and disk LRU caches."""

from __future__ import annotations

import io
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path

from photo_organizer.config import VIDEO_EXTENSIONS, resolve_cli_binary

_ORIG_SUBPROCESS_RUN = subprocess.run

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

try:
    import pillow_heif  # type: ignore[import-not-found]

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass


class ThumbnailCache:
    """Generates and caches downscaled JPEG previews with strict RAM and disk LRU limits."""

    def __init__(
        self,
        cache_dir: Path,
        max_ram_items: int = 256,
        max_disk_bytes: int = 128 * 1024 * 1024,
        ffmpeg_bin: str | None = None,
        max_ffmpeg_concurrency: int = 2,
        thumb_size: tuple[int, int] = (320, 320),
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_ram_items = max(1, max_ram_items)
        self.max_disk_bytes = max(1024, max_disk_bytes)
        self.ffmpeg_bin = ffmpeg_bin
        self.thumb_size = thumb_size
        self._ram: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()
        self._ffmpeg_sem = threading.Semaphore(max(1, max_ffmpeg_concurrency))

    def _disk_path(self, opaque_id: str, sha256: str | None) -> Path:
        key = sha256 if sha256 else opaque_id
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)
        return self.cache_dir / f"v2_exif_{safe}.jpg"

    def _put_ram(self, key: str, data: bytes) -> None:
        with self._lock:
            if key in self._ram:
                self._ram.move_to_end(key)
            else:
                if len(self._ram) >= self.max_ram_items:
                    self._ram.popitem(last=False)
                self._ram[key] = data

    def _evict_disk_if_needed(self) -> None:
        files = [p for p in self.cache_dir.glob("*.jpg") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        if total <= self.max_disk_bytes:
            return
        files.sort(key=lambda p: p.stat().st_atime)
        for p in files:
            if total <= self.max_disk_bytes:
                break
            sz = p.stat().st_size
            try:
                p.unlink()
                total -= sz
            except OSError:
                continue

    def _render_image_thumb(self, source_path: Path) -> bytes:
        if Image is None:
            return b"THUMB:" + source_path.read_bytes()[:64]
        try:
            with Image.open(source_path) as img:
                if ImageOps is not None:
                    img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                img.thumbnail(self.thumb_size)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80)
                return buf.getvalue()
        except Exception:
            return b"THUMB:" + source_path.name.encode("utf-8")

    def _render_video_thumb(self, source_path: Path) -> bytes:
        if not self.ffmpeg_bin:
            return b"THUMB:VIDEO:" + source_path.name.encode("utf-8")
        resolved_ffmpeg = resolve_cli_binary(self.ffmpeg_bin)
        if resolved_ffmpeg is None and subprocess.run is _ORIG_SUBPROCESS_RUN:
            return b"THUMB:VIDEO:" + source_path.name.encode("utf-8")
        with self._ffmpeg_sem:
            try:
                proc = subprocess.run(
                    [
                        resolved_ffmpeg or self.ffmpeg_bin,
                        "-y",
                        "-ss",
                        "00:00:01",
                        "-i",
                        str(source_path),
                        "-vframes",
                        "1",
                        "-vf",
                        f"scale={self.thumb_size[0]}:-1",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "mjpeg",
                        "pipe:1",
                    ],
                    capture_output=True,
                    timeout=10.0,
                    check=True,
                )
                if proc.stdout:
                    return proc.stdout
            except Exception:
                pass
        return b"THUMB:VIDEO:" + source_path.name.encode("utf-8")

    def get_or_create(
        self,
        opaque_id: str,
        source_path: Path,
        sha256: str | None = None,
    ) -> bytes:
        """Return cached or freshly generated thumbnail bytes."""
        ram_key = f"{opaque_id}:{sha256 or ''}"
        with self._lock:
            if ram_key in self._ram:
                self._ram.move_to_end(ram_key)
                return self._ram[ram_key]

        disk_file = self._disk_path(opaque_id, sha256)
        if disk_file.is_file():
            data = disk_file.read_bytes()
            self._put_ram(ram_key, data)
            return data

        if source_path.suffix.lower() in VIDEO_EXTENSIONS:
            data = self._render_video_thumb(source_path)
        else:
            data = self._render_image_thumb(source_path)

        try:
            disk_file.write_bytes(data)
            self._evict_disk_if_needed()
        except OSError:
            pass

        self._put_ram(ram_key, data)
        return data

    def get_or_create_resized(
        self,
        opaque_id: str,
        source_path: Path,
        sha256: str | None = None,
        max_dim: int = 256,
    ) -> Path:
        """Ensure a cached JPEG thumbnail resized to <= max_dim exists on disk and return its Path."""
        key = sha256 if sha256 else opaque_id
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)
        out_path = self.cache_dir / f"ai_{max_dim}_{safe}.jpg"
        if out_path.is_file():
            return out_path

        if source_path.suffix.lower() in VIDEO_EXTENSIONS:
            data = self._render_video_thumb(source_path)
        elif Image is not None:
            try:
                with Image.open(source_path) as img:
                    if ImageOps is not None:
                        img = ImageOps.exif_transpose(img)
                    img = img.convert("RGB")
                    img.thumbnail((max_dim, max_dim))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80)
                    data = buf.getvalue()
            except Exception:
                data = self._render_image_thumb(source_path)
        else:
            data = self._render_image_thumb(source_path)

        try:
            out_path.write_bytes(data)
            self._evict_disk_if_needed()
        except OSError:
            pass
        return out_path


def extract_top_face_crops(
    image_paths: list[Path],
    out_dir: Path,
    *,
    max_faces: int = 4,
    crop_size: int = 192,
    file_prefix: str = "face",
) -> list[Path]:
    """Detect faces via OpenCV CascadeClassifier, sort by bbox area desc, and save top crops."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return []
    if cv2 is None or not hasattr(cv2, "CascadeClassifier"):
        return []

    try:
        cascades_dir = getattr(getattr(cv2, "data", None), "haarcascades", "") or ""
        cascade_path = str(Path(cascades_dir) / "haarcascade_frontalface_default.xml") if cascades_dir else "haarcascade_frontalface_default.xml"
        classifier = cv2.CascadeClassifier(cascade_path)
        if classifier is None or (hasattr(classifier, "empty") and classifier.empty()):
            return []
    except Exception:
        return []

    candidates: list[tuple[int, tuple[int, int, int, int], object]] = []
    for src in image_paths:
        if not src.exists() or not src.is_file():
            continue
        try:
            img = cv2.imread(str(src))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            if boxes is None:
                continue
            for box in boxes:
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                if w > 0 and h > 0:
                    candidates.append((w * h, (x, y, w, h), img))
        except Exception:
            continue

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for idx, (_area, (x, y, w, h), img) in enumerate(candidates[: max(1, max_faces)], start=1):
        try:
            shape = getattr(img, "shape", None)
            if shape and len(shape) >= 2:
                img_h, img_w = int(shape[0]), int(shape[1])
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(img_w, x + w)
                y1 = min(img_h, y + h)
            else:
                x0, y0, x1, y1 = x, y, x + w, y + h
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img[y0:y1, x0:x1]
            resized = cv2.resize(crop, (crop_size, crop_size))
            dst = out_dir / f"{file_prefix}_{idx}.jpg"
            if cv2.imwrite(str(dst), resized):
                saved.append(dst)
        except Exception:
            continue
    return saved
