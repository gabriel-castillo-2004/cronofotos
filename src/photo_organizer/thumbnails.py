"""On-demand JPEG thumbnail generator with bounded RAM and disk LRU caches."""

from __future__ import annotations

import io
import json
import math
import subprocess
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

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


YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_NAME = "face_recognition_sface_2021dec.onnx"
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
_FALLBACK_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xfe\x00\x08face\xff\xd9"
)


def _ensure_face_models(models_dir: Path, timeout: float = 8.0) -> tuple[Path, Path] | None:
    """Ensure YuNet and SFace ONNX models exist in `models_dir`, downloading if needed."""
    models_dir.mkdir(parents=True, exist_ok=True)
    yunet_path = models_dir / YUNET_MODEL_NAME
    sface_path = models_dir / SFACE_MODEL_NAME
    targets = ((yunet_path, YUNET_MODEL_URL), (sface_path, SFACE_MODEL_URL))
    for path, url in targets:
        if path.is_file() and path.stat().st_size > 0:
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CronoFotos/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                return None
            tmp = path.with_suffix(".onnx.tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        except Exception:
            return None
    if yunet_path.is_file() and sface_path.is_file():
        return yunet_path, sface_path
    return None


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 <= 0.0 or n2 <= 0.0:
        return 0.0
    return dot / (n1 * n2)


def _fallback_feature_and_crop(src: Path, crop_size: int = 192) -> tuple[list[float], bytes]:
    """Compute a safe histogram feature vector and JPEG crop bytes without OpenCV."""
    if Image is not None:
        try:
            with Image.open(src) as img:
                if ImageOps is not None:
                    img = ImageOps.exif_transpose(img)
                rgb = img.convert("RGB")
                w, h = rgb.size
                left = max(0, int(w * 0.2))
                top = max(0, int(h * 0.15))
                right = min(w, int(w * 0.8))
                bottom = min(h, int(h * 0.75))
                crop = rgb.crop((left, top, right, bottom)) if right > left and bottom > top else rgb
                thumb = crop.resize((crop_size, crop_size))
                small = thumb.resize((32, 32))
                bins = [0.0] * 24
                for r, g, b in small.getdata():
                    bins[min(7, r // 32)] += 1.0
                    bins[8 + min(7, g // 32)] += 1.0
                    bins[16 + min(7, b // 32)] += 1.0
                buf = io.BytesIO()
                thumb.save(buf, format="JPEG", quality=82)
                return bins, buf.getvalue()
        except Exception:
            pass

    try:
        raw = src.read_bytes()
    except OSError:
        return [1.0] + [0.0] * 15, _FALLBACK_JPEG
    payload = raw[24:] if len(raw) > 24 else raw
    bins16 = [0.0] * 16
    for byte in payload[:4096]:
        bins16[byte >> 4] += 1.0
    if sum(bins16) <= 0.0:
        bins16[0] = 1.0
    crop_bytes = raw if raw.startswith(b"\xff\xd8") else _FALLBACK_JPEG
    return bins16, crop_bytes


def load_face_clusters(face_clusters_dir: Path, *, include_ignored: bool = True) -> list[dict[str, Any]]:
    """Load persisted face cluster records ordered by count descending."""
    if not face_clusters_dir.exists() or not face_clusters_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for jf in sorted(face_clusters_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("cluster_id"):
            items = [data]
        elif isinstance(data, list):
            items = [it for it in data if isinstance(it, dict) and it.get("cluster_id")]
        else:
            continue
        for rec in items:
            ignored = bool(rec.get("ignored", False))
            if not include_ignored and ignored:
                continue
            cid = str(rec["cluster_id"])
            media_ids = [str(m) for m in (rec.get("media_ids") or []) if m]
            count = int(rec.get("count", len(media_ids)))
            records.append(
                {
                    "cluster_id": cid,
                    "count": count,
                    "media_ids": media_ids,
                    "label": str(rec.get("label") or ""),
                    "ignored": ignored,
                    "crop_path": str(rec.get("crop_path") or (face_clusters_dir / f"{cid}.jpg")),
                    "url": str(rec.get("url") or f"/api/faces/clusters/{cid}/image"),
                }
            )
    records.sort(key=lambda r: (-int(r["count"]), str(r["cluster_id"])))
    return records


def update_face_cluster(
    face_clusters_dir: Path,
    cluster_id: str,
    *,
    label: str | None = None,
    ignored: bool | None = None,
) -> dict[str, Any] | None:
    """Update `label` and/or `ignored` on a persisted face cluster JSON record."""
    jf = face_clusters_dir / f"{cluster_id}.json"
    if not jf.is_file():
        return None
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if label is not None:
        data["label"] = str(label).strip()
    if ignored is not None:
        data["ignored"] = bool(ignored)
    jf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def discover_frequent_faces(
    media_records: list[dict[str, Any]],
    data_dir: Path,
    *,
    progress_cb: Callable[[int, int, int], None] | None = None,
    cosine_threshold: float = 0.45,
    crop_size: int = 192,
) -> list[dict[str, Any]]:
    """Detect faces across scanned photos, cluster by embedding similarity, and persist top crops."""
    face_clusters_dir = data_dir / "face_clusters"
    face_clusters_dir.mkdir(parents=True, exist_ok=True)
    models_dir = data_dir / "models"

    cv2_mod: Any = None
    detector: Any = None
    recognizer: Any = None
    try:
        import cv2 as _cv2  # type: ignore[import-not-found]

        if (
            _cv2 is not None
            and hasattr(_cv2, "FaceDetectorYN")
            and hasattr(_cv2, "FaceRecognizerSF")
        ):
            model_paths = _ensure_face_models(models_dir)
            if model_paths is not None:
                yunet_path, sface_path = model_paths
                detector = _cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), 0.7, 0.3, 5000)
                recognizer = _cv2.FaceRecognizerSF.create(str(sface_path), "")
                cv2_mod = _cv2
    except Exception:
        cv2_mod = None
        detector = None
        recognizer = None

    use_sface = cv2_mod is not None and detector is not None and recognizer is not None
    match_threshold = cosine_threshold if use_sface else 0.90

    clusters: list[dict[str, Any]] = []
    total = len(media_records)

    for idx, item in enumerate(media_records, start=1):
        media_id = str(item.get("id") or "")
        src = Path(str(item.get("source_path") or ""))
        if not media_id or not src.is_file():
            if progress_cb is not None:
                progress_cb(idx, total, len(clusters))
            continue

        extracted: list[tuple[Any, Any, bytes | None]] = []
        if use_sface:
            try:
                img = cv2_mod.imread(str(src))
                if img is not None:
                    shape = getattr(img, "shape", None)
                    if shape and len(shape) >= 2:
                        detector.setInputSize((int(shape[1]), int(shape[0])))
                    _ret, faces = detector.detect(img)
                    if faces is not None and len(faces) > 0:
                        sorted_faces = sorted(
                            faces,
                            key=lambda fb: float(fb[2]) * float(fb[3]) if len(fb) >= 4 else 0.0,
                            reverse=True,
                        )
                        for fb in sorted_faces:
                            x, y, w, h = int(fb[0]), int(fb[1]), int(fb[2]), int(fb[3])
                            if w <= 0 or h <= 0:
                                continue
                            aligned = recognizer.alignCrop(img, fb)
                            feat = recognizer.feature(aligned)
                            if shape and len(shape) >= 2:
                                x0 = max(0, x)
                                y0 = max(0, y)
                                x1 = min(int(shape[1]), x + w)
                                y1 = min(int(shape[0]), y + h)
                            else:
                                x0, y0, x1, y1 = x, y, x + w, y + h
                            crop_img = img[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else aligned
                            resized_crop = cv2_mod.resize(crop_img, (crop_size, crop_size))
                            extracted.append((feat, resized_crop, None))
            except Exception:
                extracted = []
        else:
            vec, crop_bytes = _fallback_feature_and_crop(src, crop_size=crop_size)
            extracted.append((vec, None, crop_bytes))

        for feat_or_vec, cv_crop, raw_crop_bytes in extracted:
            best_cluster: dict[str, Any] | None = None
            best_score = -1.0
            for cl in clusters:
                if use_sface:
                    score = float(
                        recognizer.match(
                            feat_or_vec,
                            cl["rep_feat"],
                            getattr(cv2_mod, "FR_COSINE", 0),
                        )
                    )
                else:
                    score = _cosine_similarity(feat_or_vec, cl["rep_feat"])
                if score >= match_threshold and score > best_score:
                    best_score = score
                    best_cluster = cl

            if best_cluster is None:
                clusters.append(
                    {
                        "rep_feat": feat_or_vec,
                        "cv_crop": cv_crop,
                        "raw_crop_bytes": raw_crop_bytes,
                        "media_ids": [media_id],
                    }
                )
            else:
                if media_id not in best_cluster["media_ids"]:
                    best_cluster["media_ids"].append(media_id)

        if progress_cb is not None:
            progress_cb(idx, total, len(clusters))

    clusters.sort(key=lambda c: len(c["media_ids"]), reverse=True)

    # Clean previous auto-generated cluster files before writing updated set
    for old_file in face_clusters_dir.iterdir():
        if old_file.is_file() and old_file.suffix.lower() in {".json", ".jpg", ".jpeg", ".png"}:
            try:
                old_file.unlink()
            except OSError:
                pass

    persisted: list[dict[str, Any]] = []
    for idx, cl in enumerate(clusters, start=1):
        cid = f"fc_{idx:03d}"
        crop_path = face_clusters_dir / f"{cid}.jpg"
        if cl["cv_crop"] is not None and cv2_mod is not None:
            try:
                cv2_mod.imwrite(str(crop_path), cl["cv_crop"])
            except Exception:
                crop_path.write_bytes(_FALLBACK_JPEG)
        elif cl["raw_crop_bytes"] is not None:
            crop_path.write_bytes(cl["raw_crop_bytes"])
        else:
            crop_path.write_bytes(_FALLBACK_JPEG)

        record = {
            "cluster_id": cid,
            "count": len(cl["media_ids"]),
            "media_ids": list(cl["media_ids"]),
            "label": "",
            "ignored": False,
            "crop_path": str(crop_path),
            "url": f"/api/faces/clusters/{cid}/image",
        }
        (face_clusters_dir / f"{cid}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        persisted.append(record)

    return persisted
