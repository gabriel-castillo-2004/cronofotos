# A single-process local FastAPI + SQLite (WAL) + vanilla HTML/CSS/JS web application for non-destructively organizing, deduplicating (SHA-256), enriching (EXIF/XMP/QuickTime + Google Takeout JSON sidecars), and exporting 17,000+ photos and videos into a clean 'Año / AAAA-MM - Evento o Categoría / AAAA-MM-DD_HH-MM-SS_###.ext' tree, hardened with a single-writer DB queue, thread-confined ExifTool pools, deterministic collision numbering, an idempotent two-phase physical exporter, LRU thumbnail caching, and a rate-limited optional Gemini classifier.

## Objective
A single-process local FastAPI + SQLite (WAL) + vanilla HTML/CSS/JS web application for non-destructively organizing, deduplicating (SHA-256), enriching (EXIF/XMP/QuickTime + Google Takeout JSON sidecars), and exporting 17,000+ photos and videos into a clean 'Año / AAAA-MM - Evento o Categoría / AAAA-MM-DD_HH-MM-SS_###.ext' tree, hardened with a single-writer DB queue, thread-confined ExifTool pools, deterministic collision numbering, an idempotent two-phase physical exporter, LRU thumbnail caching, and a rate-limited optional Gemini classifier.

## Rationale
Reconciles the Pragmatist's lightweight single-process FastAPI + vanilla SPA architecture (avoiding Celery, Redis, and Node build chains on Windows) with the Structuralist's reliability and safety invariants: eliminating SQLite lock contention via a single-writer batched queue, preventing ExifTool '-stay_open' thread interleaving via thread-confined processes with sentinel framing and Pillow/piexif fallback, ensuring deterministic '###' collision suffixes via SQLite canonical ordering and UNIQUE constraints, guaranteeing crash-safe idempotent exports via temp-file atomic renames and per-item state checkpoints, protecting Windows/OneDrive filesystems, and securing localhost APIs with opaque media IDs, throttled 4 Hz SSE snapshots, and a circuit-broken Gemini integration.

## Deployment Target
local
