"""Tests for post-delivery modification MOD-20:
Minimalist macOS/Linear-style UI redesign for CronoFotos (`index.html`, `styles.css`, `app.js`)
with maximum visual clarity, zero telemetry loss, and all existing element IDs/endpoints intact.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "src" / "photo_organizer" / "static"


def _read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def _extract_block(html: str, marker: str, window: int = 3500) -> str:
    idx = html.find(marker)
    assert idx != -1, f"Expected marker {marker!r} in index.html"
    return html[idx : idx + window]


def test_mod20_01_compact_header_and_source_dest_bar() -> None:
    """Header is compacted to 48px, folder bar has only core inputs + primary CTA,
    duplicate delta checkbox is removed, and incremental options live inside 'Opciones incrementales'.
    """
    html = _read_static("index.html")
    css = _read_static("styles.css")

    # 1. Compact 48px topbar with title, health pills, Tema, Configuración
    assert re.search(r"\.topbar\s*\{[^}]*(?:min-height|height)\s*:\s*48px", css, re.DOTALL)
    assert "CronoFotos" in html
    assert 'id="health-badges"' in html
    assert 'id="btn-theme"' in html
    assert 'id="btn-settings"' in html

    # 2. Core visible folder controls
    assert "Carpeta con fotos nuevas" in html
    assert "Mi Biblioteca Organizada" in html
    assert "Modo prueba (100 archivos)" in html
    assert "Organizar Automáticamente" in html

    # 3. Duplicate delta checkbox removed; single incremental control + master index inside 'Opciones incrementales'
    assert "Solo archivos nuevos (Delta contra plan/índice previo)" not in html
    assert 'id="delta-mode"' not in html
    assert "Opciones incrementales" in html
    assert "Modo incremental (solo archivos nuevos)" in html
    assert "Índice maestro previo" in html

    # Both incremental controls must reside inside the Opciones incrementales collapsible block
    inc_match = re.search(
        r"<details[^>]*>[\s\S]*?Opciones incrementales[\s\S]*?</details>",
        html,
    )
    assert inc_match is not None, "Expected <details> dropdown for 'Opciones incrementales'"
    inc_block = inc_match.group(0)
    assert 'id="incremental-mode"' in inc_block
    assert 'id="master-index-path"' in inc_block


def test_mod20_02_slim_progress_stepper_and_inline_telemetry() -> None:
    """4 large phase cards replaced by slim horizontal stepper (1. Duplicados -> 4. Listo),
    6px progress bar, and single-line metrics with inline live activity + expand toggle.
    """
    html = _read_static("index.html")
    css = _read_static("styles.css")
    js = _read_static("app.js")

    for step_label in ("1. Duplicados", "2. Clasificación", "3. IA", "4. Listo"):
        assert step_label in html, f"Missing slim stepper label: {step_label}"

    # 6px progress bar
    assert re.search(r"(?:#progress-bar|progress)\s*\{[^}]*height\s*:\s*6px", css, re.DOTALL)

    # Single-line metrics & inline latest activity event with log expand control
    assert 'id="cnt-new"' in html
    assert 'id="cnt-dups"' in html
    assert 'id="cnt-screenshots"' in html
    assert "archivos/s" in html or "archivos/s" in js
    assert "Actividad en vivo" in html
    assert 'id="activity-log"' in html
    assert 'id="activity-log-panel"' in html


def test_mod20_03_simplified_action_bar_two_menus_plus_primary() -> None:
    """Action bar groups AI tools under '✨ Mejorar con IA', secondary/plan/advanced tools
    under '⋯ Más opciones', and keeps 'Aplicar y Exportar Árbol' as prominent primary button.
    """
    html = _read_static("index.html")

    assert "✨ Mejorar con IA" in html
    assert "⋯ Más opciones" in html
    assert "Aplicar y Exportar Árbol" in html

    # AI dropdown groups the 4 AI buttons with their original IDs
    ai_block = _extract_block(html, "✨ Mejorar con IA", window=2500)
    for btn_id in (
        "btn-ai-debate",
        "btn-ai-optimize",
        "btn-ai-regenerate",
        "btn-ai-debate-resume",
    ):
        assert f'id="{btn_id}"' in ai_block, f"{btn_id} must be inside '✨ Mejorar con IA' menu"

    # More options dropdown groups plan export/import, refine tree, and advanced settings
    more_block = _extract_block(html, "⋯ Más opciones", window=3500)
    for elem_id in (
        "btn-export-plan",
        "btn-import-plan",
        "btn-refine-tree",
        "advanced-menu",
    ):
        assert f'id="{elem_id}"' in more_block, f"{elem_id} must be inside '⋯ Más opciones' menu"


def test_mod20_04_existing_ids_preserved_and_workspace_prioritized() -> None:
    """All critical existing element IDs remain intact and workspace occupies main viewport."""
    html = _read_static("index.html")
    css = _read_static("styles.css")

    required_ids = (
        "health-badges",
        "btn-theme",
        "btn-settings",
        "source-dir",
        "btn-browse-source",
        "dest-dir",
        "btn-browse-dest",
        "master-index-path",
        "btn-start",
        "solo-primer-lote",
        "incremental-mode",
        "progress-phase",
        "progress-stage",
        "progress-stats",
        "progress-bar",
        "cnt-new",
        "cnt-dups",
        "cnt-screenshots",
        "btn-ai-regenerate",
        "btn-ai-optimize",
        "btn-ai-debate",
        "btn-ai-debate-resume",
        "btn-export-plan",
        "btn-import-plan",
        "import-plan-file",
        "btn-refine-tree",
        "btn-export",
        "btn-open-dest",
        "ai-optimizer-panel",
        "advanced-menu",
        "batch-size",
        "time-gap-hours",
        "gps-distance-km",
        "btn-rename-folder",
        "btn-merge-folder",
        "btn-create-folder",
        "result-toast",
        "activity-log-panel",
        "activity-log",
        "folder-list",
        "items-grid",
        "preview-box",
        "settings-modal",
        "ai-confirm-modal",
    )
    for elem_id in required_ids:
        assert f'id="{elem_id}"' in html, f"Missing preserved element ID: {elem_id}"

    # Workspace 3-column layout and compact top panel
    assert ".workspace" in css
    assert "Árbol Propuesto" in html
    assert "Vista Previa" in html
    assert re.search(r"\.workspace\s*\{[^}]*flex\s*:\s*1", css, re.DOTALL)
