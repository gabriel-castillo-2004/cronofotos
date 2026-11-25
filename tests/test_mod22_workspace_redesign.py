"""Tests for post-delivery modification MOD-22:
Apple Photos / Finder-inspired lower workspace redesign (`index.html`, `styles.css`, `app.js`)
preserving all existing element IDs, endpoints, and tests:
1. Hierarchical, resizable, instantly filterable `Árbol Propuesto`:
   - Collapsible group headers by section and year (e.g. `01_Recuerdos · 2024`, `02_Capturas_y_WhatsApp`).
   - Full subfolder name with line wrapping (`white-space: normal`, up to 2 clean lines),
     discreet count pill on the right, aligned `✏️` button, and full-path tooltip (`title`).
   - Compact search input (`Filtrar carpetas...`) inside `Árbol Propuesto` header.
   - Initial `360px` width for `Árbol Propuesto` and draggable vertical `resizer` between
     `Árbol Propuesto` and `Elementos`.
2. On-demand `Vista Previa` (2 wide columns by default):
   - Right `Vista Previa` panel hidden when no item is selected, expanding `Elementos`.
   - Clicking an item shows `Vista Previa` with a `× Cerrar` button in its top-right corner.
3. Clean gallery cards without visual noise:
   - No full folder path line below each thumbnail in `.items-grid`.
   - Square cards (`aspect-ratio: 1 / 1`) with rounded borders and a 1-line filename footer.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "src" / "photo_organizer" / "static"


def _read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def test_mod22_01_existing_workspace_ids_preserved() -> None:
    """All existing workspace element IDs and landmarks remain intact."""
    html = _read_static("index.html")
    for elem_id in (
        "folder-list",
        "current-folder-title",
        "btn-clear-folder-filter",
        "btn-prev-page",
        "page-info",
        "btn-next-page",
        "items-grid",
        "preview-box",
    ):
        assert f'id="{elem_id}"' in html, f"Missing preserved workspace ID: {elem_id}"

    assert "Árbol Propuesto" in html
    assert "Elementos" in html
    assert "Vista Previa" in html


def test_mod22_02_tree_hierarchical_groups_wrapping_pill_tooltip_and_filter() -> None:
    """Árbol Propuesto groups folders under collapsible section/year headers, wraps names
    up to 2 lines with a count pill, aligned rename button, full-path tooltip, and filter input.
    """
    html = _read_static("index.html")
    css = _read_static("styles.css")
    js = _read_static("app.js")

    # 1. Compact search input ('Filtrar carpetas...') inside Árbol Propuesto header
    tree_pane_match = re.search(
        r'<aside\b[^>]*\bclass=["\'][^"\']*\btree-pane\b[^"\']*["\'][^>]*>([\s\S]*?)</aside>',
        html,
    )
    assert tree_pane_match is not None, "Expected .tree-pane aside in index.html"
    tree_pane_html = tree_pane_match.group(1)
    assert re.search(
        r'<input\b[^>]*\bplaceholder=["\']Filtrar carpetas\.\.\.["\']',
        tree_pane_html,
    ), "Expected 'Filtrar carpetas...' search input inside Árbol Propuesto"

    # 2. CSS: subfolder label allows wrapping (white-space: normal) up to 2 lines
    folder_label_match = re.search(r"\.folder-label\s*\{([^}]*)\}", css, re.DOTALL)
    assert folder_label_match is not None, "Expected .folder-label rule in styles.css"
    folder_label_css = folder_label_match.group(1)
    assert re.search(r"white-space\s*:\s*normal", folder_label_css), (
        ".folder-label must use white-space: normal instead of nowrap"
    )
    assert re.search(r"(?:-webkit-)?line-clamp\s*:\s*2", folder_label_css), (
        ".folder-label must clamp cleanly to 2 lines"
    )

    # 3. JS: collapsible section/year grouping, tooltip with full path, count pill, and filter logic
    assert re.search(r"·", js), "app.js must format section/year group headers (e.g. '01_Recuerdos · 2024')"
    assert re.search(r"aria-expanded|<details|<summary", js), (
        "app.js must render collapsible group headers for folder sections/years"
    )
    assert re.search(r"\.title\s*=\s*f\.folder|setAttribute\(\s*['\"]title['\"]\s*,\s*f\.folder", js), (
        "app.js must set a tooltip (title) with the full folder path on subfolder items"
    )
    assert "btn-inline-rename" in js
    assert "✏️" in js
    assert re.search(r"folder-count|count-pill|folder-badge", css) and re.search(
        r"folder-count|count-pill|folder-badge", js
    ), "Subfolder count must be rendered in a discreet pill/badge on the right"
    assert re.search(r"Filtrar|folderFilter|filterText|searchQuery", js, re.IGNORECASE), (
        "app.js must implement instant folder filtering"
    )


def test_mod22_03_tree_initial_width_360px_and_draggable_resizer() -> None:
    """Árbol Propuesto starts at 360px wide with a draggable vertical resizer before Elementos."""
    html = _read_static("index.html")
    css = _read_static("styles.css")
    js = _read_static("app.js")

    # Resizer element between tree-pane and items-pane
    workspace_match = re.search(
        r'<main\b[^>]*\bclass=["\'][^"\']*\bworkspace\b[^"\']*["\'][^>]*>([\s\S]*?)</main>',
        html,
    )
    assert workspace_match is not None, "Expected <main class='workspace'> in index.html"
    workspace_html = workspace_match.group(1)
    assert re.search(
        r'tree-pane[\s\S]*?(?:resizer|role=["\']separator["\'])[\s\S]*?items-pane',
        workspace_html,
    ), "Expected a vertical resizer element between .tree-pane and .items-pane"

    # Initial 360px width for tree pane and col-resize cursor on resizer
    assert re.search(r"360px", css), "styles.css must set initial width of Árbol Propuesto to 360px"
    assert re.search(r"cursor\s*:\s*col-resize", css), (
        "styles.css must style the vertical resizer with cursor: col-resize"
    )

    # Dragging handlers in app.js
    assert re.search(r"pointerdown|mousedown", js) and re.search(r"pointermove|mousemove", js), (
        "app.js must handle drag events on the vertical resizer to resize Árbol Propuesto"
    )


def test_mod22_04_preview_pane_hidden_by_default_and_closable_on_demand() -> None:
    """Vista Previa starts hidden so Elementos spans 2 wide columns, and opens on item click
    with a '× Cerrar' button in its top-right corner.
    """
    html = _read_static("index.html")
    css = _read_static("styles.css")
    js = _read_static("app.js")

    # Preview pane hidden initially in HTML
    preview_aside_match = re.search(
        r'<aside\b[^>]*\bclass=["\'][^"\']*\bpreview-pane\b[^"\']*["\'][^>]*>([\s\S]*?)</aside>',
        html,
    )
    assert preview_aside_match is not None, "Expected .preview-pane aside in index.html"
    preview_open_tag = re.search(r"<aside\b[^>]*>", preview_aside_match.group(0)).group(0)
    assert re.search(r'\bhidden\b', preview_open_tag), (
        ".preview-pane must be hidden by default when no item is selected"
    )

    # Close button '× Cerrar' inside preview pane header
    preview_inner = preview_aside_match.group(1)
    assert re.search(
        r"<button\b[^>]*>\s*(?:×|&times;)\s*Cerrar\s*</button>",
        preview_inner,
    ), "Expected '× Cerrar' button inside .preview-pane header"

    # CSS ensures hidden preview-pane collapses cleanly so items-pane expands
    assert re.search(r"\.preview-pane\.hidden|\.workspace[^\n{]*preview", css), (
        "styles.css must handle hidden .preview-pane so .items-pane expands to full remaining width"
    )

    # JS opens preview-pane in showPreview and closes it on close button click
    show_preview_match = re.search(r"function\s+showPreview\s*\([^)]*\)\s*\{([\s\S]*?)\n\}", js)
    assert show_preview_match is not None, "Expected showPreview(item) function in app.js"
    show_preview_body = show_preview_match.group(1)
    assert re.search(r"preview-pane|previewPane", show_preview_body), (
        "showPreview must reveal .preview-pane when an item is clicked"
    )
    assert re.search(r"classList\.add\(['\"]hidden['\"]\)|\.hidden\s*=\s*true", js), (
        "app.js must hide .preview-pane again when '× Cerrar' is clicked"
    )


def test_mod22_05_clean_square_gallery_cards_without_folder_path() -> None:
    """Gallery cards in .items-grid are square (aspect-ratio: 1 / 1) with rounded borders,
    a 1-line filename footer, and no full folder path line underneath.
    """
    css = _read_static("styles.css")
    js = _read_static("app.js")

    # Square aspect-ratio and rounded borders on card / thumbnail
    assert re.search(r"\.item-card\b[^}]*aspect-ratio\s*:\s*1\s*/\s*1", css, re.DOTALL), (
        ".item-card must use square aspect-ratio: 1 / 1"
    )
    assert re.search(r"\.item-card\b[^}]*border-radius\s*:", css, re.DOTALL), (
        ".item-card must have rounded borders"
    )

    # 1-line filename footer
    item_name_match = re.search(r"\.item-name\s*\{([^}]*)\}", css, re.DOTALL)
    assert item_name_match is not None, "Expected .item-name rule in styles.css"
    item_name_css = item_name_match.group(1)
    assert re.search(r"white-space\s*:\s*nowrap", item_name_css)
    assert re.search(r"text-overflow\s*:\s*ellipsis", item_name_css)

    # Full folder path (.item-sub) removed from item cards in app.js
    assert "item-sub" not in js, (
        "app.js must not render .item-sub (full folder path) inside gallery .item-card elements"
    )
