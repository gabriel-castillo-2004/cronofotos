"""Tests for post-delivery modification MOD-21:
3 final UX clarity adjustments in `index.html`, `styles.css`, and `app.js`
preserving all existing element IDs and tests:
1. `Aplicar y Exportar Árbol` (`#btn-export`) starts disabled, is not a blue
   primary button at startup (leaving `Organizar Automáticamente` as the sole
   blue primary CTA), and is styled green (`#16a34a`) only when ready to export.
2. Initial status text `Estado: idle` is translated to `Estado: En espera`.
3. Central empty state (`Aún no hay elementos escaneados`) includes a discreet
   reassurance line: "Tus archivos originales nunca se modifican ni se eliminan."
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "src" / "photo_organizer" / "static"


def _read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def test_mod21_01_export_button_disabled_initially_and_green_when_ready() -> None:
    """#btn-export starts disabled without primary blue class and turns #16a34a when ready."""
    html = _read_static("index.html")
    css = _read_static("styles.css")
    js = _read_static("app.js")

    export_btn_match = re.search(r"<button\b[^>]*\bid=[\"']btn-export[\"'][^>]*>", html)
    assert export_btn_match is not None, "Expected #btn-export button in index.html"
    export_tag = export_btn_match.group(0)

    assert re.search(r"\bdisabled\b", export_tag), (
        "#btn-export must be disabled initially when no scanned items exist"
    )
    class_match = re.search(r'\bclass=["\']([^"\']*)["\']', export_tag)
    export_classes = class_match.group(1).split() if class_match else []
    assert "primary" not in export_classes, (
        "#btn-export must not use .primary blue styling at startup"
    )

    start_btn_match = re.search(r"<button\b[^>]*\bid=[\"']btn-start[\"'][^>]*>", html)
    assert start_btn_match is not None, "Expected #btn-start button in index.html"
    start_class_match = re.search(r'\bclass=["\']([^"\']*)["\']', start_btn_match.group(0))
    start_classes = start_class_match.group(1).split() if start_class_match else []
    assert "primary" in start_classes, (
        "#btn-start (Organizar Automáticamente) must remain the primary blue button"
    )

    assert re.search(r"#16a34a\b", css, re.IGNORECASE), (
        "styles.css must style the ready export button in green (#16a34a)"
    )
    assert re.search(
        r"(?:#btn-export[^\n{]*:not\(:disabled\)|export-ready)[^}]*#16a34a",
        css,
        re.IGNORECASE | re.DOTALL,
    ), "Green #16a34a styling must apply only when #btn-export is enabled/ready"

    assert "btn-export" in js
    assert re.search(
        r"btn-export[^\n;]*\.disabled\s*=|exportBtn\.disabled\s*=",
        js,
    ), "app.js must dynamically update #btn-export.disabled based on scanned items"


def test_mod21_02_initial_status_translated_to_en_espera() -> None:
    """Initial status displays 'Estado: En espera' instead of 'Estado: idle'."""
    html = _read_static("index.html")
    js = _read_static("app.js")

    stage_match = re.search(
        r'<span\b[^>]*\bid=["\']progress-stage["\'][^>]*>(.*?)</span>',
        html,
        re.DOTALL,
    )
    assert stage_match is not None, "Expected #progress-stage in index.html"
    assert stage_match.group(1).strip() == "Estado: En espera"

    assert "Estado: idle" not in html
    assert "Estado: idle" not in js
    assert re.search(r"""['"]idle['"]\s*:\s*['"]En espera['"]|===?\s*['"]idle['"][^\n]*['"]En espera['"]""", js), (
        "app.js must translate 'idle' status to 'En espera'"
    )


def test_mod21_03_empty_state_includes_non_destructive_note() -> None:
    """Empty state includes discreet reassurance that original files are never modified or deleted."""
    html = _read_static("index.html")
    js = _read_static("app.js")
    note = "Tus archivos originales nunca se modifican ni se eliminan."

    empty_match = re.search(
        r'<div\b[^>]*\bclass=["\'][^"\']*\bempty-state\b[^"\']*["\'][^>]*>([\s\S]*?)</div>',
        html,
    )
    assert empty_match is not None, "Expected .empty-state block in index.html"
    empty_block = empty_match.group(1)
    assert "Aún no hay elementos escaneados" in empty_block
    assert note in empty_block

    assert "Aún no hay elementos escaneados" in js
    assert note in js
