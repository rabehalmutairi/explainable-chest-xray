"""Minimal inline SVG icon set (Feather-style line icons).

No emoji anywhere in the UI - these render crisply at any size and pick up
the current text color via stroke="currentColor", so they inherit theme
colors automatically without separate light/dark icon assets.
"""

from __future__ import annotations

_PATHS: dict[str, str] = {
    "upload-cloud": (
        '<polyline points="16 16 12 12 8 16"></polyline>'
        '<line x1="12" y1="12" x2="12" y2="21"></line>'
        '<path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3"></path>'
    ),
    "user": (
        '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>'
        '<circle cx="12" cy="7" r="4"></circle>'
    ),
    "calendar": (
        '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>'
        '<line x1="16" y1="2" x2="16" y2="6"></line>'
        '<line x1="8" y1="2" x2="8" y2="6"></line>'
        '<line x1="3" y1="10" x2="21" y2="10"></line>'
    ),
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>',
    "heart": (
        '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 '
        '7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>'
    ),
    "droplet": '<path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>',
    "wind": (
        '<path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27'
        'A2.5 2.5 0 1 1 19.5 12H2"></path>'
    ),
    "alert-triangle": (
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>'
        '<line x1="12" y1="9" x2="12" y2="13"></line>'
        '<line x1="12" y1="17" x2="12.01" y2="17"></line>'
    ),
    "check-circle": (
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>'
        '<polyline points="22 4 12 14.01 9 11.01"></polyline>'
    ),
    "x-circle": (
        '<circle cx="12" cy="12" r="10"></circle>'
        '<line x1="15" y1="9" x2="9" y2="15"></line>'
        '<line x1="9" y1="9" x2="15" y2="15"></line>'
    ),
    "clock": (
        '<circle cx="12" cy="12" r="10"></circle>'
        '<polyline points="12 6 12 12 16 14"></polyline>'
    ),
    "download": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>'
        '<polyline points="7 10 12 15 17 10"></polyline>'
        '<line x1="12" y1="15" x2="12" y2="3"></line>'
    ),
    "chevron-right": '<polyline points="9 18 15 12 9 6"></polyline>',
    "image": (
        '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>'
        '<circle cx="8.5" cy="8.5" r="1.5"></circle>'
        '<polyline points="21 15 16 10 5 21"></polyline>'
    ),
    "search": (
        '<circle cx="11" cy="11" r="8"></circle>'
        '<line x1="21" y1="21" x2="16.65" y2="16.65"></line>'
    ),
    "trash": (
        '<polyline points="3 6 5 6 21 6"></polyline>'
        '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>'
        '<line x1="10" y1="11" x2="10" y2="17"></line>'
        '<line x1="14" y1="11" x2="14" y2="17"></line>'
    ),
    "layers": (
        '<polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>'
        '<polyline points="2 17 12 22 22 17"></polyline>'
        '<polyline points="2 12 12 17 22 12"></polyline>'
    ),
    "plus-circle": (
        '<circle cx="12" cy="12" r="10"></circle>'
        '<line x1="12" y1="8" x2="12" y2="16"></line>'
        '<line x1="8" y1="12" x2="16" y2="12"></line>'
    ),
    "bar-chart": (
        '<line x1="12" y1="20" x2="12" y2="10"></line>'
        '<line x1="18" y1="20" x2="18" y2="4"></line>'
        '<line x1="6" y1="20" x2="6" y2="16"></line>'
    ),
}

PATHOLOGY_ICON_NAME: dict[str, str] = {
    "Atelectasis": "wind",
    "Consolidation": "wind",
    "Infiltration": "wind",
    "Pneumothorax": "wind",
    "Emphysema": "wind",
    "Fibrosis": "wind",
    "Pleural_Thickening": "wind",
    "Edema": "droplet",
    "Effusion": "droplet",
    "Pneumonia": "alert-triangle",
    "Cardiomegaly": "heart",
    "Nodule": "activity",
    "Mass": "activity",
    "Hernia": "alert-triangle",
}


def icon(name: str, size: int = 18, color: str = "currentColor", stroke_width: float = 1.8) -> str:
    """Returns a standalone <svg> string. Embed via st.markdown(unsafe_allow_html=True)."""
    inner = _PATHS.get(name, _PATHS["activity"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;">'
        f'{inner}</svg>'
    )


def pathology_icon(pathology: str, size: int = 18, color: str = "currentColor") -> str:
    return icon(PATHOLOGY_ICON_NAME.get(pathology, "activity"), size=size, color=color)


def ring_svg(
    fraction: float,
    center_text: str,
    size: int = 58,
    stroke: int = 6,
    color: str = "#FBBF24",
    track_color: str = "rgba(255,255,255,0.08)",
    text_color: str = "#F1F5F9",
) -> str:
    """A pure-SVG progress ring. Used instead of a Plotly widget so it can be
    embedded inline inside a raw-HTML card (Streamlit cannot nest a widget
    inside markdown-rendered HTML - each st.* call is its own DOM node)."""
    r = (size - stroke) / 2
    circumference = 2 * 3.14159265 * r
    fraction = max(0.0, min(1.0, fraction))
    dash = circumference * fraction
    cx = cy = size / 2
    # Built as ONE physical line, no embedded newlines. A multi-line
    # triple-quoted string here would introduce a whitespace-only line at
    # the point where this gets interpolated into an outer <div> block,
    # which terminates Markdown's raw-HTML passthrough (CommonMark HTML
    # block type 6 ends at the first blank line) and dumps the rest of the
    # block as a literal code block instead of rendering it. Confirmed by
    # screenshot: this was rendering as visible raw SVG source before the fix.
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="flex-shrink:0;">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{track_color}" stroke-width="{stroke}"></circle>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-dasharray="{dash:.2f} {circumference:.2f}" stroke-linecap="round" '
        f'transform="rotate(-90 {cx} {cy})"></circle>'
        f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" '
        f'font-size="{size * 0.30:.0f}" font-weight="800" fill="{text_color}" '
        f'font-family="Inter, sans-serif">{center_text}</text>'
        f'</svg>'
    )
