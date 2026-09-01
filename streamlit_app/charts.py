"""Plotly visualizations for the chest X-ray dashboard.

Kept separate from components.py (markup helpers) since chart construction
is a distinct concern with its own library dependency. Colors are dark-theme
aware since the whole app runs on a dark canvas.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from streamlit_app.icons import PATHOLOGY_ICON_NAME  # noqa: F401  (re-exported for app.py)
from streamlit_app.style import COLORS


def probability_bar_chart(predictions: list[dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart, sorted ascending (Plotly renders bottom-to-top,
    so the highest probability lands at the top). A diamond marker on each
    bar shows that class's decision threshold.
    """
    ordered = sorted(predictions, key=lambda p: p["probability"])

    labels = [p["pathology"] for p in ordered]
    probs = [p["probability"] * 100 for p in ordered]
    thresholds = [p["threshold"] * 100 for p in ordered]
    colors = [COLORS["amber"] if p["above_threshold"] else "rgba(255,255,255,0.18)" for p in ordered]
    hover = [
        f"<b>{p['pathology']}</b><br>Probability: {p['probability']:.1%}"
        f"<br>Threshold: {p['threshold']:.0%}"
        f"<br>{'Above' if p['above_threshold'] else 'Below'} threshold"
        for p in ordered
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=probs, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertext=hover, hoverinfo="text", width=0.62, name="Probability",
    ))
    fig.add_trace(go.Scatter(
        y=labels, x=thresholds, mode="markers",
        marker=dict(symbol="diamond", size=9, color=COLORS["primary"], opacity=0.85,
                    line=dict(width=1, color=COLORS["bg"])),
        hovertext=[f"Threshold: {t:.0f}%" for t in thresholds],
        hoverinfo="text", name="Threshold",
    ))

    fig.update_layout(
        height=430,
        margin=dict(l=4, r=16, t=8, b=28),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(family="Inter, sans-serif", size=12.5, color=COLORS["text"]),
        xaxis=dict(
            range=[0, 100], ticksuffix="%", gridcolor="rgba(255,255,255,0.08)",
            zeroline=False, fixedrange=True, color=COLORS["text_muted"],
        ),
        yaxis=dict(fixedrange=True, color=COLORS["text"]),
        hoverlabel=dict(bgcolor=COLORS["surface_solid"], font_size=12, bordercolor=COLORS["border"],
                         font_color=COLORS["text"]),
    )
    return fig


def findings_donut(n_above: int, n_total: int) -> go.Figure:
    """Compact donut: findings above threshold vs total classes screened."""
    n_below = n_total - n_above
    color_above = COLORS["amber"] if n_above else "rgba(255,255,255,0.15)"

    fig = go.Figure(go.Pie(
        values=[n_above, n_below] if n_above else [1],
        labels=["Above threshold", "Below threshold"] if n_above else ["Below threshold"],
        hole=0.72,
        marker=dict(colors=[color_above, "rgba(255,255,255,0.06)"] if n_above else ["rgba(255,255,255,0.08)"]),
        textinfo="none", hoverinfo="label+value", sort=False,
    ))
    fig.update_layout(
        height=110, width=110,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b>{n_above}</b><span style='font-size:10px;color:{COLORS['text_faint']}'>/{n_total}</span>",
            x=0.5, y=0.5, font=dict(size=17, color=COLORS["text"], family="Inter, sans-serif"),
            showarrow=False,
        )],
    )
    return fig
