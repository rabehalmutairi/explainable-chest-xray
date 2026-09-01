"""Reusable render helpers for the chest X-ray dashboard.

Pure display content (badges, finding rows, cards) is rendered as raw HTML
via st.markdown - see style.py's module docstring for why st.container(
border=True) is deliberately avoided for anything that needs custom styling.
Interaction (uploader, buttons, form inputs, segmented control) uses native
Streamlit widgets, themed via .streamlit/config.toml + style.py.

Model architecture/checkpoint details are intentionally NOT surfaced in the
main clinical view - they live in a collapsed footer expander only, per
product direction: a radiologist reviewing findings shouldn't see ML
internals front and center.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from streamlit_app.icons import icon, pathology_icon, ring_svg
from streamlit_app.style import COLORS

DISCLAIMER = (
    "Educational research prototype (Samsung Innovation Campus AI capstone). "
    "Not a certified medical device and not intended for clinical decision-making."
)


def render_header(health: dict[str, Any] | None) -> None:
    online = health is not None and health.get("model_loaded")
    dot_class = "xr-dot-ok" if online else "xr-dot-bad"
    status_text = "System online" if online else "Backend unavailable"

    st.markdown(
        f"""
        <div class="xr-hero">
            <div class="xr-header-left">
                <div class="xr-logo">{icon('layers', size=24, color='white')}</div>
                <div>
                    <div class="xr-title">Radiograph Intelligence</div>
                    <div class="xr-subtitle">AI-assisted chest X-ray screening &amp; explainability</div>
                </div>
            </div>
            <div class="xr-status-pill">
                <span class="xr-dot {dot_class}"></span>{status_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing_hero() -> None:
    st.markdown(
        f"""
        <div class="xr-landing">
            <div class="xr-landing-eyebrow">AI-Assisted Radiograph Screening</div>
            <div class="xr-landing-title">Fourteen findings.<br/><span>One glance.</span></div>
            <div class="xr-landing-desc">
                Upload a chest radiograph to screen for thoracic pathologies and see exactly
                which regions the model attended to for every finding.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_backend_offline(error: str | None) -> None:
    st.markdown(
        f"""
        <div class="xr-empty">
            <div class="xr-empty-icon">{icon('alert-triangle', size=22)}</div>
            <div class="xr-empty-title">Analysis backend is not reachable</div>
            <div class="xr-empty-desc">
                Start the API with <code>uvicorn app.main:app</code>, then reload this page.
                {f'<br><br><span style="color:{COLORS["text_faint"]}">{error}</span>' if error else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_patient_form() -> dict[str, Any]:
    """Optional demo metadata, display-only. Never sent to the backend -
    the API receives image bytes only."""
    st.markdown(f'<div class="xr-card-title">{icon("user", 14)}&nbsp; Patient Context (optional)</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Name", placeholder="e.g. Jane Doe", label_visibility="collapsed", key="pf_name")
    with c2:
        pid = st.text_input("Patient ID", placeholder="Patient ID", label_visibility="collapsed", key="pf_id")
    with c3:
        age = st.number_input("Age", min_value=0, max_value=130, value=None, placeholder="Age",
                               label_visibility="collapsed", key="pf_age")
    return {"name": name.strip(), "patient_id": pid.strip(), "age": age}


def render_sidebar_studies(studies: dict[str, dict], active_id: str | None) -> str | None:
    """Renders the patient study worklist. Returns a study_id if the user
    clicked one, else None."""
    st.markdown(f'##### {icon("clock", 15)}&nbsp; Study History', unsafe_allow_html=True)

    if not studies:
        st.caption("Analyzed studies appear here for this session.")
        return None

    clicked = None
    for sid, study in sorted(studies.items(), key=lambda kv: kv[1]["timestamp"], reverse=True):
        label = study["patient_name"] or "Unnamed patient"
        meta_bits = []
        if study["patient_id"]:
            meta_bits.append(f"ID {study['patient_id']}")
        if study["patient_age"] is not None:
            meta_bits.append(f"{int(study['patient_age'])}y")
        meta_bits.append(study["timestamp"].strftime("%H:%M"))
        finding = study["top_finding"]

        is_active = sid == active_id
        with st.container(border=is_active):
            if st.button(
                label,
                key=f"study_btn_{sid}",
                width="stretch",
                type="secondary" if is_active else "tertiary",
                icon=":material/check_circle:" if is_active else ":material/history:",
            ):
                clicked = sid
            st.caption(f"{' · '.join(meta_bits)}  ·  {finding}")
    return clicked


def render_stat_cards(result: dict[str, Any]) -> None:
    """Each card is a single st.markdown call producing one complete
    <div class="xr-card">...</div>. Streamlit cannot nest a live widget
    inside markdown-rendered HTML (every st.* call is its own top-level DOM
    node, not a child of prior markdown), so the "above threshold" ring is a
    pure-SVG helper (icons.ring_svg), not a Plotly widget."""
    predictions = result["predictions"]
    positives = result["positive_findings"]
    top = max(predictions, key=lambda p: p["probability"]) if predictions else None
    n_total = len(predictions) or 1

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        ring = ring_svg(len(positives) / n_total, str(len(positives)),
                         color=COLORS["amber"] if positives else "rgba(255,255,255,0.25)")
        st.markdown(
            f"""<div class="xr-card" style="display:flex; align-items:center; gap:0.9rem;">
                {ring}
                <div><div class="xr-stat-value">{len(positives)}</div>
                <div class="xr-stat-label">Above threshold</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

    with c2:
        top_icon = pathology_icon(top["pathology"], 17) if top else icon("search", 17)
        top_prob_str = f"{top['probability']:.0%}" if top else "—"
        st.markdown(
            f"""<div class="xr-card">
                <div class="xr-stat-icon" style="background:{COLORS['amber_soft']};color:{COLORS['amber_text']};">{top_icon}</div>
                <div class="xr-stat-value">{top_prob_str}</div>
                <div class="xr-stat-label">Highest probability</div>
                <div class="xr-stat-sub">{top['pathology'] if top else '—'}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with c3:
        if result["no_finding"]:
            st.markdown(
                f"""<div class="xr-card">
                    <div class="xr-stat-icon" style="background:{COLORS['green_soft']};color:{COLORS['green']};">{icon('check-circle', 17)}</div>
                    <div class="xr-stat-value" style="color:{COLORS['green']};">Clear</div>
                    <div class="xr-stat-label">Overall status</div>
                    <div class="xr-stat-sub">No finding above threshold</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            n = len(positives)
            st.markdown(
                f"""<div class="xr-card">
                    <div class="xr-stat-icon" style="background:{COLORS['amber_soft']};color:{COLORS['amber_text']};">{icon('alert-triangle', 17)}</div>
                    <div class="xr-stat-value" style="color:{COLORS['amber_text']};">Review</div>
                    <div class="xr-stat-label">Overall status</div>
                    <div class="xr-stat-sub">{n} finding{'s' if n != 1 else ''} flagged</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with c4:
        st.markdown(
            f"""<div class="xr-card">
                <div class="xr-stat-icon" style="background:rgba(34,211,238,0.10);color:{COLORS['primary']};">{icon('activity', 17)}</div>
                <div class="xr-stat-value">{result.get('inference_ms', 0):.0f}<span style="font-size:0.9rem;">ms</span></div>
                <div class="xr-stat-label">Inference time</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_positive_findings(result: dict[str, Any]) -> None:
    positives = [p for p in result["predictions"] if p["above_threshold"]]

    st.markdown(
        f'<div class="xr-section-title">{icon("alert-triangle", 17, COLORS["amber"])} Findings above threshold</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="xr-section-desc">Each pathology has its own validation-tuned threshold, not a fixed 50% '
        f'cutoff. Rare, hard-to-detect findings intentionally use a low threshold so the model still catches '
        f'them - a modest probability crossing a low threshold is still a genuine flag, not noise.</div>',
        unsafe_allow_html=True,
    )

    if not positives:
        st.markdown(
            f"""
            <div class="xr-card">
                <span class="xr-badge xr-badge-green">{icon('check-circle', 13)} No Finding</span>
                <div style="font-size:0.82rem; color:{COLORS['text_muted']}; margin-top:0.6rem;">
                    None of the 14 pathology outputs crossed its threshold for this image.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    rows = []
    for i, p in enumerate(positives):
        border = f"border-bottom:1px solid {COLORS['border']};" if i < len(positives) - 1 else ""
        # Single physical line per row - see icons.ring_svg for why: joining
        # multi-line fragments creates a whitespace-only line at each seam,
        # which breaks Markdown's raw-HTML block after the first row.
        rows.append(
            f'<div style="display:flex; align-items:center; justify-content:space-between; padding:0.55rem 0; {border}">'
            f'<div><div class="xr-finding-name">{pathology_icon(p["pathology"], 16, COLORS["amber_text"])} {p["pathology"]}</div>'
            f'<div class="xr-finding-meta">{p["probability"]:.0%} probability crosses this class\'s {p["threshold"]:.0%} threshold</div></div>'
            f'<span class="xr-badge xr-badge-amber">{p["probability"]:.0%}</span>'
            f'</div>'
        )
    st.markdown(f'<div class="xr-card">{"".join(rows)}</div>', unsafe_allow_html=True)


def render_footer(model_info: dict[str, Any] | None) -> None:
    st.markdown(f'<div class="xr-footer">{DISCLAIMER}</div>', unsafe_allow_html=True)
    if model_info:
        with st.expander("System diagnostics"):
            auc = model_info.get("val_macro_auc")
            st.caption(
                f"{model_info.get('model_version')} · epoch {model_info.get('checkpoint_epoch')} · "
                f"val AUROC {f'{auc:.4f}' if auc else '—'} · "
                f"explainability layer {model_info.get('target_layer')}"
            )
