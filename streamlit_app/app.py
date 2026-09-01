"""Chest X-Ray AI Analysis - Streamlit frontend.

Run:
    streamlit run streamlit_app/app.py

This app only calls the FastAPI backend (streamlit_app/api_client.py) and
renders its responses. No preprocessing, inference, or Grad-CAM logic lives
here - see src/inference.py and src/gradcam.py for that.

State model: each analyzed image becomes a "study" (session-scoped worklist
entry with optional patient name/ID/age - display-only, never sent to the
backend, which only ever receives image bytes). Switching studies in the
sidebar re-renders a previous result instantly from session_state; nothing
is re-uploaded unless the user picks a new Grad-CAM pathology that wasn't
already cached for that study.

Research prototype - not for clinical use (see footer).
"""

from __future__ import annotations

import base64
import sys
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit_app import api_client, components, report
from streamlit_app.icons import PATHOLOGY_ICON_NAME, icon  # noqa: F401
from streamlit_app.charts import probability_bar_chart
from streamlit_app.style import COLORS, inject_base_css

st.set_page_config(
    page_title="Radiograph Intelligence",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(inject_base_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------- state ----
def _init_state() -> None:
    st.session_state.setdefault("studies", {})
    st.session_state.setdefault("active_study_id", None)
    st.session_state.setdefault("pending_image_bytes", None)
    st.session_state.setdefault("pending_filename", None)


_init_state()


def _clear_patient_form() -> None:
    for k in ("pf_name", "pf_id", "pf_age"):
        st.session_state.pop(k, None)


def _new_study() -> None:
    st.session_state["active_study_id"] = None
    st.session_state["pending_image_bytes"] = None
    st.session_state["pending_filename"] = None
    _clear_patient_form()


# --------------------------------------------------------------- backend ----
health = api_client.check_health()
model_info = api_client.get_model_info() if health and health.get("model_loaded") else None

components.render_header(health)

with st.sidebar:
    if st.button("New Study", width="stretch", type="primary", icon=":material/add:"):
        _new_study()
        st.rerun()
    st.divider()
    clicked_study = components.render_sidebar_studies(
        st.session_state["studies"], st.session_state["active_study_id"]
    )
    if clicked_study:
        st.session_state["active_study_id"] = clicked_study
        st.session_state["pending_image_bytes"] = None
        st.rerun()

if health is None or not health.get("model_loaded"):
    components.render_backend_offline(health.get("error") if health else None)
    st.stop()


# ============================================================ VIEW STUDY ====
active_id = st.session_state["active_study_id"]
studies = st.session_state["studies"]

if active_id and active_id in studies:
    study = studies[active_id]
    result = study["predict_result"]

    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        patient_line = study["patient_name"] or "Unnamed patient"
        details = []
        if study["patient_id"]:
            details.append(f"ID {study['patient_id']}")
        if study["patient_age"] is not None:
            details.append(f"{int(study['patient_age'])} yrs")
        details.append(study["timestamp"].strftime("%b %d, %Y · %H:%M"))
        st.markdown(
            f'<div class="xr-section-title" style="font-size:1.15rem;">{icon("user", 18, COLORS["primary"])} {patient_line}</div>'
            f'<div class="xr-section-desc">{"  ·  ".join(details)}</div>',
            unsafe_allow_html=True,
        )
    with hdr_r:
        st.write("")

    st.write("")
    components.render_stat_cards(result)
    st.write("")

    left, right = st.columns([2, 3], gap="large")

    with left:
        components.render_positive_findings(result)

    with right:
        st.markdown(
            f'<div class="xr-section-title">{icon("image", 17, COLORS["primary"])} Original &amp; Grad-CAM Explanation</div>',
            unsafe_allow_html=True,
        )

        all_labels = [p["pathology"] for p in result["predictions"]]
        positives = result["positive_findings"]
        ordered_labels = positives + [l for l in all_labels if l not in positives]

        current = study.get("selected_pathology") or ordered_labels[0]

        if positives:
            quick_pick = st.segmented_control(
                "Quick select",
                options=positives,
                default=current if current in positives else positives[0],
                format_func=lambda p: p,
                label_visibility="collapsed",
                key=f"quick_pick_{active_id}",
            )
            if quick_pick:
                current = quick_pick
                study["selected_pathology"] = quick_pick

        default_idx = ordered_labels.index(current) if current in ordered_labels else 0
        selected = st.selectbox(
            "Or explore any of the 14 classes",
            options=ordered_labels,
            index=default_idx,
            format_func=lambda p: p + ("  ⚠" if p in positives else ""),
            key=f"select_{active_id}",
        )
        study["selected_pathology"] = selected

        cache = study["gradcam_cache"]
        if selected not in cache:
            with st.spinner(f"Generating Grad-CAM for {selected}..."):
                try:
                    cache[selected] = api_client.get_gradcam(
                        study["image_bytes"], study["filename"], selected
                    )
                except api_client.ApiError as exc:
                    st.error(f"Grad-CAM failed: {exc}")
                    cache[selected] = None

        gc_result = cache.get(selected)

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(study["image_bytes"], caption="Original", width="stretch")
        with img_col2:
            if gc_result:
                st.image(gc_result.image_bytes, caption=f"Grad-CAM: {selected}", width="stretch")
            else:
                st.markdown(
                    '<div class="xr-empty" style="padding:1.2rem;"><div class="xr-empty-desc">Explanation unavailable</div></div>',
                    unsafe_allow_html=True,
                )

        if gc_result:
            badge = (
                '<span class="xr-badge xr-badge-amber">Above threshold</span>'
                if gc_result.above_threshold
                else '<span class="xr-badge xr-badge-muted">Below threshold</span>'
            )
            st.markdown(
                f"""
                <div style="display:flex; gap:0.6rem; align-items:center; margin-top:0.4rem;">
                    {badge}
                    <span style="font-size:0.85rem; color:{COLORS['text']};">
                        {selected}: <b>{gc_result.probability:.0%}</b> probability (threshold {gc_result.threshold:.0%})
                    </span>
                </div>
                <div class="xr-explain-note">{gc_result.note}</div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown(
        f'<div class="xr-section-title">{icon("bar-chart", 17, COLORS["primary"])} All 14 Pathology Probabilities</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="xr-section-desc">Amber bars crossed their own threshold (◆ marker) - thresholds are tuned '
        'per class, not fixed at 50%, so a low bar can still be flagged if it clears that class\'s cutoff. '
        'Sorted by model probability.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="xr-card">', unsafe_allow_html=True)
    st.plotly_chart(
        probability_bar_chart(result["predictions"]),
        config={"displayModeBar": False}, width="stretch", key=f"chart_{active_id}",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(f'<div class="xr-section-title">{icon("download", 17, COLORS["primary"])} Export</div>', unsafe_allow_html=True)
    exp1, exp2, _ = st.columns([1, 1, 2])
    with exp1:
        st.download_button(
            "Download JSON", icon=":material/download:",
            data=report.build_json_report(result, model_info or {}, study["filename"]),
            file_name=f"{Path(study['filename']).stem}_analysis.json",
            mime="application/json", width="stretch",
        )
    with exp2:
        gc_for_pdf = study["gradcam_cache"].get(study.get("selected_pathology"))
        try:
            pdf_bytes = report.build_pdf_report(
                result, model_info or {}, study["filename"], study["image_bytes"],
                gradcam_bytes=gc_for_pdf.image_bytes if gc_for_pdf else None,
                gradcam_pathology=study.get("selected_pathology"),
            )
            st.download_button(
                "Download PDF Report", icon=":material/picture_as_pdf:",
                data=pdf_bytes,
                file_name=f"{Path(study['filename']).stem}_report.pdf",
                mime="application/pdf", width="stretch",
            )
        except Exception as exc:
            st.caption(f"PDF export unavailable: {exc}")

# ============================================================ NEW STUDY ====
else:
    if st.session_state["pending_image_bytes"] is None:
        components.render_landing_hero()
        st.markdown(f'<div style="text-align:center; color:{COLORS["text_faint"]}; font-size:0.8rem; margin-bottom:0.6rem;">PNG · JPEG · BMP · TIFF, up to 20 MB</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload a chest radiograph", type=["png", "jpg", "jpeg", "bmp", "tiff"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.session_state["pending_image_bytes"] = uploaded.getvalue()
            st.session_state["pending_filename"] = uploaded.name
            st.rerun()

    else:
        img_bytes = st.session_state["pending_image_bytes"]
        filename = st.session_state["pending_filename"]

        st.markdown(f'<div class="xr-section-title">{icon("image", 17, COLORS["primary"])} New Study</div>', unsafe_allow_html=True)
        prev_col, info_col = st.columns([1, 2], gap="large")
        with prev_col:
            st.image(img_bytes, caption=filename, width="stretch")
            if st.button("Choose a different image", width="stretch", type="tertiary",
                         icon=":material/refresh:"):
                _new_study()
                st.rerun()

        with info_col:
            st.markdown('<div class="xr-card">', unsafe_allow_html=True)
            patient = components.render_patient_form()
            st.markdown('</div>', unsafe_allow_html=True)
            st.write("")

            analyze_clicked = st.button(
                "Analyze X-Ray", type="primary", width="stretch", icon=":material/monitor_heart:"
            )

            if analyze_clicked:
                b64 = base64.b64encode(img_bytes).decode()
                scan_slot = st.empty()
                with scan_slot.container():
                    st.markdown(
                        f"""
                        <div class="xr-scan-wrap">
                            <img src="data:image/png;base64,{b64}"/>
                            <div class="xr-scan-line"></div>
                        </div>
                        <div class="xr-scan-caption">Analyzing radiograph across 14 pathology classes…</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    try:
                        result = api_client.predict(img_bytes, filename)
                    except api_client.ApiError as exc:
                        scan_slot.empty()
                        st.error(f"Analysis failed: {exc}")
                        result = None

                if result is not None:
                    scan_slot.empty()
                    positives = result["positive_findings"]
                    sid = str(uuid.uuid4())[:8]
                    st.session_state["studies"][sid] = {
                        "patient_name": patient["name"],
                        "patient_id": patient["patient_id"],
                        "patient_age": patient["age"],
                        "filename": filename,
                        "image_bytes": img_bytes,
                        "timestamp": datetime.now(),
                        "predict_result": result,
                        "gradcam_cache": {},
                        "selected_pathology": positives[0] if positives else result["predictions"][0]["pathology"],
                        "top_finding": positives[0] if positives else "No Finding",
                    }
                    st.session_state["active_study_id"] = sid
                    st.session_state["pending_image_bytes"] = None
                    st.session_state["pending_filename"] = None
                    _clear_patient_form()
                    st.rerun()

components.render_footer(model_info)
