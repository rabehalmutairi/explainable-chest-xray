"""Design system for the chest X-ray dashboard.

Dark, glass-panel aesthetic inspired by modern radiology AI products
(Aidoc/Qure.ai-style worklists) rather than a default light Streamlit page.
Cards are authored as raw HTML (`.xr-card`) rather than st.container(border=True):
Streamlit's native bordered container has no stable CSS hook to restyle (the
wrapper class is a build-hashed emotion class, verified against the installed
JS bundle), so custom markup is the only reliable way to get the glass look.
`[data-testid="stFileUploaderDropzone"]` IS a stable, real hook - confirmed
in the installed Streamlit JS bundle - so the uploader is safely restyled.
"""

from __future__ import annotations

COLORS = {
    "bg": "#0A0F1C",
    "bg_soft": "#0D1526",
    "surface": "rgba(255,255,255,0.045)",
    "surface_hover": "rgba(255,255,255,0.07)",
    "surface_solid": "#131B2E",
    "border": "rgba(255,255,255,0.09)",
    "border_strong": "rgba(255,255,255,0.18)",
    "text": "#F1F5F9",
    "text_muted": "#94A3B8",
    "text_faint": "#64748B",
    "primary": "#22D3EE",
    "primary_deep": "#0891B2",
    "accent": "#34D399",
    "amber": "#FBBF24",
    "amber_text": "#FDE68A",
    "amber_soft": "rgba(251,191,36,0.12)",
    "amber_border": "rgba(251,191,36,0.35)",
    "green": "#34D399",
    "green_soft": "rgba(52,211,153,0.12)",
    "green_border": "rgba(52,211,153,0.35)",
    "red_muted": "#F87171",
}


def inject_base_css() -> str:
    c = COLORS
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

@keyframes gradientShift {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulseDot {{
    0% {{ box-shadow: 0 0 0 0 rgba(52,211,153,0.55); }}
    70% {{ box-shadow: 0 0 0 8px rgba(52,211,153,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(52,211,153,0); }}
}}
@keyframes scanSweep {{
    0% {{ top: 0%; opacity: 0.9; }}
    50% {{ opacity: 1; }}
    100% {{ top: 100%; opacity: 0.9; }}
}}
@keyframes shimmer {{
    0% {{ background-position: -400px 0; }}
    100% {{ background-position: 400px 0; }}
}}

.stApp {{
    background:
        radial-gradient(circle at 12% 8%, rgba(34,211,238,0.10), transparent 42%),
        radial-gradient(circle at 88% 82%, rgba(52,211,153,0.09), transparent 46%),
        radial-gradient(circle at 50% 50%, rgba(99,102,241,0.05), transparent 60%),
        linear-gradient(160deg, #0A0F1C 0%, #0D1526 45%, #0A0F1C 100%);
    background-size: 180% 180%;
    animation: gradientShift 24s ease infinite;
}}

.block-container {{
    padding-top: 1.25rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{background: transparent;}}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0B1220 0%, #0A0E1A 100%);
    border-right: 1px solid {c['border']};
}}

/* ---------- Hero header ---------- */
.xr-hero {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 1.6rem; border-radius: 18px; margin-bottom: 1.4rem;
    background: linear-gradient(120deg, rgba(34,211,238,0.10), rgba(52,211,153,0.06));
    border: 1px solid {c['border']};
    backdrop-filter: blur(20px);
    animation: fadeInUp 0.5s ease both;
}}
.xr-header-left {{ display: flex; align-items: center; gap: 0.9rem; }}
.xr-logo {{
    width: 46px; height: 46px; border-radius: 13px;
    background: linear-gradient(135deg, {c['primary']}, {c['accent']});
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    box-shadow: 0 4px 18px rgba(34,211,238,0.35);
}}
.xr-title {{ font-size: 1.3rem; font-weight: 800; color: #FFFFFF; line-height: 1.2; letter-spacing: -0.01em; }}
.xr-subtitle {{ font-size: 0.83rem; color: {c['text_muted']}; margin-top: 2px; }}
.xr-status-pill {{
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.4rem 0.9rem; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700; color: {c['text']};
    border: 1px solid {c['border']}; background: rgba(255,255,255,0.04);
}}
.xr-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.xr-dot-ok {{ background: #34D399; animation: pulseDot 2s infinite; }}
.xr-dot-bad {{ background: {c['red_muted']}; }}

/* ---------- Landing hero ---------- */
.xr-landing {{
    text-align: center; padding: 2.2rem 1rem 1.4rem;
    animation: fadeInUp 0.6s ease both;
}}
.xr-landing-eyebrow {{
    display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
    color: {c['primary']}; text-transform: uppercase; margin-bottom: 0.9rem;
    padding: 0.3rem 0.8rem; border-radius: 999px; border: 1px solid rgba(34,211,238,0.3);
    background: rgba(34,211,238,0.08);
}}
.xr-landing-title {{
    font-size: 2.6rem; font-weight: 900; color: #FFFFFF; line-height: 1.12;
    letter-spacing: -0.02em; margin-bottom: 0.9rem;
}}
.xr-landing-title span {{
    background: linear-gradient(120deg, {c['primary']}, {c['accent']});
    -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.xr-landing-desc {{
    font-size: 1rem; color: {c['text_muted']}; max-width: 480px; margin: 0 auto;
}}

/* ---------- Glass card ---------- */
.xr-card {{
    background: {c['surface']};
    border: 1px solid {c['border']};
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    backdrop-filter: blur(14px);
    transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeInUp 0.45s ease both;
}}
.xr-card:hover {{
    border-color: {c['border_strong']};
    box-shadow: 0 8px 28px rgba(0,0,0,0.25);
}}
.xr-card-title {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: {c['text_muted']}; margin-bottom: 0.6rem;
}}

/* ---------- Section titles ---------- */
.xr-section-title {{
    font-size: 1.02rem; font-weight: 700; color: {c['text']};
    display: flex; align-items: center; gap: 0.5rem; margin: 0 0 0.15rem 0;
}}
.xr-section-desc {{ font-size: 0.82rem; color: {c['text_muted']}; margin-bottom: 0.9rem; }}

/* ---------- Stat cards ---------- */
.xr-stat-value {{ font-size: 1.65rem; font-weight: 800; color: {c['text']}; line-height: 1.1; }}
.xr-stat-label {{ font-size: 0.76rem; color: {c['text_muted']}; margin-top: 0.3rem; font-weight: 500; }}
.xr-stat-sub {{ font-size: 0.78rem; color: {c['text_faint']}; margin-top: 0.15rem; font-weight: 600; }}
.xr-stat-icon {{
    width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center;
    justify-content: center; margin-bottom: 0.6rem;
}}

/* ---------- Badges ---------- */
.xr-badge {{
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.22rem 0.65rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.01em; border: 1px solid transparent;
}}
.xr-badge-amber {{ background: {c['amber_soft']}; color: {c['amber_text']}; border-color: {c['amber_border']}; }}
.xr-badge-green {{ background: {c['green_soft']}; color: {c['green']}; border-color: {c['green_border']}; }}
.xr-badge-muted {{ background: rgba(255,255,255,0.05); color: {c['text_faint']}; border-color: {c['border']}; }}

/* ---------- Finding rows ---------- */
.xr-finding-name {{ font-size: 0.92rem; font-weight: 700; color: {c['text']}; display: flex; align-items: center; gap: 0.5rem; }}
.xr-finding-meta {{ font-size: 0.72rem; color: {c['text_faint']}; margin-top: 2px; margin-left: 1.6rem; }}

/* ---------- Explain note ---------- */
.xr-explain-note {{
    font-size: 0.78rem; color: {c['text_muted']}; font-style: italic;
    padding: 0.55rem 0.8rem; background: rgba(255,255,255,0.03);
    border-radius: 10px; border: 1px solid {c['border']}; margin-top: 0.6rem;
}}

/* ---------- Footer disclaimer ---------- */
.xr-footer {{
    text-align: center; font-size: 0.72rem; color: {c['text_faint']};
    padding: 1.5rem 1rem 0.5rem; opacity: 0.75;
}}

/* ---------- Empty / offline states ---------- */
.xr-empty {{
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 2.2rem 1.5rem; text-align: center;
    background: {c['surface']}; border: 1px dashed {c['border_strong']}; border-radius: 16px;
}}
.xr-empty-icon {{
    width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center;
    justify-content: center; margin-bottom: 0.8rem; color: {c['primary']};
    background: rgba(34,211,238,0.08); border: 1px solid rgba(34,211,238,0.25);
}}
.xr-empty-title {{ font-size: 1rem; font-weight: 700; color: {c['text']}; margin-bottom: 0.3rem; }}
.xr-empty-desc {{ font-size: 0.85rem; color: {c['text_muted']}; max-width: 360px; }}

/* ---------- Scan animation (loading overlay on the X-ray) ---------- */
.xr-scan-wrap {{
    position: relative; width: 100%; border-radius: 14px; overflow: hidden;
    border: 1px solid {c['border']};
}}
.xr-scan-wrap img {{ width: 100%; display: block; filter: brightness(0.85); }}
.xr-scan-line {{
    position: absolute; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, {c['primary']}, {c['accent']}, transparent);
    box-shadow: 0 0 14px 2px rgba(34,211,238,0.7);
    animation: scanSweep 1.7s ease-in-out infinite;
}}
.xr-scan-caption {{
    text-align: center; font-size: 0.8rem; color: {c['primary']}; font-weight: 600;
    margin-top: 0.6rem; letter-spacing: 0.02em;
}}

/* ---------- Patient history (sidebar worklist) ---------- */
.xr-study-item {{
    padding: 0.6rem 0.7rem; border-radius: 10px; margin-bottom: 0.4rem;
    border: 1px solid transparent; transition: background 0.15s ease;
}}
.xr-study-active {{ background: rgba(34,211,238,0.10); border-color: rgba(34,211,238,0.3); }}
.xr-study-name {{ font-size: 0.84rem; font-weight: 700; color: {c['text']}; }}
.xr-study-meta {{ font-size: 0.7rem; color: {c['text_faint']}; margin-top: 1px; }}

/* ---------- Streamlit widget theming ---------- */
[data-testid="stFileUploaderDropzone"] {{
    background: linear-gradient(160deg, rgba(34,211,238,0.06), rgba(52,211,153,0.03));
    border: 1.5px dashed rgba(34,211,238,0.35) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    transition: border-color 0.2s ease, background 0.2s ease;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: {c['primary']} !important;
    background: linear-gradient(160deg, rgba(34,211,238,0.12), rgba(52,211,153,0.06));
}}
[data-testid="stFileUploaderDropzoneInstructions"] svg {{ color: {c['primary']}; }}

.stButton > button {{
    border-radius: 10px; font-weight: 700; transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.stButton > button:hover {{ transform: translateY(-1px); }}
button[kind="primary"] {{
    background: linear-gradient(120deg, {c['primary']}, {c['primary_deep']}) !important;
    border: none !important; box-shadow: 0 6px 20px rgba(34,211,238,0.28);
}}
.stDownloadButton > button {{ border-radius: 10px; font-weight: 700; }}
[data-testid="stMetric"] {{
    background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 12px;
    padding: 0.7rem 0.9rem;
}}
div[data-testid="stSegmentedControl"] label {{ font-weight: 600; }}
hr {{ border-color: {c['border']}; margin: 1.25rem 0; }}
[data-testid="stExpander"] {{
    background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 14px;
}}
</style>
"""
